#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Create and verify strict manifests for CWAM serving checkpoints.

The manifest binds a canonical absolute checkpoint path to the exact regular
file tree below it. Verification always enumerates the live tree and computes
the full SHA-256 of every file; the aggregate tree digest is not a substitute
for those per-file checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_KIND = "cwam_serving_checkpoint_manifest"
HASH_CHUNK_BYTES = 8 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "checkpoint_path",
        "files",
        "tree_sha256",
    }
)
_FILE_KEYS = frozenset({"relative_path", "size_bytes", "sha256"})


class ServingCheckpointManifestError(RuntimeError):
    """Raised when a serving checkpoint manifest contract is violated."""


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _reject_incomplete_name(name: str, label: str) -> None:
    if ".incomplete" in name:
        raise ServingCheckpointManifestError(f"{label} contains a forbidden .incomplete name: {name!r}")


def _require_directory(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ServingCheckpointManifestError(f"{label} does not exist: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ServingCheckpointManifestError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ServingCheckpointManifestError(f"{label} must be a directory: {path}")
    return metadata


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ServingCheckpointManifestError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ServingCheckpointManifestError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ServingCheckpointManifestError(f"{label} must be a regular file: {path}")
    return metadata


def _resolve_checkpoint_root(checkpoint_path: str | os.PathLike[str]) -> Path:
    raw_path = os.fspath(checkpoint_path)
    if not raw_path:
        raise ServingCheckpointManifestError("checkpoint path must not be empty")

    lexical_path = Path(os.path.abspath(os.path.expanduser(raw_path)))
    try:
        resolved_path = lexical_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ServingCheckpointManifestError(f"checkpoint path does not exist: {lexical_path}") from error

    if lexical_path != resolved_path:
        raise ServingCheckpointManifestError(
            "checkpoint path must be canonical and contain no symlink components: "
            f"provided={lexical_path}, resolved={resolved_path}"
        )
    for component in resolved_path.parts:
        _reject_incomplete_name(component, "checkpoint path")
    _require_directory(resolved_path, "checkpoint root")
    return resolved_path


def _hash_regular_file(path: Path) -> tuple[int, str]:
    before = _require_regular_file(path, "checkpoint entry")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ServingCheckpointManifestError(f"cannot safely open checkpoint file {path}: {error}") from error

    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ServingCheckpointManifestError(f"checkpoint entry must be a regular file: {path}")
        if _stable_identity(before) != _stable_identity(opened):
            raise ServingCheckpointManifestError(f"checkpoint file changed before hashing: {path}")

        while chunk := os.read(descriptor, HASH_CHUNK_BYTES):
            digest.update(chunk)

        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    after_path = _require_regular_file(path, "checkpoint entry")
    identity = _stable_identity(before)
    if identity != _stable_identity(after_read) or identity != _stable_identity(after_path):
        raise ServingCheckpointManifestError(f"checkpoint file changed while hashing: {path}")
    return int(before.st_size), digest.hexdigest()


def _relative_path(base: PurePosixPath, name: str) -> PurePosixPath:
    _reject_incomplete_name(name, "checkpoint entry")
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ServingCheckpointManifestError(f"checkpoint entry has a non-canonical name: {name!r}")
    if unicodedata.normalize("NFC", name) != name:
        raise ServingCheckpointManifestError(f"checkpoint entry name must use NFC normalization: {name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ServingCheckpointManifestError(f"checkpoint entry contains a control character: {name!r}")
    return base / name


def _scan_checkpoint_tree(root: Path) -> list[dict[str, str | int]]:
    root_before = _require_directory(root, "checkpoint root")
    records: list[dict[str, str | int]] = []
    pending: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath())]

    while pending:
        directory, relative_base = pending.pop()
        directory_before = _require_directory(directory, "checkpoint directory")
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise ServingCheckpointManifestError(
                f"cannot enumerate checkpoint directory {directory}: {error}"
            ) from error

        child_directories: list[tuple[Path, PurePosixPath]] = []
        for entry in entries:
            relative = _relative_path(relative_base, entry.name)
            entry_path = directory / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError as error:
                raise ServingCheckpointManifestError(
                    f"checkpoint entry disappeared during scan: {entry_path}"
                ) from error
            except OSError as error:
                raise ServingCheckpointManifestError(
                    f"cannot inspect checkpoint entry {entry_path}: {error}"
                ) from error

            if stat.S_ISLNK(metadata.st_mode):
                raise ServingCheckpointManifestError(f"checkpoint tree must not contain symlinks: {entry_path}")
            if stat.S_ISDIR(metadata.st_mode):
                child_directories.append((entry_path, relative))
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ServingCheckpointManifestError(f"checkpoint tree contains a special file: {entry_path}")

            size_bytes, sha256 = _hash_regular_file(entry_path)
            records.append(
                {
                    "relative_path": relative.as_posix(),
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                }
            )

        directory_after = _require_directory(directory, "checkpoint directory")
        if _stable_identity(directory_before) != _stable_identity(directory_after):
            raise ServingCheckpointManifestError(f"checkpoint directory changed during scan: {directory}")
        pending.extend(reversed(child_directories))

    root_after = _require_directory(root, "checkpoint root")
    if _stable_identity(root_before) != _stable_identity(root_after):
        raise ServingCheckpointManifestError(f"checkpoint root changed during scan: {root}")
    if not records:
        raise ServingCheckpointManifestError(f"checkpoint tree must contain at least one regular file: {root}")

    records.sort(key=lambda record: str(record["relative_path"]))
    return records


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted((repr(key) for key in observed - expected))
        raise ServingCheckpointManifestError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ServingCheckpointManifestError(f"{label} must be a lowercase SHA-256, got {value!r}")
    return value


def _canonical_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ServingCheckpointManifestError(f"{label} must be a non-empty string")
    if "\x00" in value or "\\" in value:
        raise ServingCheckpointManifestError(f"{label} is not a canonical POSIX relative path: {value!r}")
    if unicodedata.normalize("NFC", value) != value:
        raise ServingCheckpointManifestError(f"{label} must use NFC normalization: {value!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ServingCheckpointManifestError(f"{label} contains a control character: {value!r}")

    path = PurePosixPath(value)
    if path.is_absolute() or any(component in {"", ".", ".."} for component in path.parts):
        raise ServingCheckpointManifestError(f"{label} is not a canonical relative path: {value!r}")
    if path.as_posix() != value:
        raise ServingCheckpointManifestError(f"{label} is not normalized: {value!r}")
    for component in path.parts:
        _reject_incomplete_name(component, label)
    return value


def _tree_sha256(files: list[dict[str, str | int]]) -> str:
    payload = json.dumps(
        files,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_manifest(
    manifest: Mapping[str, Any],
    *,
    checkpoint_path: str,
) -> tuple[list[dict[str, str | int]], str]:
    _require_exact_keys(manifest, _MANIFEST_KEYS, "manifest")

    schema_version = manifest["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ServingCheckpointManifestError(
            f"manifest schema_version must be {SCHEMA_VERSION}, got {schema_version!r}"
        )
    if manifest["kind"] != MANIFEST_KIND:
        raise ServingCheckpointManifestError(f"manifest kind must be {MANIFEST_KIND!r}, got {manifest['kind']!r}")
    if manifest["checkpoint_path"] != checkpoint_path:
        raise ServingCheckpointManifestError(
            "manifest checkpoint_path does not match the exact resolved checkpoint path: "
            f"expected={checkpoint_path!r}, actual={manifest['checkpoint_path']!r}"
        )

    raw_files = manifest["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ServingCheckpointManifestError("manifest files must be a non-empty JSON array")

    files: list[dict[str, str | int]] = []
    seen_paths: set[str] = set()
    for index, raw_record in enumerate(raw_files):
        if not isinstance(raw_record, Mapping):
            raise ServingCheckpointManifestError(f"manifest files[{index}] must be a JSON object")
        _require_exact_keys(raw_record, _FILE_KEYS, f"manifest files[{index}]")
        relative_path = _canonical_relative_path(
            raw_record["relative_path"],
            f"manifest files[{index}].relative_path",
        )
        if relative_path in seen_paths:
            raise ServingCheckpointManifestError(f"manifest contains duplicate relative_path: {relative_path!r}")
        seen_paths.add(relative_path)

        size_bytes = raw_record["size_bytes"]
        if type(size_bytes) is not int or size_bytes < 0:
            raise ServingCheckpointManifestError(
                f"manifest files[{index}].size_bytes must be a non-negative integer, got {size_bytes!r}"
            )
        sha256 = _require_sha256(raw_record["sha256"], f"manifest files[{index}].sha256")
        files.append(
            {
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )

    paths = [str(record["relative_path"]) for record in files]
    if paths != sorted(paths):
        raise ServingCheckpointManifestError("manifest files must be sorted by relative_path")

    expected_tree_sha = _require_sha256(manifest["tree_sha256"], "manifest tree_sha256")
    computed_tree_sha = _tree_sha256(files)
    if computed_tree_sha != expected_tree_sha:
        raise ServingCheckpointManifestError(
            f"manifest tree_sha256 mismatch: expected={expected_tree_sha}, computed={computed_tree_sha}"
        )
    return files, expected_tree_sha


def create_serving_checkpoint_manifest(
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return a strict manifest for the exact checkpoint tree."""

    root = _resolve_checkpoint_root(checkpoint_path)
    files = _scan_checkpoint_tree(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "checkpoint_path": str(root),
        "files": files,
        "tree_sha256": _tree_sha256(files),
    }


def verify_serving_checkpoint_manifest(
    manifest: Mapping[str, Any],
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, str | int]:
    """Verify path identity, exact file set, sizes, and full file checksums."""

    root = _resolve_checkpoint_root(checkpoint_path)
    expected_files, expected_tree_sha = _validated_manifest(manifest, checkpoint_path=str(root))
    actual_files = _scan_checkpoint_tree(root)

    expected_by_path = {str(record["relative_path"]): record for record in expected_files}
    actual_by_path = {str(record["relative_path"]): record for record in actual_files}
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    extra = sorted(set(actual_by_path) - set(expected_by_path))
    if missing or extra:
        raise ServingCheckpointManifestError(f"checkpoint file set mismatch: missing={missing}, extra={extra}")

    for relative_path in sorted(expected_by_path):
        expected = expected_by_path[relative_path]
        actual = actual_by_path[relative_path]
        if actual["size_bytes"] != expected["size_bytes"]:
            raise ServingCheckpointManifestError(
                f"checkpoint size mismatch for {relative_path}: "
                f"expected={expected['size_bytes']}, actual={actual['size_bytes']}"
            )
        if actual["sha256"] != expected["sha256"]:
            raise ServingCheckpointManifestError(
                f"checkpoint SHA-256 mismatch for {relative_path}: "
                f"expected={expected['sha256']}, actual={actual['sha256']}"
            )

    actual_tree_sha = _tree_sha256(actual_files)
    if actual_tree_sha != expected_tree_sha:
        raise ServingCheckpointManifestError(
            f"checkpoint tree_sha256 mismatch: expected={expected_tree_sha}, actual={actual_tree_sha}"
        )
    return {
        "status": "passed",
        "checkpoint_path": str(root),
        "file_count": len(actual_files),
        "tree_sha256": actual_tree_sha,
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ServingCheckpointManifestError(f"manifest JSON contains a duplicate object key: {key!r}")
        value[key] = item
    return value


def load_serving_checkpoint_manifest(manifest_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load JSON while rejecting symlinked, special, incomplete, or duplicate-key manifests."""

    path = Path(os.path.abspath(os.path.expanduser(os.fspath(manifest_path))))
    for component in path.parts:
        _reject_incomplete_name(component, "manifest path")
    _require_regular_file(path, "manifest file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ServingCheckpointManifestError(f"cannot load manifest JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ServingCheckpointManifestError("manifest JSON root must be an object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="print a canonical checkpoint manifest")
    create.add_argument("--checkpoint-path", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify a checkpoint against a manifest")
    verify.add_argument("--checkpoint-path", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            value = create_serving_checkpoint_manifest(args.checkpoint_path)
        else:
            manifest = load_serving_checkpoint_manifest(args.manifest)
            value = verify_serving_checkpoint_manifest(manifest, args.checkpoint_path)
    except ServingCheckpointManifestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
