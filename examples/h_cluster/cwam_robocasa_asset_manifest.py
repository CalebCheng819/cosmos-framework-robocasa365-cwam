#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strictly verify a published RoboCasa asset tree and its provenance.

The publisher stores a JSON receipt and a JSONL payload manifest below
``.cwam_provenance``.  This helper treats both files as untrusted input.  It
requires their exact version-1 schemas, binds them to the canonical asset root
and the fixed top-level exclusion policy, and then enumerates the live payload.
Every live regular file is read in full and SHA-256 hashed.  Symlinks, special
files, ``.incomplete`` paths, missing entries, extra entries, and all
type/mode/size/hash drift fail verification.

The returned dictionary is a deterministic verification receipt intended to be
embedded directly in the formal-evaluation runtime receipt.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
import uuid
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_KIND = "cwam_robocasa_asset_tree_manifest"
PUBLISHER_RECEIPT_KIND = "cwam_robocasa_asset_tree_receipt"
VERIFICATION_RECEIPT_KIND = "cwam_robocasa_asset_tree_verification_receipt"
GENERATOR_ID = "cwam_robocasa_asset_tree_manifest_v1_20260729"

PROVENANCE_DIRECTORY = ".cwam_provenance"
MANIFEST_RELATIVE_PATH = ".cwam_provenance/asset-tree-manifest.jsonl"
PUBLISHER_RECEIPT_RELATIVE_PATH = ".cwam_provenance/asset-tree-receipt.json"
EXCLUDED_TOP_LEVEL_PATHS = (".cwam_provenance", ".git")
PROVENANCE_FILE_MODE = 0o444

HASH_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_HASH_WORKERS = 1
MAX_HASH_WORKERS = 64
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_LINE_BYTES = 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GENERATOR_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_RECEIPT_KEYS = frozenset(
    {
        "excluded_top_level_paths",
        "generated_at_utc",
        "generator_id",
        "git_commit",
        "git_status_porcelain",
        "kind",
        "manifest_relative_path",
        "manifest_sha256",
        "payload_directory_count",
        "payload_file_count",
        "payload_root",
        "payload_symlink_count",
        "payload_total_bytes",
        "schema_version",
        "source_archive_directory",
        "source_sha256sums_path",
        "source_sha256sums_sha256",
        "source_sizes_path",
        "source_sizes_sha256",
        "status",
    }
)
_MANIFEST_HEADER_KEYS = frozenset(
    {
        "excluded_top_level_paths",
        "kind",
        "payload_root",
        "schema_version",
    }
)
_DIRECTORY_RECORD_KEYS = frozenset({"mode", "path", "type"})
_FILE_RECORD_KEYS = frozenset({"mode", "path", "sha256", "size_bytes", "type"})


class RoboCasaAssetManifestError(RuntimeError):
    """Raised when the published asset-tree contract is violated."""


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _reject_incomplete_component(component: str, label: str) -> None:
    if ".incomplete" in component.lower():
        raise RoboCasaAssetManifestError(f"{label} contains a forbidden .incomplete component: {component!r}")


def _validate_component(component: str, label: str) -> None:
    if not component or component in {".", ".."} or "/" in component or "\\" in component:
        raise RoboCasaAssetManifestError(f"{label} contains a non-canonical component: {component!r}")
    if unicodedata.normalize("NFC", component) != component:
        raise RoboCasaAssetManifestError(f"{label} component must use NFC normalization: {component!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in component):
        raise RoboCasaAssetManifestError(f"{label} contains a control character: {component!r}")
    _reject_incomplete_component(component, label)


def _lexical_absolute_path(path: str | os.PathLike[str], label: str) -> Path:
    raw = os.fspath(path)
    if not raw:
        raise RoboCasaAssetManifestError(f"{label} must not be empty")
    supplied = Path(raw)
    lexical = Path(os.path.abspath(os.path.expanduser(raw)))
    if supplied != lexical:
        raise RoboCasaAssetManifestError(f"{label} must be a normalized absolute path: {supplied}")
    for component in lexical.parts:
        if component not in {lexical.anchor, os.sep}:
            _validate_component(component, label)
    return lexical


def _require_path(
    path: str | os.PathLike[str],
    label: str,
    *,
    expected_kind: str,
    expected_mode: int | None = None,
) -> tuple[Path, os.stat_result]:
    lexical = _lexical_absolute_path(path, label)
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as error:
        raise RoboCasaAssetManifestError(f"{label} does not exist: {lexical}") from error
    except (OSError, RuntimeError) as error:
        raise RoboCasaAssetManifestError(f"cannot resolve {label} {lexical}: {error}") from error
    if lexical != resolved:
        raise RoboCasaAssetManifestError(
            f"{label} must be canonical and contain no symlink components: provided={lexical}, resolved={resolved}"
        )
    try:
        metadata = resolved.lstat()
    except OSError as error:
        raise RoboCasaAssetManifestError(f"cannot inspect {label} {resolved}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise RoboCasaAssetManifestError(f"{label} must not be a symlink: {resolved}")
    if expected_kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise RoboCasaAssetManifestError(f"{label} must be a regular file: {resolved}")
    if expected_kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise RoboCasaAssetManifestError(f"{label} must be a directory: {resolved}")
    mode = stat.S_IMODE(metadata.st_mode)
    if expected_mode is not None and mode != expected_mode:
        raise RoboCasaAssetManifestError(
            f"{label} mode mismatch: expected={oct(expected_mode)}, actual={oct(mode)}, path={resolved}"
        )
    return resolved, metadata


def _read_stable_regular_file(
    path: str | os.PathLike[str],
    label: str,
    *,
    maximum_bytes: int | None = None,
    expected_mode: int | None = None,
) -> tuple[bytes, int, str]:
    canonical, before = _require_path(
        path,
        label,
        expected_kind="file",
        expected_mode=expected_mode,
    )
    if maximum_bytes is not None and before.st_size > maximum_bytes:
        raise RoboCasaAssetManifestError(
            f"{label} exceeds {maximum_bytes} bytes: size={before.st_size}, path={canonical}"
        )

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(canonical, flags)
    except OSError as error:
        raise RoboCasaAssetManifestError(f"cannot safely open {label} {canonical}: {error}") from error

    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RoboCasaAssetManifestError(f"{label} must be a regular file: {canonical}")
        if _stable_identity(before) != _stable_identity(opened):
            raise RoboCasaAssetManifestError(f"{label} changed before reading: {canonical}")
        while chunk := os.read(descriptor, HASH_CHUNK_BYTES):
            chunks.append(chunk)
            digest.update(chunk)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    _, after_path = _require_path(
        canonical,
        label,
        expected_kind="file",
        expected_mode=expected_mode,
    )
    identity = _stable_identity(before)
    if identity != _stable_identity(after_read) or identity != _stable_identity(after_path):
        raise RoboCasaAssetManifestError(f"{label} changed while reading: {canonical}")
    return b"".join(chunks), int(before.st_size), digest.hexdigest()


def _hash_stable_regular_file(path: Path, label: str) -> tuple[int, int, str]:
    canonical, before = _require_path(path, label, expected_kind="file")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(canonical, flags)
    except OSError as error:
        raise RoboCasaAssetManifestError(f"cannot safely open {label} {canonical}: {error}") from error

    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RoboCasaAssetManifestError(f"{label} must be a regular file: {canonical}")
        if _stable_identity(before) != _stable_identity(opened):
            raise RoboCasaAssetManifestError(f"{label} changed before hashing: {canonical}")
        while chunk := os.read(descriptor, HASH_CHUNK_BYTES):
            digest.update(chunk)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    _, after_path = _require_path(canonical, label, expected_kind="file")
    identity = _stable_identity(before)
    if identity != _stable_identity(after_read) or identity != _stable_identity(after_path):
        raise RoboCasaAssetManifestError(f"{label} changed while hashing: {canonical}")
    return stat.S_IMODE(before.st_mode), int(before.st_size), digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RoboCasaAssetManifestError(f"JSON contains a duplicate object key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise RoboCasaAssetManifestError(f"JSON contains a forbidden constant: {value}")


def _decode_json(payload: bytes, label: str) -> Any:
    if payload.startswith(codecs.BOM_UTF8):
        raise RoboCasaAssetManifestError(f"{label} must not contain a UTF-8 BOM")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RoboCasaAssetManifestError(f"{label} is not valid UTF-8: {error}") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise RoboCasaAssetManifestError(f"{label} is not valid JSON: {error}") from error


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoboCasaAssetManifestError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(repr(key) for key in observed - expected)
        raise RoboCasaAssetManifestError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RoboCasaAssetManifestError(f"{label} must be a non-empty, trimmed string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RoboCasaAssetManifestError(f"{label} contains a control character")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RoboCasaAssetManifestError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _require_positive_int(value: object, label: str) -> int:
    integer = _require_nonnegative_int(value, label)
    if integer == 0:
        raise RoboCasaAssetManifestError(f"{label} must be positive")
    return integer


def _validated_hash_workers(value: object) -> int:
    workers = _require_positive_int(value, "asset hash workers")
    if workers > MAX_HASH_WORKERS:
        raise RoboCasaAssetManifestError(f"asset hash workers must be <= {MAX_HASH_WORKERS}, got {workers}")
    return workers


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RoboCasaAssetManifestError(f"{label} must be a lowercase SHA-256, got {value!r}")
    return value


def _require_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise RoboCasaAssetManifestError(f"{label} must be a full lowercase Git commit, got {value!r}")
    return value


def _require_utc_timestamp(value: object, label: str) -> str:
    timestamp = _require_nonempty_string(value, label)
    if not timestamp.endswith("Z"):
        raise RoboCasaAssetManifestError(f"{label} must use an explicit UTC Z suffix")
    try:
        parsed = datetime.fromisoformat(f"{timestamp[:-1]}+00:00")
    except ValueError as error:
        raise RoboCasaAssetManifestError(f"{label} is not a valid ISO-8601 timestamp: {timestamp!r}") from error
    if parsed.tzinfo != timezone.utc:
        raise RoboCasaAssetManifestError(f"{label} must be UTC")
    return timestamp


def _normalized_absolute_descriptor_path(value: object, label: str) -> PurePosixPath:
    raw = _require_nonempty_string(value, label)
    if "\\" in raw:
        raise RoboCasaAssetManifestError(f"{label} must be a canonical POSIX absolute path")
    path = PurePosixPath(raw)
    if not path.is_absolute() or path.as_posix() != raw:
        raise RoboCasaAssetManifestError(f"{label} must be a normalized POSIX absolute path: {raw!r}")
    for component in path.parts[1:]:
        _validate_component(component, label)
    return path


def _canonical_relative_path(value: object, label: str) -> str:
    raw = _require_nonempty_string(value, label)
    if "\\" in raw:
        raise RoboCasaAssetManifestError(f"{label} must be a canonical POSIX relative path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.as_posix() != raw:
        raise RoboCasaAssetManifestError(f"{label} must be a normalized POSIX relative path: {raw!r}")
    for component in path.parts:
        _validate_component(component, label)
    return raw


def _require_exclusion_policy(value: object, label: str) -> list[str]:
    expected = list(EXCLUDED_TOP_LEVEL_PATHS)
    if type(value) is not list or value != expected:
        raise RoboCasaAssetManifestError(f"{label} must equal the exact exclusion policy {expected!r}, got {value!r}")
    return expected


def _validate_publisher_receipt(value: object, asset_root: Path) -> dict[str, Any]:
    receipt = _require_mapping(value, "publisher receipt")
    _require_exact_keys(receipt, _RECEIPT_KEYS, "publisher receipt")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != SCHEMA_VERSION:
        raise RoboCasaAssetManifestError(
            f"publisher receipt schema_version must be {SCHEMA_VERSION}, got {receipt['schema_version']!r}"
        )
    if receipt["kind"] != PUBLISHER_RECEIPT_KIND:
        raise RoboCasaAssetManifestError(
            f"publisher receipt kind must be {PUBLISHER_RECEIPT_KIND!r}, got {receipt['kind']!r}"
        )
    if receipt["status"] != "passed":
        raise RoboCasaAssetManifestError(f"publisher receipt status must be 'passed', got {receipt['status']!r}")
    exclusions = _require_exclusion_policy(
        receipt["excluded_top_level_paths"],
        "publisher receipt excluded_top_level_paths",
    )
    if receipt["manifest_relative_path"] != MANIFEST_RELATIVE_PATH:
        raise RoboCasaAssetManifestError(
            "publisher receipt manifest_relative_path mismatch: "
            f"expected={MANIFEST_RELATIVE_PATH!r}, actual={receipt['manifest_relative_path']!r}"
        )
    if receipt["payload_root"] != str(asset_root):
        raise RoboCasaAssetManifestError(
            "publisher receipt payload_root does not match the exact canonical asset root: "
            f"expected={str(asset_root)!r}, actual={receipt['payload_root']!r}"
        )
    generator_id = _require_nonempty_string(receipt["generator_id"], "publisher receipt generator_id")
    if _GENERATOR_ID_RE.fullmatch(generator_id) is None or generator_id != GENERATOR_ID:
        raise RoboCasaAssetManifestError(
            f"publisher receipt generator_id must be the supported generator {GENERATOR_ID!r}, got {generator_id!r}"
        )
    generated_at_utc = _require_utc_timestamp(
        receipt["generated_at_utc"],
        "publisher receipt generated_at_utc",
    )
    git_commit = _require_commit(receipt["git_commit"], "publisher receipt git_commit")
    if receipt["git_status_porcelain"] != "":
        raise RoboCasaAssetManifestError("publisher receipt git_status_porcelain must be empty")
    manifest_sha256 = _require_sha256(
        receipt["manifest_sha256"],
        "publisher receipt manifest_sha256",
    )
    directory_count = _require_nonnegative_int(
        receipt["payload_directory_count"],
        "publisher receipt payload_directory_count",
    )
    file_count = _require_positive_int(
        receipt["payload_file_count"],
        "publisher receipt payload_file_count",
    )
    symlink_count = _require_nonnegative_int(
        receipt["payload_symlink_count"],
        "publisher receipt payload_symlink_count",
    )
    if symlink_count != 0:
        raise RoboCasaAssetManifestError(f"publisher receipt payload_symlink_count must be zero, got {symlink_count}")
    total_bytes = _require_positive_int(
        receipt["payload_total_bytes"],
        "publisher receipt payload_total_bytes",
    )

    source_root = _normalized_absolute_descriptor_path(
        receipt["source_archive_directory"],
        "publisher receipt source_archive_directory",
    )
    source_sha256sums_path = _normalized_absolute_descriptor_path(
        receipt["source_sha256sums_path"],
        "publisher receipt source_sha256sums_path",
    )
    source_sizes_path = _normalized_absolute_descriptor_path(
        receipt["source_sizes_path"],
        "publisher receipt source_sizes_path",
    )
    if source_sha256sums_path != source_root / "SHA256SUMS":
        raise RoboCasaAssetManifestError(
            "publisher receipt source_sha256sums_path must be SHA256SUMS directly below source_archive_directory"
        )
    if source_sizes_path != source_root / "SIZES":
        raise RoboCasaAssetManifestError(
            "publisher receipt source_sizes_path must be SIZES directly below source_archive_directory"
        )
    source_sha256sums_sha256 = _require_sha256(
        receipt["source_sha256sums_sha256"],
        "publisher receipt source_sha256sums_sha256",
    )
    source_sizes_sha256 = _require_sha256(
        receipt["source_sizes_sha256"],
        "publisher receipt source_sizes_sha256",
    )
    return {
        "excluded_top_level_paths": exclusions,
        "generated_at_utc": generated_at_utc,
        "generator_id": generator_id,
        "git_commit": git_commit,
        "manifest_sha256": manifest_sha256,
        "payload_directory_count": directory_count,
        "payload_file_count": file_count,
        "payload_symlink_count": symlink_count,
        "payload_total_bytes": total_bytes,
        "source_archive_directory": source_root.as_posix(),
        "source_sha256sums_sha256": source_sha256sums_sha256,
        "source_sizes_sha256": source_sizes_sha256,
    }


def _validated_mode(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > 0o7777:
        raise RoboCasaAssetManifestError(f"{label} must be a permission-mode integer, got {value!r}")
    return value


def _parse_manifest_line(line: bytes, line_number: int) -> Mapping[str, Any]:
    if not line:
        raise RoboCasaAssetManifestError(f"asset manifest contains a blank line at line {line_number}")
    if len(line) > MAX_MANIFEST_LINE_BYTES:
        raise RoboCasaAssetManifestError(f"asset manifest line {line_number} exceeds {MAX_MANIFEST_LINE_BYTES} bytes")
    value = _decode_json(line, f"asset manifest line {line_number}")
    return _require_mapping(value, f"asset manifest line {line_number}")


def _parse_asset_manifest(payload: bytes, asset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not payload:
        raise RoboCasaAssetManifestError("asset manifest must not be empty")
    if payload.startswith(codecs.BOM_UTF8):
        raise RoboCasaAssetManifestError("asset manifest must not contain a UTF-8 BOM")
    lines = payload.splitlines()
    if not lines:
        raise RoboCasaAssetManifestError("asset manifest must contain a header")

    raw_header = _parse_manifest_line(lines[0], 1)
    _require_exact_keys(raw_header, _MANIFEST_HEADER_KEYS, "asset manifest header")
    if type(raw_header["schema_version"]) is not int or raw_header["schema_version"] != SCHEMA_VERSION:
        raise RoboCasaAssetManifestError(
            f"asset manifest schema_version must be {SCHEMA_VERSION}, got {raw_header['schema_version']!r}"
        )
    if raw_header["kind"] != MANIFEST_KIND:
        raise RoboCasaAssetManifestError(f"asset manifest kind must be {MANIFEST_KIND!r}, got {raw_header['kind']!r}")
    exclusions = _require_exclusion_policy(
        raw_header["excluded_top_level_paths"],
        "asset manifest excluded_top_level_paths",
    )
    if raw_header["payload_root"] != str(asset_root):
        raise RoboCasaAssetManifestError(
            "asset manifest payload_root does not match the exact canonical asset root: "
            f"expected={str(asset_root)!r}, actual={raw_header['payload_root']!r}"
        )
    header = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "payload_root": str(asset_root),
        "excluded_top_level_paths": exclusions,
    }

    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        raw_record = _parse_manifest_line(line, line_number)
        record_type = raw_record.get("type")
        if record_type == "directory":
            _require_exact_keys(raw_record, _DIRECTORY_RECORD_KEYS, f"asset manifest line {line_number}")
        elif record_type == "file":
            _require_exact_keys(raw_record, _FILE_RECORD_KEYS, f"asset manifest line {line_number}")
        else:
            raise RoboCasaAssetManifestError(
                f"asset manifest line {line_number} type must be 'directory' or 'file', got {record_type!r}"
            )
        relative_path = _canonical_relative_path(
            raw_record["path"],
            f"asset manifest line {line_number} path",
        )
        top_level = PurePosixPath(relative_path).parts[0]
        if top_level in EXCLUDED_TOP_LEVEL_PATHS:
            raise RoboCasaAssetManifestError(
                f"asset manifest line {line_number} illegally records excluded path {relative_path!r}"
            )
        if relative_path in seen_paths:
            raise RoboCasaAssetManifestError(f"asset manifest contains duplicate path: {relative_path!r}")
        seen_paths.add(relative_path)
        record: dict[str, Any] = {
            "mode": _validated_mode(raw_record["mode"], f"asset manifest line {line_number} mode"),
            "path": relative_path,
            "type": record_type,
        }
        if record_type == "file":
            record["sha256"] = _require_sha256(
                raw_record["sha256"],
                f"asset manifest line {line_number} sha256",
            )
            record["size_bytes"] = _require_nonnegative_int(
                raw_record["size_bytes"],
                f"asset manifest line {line_number} size_bytes",
            )
        records.append(record)
    if not records:
        raise RoboCasaAssetManifestError("asset manifest must contain at least one payload record")
    return header, records


def _relative_child_path(base: PurePosixPath, name: str, label: str) -> PurePosixPath:
    _validate_component(name, label)
    return base / name


def _scan_payload_tree_serial(asset_root: Path) -> tuple[list[dict[str, Any]], int]:
    root_before = _require_path(asset_root, "asset root", expected_kind="directory")[1]
    records: list[dict[str, Any]] = []
    observed_exclusions: set[str] = set()

    def scan_directory(directory: Path, relative_base: PurePosixPath) -> None:
        directory_before = _require_path(
            directory,
            f"asset directory {relative_base.as_posix() or '.'}",
            expected_kind="directory",
        )[1]
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise RoboCasaAssetManifestError(f"cannot enumerate asset directory {directory}: {error}") from error

        for entry in entries:
            entry_label = f"asset entry {entry.path}"
            _validate_component(entry.name, entry_label)
            entry_path = directory / entry.name
            if not relative_base.parts and entry.name in EXCLUDED_TOP_LEVEL_PATHS:
                _require_path(entry_path, f"excluded top-level path {entry.name}", expected_kind="directory")
                observed_exclusions.add(entry.name)
                continue

            relative = _relative_child_path(relative_base, entry.name, entry_label)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError as error:
                raise RoboCasaAssetManifestError(f"asset entry disappeared during scan: {entry_path}") from error
            except OSError as error:
                raise RoboCasaAssetManifestError(f"cannot inspect asset entry {entry_path}: {error}") from error

            if stat.S_ISLNK(metadata.st_mode):
                raise RoboCasaAssetManifestError(f"asset tree must not contain symlinks: {entry_path}")
            if stat.S_ISDIR(metadata.st_mode):
                records.append(
                    {
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "path": relative.as_posix(),
                        "type": "directory",
                    }
                )
                scan_directory(entry_path, relative)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RoboCasaAssetManifestError(f"asset tree contains a special file: {entry_path}")
            mode, size_bytes, sha256 = _hash_stable_regular_file(entry_path, entry_label)
            records.append(
                {
                    "mode": mode,
                    "path": relative.as_posix(),
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                    "type": "file",
                }
            )

        directory_after = _require_path(
            directory,
            f"asset directory {relative_base.as_posix() or '.'}",
            expected_kind="directory",
        )[1]
        if _stable_identity(directory_before) != _stable_identity(directory_after):
            raise RoboCasaAssetManifestError(f"asset directory changed during scan: {directory}")

    scan_directory(asset_root, PurePosixPath())
    missing_exclusions = sorted(set(EXCLUDED_TOP_LEVEL_PATHS) - observed_exclusions)
    if missing_exclusions:
        raise RoboCasaAssetManifestError(f"required excluded top-level paths are missing: {missing_exclusions}")
    root_after = _require_path(asset_root, "asset root", expected_kind="directory")[1]
    if _stable_identity(root_before) != _stable_identity(root_after):
        raise RoboCasaAssetManifestError(f"asset root changed during scan: {asset_root}")
    return records, stat.S_IMODE(root_after.st_mode)


def _hash_file_tasks_bounded(
    tasks: list[tuple[Path, str]],
    workers: int,
) -> list[tuple[int, int, str]]:
    """Hash files with at most ``workers`` submitted futures at any time."""

    results: dict[int, tuple[int, int, str]] = {}
    failures: dict[int, Exception] = {}
    task_iterator = iter(enumerate(tasks))

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asset-sha256") as executor:
        pending: dict[Future[tuple[int, int, str]], int] = {}

        def fill_available_slots() -> None:
            while len(pending) < workers:
                try:
                    task_index, (path, label) = next(task_iterator)
                except StopIteration:
                    return
                pending[executor.submit(_hash_stable_regular_file, path, label)] = task_index

        fill_available_slots()
        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                task_index = pending.pop(future)
                try:
                    results[task_index] = future.result()
                except Exception as error:
                    failures[task_index] = error
            fill_available_slots()

    if failures:
        raise failures[min(failures)]
    return [results[index] for index in range(len(tasks))]


def _scan_payload_tree_parallel(
    asset_root: Path,
    *,
    workers: int,
) -> tuple[list[dict[str, Any]], int]:
    """Discover deterministically, hash concurrently, and retain record order."""

    root_before = _require_path(asset_root, "asset root", expected_kind="directory")[1]
    records: list[dict[str, Any] | None] = []
    file_tasks: list[tuple[Path, str]] = []
    file_record_indexes: list[int] = []
    file_relative_paths: list[str] = []
    observed_exclusions: set[str] = set()
    directory_checks: list[tuple[Path, str, os.stat_result]] = []

    def scan_directory(directory: Path, relative_base: PurePosixPath) -> None:
        directory_label = f"asset directory {relative_base.as_posix() or '.'}"
        directory_before = _require_path(
            directory,
            directory_label,
            expected_kind="directory",
        )[1]
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise RoboCasaAssetManifestError(f"cannot enumerate asset directory {directory}: {error}") from error

        for entry in entries:
            entry_label = f"asset entry {entry.path}"
            _validate_component(entry.name, entry_label)
            entry_path = directory / entry.name
            if not relative_base.parts and entry.name in EXCLUDED_TOP_LEVEL_PATHS:
                _require_path(entry_path, f"excluded top-level path {entry.name}", expected_kind="directory")
                observed_exclusions.add(entry.name)
                continue

            relative = _relative_child_path(relative_base, entry.name, entry_label)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError as error:
                raise RoboCasaAssetManifestError(f"asset entry disappeared during scan: {entry_path}") from error
            except OSError as error:
                raise RoboCasaAssetManifestError(f"cannot inspect asset entry {entry_path}: {error}") from error

            if stat.S_ISLNK(metadata.st_mode):
                raise RoboCasaAssetManifestError(f"asset tree must not contain symlinks: {entry_path}")
            if stat.S_ISDIR(metadata.st_mode):
                records.append(
                    {
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "path": relative.as_posix(),
                        "type": "directory",
                    }
                )
                scan_directory(entry_path, relative)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RoboCasaAssetManifestError(f"asset tree contains a special file: {entry_path}")
            file_record_indexes.append(len(records))
            file_tasks.append((entry_path, entry_label))
            file_relative_paths.append(relative.as_posix())
            records.append(None)

        # Preserve the serial scanner's depth-first directory validation order,
        # but perform the checks after every concurrent file hash has finished.
        directory_checks.append((directory, directory_label, directory_before))

    scan_directory(asset_root, PurePosixPath())
    missing_exclusions = sorted(set(EXCLUDED_TOP_LEVEL_PATHS) - observed_exclusions)
    if missing_exclusions:
        raise RoboCasaAssetManifestError(f"required excluded top-level paths are missing: {missing_exclusions}")

    hash_results = _hash_file_tasks_bounded(file_tasks, workers)
    for task_index, (record_index, (mode, size_bytes, sha256)) in enumerate(
        zip(file_record_indexes, hash_results, strict=True)
    ):
        relative_path = records[record_index]
        if relative_path is not None:
            raise RuntimeError("internal asset record slot was unexpectedly populated")
        records[record_index] = {
            "mode": mode,
            "path": file_relative_paths[task_index],
            "sha256": sha256,
            "size_bytes": size_bytes,
            "type": "file",
        }

    for directory, directory_label, directory_before in directory_checks:
        directory_after = _require_path(
            directory,
            directory_label,
            expected_kind="directory",
        )[1]
        if _stable_identity(directory_before) != _stable_identity(directory_after):
            raise RoboCasaAssetManifestError(f"asset directory changed during scan: {directory}")

    root_after = _require_path(asset_root, "asset root", expected_kind="directory")[1]
    if _stable_identity(root_before) != _stable_identity(root_after):
        raise RoboCasaAssetManifestError(f"asset root changed during scan: {asset_root}")
    if any(record is None for record in records):
        raise RuntimeError("internal asset record slot was not populated")
    return [record for record in records if record is not None], stat.S_IMODE(root_after.st_mode)


def _scan_payload_tree(
    asset_root: Path,
    *,
    workers: int = DEFAULT_HASH_WORKERS,
) -> tuple[list[dict[str, Any]], int]:
    workers = _validated_hash_workers(workers)
    if workers == 1:
        return _scan_payload_tree_serial(asset_root)
    return _scan_payload_tree_parallel(asset_root, workers=workers)


def _payload_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    directories = sum(record["type"] == "directory" for record in records)
    files = sum(record["type"] == "file" for record in records)
    total_bytes = sum(int(record["size_bytes"]) for record in records if record["type"] == "file")
    return {
        "directory_count": directories,
        "file_count": files,
        "symlink_count": 0,
        "total_bytes": total_bytes,
    }


def _compare_manifest_and_live_tree(
    expected_records: list[dict[str, Any]],
    actual_records: list[dict[str, Any]],
) -> None:
    expected_by_path = {record["path"]: record for record in expected_records}
    actual_by_path = {record["path"]: record for record in actual_records}
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    extra = sorted(set(actual_by_path) - set(expected_by_path))
    if missing or extra:
        raise RoboCasaAssetManifestError(f"asset payload set mismatch: missing={missing}, extra={extra}")

    for relative_path in expected_by_path:
        expected = expected_by_path[relative_path]
        actual = actual_by_path[relative_path]
        if actual["type"] != expected["type"]:
            raise RoboCasaAssetManifestError(
                f"asset type mismatch for {relative_path}: expected={expected['type']!r}, actual={actual['type']!r}"
            )
        if actual["mode"] != expected["mode"]:
            raise RoboCasaAssetManifestError(
                f"asset mode mismatch for {relative_path}: "
                f"expected={oct(expected['mode'])}, actual={oct(actual['mode'])}"
            )
        if expected["type"] == "file":
            if actual["size_bytes"] != expected["size_bytes"]:
                raise RoboCasaAssetManifestError(
                    f"asset size mismatch for {relative_path}: "
                    f"expected={expected['size_bytes']}, actual={actual['size_bytes']}"
                )
            if actual["sha256"] != expected["sha256"]:
                raise RoboCasaAssetManifestError(
                    f"asset SHA-256 mismatch for {relative_path}: "
                    f"expected={expected['sha256']}, actual={actual['sha256']}"
                )

    expected_order = [record["path"] for record in expected_records]
    actual_order = [record["path"] for record in actual_records]
    if expected_order != actual_order:
        first_mismatch = next(
            index for index, (expected, actual) in enumerate(zip(expected_order, actual_order)) if expected != actual
        )
        raise RoboCasaAssetManifestError(
            "asset manifest record order does not match deterministic depth-first traversal: "
            f"index={first_mismatch}, expected={expected_order[first_mismatch]!r}, "
            f"actual={actual_order[first_mismatch]!r}"
        )


def verify_robocasa_asset_tree(
    asset_root: str | os.PathLike[str],
    *,
    workers: int = DEFAULT_HASH_WORKERS,
) -> dict[str, Any]:
    """Verify a published asset tree and return a structured pass receipt."""

    workers = _validated_hash_workers(workers)
    root, _ = _require_path(asset_root, "asset root", expected_kind="directory")
    provenance_root, _ = _require_path(
        root / PROVENANCE_DIRECTORY,
        "asset provenance directory",
        expected_kind="directory",
    )
    receipt_path = provenance_root / Path(PUBLISHER_RECEIPT_RELATIVE_PATH).name
    manifest_path = provenance_root / Path(MANIFEST_RELATIVE_PATH).name

    receipt_bytes, receipt_size, receipt_sha256 = _read_stable_regular_file(
        receipt_path,
        "asset publisher receipt",
        maximum_bytes=MAX_RECEIPT_BYTES,
        expected_mode=PROVENANCE_FILE_MODE,
    )
    receipt = _validate_publisher_receipt(
        _decode_json(receipt_bytes, "asset publisher receipt"),
        root,
    )

    manifest_bytes, manifest_size, manifest_sha256 = _read_stable_regular_file(
        manifest_path,
        "asset manifest",
        maximum_bytes=MAX_MANIFEST_BYTES,
        expected_mode=PROVENANCE_FILE_MODE,
    )
    if manifest_sha256 != receipt["manifest_sha256"]:
        raise RoboCasaAssetManifestError(
            "asset manifest SHA-256 does not match publisher receipt: "
            f"expected={receipt['manifest_sha256']}, actual={manifest_sha256}"
        )
    header, expected_records = _parse_asset_manifest(manifest_bytes, root)
    if header["excluded_top_level_paths"] != receipt["excluded_top_level_paths"]:
        raise RoboCasaAssetManifestError("asset manifest and publisher receipt exclusion policies do not match")

    expected_summary = _payload_summary(expected_records)
    for summary_key, receipt_key in (
        ("directory_count", "payload_directory_count"),
        ("file_count", "payload_file_count"),
        ("symlink_count", "payload_symlink_count"),
        ("total_bytes", "payload_total_bytes"),
    ):
        if expected_summary[summary_key] != receipt[receipt_key]:
            raise RoboCasaAssetManifestError(
                f"asset manifest {summary_key} does not match publisher receipt: "
                f"manifest={expected_summary[summary_key]}, receipt={receipt[receipt_key]}"
            )

    actual_records, asset_root_mode = _scan_payload_tree(root, workers=workers)
    _compare_manifest_and_live_tree(expected_records, actual_records)
    actual_summary = _payload_summary(actual_records)
    if actual_summary != expected_summary:
        raise RoboCasaAssetManifestError(
            f"asset payload summary mismatch: expected={expected_summary}, actual={actual_summary}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": VERIFICATION_RECEIPT_KIND,
        "status": "passed",
        "asset_root": str(root),
        "asset_root_mode": asset_root_mode,
        "excluded_top_level_paths": list(EXCLUDED_TOP_LEVEL_PATHS),
        "manifest_path": str(manifest_path),
        "manifest_size_bytes": manifest_size,
        "manifest_sha256": manifest_sha256,
        "publisher_receipt_path": str(receipt_path),
        "publisher_receipt_size_bytes": receipt_size,
        "publisher_receipt_sha256": receipt_sha256,
        "publisher_generator_id": receipt["generator_id"],
        "publisher_generated_at_utc": receipt["generated_at_utc"],
        "robocasa_git_commit": receipt["git_commit"],
        "payload_directory_count": actual_summary["directory_count"],
        "payload_file_count": actual_summary["file_count"],
        "payload_symlink_count": actual_summary["symlink_count"],
        "payload_total_bytes": actual_summary["total_bytes"],
        "source_archive_directory": receipt["source_archive_directory"],
        "source_sha256sums_sha256": receipt["source_sha256sums_sha256"],
        "source_sizes_sha256": receipt["source_sizes_sha256"],
    }


def write_verification_receipt(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically publish one immutable verification receipt without overwrite."""

    target = Path(os.path.abspath(path))
    if target.exists() or target.is_symlink():
        raise RoboCasaAssetManifestError(f"refusing to overwrite asset verification receipt: {target}")
    parent, _ = _require_path(
        target.parent,
        "asset verification receipt parent",
        expected_kind="directory",
    )
    if target.parent != parent:
        raise RoboCasaAssetManifestError("asset verification receipt parent must be an exact canonical directory")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o444)
    published = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target, follow_symlinks=False)
        published = True
        os.chmod(target, PROVENANCE_FILE_MODE, follow_symlinks=False)
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if published:
            target.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify the exact published RoboCasa asset tree")
    verify.add_argument("--asset-root", type=Path, required=True)
    verify.add_argument("--output", type=Path)
    verify.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_HASH_WORKERS,
        help=f"bounded regular-file SHA-256 workers (default: {DEFAULT_HASH_WORKERS}, max: {MAX_HASH_WORKERS})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = verify_robocasa_asset_tree(args.asset_root, workers=args.workers)
        if args.output is not None:
            digest = write_verification_receipt(args.output, result)
    except RoboCasaAssetManifestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.output is not None:
        print(f"ROBOCASA_ASSET_TREE_VERIFIED sha256={digest} path={args.output}")
        return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
