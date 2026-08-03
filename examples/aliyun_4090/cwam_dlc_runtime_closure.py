#!/usr/bin/env python3
"""Build and verify a relocatable manifest for the DLC runtime closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_KIND = "cwam_dlc_runtime_closure_manifest"
RECEIPT_KIND = "cwam_dlc_runtime_closure_verification"
HASH_CHUNK_BYTES = 8 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "artifact_name",
    "entries",
    "entry_count",
    "regular_file_count",
    "symlink_count",
    "directory_count",
    "total_regular_file_bytes",
    "tree_sha256",
}


class RuntimeClosureError(RuntimeError):
    """Raised when a runtime closure violates its manifest contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha256_file(path: Path) -> str:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeClosureError(f"expected regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _identity(before) != _identity(opened):
            raise RuntimeClosureError(f"file changed before hashing: {path}")
        while chunk := os.read(descriptor, HASH_CHUNK_BYTES):
            digest.update(chunk)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    if _identity(before) != _identity(after_read) or _identity(before) != _identity(after_path):
        raise RuntimeClosureError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _canonical_root(value: str | os.PathLike[str]) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    resolved = lexical.resolve(strict=True)
    if lexical != resolved:
        raise RuntimeClosureError(
            f"runtime closure root must be canonical and contain no symlink components: {lexical}"
        )
    metadata = resolved.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeClosureError(f"runtime closure root must be a directory: {resolved}")
    return resolved


def _relative_path(base: PurePosixPath, name: str) -> PurePosixPath:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or ".incomplete" in name
        or unicodedata.normalize("NFC", name) != name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise RuntimeClosureError(f"non-canonical closure entry name: {name!r}")
    return base / name


def _mode(metadata: os.stat_result) -> str:
    return f"{stat.S_IMODE(metadata.st_mode):04o}"


def scan(root_value: str | os.PathLike[str]) -> list[dict[str, Any]]:
    root = _canonical_root(root_value)
    root_before = root.lstat()
    pending: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath())]
    records: list[dict[str, Any]] = []

    while pending:
        directory, relative_base = pending.pop()
        directory_before = directory.lstat()
        if not stat.S_ISDIR(directory_before.st_mode):
            raise RuntimeClosureError(f"closure directory changed type: {directory}")
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        children: list[tuple[Path, PurePosixPath]] = []
        for entry in entries:
            relative = _relative_path(relative_base, entry.name)
            path = directory / entry.name
            metadata = entry.stat(follow_symlinks=False)
            common = {"relative_path": relative.as_posix(), "mode": _mode(metadata)}
            if stat.S_ISDIR(metadata.st_mode):
                records.append({**common, "type": "directory"})
                children.append((path, relative))
            elif stat.S_ISREG(metadata.st_mode):
                records.append(
                    {
                        **common,
                        "type": "regular_file",
                        "size_bytes": int(metadata.st_size),
                        "sha256": _sha256_file(path),
                    }
                )
            elif stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(path)
                if not target or "\x00" in target:
                    raise RuntimeClosureError(f"invalid symlink target: {path}")
                after = path.lstat()
                if _identity(metadata) != _identity(after) or os.readlink(path) != target:
                    raise RuntimeClosureError(f"symlink changed while scanning: {path}")
                records.append({**common, "type": "symlink", "target": target})
            else:
                raise RuntimeClosureError(f"closure contains a special file: {path}")
        if _identity(directory_before) != _identity(directory.lstat()):
            raise RuntimeClosureError(f"closure directory changed while scanning: {directory}")
        pending.extend(reversed(children))

    if _identity(root_before) != _identity(root.lstat()):
        raise RuntimeClosureError(f"closure root changed while scanning: {root}")
    if not records:
        raise RuntimeClosureError("runtime closure must not be empty")
    records.sort(key=lambda item: str(item["relative_path"]))
    return records


def build_manifest(root: str | os.PathLike[str], artifact_name: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", artifact_name):
        raise RuntimeClosureError(f"invalid artifact name: {artifact_name!r}")
    entries = scan(root)
    regular_files = [entry for entry in entries if entry["type"] == "regular_file"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "artifact_name": artifact_name,
        "entries": entries,
        "entry_count": len(entries),
        "regular_file_count": len(regular_files),
        "symlink_count": sum(entry["type"] == "symlink" for entry in entries),
        "directory_count": sum(entry["type"] == "directory" for entry in entries),
        "total_regular_file_bytes": sum(int(entry["size_bytes"]) for entry in regular_files),
        "tree_sha256": hashlib.sha256(_canonical_json(entries)).hexdigest(),
    }


def _require_sha256(value: str, label: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise RuntimeClosureError(f"{label} must be a lowercase SHA-256")
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise RuntimeClosureError("runtime closure manifest keys mismatch")
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != MANIFEST_KIND:
        raise RuntimeClosureError("runtime closure manifest identity mismatch")
    entries = value["entries"]
    if not isinstance(entries, list) or value["entry_count"] != len(entries):
        raise RuntimeClosureError("runtime closure manifest entry count mismatch")
    if not isinstance(value["artifact_name"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", value["artifact_name"]
    ):
        raise RuntimeClosureError("runtime closure manifest artifact name is invalid")
    expected_paths: list[str] = []
    regular_file_count = 0
    symlink_count = 0
    directory_count = 0
    total_regular_file_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeClosureError("runtime closure manifest entry must be an object")
        entry_type = entry.get("type")
        expected_keys = {
            "directory": {"relative_path", "mode", "type"},
            "symlink": {"relative_path", "mode", "type", "target"},
            "regular_file": {
                "relative_path",
                "mode",
                "type",
                "size_bytes",
                "sha256",
            },
        }.get(entry_type)
        if expected_keys is None or set(entry) != expected_keys:
            raise RuntimeClosureError("runtime closure manifest entry schema mismatch")
        relative_path = entry["relative_path"]
        if not isinstance(relative_path, str) or PurePosixPath(relative_path).as_posix() != relative_path:
            raise RuntimeClosureError("runtime closure manifest relative path is invalid")
        path = PurePosixPath(relative_path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise RuntimeClosureError("runtime closure manifest relative path is unsafe")
        for part in path.parts:
            _relative_path(PurePosixPath(), part)
        expected_paths.append(relative_path)
        if not isinstance(entry["mode"], str) or re.fullmatch(r"[0-7]{4}", entry["mode"]) is None:
            raise RuntimeClosureError("runtime closure manifest mode is invalid")
        if entry_type == "regular_file":
            if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] < 0:
                raise RuntimeClosureError("runtime closure manifest file size is invalid")
            _require_sha256(entry["sha256"], "entry SHA-256")
            regular_file_count += 1
            total_regular_file_bytes += entry["size_bytes"]
        elif entry_type == "symlink":
            if not isinstance(entry["target"], str) or not entry["target"] or "\x00" in entry["target"]:
                raise RuntimeClosureError("runtime closure manifest symlink target is invalid")
            symlink_count += 1
        else:
            directory_count += 1
    if expected_paths != sorted(set(expected_paths)):
        raise RuntimeClosureError("runtime closure manifest paths must be unique and sorted")
    expected_counts = {
        "regular_file_count": regular_file_count,
        "symlink_count": symlink_count,
        "directory_count": directory_count,
        "total_regular_file_bytes": total_regular_file_bytes,
    }
    if any(value[key] != expected for key, expected in expected_counts.items()):
        raise RuntimeClosureError("runtime closure manifest aggregate counts mismatch")
    if value["tree_sha256"] != hashlib.sha256(_canonical_json(entries)).hexdigest():
        raise RuntimeClosureError("runtime closure manifest tree digest mismatch")
    return value


def verify(
    root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    expected_manifest_sha256: str,
    expected_tree_sha256: str,
    phase: str,
) -> dict[str, Any]:
    lexical_manifest_file = Path(os.path.abspath(os.path.expanduser(os.fspath(manifest_path))))
    if lexical_manifest_file.is_symlink():
        raise RuntimeClosureError("runtime closure manifest must be a regular non-symlink file")
    manifest_file = lexical_manifest_file.resolve(strict=True)
    if lexical_manifest_file != manifest_file or not manifest_file.is_file():
        raise RuntimeClosureError("runtime closure manifest path must be canonical and contain no symlink components")
    actual_manifest_sha256 = _sha256_file(manifest_file)
    if actual_manifest_sha256 != _require_sha256(expected_manifest_sha256, "manifest SHA-256"):
        raise RuntimeClosureError("runtime closure manifest SHA-256 mismatch")
    manifest = _load_manifest(manifest_file)
    expected_tree_sha256 = _require_sha256(expected_tree_sha256, "tree SHA-256")
    if manifest["tree_sha256"] != expected_tree_sha256:
        raise RuntimeClosureError("runtime closure expected tree SHA-256 mismatch")
    live_entries = scan(root)
    live_tree_sha256 = hashlib.sha256(_canonical_json(live_entries)).hexdigest()
    if live_entries != manifest["entries"] or live_tree_sha256 != expected_tree_sha256:
        raise RuntimeClosureError("live runtime closure differs from the immutable manifest")
    if phase not in {"pre", "post", "publication"}:
        raise RuntimeClosureError(f"invalid verification phase: {phase!r}")
    verifier_path = Path(__file__).resolve(strict=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "status": "passed",
        "phase": phase,
        "root": str(_canonical_root(root)),
        "manifest_path": str(manifest_file),
        "manifest_sha256": actual_manifest_sha256,
        "artifact_name": manifest["artifact_name"],
        "entry_count": manifest["entry_count"],
        "regular_file_count": manifest["regular_file_count"],
        "symlink_count": manifest["symlink_count"],
        "directory_count": manifest["directory_count"],
        "total_regular_file_bytes": manifest["total_regular_file_bytes"],
        "tree_sha256": live_tree_sha256,
        "verifier": {
            "path": str(verifier_path),
            "sha256": _sha256_file(verifier_path),
            "python": sys.version,
        },
    }


def _write_json(path_value: str | os.PathLike[str], value: dict[str, Any]) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeClosureError(f"refusing to overwrite output: {path}")
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--root", required=True)
    build_parser.add_argument("--artifact-name", required=True)
    build_parser.add_argument("--output")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", required=True)
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--expected-manifest-sha256", required=True)
    verify_parser.add_argument("--expected-tree-sha256", required=True)
    verify_parser.add_argument("--phase", choices=("pre", "post", "publication"), required=True)
    verify_parser.add_argument("--output")
    args = parser.parse_args()

    if args.command == "build":
        result = build_manifest(args.root, args.artifact_name)
    else:
        result = verify(
            args.root,
            args.manifest,
            args.expected_manifest_sha256,
            args.expected_tree_sha256,
            args.phase,
        )
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeClosureError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
