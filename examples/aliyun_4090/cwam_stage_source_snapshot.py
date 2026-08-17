#!/usr/bin/env python3
"""Materialize an exact manifest-listed source tree on worker-local storage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Any


class SourceStagingError(RuntimeError):
    """Raised when a source snapshot cannot be staged safely."""


def _canonical_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise SourceStagingError(f"{label} must be a regular directory: {path}")
    return path.resolve(strict=True)


def _relative_manifest_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise SourceStagingError("manifest file path must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise SourceStagingError(f"manifest file path is unsafe: {value!r}")
    return Path(*pure.parts)


def _load_inventory(manifest_path: Path, source_root: Path) -> tuple[dict[str, Any], list[tuple[Path, int]]]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SourceStagingError(f"source manifest must be a regular file: {manifest_path}")
    if manifest_path.resolve(strict=True) != source_root / "SNAPSHOT_MANIFEST.json":
        raise SourceStagingError("source manifest must be SNAPSHOT_MANIFEST.json under source root")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = document.get("files")
    if (
        document.get("schema_version") != 1
        or document.get("source_dirty") is not False
        or not isinstance(files, list)
        or document.get("file_count") != len(files)
    ):
        raise SourceStagingError("source manifest schema, clean state, or file count is invalid")

    inventory: list[tuple[Path, int]] = []
    seen: set[Path] = set()
    for item in files:
        if not isinstance(item, dict):
            raise SourceStagingError("manifest file entry must be an object")
        relative = _relative_manifest_path(item.get("path"))
        expected_bytes = item.get("bytes")
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
            raise SourceStagingError(f"manifest byte count is invalid for {relative}")
        if relative in seen:
            raise SourceStagingError(f"manifest file path is duplicated: {relative}")
        seen.add(relative)
        inventory.append((relative, expected_bytes))
    return document, inventory


def _copy_regular_file(source: Path, destination: Path, expected_bytes: int, expected_source: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise SourceStagingError(f"manifest source is not a regular file: {source}")
    if source.resolve(strict=True) != expected_source:
        raise SourceStagingError(f"manifest source traverses a symlinked parent: {source}")
    metadata = source.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
        raise SourceStagingError(f"manifest source size mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copyfile(source, destination)
    destination.chmod(0o444)


def stage_source_snapshot(*, source_root: Path, manifest_path: Path, destination_root: Path) -> dict[str, Any]:
    source_root = _canonical_directory(source_root, "source snapshot")
    manifest_path = manifest_path.resolve(strict=True)
    _, inventory = _load_inventory(manifest_path, source_root)

    destination_root = destination_root.absolute()
    if destination_root.exists() or destination_root.is_symlink():
        raise SourceStagingError(f"destination root already exists: {destination_root}")
    destination_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_root = destination_root.with_name(f".{destination_root.name}.incomplete-{os.getpid()}")
    if temporary_root.exists() or temporary_root.is_symlink():
        raise SourceStagingError(f"temporary destination already exists: {temporary_root}")

    copied_bytes = 0
    try:
        temporary_root.mkdir(mode=0o700)
        for relative, expected_bytes in inventory:
            _copy_regular_file(
                source_root / relative,
                temporary_root / relative,
                expected_bytes,
                source_root / relative,
            )
            copied_bytes += expected_bytes
        _copy_regular_file(
            manifest_path,
            temporary_root / "SNAPSHOT_MANIFEST.json",
            manifest_path.stat().st_size,
            source_root / "SNAPSHOT_MANIFEST.json",
        )
        for directory in sorted(
            (path for path in temporary_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        temporary_root.chmod(0o555)
        os.replace(temporary_root, destination_root)
    except BaseException:
        if temporary_root.exists():
            for directory in [temporary_root, *temporary_root.rglob("*")]:
                if directory.is_dir():
                    directory.chmod(0o700)
            shutil.rmtree(temporary_root)
        raise

    return {
        "schema_version": "manifest_only_source_stage_v1",
        "status": "ready",
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "manifest_file_count": len(inventory),
        "manifest_file_bytes": copied_bytes,
        "ignored_unlisted_entries": True,
        "destination_files_mode": "0444",
        "destination_directories_mode": "0555",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = stage_source_snapshot(
        source_root=args.source_root,
        manifest_path=args.manifest_path,
        destination_root=args.destination_root,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
