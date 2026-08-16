#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Persist a local formal evaluation tree to the fixed OSS namespace.

OSS FUSE does not provide the Unix link and mode semantics required while the
formal evaluator is writing evidence. The evaluator therefore writes to local
storage. This helper runs after it exits and publishes both an inspectable OSS
mirror and a tar archive that preserves the formal tree's Unix metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ARCHIVE_NAME = "formal-evidence-tree.tar"
STATUS_NAME = "persistence-status.json"


class PersistenceError(RuntimeError):
    """Raised when a persistence contract is not satisfied."""


def _require_safe_id(value: str, label: str) -> None:
    if not SAFE_ID.fullmatch(value):
        raise PersistenceError(f"{label} is not a safe path component: {value!r}")


def _canonical_existing_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise PersistenceError(f"{label} must be absolute: {path}")
    if path.is_symlink() or not path.is_dir():
        raise PersistenceError(f"{label} must be an existing non-symlink directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise PersistenceError(f"{label} must be canonical: {path} -> {resolved}")
    return resolved


def _require_exact_path(actual: Path, expected: Path, label: str) -> None:
    if actual != expected:
        raise PersistenceError(f"{label} must be exactly {expected}, observed {actual}")


def _scan_regular_tree(root: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            candidate = current / name
            mode = os.lstat(candidate).st_mode
            if not stat.S_ISDIR(mode):
                raise PersistenceError(f"formal output contains an unsupported entry: {candidate}")
        for name in file_names:
            candidate = current / name
            metadata = os.lstat(candidate)
            if not stat.S_ISREG(metadata.st_mode):
                raise PersistenceError(f"formal output contains an unsupported entry: {candidate}")
            file_count += 1
            total_bytes += metadata.st_size
    return file_count, total_bytes


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_regular_file(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise PersistenceError(f"refusing to replace persisted file: {destination}")
    shutil.copyfile(source, destination)
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_tree_without_metadata(source: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    for entry in sorted(source.iterdir(), key=lambda item: item.name):
        target = destination / entry.name
        mode = os.lstat(entry).st_mode
        if stat.S_ISDIR(mode):
            _copy_tree_without_metadata(entry, target)
        elif stat.S_ISREG(mode):
            _copy_regular_file(entry, target)
        else:
            raise PersistenceError(f"formal output contains an unsupported entry: {entry}")
    _fsync_directory(destination)


def _create_archive(source: Path, archive: Path) -> int:
    if archive.exists() or archive.is_symlink():
        raise PersistenceError(f"refusing to replace local archive: {archive}")
    with tarfile.open(archive, mode="x") as handle:
        for entry in sorted(source.iterdir(), key=lambda item: item.name):
            handle.add(entry, arcname=entry.name, recursive=True)
    os.chmod(archive, 0o444)
    descriptor = os.open(archive, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return archive.stat().st_size


def _write_exclusive_json(path: Path, document: dict[str, Any]) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def persist(args: argparse.Namespace) -> Path:
    _require_safe_id(args.experiment_id, "experiment_id")
    _require_safe_id(args.run_id, "run_id")
    if not re.fullmatch(r"[0-9a-f]{40}", args.control_commit):
        raise PersistenceError("control_commit must be a lowercase 40-character Git commit")
    if args.launcher_exit_code < 0 or args.launcher_exit_code > 255:
        raise PersistenceError("launcher_exit_code must be in [0, 255]")

    local_parent = _canonical_existing_directory(
        Path(args.local_output_parent), "local_output_parent"
    )
    persistent_parent = _canonical_existing_directory(
        Path(args.persistent_output_parent), "persistent_output_parent"
    )
    if not os.access(persistent_parent, os.W_OK):
        raise PersistenceError(f"persistent_output_parent is not writable: {persistent_parent}")

    local_root = Path(args.local_output_root)
    persistent_root = Path(args.persistent_output_root)
    expected_local_root = local_parent / args.experiment_id / args.run_id
    expected_persistent_root = persistent_parent / args.experiment_id / args.run_id
    _require_exact_path(local_root, expected_local_root, "local_output_root")
    _require_exact_path(persistent_root, expected_persistent_root, "persistent_output_root")
    if persistent_root.exists() or persistent_root.is_symlink():
        raise PersistenceError(f"refusing to replace persistent run root: {persistent_root}")

    local_root_exists = local_root.exists() or local_root.is_symlink()
    if local_root_exists:
        local_root = _canonical_existing_directory(local_root, "local_output_root")
    elif args.launcher_exit_code == 0:
        raise PersistenceError("successful launcher did not produce a formal output tree")

    experiment_root = persistent_parent / args.experiment_id
    if experiment_root.exists() or experiment_root.is_symlink():
        raise PersistenceError(f"refusing to reuse persistent experiment root: {experiment_root}")
    experiment_root.mkdir(mode=0o700)
    stage_root = experiment_root / f".{args.run_id}.incomplete-{os.getpid()}"
    stage_root.mkdir(mode=0o700)

    archive_path: Path | None = None
    archive_bytes: int | None = None
    local_file_count = 0
    local_total_bytes = 0
    try:
        if local_root_exists:
            local_file_count, local_total_bytes = _scan_regular_tree(local_root)
            archive_path = local_parent / f".{args.run_id}.{os.getpid()}.{ARCHIVE_NAME}.tmp"
            archive_bytes = _create_archive(local_root, archive_path)
            for entry in sorted(local_root.iterdir(), key=lambda item: item.name):
                target = stage_root / entry.name
                mode = os.lstat(entry).st_mode
                if stat.S_ISDIR(mode):
                    _copy_tree_without_metadata(entry, target)
                elif stat.S_ISREG(mode):
                    _copy_regular_file(entry, target)
                else:
                    raise PersistenceError(
                        f"formal output contains an unsupported entry: {entry}"
                    )
            _copy_regular_file(archive_path, stage_root / ARCHIVE_NAME)

        terminal_status = (
            "complete"
            if args.launcher_exit_code == 0
            else "failed_partial" if local_root_exists else "failed_before_formal_output"
        )
        status_document: dict[str, Any] = {
            "schema_version": "local-formal-evidence-oss-mirror-v1",
            "experiment_id": args.experiment_id,
            "run_id": args.run_id,
            "control_commit": args.control_commit,
            "job_id": args.job_id,
            "launcher_exit_code": args.launcher_exit_code,
            "terminal_status": terminal_status,
            "publication_complete": True,
            "persisted_at": datetime.now(timezone.utc).isoformat(),
            "source_local_root": str(expected_local_root),
            "persistent_root": str(expected_persistent_root),
            "local_output_tree_present": local_root_exists,
            "local_regular_file_count": local_file_count,
            "local_regular_file_bytes": local_total_bytes,
            "formal_evidence_archive": ARCHIVE_NAME if local_root_exists else None,
            "formal_evidence_archive_bytes": archive_bytes,
            "expanded_mirror": "." if local_root_exists else None,
            "expanded_mirror_preserves_unix_modes": False,
            "formal_evidence_archive_preserves_unix_modes": bool(local_root_exists),
            "artifact_hashes": None,
        }
        _write_exclusive_json(stage_root / STATUS_NAME, status_document)
        _fsync_directory(stage_root)
        _fsync_directory(experiment_root)
        os.rename(stage_root, persistent_root)
        _fsync_directory(experiment_root)
        _fsync_directory(persistent_parent)
    finally:
        if archive_path is not None:
            try:
                os.chmod(archive_path, 0o600)
                archive_path.unlink()
            except FileNotFoundError:
                pass

    return persistent_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-output-parent", required=True)
    parser.add_argument("--local-output-root", required=True)
    parser.add_argument("--persistent-output-parent", required=True)
    parser.add_argument("--persistent-output-root", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--control-commit", required=True)
    parser.add_argument("--launcher-exit-code", required=True, type=int)
    parser.add_argument("--job-id", default="unknown-at-runtime")
    return parser


def main() -> int:
    try:
        persistent_root = persist(_parser().parse_args())
    except (OSError, PersistenceError, tarfile.TarError) as error:
        print(f"ERROR DLC_EVAL_PERSISTENCE_FAILED: {error}", file=sys.stderr)
        return 1
    print(f"DLC_EVAL_PERSISTENCE_READY root={persistent_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
