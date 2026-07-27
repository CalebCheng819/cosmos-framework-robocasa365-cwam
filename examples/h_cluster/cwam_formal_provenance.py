#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Create and verify fail-closed provenance for a formal CWAM Cosmos3 run.

The artifact inventory is expensive to create exactly once: every selected
artifact receives a full SHA-256.  Runtime verification re-hashes small files
in full, while large immutable artifacts use their pinned size plus
deterministic first/middle/last samples.  The latter is portable across
containers, but deliberately does not pretend that a sampled check is a
second full audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_INVENTORY_KIND = "cwam_cosmos3_formal_artifact_inventory"
_DEFAULT_LARGE_FILE_THRESHOLD = 64 * 1024 * 1024
_DEFAULT_SAMPLE_BYTES = 1024 * 1024
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_EXPECTED_CONFIG_PATH = "examples/toml/sft_config/action_policy_robocasa365_nano.toml"
_FORMAL_OUTPUT_ROOT = "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-formal"
_FORMAL_MIN_FREE_BYTES = 5 * 1024**4
_FORMAL_ESTIMATED_REQUIRED_BYTES = 4 * 1024**4
_FORMAL_QUOTA_GROUP = "eailabagent_gpu"
_FORMAL_EXCLUDED_NODE_TAG = "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"
_PROTOCOL_FIELDS = (
    "experiment_family",
    "world_size",
    "per_rank_samples",
    "grad_accum_iter",
    "global_batch",
    "learning_rate",
    "max_iter",
    "warm_up_steps",
    "cycle_length",
    "validation_iter",
    "max_val_iter",
    "save_iter",
    "train_sample_budget",
)
_PROTOCOL_BASE = {
    8: {
        "experiment_family": "CWAM-E35",
        "learning_rate": 5.0e-5,
        "max_iter": 10_000,
        "warm_up_steps": 500,
        "cycle_length": 16_000,
        "validation_iter": 500,
        "max_val_iter": 8,
        "save_iter": 500,
    },
    6: {
        "experiment_family": "CWAM-E35A",
        "learning_rate": 3.75e-5,
        "max_iter": 13_340,
        "warm_up_steps": 667,
        "cycle_length": 21_344,
        "validation_iter": 667,
        "max_val_iter": 11,
        "save_iter": 667,
    },
}

_QWEN_PROCESSOR_FILES = (
    "chat_template.json",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)

_ENV_METADATA_NAMES = frozenset({"METADATA", "RECORD", "direct_url.json"})
_ENV_CRITICAL_GLOB_GROUPS = {
    "cuda_runtime": ("lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",),
    "torch_extension": ("lib/python*/site-packages/torch/_C*.so",),
    "torch_runtime": (
        "lib/python*/site-packages/torch/lib/libc10.so",
        "lib/python*/site-packages/torch/lib/libc10_cuda.so",
        "lib/python*/site-packages/torch/lib/libtorch_cpu.so",
        "lib/python*/site-packages/torch/lib/libtorch_cuda.so",
        "lib/python*/site-packages/torch/lib/libtorch_python.so",
    ),
    "torchcodec": ("lib/python*/site-packages/torchcodec/libtorchcodec_core8.so",),
    "pyav": ("lib/python*/site-packages/av/_core*.so",),
    "pydantic_core": ("lib/python*/site-packages/pydantic_core/_pydantic_core*.so",),
}


class ProvenanceError(RuntimeError):
    """Raised when a formal-run provenance contract is not satisfied."""


def _formal_protocol_for_world_size(world_size: int) -> dict[str, str | int | float]:
    try:
        base = _PROTOCOL_BASE[world_size]
    except KeyError as error:
        raise ProvenanceError(f"formal world size must be 6 or 8, got {world_size}") from error
    per_rank_samples = 128
    grad_accum_iter = 2
    global_batch = world_size * per_rank_samples * grad_accum_iter
    protocol = {
        "experiment_family": base["experiment_family"],
        "world_size": world_size,
        "per_rank_samples": per_rank_samples,
        "grad_accum_iter": grad_accum_iter,
        "global_batch": global_batch,
        "learning_rate": base["learning_rate"],
        "max_iter": base["max_iter"],
        "warm_up_steps": base["warm_up_steps"],
        "cycle_length": base["cycle_length"],
        "validation_iter": base["validation_iter"],
        "max_val_iter": base["max_val_iter"],
        "save_iter": base["save_iter"],
        "train_sample_budget": global_batch * int(base["max_iter"]),
    }
    if tuple(protocol) != _PROTOCOL_FIELDS:
        raise AssertionError("internal formal protocol field order drifted")
    return protocol


def _verify_formal_protocol(
    formal: dict[str, Any],
    *,
    experiment_id: str,
    world_size: int,
) -> dict[str, str | int | float]:
    expected = _formal_protocol_for_world_size(world_size)
    expected_family = str(expected["experiment_family"])
    if not experiment_id.startswith(f"{expected_family}-"):
        raise ProvenanceError(
            f"world_size={world_size} requires an experiment ID prefixed by {expected_family}-, got {experiment_id!r}"
        )

    observed = _require_mapping(formal.get("protocol"), "formal protocol")
    expected_keys = set(_PROTOCOL_FIELDS)
    observed_keys = set(observed)
    if observed_keys != expected_keys:
        raise ProvenanceError(
            "formal protocol key mismatch: "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}"
        )
    for field in _PROTOCOL_FIELDS:
        actual_value = observed[field]
        expected_value = expected[field]
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ProvenanceError(
                f"formal protocol {field} mismatch for {expected_family}: "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )
    return expected


def _verify_exact_contract(
    formal: dict[str, Any],
    *,
    key: str,
    expected: dict[str, str | int],
) -> dict[str, str | int]:
    observed = _require_mapping(formal.get(key), f"formal {key}")
    if set(observed) != set(expected):
        raise ProvenanceError(
            f"formal {key} key mismatch: "
            f"missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    for field, expected_value in expected.items():
        actual_value = observed[field]
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ProvenanceError(
                f"formal {key}.{field} mismatch: expected={expected_value!r}, actual={actual_value!r}"
            )
    return expected


def _formal_storage_contract() -> dict[str, str | int]:
    return {
        "output_root": _FORMAL_OUTPUT_ROOT,
        "minimum_free_bytes": _FORMAL_MIN_FREE_BYTES,
        "estimated_required_bytes": _FORMAL_ESTIMATED_REQUIRED_BYTES,
    }


def _formal_rjob_contract(world_size: int) -> dict[str, str | int]:
    return {
        "quota_group": _FORMAL_QUOTA_GROUP,
        "cpu": 128 if world_size == 8 else 96,
        "memory_mib": 1_500_000 if world_size == 8 else 1_125_000,
        "negative_tag": _FORMAL_EXCLUDED_NODE_TAG,
        "task_type": "normal",
        "private_machine": "group",
        "priority": 9,
        "restart_policy": "never",
        "backoff_limit": 1,
    }


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProvenanceError(f"{label} must be a lowercase SHA-256, got {value!r}")
    return value


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProvenanceError(f"{label} must be a JSON object")
    return value


def _require_regular(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ProvenanceError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ProvenanceError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ProvenanceError(f"{label} must be a regular file: {path}")
    return metadata


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def sha256_file(path: Path) -> str:
    """Return a full SHA-256 while rejecting an in-flight file replacement."""
    before = _require_regular(path, "hashed file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    after = _require_regular(path, "hashed file")
    if _stable_identity(before) != _stable_identity(after):
        raise ProvenanceError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def sampled_sha256(path: Path, *, sample_bytes: int) -> str:
    """Hash deterministic first/middle/last regions with offsets and file size."""
    if sample_bytes < 1:
        raise ProvenanceError(f"sample_bytes must be positive, got {sample_bytes}")
    before = _require_regular(path, "sampled file")
    size = int(before.st_size)
    width = min(sample_bytes, size)
    offsets = sorted({0, max(0, (size - width) // 2), max(0, size - width)})
    digest = hashlib.sha256()
    digest.update(b"CWAM-SAMPLED-FILE-V1\0")
    digest.update(size.to_bytes(16, "big"))
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            payload = handle.read(width)
            if len(payload) != width:
                raise ProvenanceError(f"short sampled read at offset {offset}: {path}")
            digest.update(offset.to_bytes(16, "big"))
            digest.update(len(payload).to_bytes(16, "big"))
            digest.update(payload)
    after = _require_regular(path, "sampled file")
    if _stable_identity(before) != _stable_identity(after):
        raise ProvenanceError(f"file changed while sampling: {path}")
    return digest.hexdigest()


def _safe_relative_path(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ProvenanceError(f"{label} must be a non-empty string")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ProvenanceError(f"{label} is not a safe relative path: {raw!r}")
    if relative.as_posix() != raw:
        raise ProvenanceError(f"{label} is not canonically encoded: {raw!r}")
    return raw


def _canonical_root(path: Path, *, allowed_root: Path | None = None) -> Path:
    raw = path.expanduser()
    if ".." in raw.parts:
        raise ProvenanceError(f"path must not contain '..': {path}")
    absolute = Path(os.path.abspath(raw))
    canonical = raw.resolve(strict=True)
    if canonical != absolute:
        raise ProvenanceError(f"path must not traverse symlinks: requested={absolute}, resolved={canonical}")
    if allowed_root is not None:
        allowed = allowed_root.expanduser().resolve(strict=True)
        try:
            relative = canonical.relative_to(allowed)
        except ValueError as error:
            raise ProvenanceError(f"path escapes allowed root {allowed}: {canonical}") from error
        if not relative.parts:
            raise ProvenanceError(f"path must be a child of the allowed root, not the root itself: {canonical}")
    metadata = canonical.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ProvenanceError(f"path is not a directory: {canonical}")
    return canonical


def _scan_regular_tree(root: Path, *, reject_writable: bool) -> dict[str, Path]:
    root_metadata = root.lstat()
    if reject_writable and root_metadata.st_mode & 0o222:
        raise ProvenanceError(f"tree root is write-enabled: {root}")
    files: dict[str, Path] = {}
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                metadata = entry.stat(follow_symlinks=False)
                relative = path.relative_to(root).as_posix()
                if stat.S_ISLNK(metadata.st_mode):
                    raise ProvenanceError(f"tree contains a symlink: {relative}")
                if stat.S_ISDIR(metadata.st_mode):
                    if reject_writable and metadata.st_mode & 0o222:
                        raise ProvenanceError(f"tree contains a write-enabled directory: {relative}")
                    stack.append(path)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise ProvenanceError(f"tree contains a special file: {relative}")
                if reject_writable and metadata.st_mode & 0o222:
                    raise ProvenanceError(f"tree contains a write-enabled file: {relative}")
                files[relative] = path
    return files


def verify_snapshot(
    *,
    snapshot_root: Path,
    allowed_root: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        "expected snapshot manifest SHA-256",
    )
    root = _canonical_root(snapshot_root, allowed_root=allowed_root)
    manifest_path = root / "SNAPSHOT_MANIFEST.json"
    manifest_metadata = _require_regular(manifest_path, "snapshot manifest")
    if manifest_metadata.st_mode & 0o222:
        raise ProvenanceError(f"snapshot manifest is write-enabled: {manifest_path}")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ProvenanceError(
            f"snapshot manifest checksum mismatch: expected={expected_manifest_sha256}, actual={actual_manifest_sha256}"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProvenanceError(f"invalid snapshot manifest JSON: {manifest_path}") from error
    if manifest.get("schema_version") != 1:
        raise ProvenanceError(f"unsupported snapshot schema_version: {manifest.get('schema_version')!r}")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ProvenanceError("snapshot manifest files must be a list")
    if manifest.get("file_count") != len(entries):
        raise ProvenanceError(
            f"snapshot file_count mismatch: declared={manifest.get('file_count')!r}, actual={len(entries)}"
        )

    expected: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(entries):
        entry = _require_mapping(value, f"snapshot files[{index}]")
        relative = _safe_relative_path(entry.get("path"), f"snapshot files[{index}].path")
        if relative == "SNAPSHOT_MANIFEST.json":
            raise ProvenanceError("snapshot manifest must not list itself")
        if relative in expected:
            raise ProvenanceError(f"duplicate snapshot manifest path: {relative}")
        size = entry.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ProvenanceError(f"invalid snapshot byte count for {relative}: {size!r}")
        _require_sha256(entry.get("sha256"), f"snapshot SHA-256 for {relative}")
        expected[relative] = entry

    actual = _scan_regular_tree(root, reject_writable=True)
    actual.pop("SNAPSHOT_MANIFEST.json", None)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ProvenanceError(f"snapshot file set mismatch: missing={missing}, extra={extra}")

    total_bytes = 0
    for relative in sorted(expected):
        path = actual[relative]
        metadata = _require_regular(path, f"snapshot file {relative}")
        expected_size = int(expected[relative]["bytes"])
        if metadata.st_size != expected_size:
            raise ProvenanceError(
                f"snapshot size mismatch for {relative}: expected={expected_size}, actual={metadata.st_size}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected[relative]["sha256"]:
            raise ProvenanceError(
                f"snapshot checksum mismatch for {relative}: "
                f"expected={expected[relative]['sha256']}, actual={actual_sha256}"
            )
        total_bytes += expected_size

    return {
        "schema_version": 1,
        "status": "passed",
        "snapshot_root": str(root),
        "snapshot_manifest": str(manifest_path),
        "snapshot_manifest_sha256": actual_manifest_sha256,
        "file_count": len(expected),
        "file_bytes": total_bytes,
    }


def _audit_file(
    path: Path,
    *,
    relative_path: str | None,
    large_file_threshold: int,
    sample_bytes: int,
    force_runtime_full_sha256: bool = False,
) -> dict[str, Any]:
    metadata = _require_regular(path, "artifact")
    full_sha256 = sha256_file(path)
    entry: dict[str, Any] = {
        "bytes": int(metadata.st_size),
        "sha256": full_sha256,
    }
    if relative_path is not None:
        entry["path"] = relative_path
    if force_runtime_full_sha256 or metadata.st_size <= large_file_threshold:
        entry["runtime_verification"] = {"mode": "full_sha256"}
    else:
        entry["runtime_verification"] = {
            "mode": "size_and_sample_sha256",
            "sample_bytes": sample_bytes,
            "sample_sha256": sampled_sha256(path, sample_bytes=sample_bytes),
        }
    after = _require_regular(path, "artifact")
    if _stable_identity(metadata) != _stable_identity(after):
        raise ProvenanceError(f"artifact changed while creating inventory: {path}")
    return entry


def _audit_paths(
    root: Path,
    relative_paths: list[str],
    *,
    large_file_threshold: int,
    sample_bytes: int,
    force_runtime_full_sha256: bool = False,
) -> list[dict[str, Any]]:
    output = []
    for relative in sorted(relative_paths):
        _safe_relative_path(relative, "artifact inventory path")
        output.append(
            _audit_file(
                root / relative,
                relative_path=relative,
                large_file_threshold=large_file_threshold,
                sample_bytes=sample_bytes,
                force_runtime_full_sha256=force_runtime_full_sha256,
            )
        )
    return output


def _environment_evidence_paths(environment: Path) -> tuple[list[str], list[str]]:
    site_packages = sorted(environment.glob("lib/python*/site-packages"))
    if len(site_packages) != 1 or not site_packages[0].is_dir():
        raise ProvenanceError(f"expected one Python site-packages tree under {environment}, got {site_packages}")

    metadata_paths = []
    for path in site_packages[0].rglob("*"):
        if path.name not in _ENV_METADATA_NAMES or not path.is_file():
            continue
        if not any(parent.name.endswith(".dist-info") for parent in path.parents):
            continue
        metadata_paths.append(path.relative_to(environment).as_posix())
    if not metadata_paths:
        raise ProvenanceError(f"environment has no dist-info evidence under {site_packages[0]}")

    nvidia_root = site_packages[0] / "nvidia"
    cudart_alias = nvidia_root / "cudart"
    if not cudart_alias.is_symlink() or os.readlink(cudart_alias) != "cuda_runtime":
        raise ProvenanceError(
            f"environment cudart compatibility alias must be exactly cudart -> cuda_runtime: {cudart_alias}"
        )
    try:
        resolved_alias = cudart_alias.resolve(strict=True)
        expected_target = (nvidia_root / "cuda_runtime").resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(f"environment cudart compatibility alias is dangling: {cudart_alias}") from error
    if resolved_alias != expected_target or not expected_target.is_dir():
        raise ProvenanceError(
            f"environment cudart compatibility alias escapes its expected target: "
            f"alias={resolved_alias}, expected={expected_target}"
        )

    critical_paths: set[str] = set()
    for label, patterns in _ENV_CRITICAL_GLOB_GROUPS.items():
        for pattern in patterns:
            matches = {path for path in environment.glob(pattern) if path.is_file()}
            if not matches:
                raise ProvenanceError(f"environment has no critical runtime file for {label}: pattern={pattern}")
            critical_paths.update(path.relative_to(environment).as_posix() for path in matches)
    critical_paths.update({".cwam_env_ready.json", "pyvenv.cfg"})
    return sorted(set(metadata_paths)), sorted(critical_paths)


def _load_json_regular(path: Path, label: str) -> dict[str, Any]:
    _require_regular(path, label)
    try:
        return _require_mapping(json.loads(path.read_text()), label)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProvenanceError(f"{label} is not valid JSON: {path}") from error


def _processed_dataset_contract(
    dataset_root: Path,
    *,
    expected_source_root: Path | None,
    verify_payload_sha256: bool,
) -> dict[str, Any]:
    root = _canonical_root(dataset_root)
    actual_files = _scan_regular_tree(root, reject_writable=False)
    success_path = root / "_SUCCESS.json"
    manifest_path = root / "dataset_manifest.json"
    success = _load_json_regular(success_path, "processed dataset _SUCCESS")
    manifest = _load_json_regular(manifest_path, "processed dataset manifest")

    if success.get("status") != "complete" or success.get("dataset_manifest") != "dataset_manifest.json":
        raise ProvenanceError("processed dataset _SUCCESS does not declare a complete dataset_manifest.json")
    manifest_sha256 = sha256_file(manifest_path)
    declared_manifest_sha256 = _require_sha256(
        success.get("dataset_manifest_sha256"),
        "processed dataset manifest SHA-256 in _SUCCESS",
    )
    if manifest_sha256 != declared_manifest_sha256:
        raise ProvenanceError(
            "processed dataset manifest does not match _SUCCESS: "
            f"declared={declared_manifest_sha256}, actual={manifest_sha256}"
        )
    if manifest.get("schema_version") != 1:
        raise ProvenanceError(f"unsupported processed dataset schema_version: {manifest.get('schema_version')!r}")
    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ProvenanceError(f"processed dataset manifest has invalid dataset_id: {dataset_id!r}")
    if success.get("dataset_id") != dataset_id:
        raise ProvenanceError(
            "processed dataset ID mismatch between _SUCCESS and manifest: "
            f"success={success.get('dataset_id')!r}, manifest={dataset_id!r}"
        )

    inputs = _require_mapping(manifest.get("inputs"), "processed dataset inputs")
    source_root_value = inputs.get("source_root")
    if not isinstance(source_root_value, str) or not source_root_value.startswith("/"):
        raise ProvenanceError(f"processed dataset source_root must be an absolute path: {source_root_value!r}")
    if expected_source_root is not None:
        canonical_source_root = _canonical_root(expected_source_root)
        if source_root_value != str(canonical_source_root):
            raise ProvenanceError(
                "processed dataset source root mismatch: "
                f"manifest={source_root_value!r}, runtime={str(canonical_source_root)!r}"
            )

    raw_entries = manifest.get("files")
    if not isinstance(raw_entries, dict):
        raise ProvenanceError("processed dataset manifest files must be an object")
    expected_files: dict[str, dict[str, Any]] = {}
    for raw_relative, raw_entry in raw_entries.items():
        relative = _safe_relative_path(raw_relative, "processed dataset manifest path")
        if relative in {"_SUCCESS.json", "dataset_manifest.json"}:
            raise ProvenanceError(f"processed dataset payload must not list control file {relative}")
        entry = _require_mapping(raw_entry, f"processed dataset manifest entry {relative}")
        size = entry.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ProvenanceError(f"processed dataset has invalid byte count for {relative}: {size!r}")
        _require_sha256(entry.get("sha256"), f"processed dataset SHA-256 for {relative}")
        expected_files[relative] = entry

    expected_tree_files = set(expected_files) | {"_SUCCESS.json", "dataset_manifest.json"}
    if set(actual_files) != expected_tree_files:
        raise ProvenanceError(
            "processed dataset file set mismatch: "
            f"missing={sorted(expected_tree_files - set(actual_files))}, "
            f"extra={sorted(set(actual_files) - expected_tree_files)}"
        )

    for relative, entry in sorted(expected_files.items()):
        path = actual_files[relative]
        metadata = _require_regular(path, f"processed dataset file {relative}")
        if metadata.st_size != entry["bytes"]:
            raise ProvenanceError(
                f"processed dataset size mismatch for {relative}: expected={entry['bytes']}, actual={metadata.st_size}"
            )
        if verify_payload_sha256:
            actual_sha256 = sha256_file(path)
            if actual_sha256 != entry["sha256"]:
                raise ProvenanceError(
                    f"processed dataset checksum mismatch for {relative}: "
                    f"expected={entry['sha256']}, actual={actual_sha256}"
                )

    return {
        "root": root,
        "success": success,
        "success_path": success_path,
        "success_sha256": sha256_file(success_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "dataset_id": dataset_id,
        "source_root": source_root_value,
        "files": expected_files,
    }


def _processed_dataset_inventory_entries(
    entries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for relative, entry in sorted(entries.items()):
        output.append(
            {
                "path": relative,
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
                "runtime_verification": {"mode": "full_sha256"},
            }
        )
    return output


def _raw_video_files(source_root: Path) -> tuple[Path, dict[str, Path]]:
    root = _canonical_root(source_root)
    video_root = _canonical_root(root / "videos", allowed_root=root)
    video_files = _scan_regular_tree(video_root, reject_writable=False)
    if not video_files:
        raise ProvenanceError(f"raw source videos tree is empty: {video_root}")
    prefixed = {f"videos/{relative}": path for relative, path in video_files.items()}
    return root, prefixed


def create_artifact_inventory(
    *,
    base_checkpoint: Path,
    wan_vae: Path,
    qwen_tokenizer: Path,
    environment: Path,
    dataset_root: Path,
    source_root: Path,
    large_file_threshold: int = _DEFAULT_LARGE_FILE_THRESHOLD,
    sample_bytes: int = _DEFAULT_SAMPLE_BYTES,
) -> dict[str, Any]:
    if large_file_threshold < 1 or sample_bytes < 1:
        raise ProvenanceError("large-file threshold and sample size must both be positive")
    checkpoint_root = _canonical_root(base_checkpoint)
    checkpoint_files = _scan_regular_tree(checkpoint_root, reject_writable=False)
    if "checkpoint.json" not in checkpoint_files or "model/.metadata" not in checkpoint_files:
        raise ProvenanceError("base checkpoint must contain checkpoint.json and model/.metadata")
    dcp_shards = sorted(path for path in checkpoint_files if path.startswith("model/") and path.endswith(".distcp"))
    if len(dcp_shards) != 6:
        raise ProvenanceError(f"base checkpoint must contain six model DCP shards, got {dcp_shards}")

    vae_path = wan_vae.expanduser().resolve(strict=True)
    _require_regular(vae_path, "Wan VAE")
    qwen_root = _canonical_root(qwen_tokenizer)
    for relative in _QWEN_PROCESSOR_FILES:
        _require_regular(qwen_root / relative, f"Qwen processor file {relative}")

    processed_dataset = _processed_dataset_contract(
        dataset_root,
        expected_source_root=source_root,
        verify_payload_sha256=True,
    )
    raw_source_root, raw_video_files = _raw_video_files(source_root)

    environment_root = _canonical_root(environment)
    marker_path = environment_root / ".cwam_env_ready.json"
    marker = _load_json_regular(marker_path, "environment marker")
    if marker.get("status") != "ready":
        raise ProvenanceError(f"environment marker is not ready: {marker.get('status')!r}")
    metadata_paths, critical_paths = _environment_evidence_paths(environment_root)

    python_link = environment_root / "bin/python"
    python_link_metadata = python_link.lstat()
    if not (stat.S_ISLNK(python_link_metadata.st_mode) or stat.S_ISREG(python_link_metadata.st_mode)):
        raise ProvenanceError(f"environment Python is not a regular file or symlink: {python_link}")
    python_target = python_link.resolve(strict=True)
    python_target_entry = _audit_file(
        python_target,
        relative_path=None,
        large_file_threshold=large_file_threshold,
        sample_bytes=sample_bytes,
    )

    return {
        "schema_version": 1,
        "kind": _INVENTORY_KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_policy": {
            "large_file_threshold_bytes": large_file_threshold,
            "large_file_verification": "size+first/middle/last sampled SHA-256",
            "large_file_boundary": (
                "A full SHA-256 is recorded during inventory creation. Runtime sampling is "
                "portable change detection only and cannot prove unsampled bytes unchanged."
            ),
            "sample_bytes_per_region": sample_bytes,
            "small_file_verification": "full SHA-256",
            "processed_dataset_payload_verification": "full SHA-256 for every manifest-listed file",
            "full_sha256_recorded_for_every_selected_file": True,
        },
        "artifacts": {
            "base_checkpoint": {
                "kind": "exact_tree",
                "root": str(checkpoint_root),
                "files": _audit_paths(
                    checkpoint_root,
                    list(checkpoint_files),
                    large_file_threshold=large_file_threshold,
                    sample_bytes=sample_bytes,
                ),
            },
            "wan_vae": {
                "kind": "file",
                "path": str(vae_path),
                "file": _audit_file(
                    vae_path,
                    relative_path=None,
                    large_file_threshold=large_file_threshold,
                    sample_bytes=sample_bytes,
                ),
            },
            "qwen_processor": {
                "kind": "selected_files",
                "root": str(qwen_root),
                "required_files": list(_QWEN_PROCESSOR_FILES),
                "files": _audit_paths(
                    qwen_root,
                    list(_QWEN_PROCESSOR_FILES),
                    large_file_threshold=large_file_threshold,
                    sample_bytes=sample_bytes,
                ),
            },
            "processed_dataset": {
                "kind": "manifest_exact_tree",
                "root": str(processed_dataset["root"]),
                "dataset_id": processed_dataset["dataset_id"],
                "source_root": processed_dataset["source_root"],
                "success": _audit_file(
                    processed_dataset["success_path"],
                    relative_path=None,
                    large_file_threshold=large_file_threshold,
                    sample_bytes=sample_bytes,
                    force_runtime_full_sha256=True,
                ),
                "manifest": _audit_file(
                    processed_dataset["manifest_path"],
                    relative_path=None,
                    large_file_threshold=large_file_threshold,
                    sample_bytes=sample_bytes,
                    force_runtime_full_sha256=True,
                ),
                "files": _processed_dataset_inventory_entries(processed_dataset["files"]),
            },
            "raw_video_source": {
                "kind": "exact_tree",
                "root": str(raw_source_root),
                "tree": "videos",
                "files": _audit_paths(
                    raw_source_root,
                    list(raw_video_files),
                    large_file_threshold=large_file_threshold,
                    sample_bytes=sample_bytes,
                ),
            },
            "environment": {
                "kind": "environment_evidence",
                "root": str(environment_root),
                "marker_payload": marker,
                "metadata_paths": metadata_paths,
                "critical_paths": critical_paths,
                "files": _audit_paths(
                    environment_root,
                    sorted(set(metadata_paths) | set(critical_paths)),
                    large_file_threshold=large_file_threshold,
                    sample_bytes=sample_bytes,
                ),
                "python": {
                    "path": "bin/python",
                    "link_target": os.readlink(python_link) if python_link.is_symlink() else None,
                    "resolved_path": str(python_target),
                    "resolved_file": python_target_entry,
                },
            },
        },
    }


def write_inventory(payload: dict[str, Any], output: Path) -> str:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ProvenanceError(f"refusing to overwrite an existing artifact inventory: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ProvenanceError(f"refusing to reuse an existing artifact inventory temporary file: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    output.chmod(0o444)
    return sha256_file(output)


def _inventory_entries(value: object, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ProvenanceError(f"{label} files must be a list")
    entries: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(value):
        entry = _require_mapping(raw_entry, f"{label} files[{index}]")
        relative = _safe_relative_path(entry.get("path"), f"{label} files[{index}].path")
        if relative in entries:
            raise ProvenanceError(f"duplicate {label} path: {relative}")
        entries[relative] = entry
    return entries


def _verify_inventory_file(path: Path, entry: dict[str, Any], label: str) -> str:
    metadata = _require_regular(path, label)
    expected_size = entry.get("bytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        raise ProvenanceError(f"{label} has an invalid inventory byte count: {expected_size!r}")
    if metadata.st_size != expected_size:
        raise ProvenanceError(f"{label} size mismatch: expected={expected_size}, actual={metadata.st_size}")
    _require_sha256(entry.get("sha256"), f"{label} full SHA-256")
    runtime = _require_mapping(entry.get("runtime_verification"), f"{label} runtime_verification")
    mode = runtime.get("mode")
    if mode == "full_sha256":
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ProvenanceError(f"{label} checksum mismatch: expected={entry['sha256']}, actual={actual}")
        return mode
    if mode != "size_and_sample_sha256":
        raise ProvenanceError(f"{label} has unsupported runtime verification mode: {mode!r}")
    sample_bytes = runtime.get("sample_bytes")
    if not isinstance(sample_bytes, int) or isinstance(sample_bytes, bool) or sample_bytes < 1:
        raise ProvenanceError(f"{label} has invalid sample_bytes: {sample_bytes!r}")
    expected_sample = _require_sha256(runtime.get("sample_sha256"), f"{label} sampled SHA-256")
    actual_sample = sampled_sha256(path, sample_bytes=sample_bytes)
    if actual_sample != expected_sample:
        raise ProvenanceError(f"{label} sampled checksum mismatch: expected={expected_sample}, actual={actual_sample}")
    return mode


def verify_artifact_inventory(
    *,
    inventory_path: Path,
    expected_inventory_sha256: str,
    base_checkpoint: Path,
    wan_vae: Path,
    qwen_tokenizer: Path,
    environment: Path,
    dataset_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    expected_inventory_sha256 = _require_sha256(expected_inventory_sha256, "artifact inventory SHA-256")
    actual_inventory_sha256 = sha256_file(inventory_path)
    if actual_inventory_sha256 != expected_inventory_sha256:
        raise ProvenanceError(
            "artifact inventory checksum mismatch: "
            f"expected={expected_inventory_sha256}, actual={actual_inventory_sha256}"
        )
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("schema_version") != 1 or inventory.get("kind") != _INVENTORY_KIND:
        raise ProvenanceError("unsupported artifact inventory schema or kind")
    artifacts = _require_mapping(inventory.get("artifacts"), "artifact inventory artifacts")

    expected_roots = {
        "base_checkpoint": _canonical_root(base_checkpoint),
        "qwen_processor": _canonical_root(qwen_tokenizer),
        "environment": _canonical_root(environment),
        "processed_dataset": _canonical_root(dataset_root),
        "raw_video_source": _canonical_root(source_root),
    }
    verification_counts = {"full_sha256": 0, "size_and_sample_sha256": 0}

    checkpoint = _require_mapping(artifacts.get("base_checkpoint"), "base_checkpoint inventory")
    if checkpoint.get("kind") != "exact_tree":
        raise ProvenanceError("base checkpoint inventory must use exact_tree")
    checkpoint_root = expected_roots["base_checkpoint"]
    if checkpoint.get("root") != str(checkpoint_root):
        raise ProvenanceError(
            f"base checkpoint root mismatch: inventory={checkpoint.get('root')!r}, runtime={str(checkpoint_root)!r}"
        )
    checkpoint_entries = _inventory_entries(checkpoint.get("files"), "base checkpoint")
    checkpoint_files = _scan_regular_tree(checkpoint_root, reject_writable=False)
    if set(checkpoint_entries) != set(checkpoint_files):
        raise ProvenanceError(
            "base checkpoint file set mismatch: "
            f"missing={sorted(set(checkpoint_entries) - set(checkpoint_files))}, "
            f"extra={sorted(set(checkpoint_files) - set(checkpoint_entries))}"
        )
    for relative, entry in checkpoint_entries.items():
        mode = _verify_inventory_file(checkpoint_files[relative], entry, f"base checkpoint {relative}")
        verification_counts[mode] += 1

    vae = _require_mapping(artifacts.get("wan_vae"), "Wan VAE inventory")
    vae_path = wan_vae.expanduser().resolve(strict=True)
    if vae.get("kind") != "file" or vae.get("path") != str(vae_path):
        raise ProvenanceError(f"Wan VAE path mismatch: inventory={vae.get('path')!r}, runtime={str(vae_path)!r}")
    mode = _verify_inventory_file(vae_path, _require_mapping(vae.get("file"), "Wan VAE file"), "Wan VAE")
    verification_counts[mode] += 1

    qwen = _require_mapping(artifacts.get("qwen_processor"), "Qwen processor inventory")
    qwen_root = expected_roots["qwen_processor"]
    if qwen.get("kind") != "selected_files" or qwen.get("root") != str(qwen_root):
        raise ProvenanceError(
            f"Qwen processor root mismatch: inventory={qwen.get('root')!r}, runtime={str(qwen_root)!r}"
        )
    if qwen.get("required_files") != list(_QWEN_PROCESSOR_FILES):
        raise ProvenanceError("Qwen processor required-file contract mismatch")
    qwen_entries = _inventory_entries(qwen.get("files"), "Qwen processor")
    if set(qwen_entries) != set(_QWEN_PROCESSOR_FILES):
        raise ProvenanceError("Qwen processor inventory does not contain exactly the required processor files")
    for relative, entry in qwen_entries.items():
        mode = _verify_inventory_file(qwen_root / relative, entry, f"Qwen processor {relative}")
        verification_counts[mode] += 1

    processed = _require_mapping(artifacts.get("processed_dataset"), "processed dataset inventory")
    processed_root = expected_roots["processed_dataset"]
    if processed.get("kind") != "manifest_exact_tree" or processed.get("root") != str(processed_root):
        raise ProvenanceError(
            f"processed dataset root mismatch: inventory={processed.get('root')!r}, runtime={str(processed_root)!r}"
        )
    processed_contract = _processed_dataset_contract(
        processed_root,
        expected_source_root=expected_roots["raw_video_source"],
        verify_payload_sha256=False,
    )
    if processed.get("dataset_id") != processed_contract["dataset_id"]:
        raise ProvenanceError(
            "processed dataset ID mismatch: "
            f"inventory={processed.get('dataset_id')!r}, runtime={processed_contract['dataset_id']!r}"
        )
    if processed.get("source_root") != processed_contract["source_root"]:
        raise ProvenanceError(
            "processed dataset source root changed: "
            f"inventory={processed.get('source_root')!r}, runtime={processed_contract['source_root']!r}"
        )
    mode = _verify_inventory_file(
        processed_contract["success_path"],
        _require_mapping(processed.get("success"), "processed dataset _SUCCESS inventory"),
        "processed dataset _SUCCESS",
    )
    if mode != "full_sha256":
        raise ProvenanceError("processed dataset _SUCCESS must use full SHA-256 at runtime")
    verification_counts[mode] += 1
    mode = _verify_inventory_file(
        processed_contract["manifest_path"],
        _require_mapping(processed.get("manifest"), "processed dataset manifest inventory"),
        "processed dataset manifest",
    )
    if mode != "full_sha256":
        raise ProvenanceError("processed dataset manifest must use full SHA-256 at runtime")
    verification_counts[mode] += 1
    processed_entries = _inventory_entries(processed.get("files"), "processed dataset")
    if set(processed_entries) != set(processed_contract["files"]):
        raise ProvenanceError(
            "processed dataset inventory file set mismatch: "
            f"missing={sorted(set(processed_contract['files']) - set(processed_entries))}, "
            f"extra={sorted(set(processed_entries) - set(processed_contract['files']))}"
        )
    for relative, entry in processed_entries.items():
        manifest_entry = processed_contract["files"][relative]
        if entry.get("bytes") != manifest_entry["bytes"] or entry.get("sha256") != manifest_entry["sha256"]:
            raise ProvenanceError(f"processed dataset inventory does not match dataset_manifest.json for {relative}")
        mode = _verify_inventory_file(
            processed_root / relative,
            entry,
            f"processed dataset {relative}",
        )
        if mode != "full_sha256":
            raise ProvenanceError(f"processed dataset payload must use full SHA-256 at runtime: {relative}")
        verification_counts[mode] += 1

    raw_source = _require_mapping(artifacts.get("raw_video_source"), "raw video source inventory")
    raw_source_root = expected_roots["raw_video_source"]
    if (
        raw_source.get("kind") != "exact_tree"
        or raw_source.get("root") != str(raw_source_root)
        or raw_source.get("tree") != "videos"
    ):
        raise ProvenanceError(
            "raw video source root/tree mismatch: "
            f"inventory_root={raw_source.get('root')!r}, runtime_root={str(raw_source_root)!r}, "
            f"tree={raw_source.get('tree')!r}"
        )
    _, current_raw_video_files = _raw_video_files(raw_source_root)
    raw_entries = _inventory_entries(raw_source.get("files"), "raw video source")
    if set(raw_entries) != set(current_raw_video_files):
        raise ProvenanceError(
            "raw video source file set mismatch: "
            f"missing={sorted(set(raw_entries) - set(current_raw_video_files))}, "
            f"extra={sorted(set(current_raw_video_files) - set(raw_entries))}"
        )
    for relative, entry in raw_entries.items():
        mode = _verify_inventory_file(
            current_raw_video_files[relative],
            entry,
            f"raw video source {relative}",
        )
        verification_counts[mode] += 1

    env = _require_mapping(artifacts.get("environment"), "environment inventory")
    environment_root = expected_roots["environment"]
    if env.get("kind") != "environment_evidence" or env.get("root") != str(environment_root):
        raise ProvenanceError(
            f"environment root mismatch: inventory={env.get('root')!r}, runtime={str(environment_root)!r}"
        )
    current_metadata_paths, current_critical_paths = _environment_evidence_paths(environment_root)
    if env.get("metadata_paths") != current_metadata_paths:
        raise ProvenanceError("environment dist-info evidence set changed after inventory creation")
    if env.get("critical_paths") != current_critical_paths:
        raise ProvenanceError("environment critical-runtime evidence set changed after inventory creation")
    env_entries = _inventory_entries(env.get("files"), "environment")
    expected_env_paths = set(current_metadata_paths) | set(current_critical_paths)
    if set(env_entries) != expected_env_paths:
        raise ProvenanceError("environment inventory file set does not match its declared evidence sets")
    for relative, entry in env_entries.items():
        mode = _verify_inventory_file(environment_root / relative, entry, f"environment {relative}")
        verification_counts[mode] += 1

    python = _require_mapping(env.get("python"), "environment Python inventory")
    python_path = environment_root / "bin/python"
    python_metadata = python_path.lstat()
    observed_link = os.readlink(python_path) if stat.S_ISLNK(python_metadata.st_mode) else None
    if python.get("path") != "bin/python" or python.get("link_target") != observed_link:
        raise ProvenanceError(
            f"environment Python link mismatch: expected={python.get('link_target')!r}, actual={observed_link!r}"
        )
    resolved_python = python_path.resolve(strict=True)
    if python.get("resolved_path") != str(resolved_python):
        raise ProvenanceError(
            f"environment Python target mismatch: expected={python.get('resolved_path')!r}, "
            f"actual={str(resolved_python)!r}"
        )
    mode = _verify_inventory_file(
        resolved_python,
        _require_mapping(python.get("resolved_file"), "environment Python target"),
        "environment Python target",
    )
    verification_counts[mode] += 1

    return {
        "schema_version": 1,
        "status": "passed",
        "inventory_path": str(inventory_path.resolve(strict=True)),
        "inventory_sha256": actual_inventory_sha256,
        "verification_counts": verification_counts,
        "runtime_large_file_check": "size_and_first/middle/last_sample_not_full_rehash",
        "runtime_large_file_boundary": (
            "Sampling detects changes only in sampled regions; the inventory's audit-time "
            "full SHA-256 is not re-proven for unsampled bytes at launch."
        ),
    }


def verify_formal_contract(
    *,
    snapshot_root: Path,
    snapshot_manifest_sha256: str,
    formal_manifest_path: Path,
    formal_manifest_sha256: str,
    experiment_id: str,
    expected_commit: str,
    expected_workflow_snapshot_sha256: str,
    nproc_per_node: int,
    expected_container_image: str,
    dataset_root: Path,
    source_root: Path,
    base_checkpoint: Path,
    wan_vae: Path,
    qwen_tokenizer: Path,
    environment: Path,
) -> dict[str, Any]:
    if _IMAGE_DIGEST_RE.fullmatch(expected_container_image) is None:
        raise ProvenanceError(
            "expected container image must be content-addressed as repository@sha256:<64 hex>: "
            f"{expected_container_image!r}"
        )
    formal_manifest_sha256 = _require_sha256(formal_manifest_sha256, "formal manifest SHA-256")
    snapshot_manifest_sha256 = _require_sha256(snapshot_manifest_sha256, "snapshot manifest SHA-256")
    expected_workflow_snapshot_sha256 = _require_sha256(
        expected_workflow_snapshot_sha256,
        "workflow snapshot SHA-256",
    )
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ProvenanceError(f"expected source commit must be a 40-character lowercase Git SHA: {expected_commit!r}")
    if nproc_per_node not in {6, 8}:
        raise ProvenanceError(f"formal world size must be 6 or 8, got {nproc_per_node}")

    snapshot_root_literal = Path(os.path.abspath(snapshot_root))
    snapshot_root_canonical = snapshot_root.resolve(strict=True)
    if (
        snapshot_root != snapshot_root_literal
        or snapshot_root_canonical != snapshot_root_literal
        or snapshot_root.is_symlink()
        or not snapshot_root.is_dir()
    ):
        raise ProvenanceError(f"formal snapshot root is not a canonical non-symlink directory: {snapshot_root}")
    snapshot_manifest_path = snapshot_root / "SNAPSHOT_MANIFEST.json"
    actual_snapshot_manifest_sha256 = sha256_file(snapshot_manifest_path)
    if actual_snapshot_manifest_sha256 != snapshot_manifest_sha256:
        raise ProvenanceError(
            "source manifest checksum changed between source and formal checks: "
            f"expected={snapshot_manifest_sha256}, actual={actual_snapshot_manifest_sha256}"
        )
    snapshot = json.loads(snapshot_manifest_path.read_text())
    if snapshot.get("experiment_id") != experiment_id:
        raise ProvenanceError(f"snapshot experiment mismatch: {snapshot.get('experiment_id')!r}")
    if snapshot.get("source_dirty") is not False:
        raise ProvenanceError("formal snapshot was captured from a dirty source tree")
    if snapshot.get("source_head") != expected_commit:
        raise ProvenanceError(f"snapshot source head mismatch: {snapshot.get('source_head')!r}")
    if snapshot.get("workflow_snapshot_sha256") != expected_workflow_snapshot_sha256:
        raise ProvenanceError("snapshot workflow SHA does not match the requested workflow snapshot")

    formal_bytes = formal_manifest_path.read_bytes()
    actual_formal_sha256 = hashlib.sha256(formal_bytes).hexdigest()
    if actual_formal_sha256 != formal_manifest_sha256:
        raise ProvenanceError(
            f"formal manifest checksum mismatch: expected={formal_manifest_sha256}, actual={actual_formal_sha256}"
        )
    formal = json.loads(formal_bytes)
    if formal.get("experiment_id") != experiment_id:
        raise ProvenanceError(f"formal experiment mismatch: {formal.get('experiment_id')!r}")
    if formal.get("status") not in {"planned", "queued"}:
        raise ProvenanceError(f"formal manifest status is not launchable: {formal.get('status')!r}")
    protocol = _verify_formal_protocol(
        formal,
        experiment_id=experiment_id,
        world_size=nproc_per_node,
    )
    storage_contract = _verify_exact_contract(
        formal,
        key="storage",
        expected=_formal_storage_contract(),
    )
    rjob_contract = _verify_exact_contract(
        formal,
        key="rjob",
        expected=_formal_rjob_contract(nproc_per_node),
    )

    git = _require_mapping(formal.get("git"), "formal git")
    if git.get("dirty") is not False or git.get("pushed") is not True:
        raise ProvenanceError("formal manifest does not declare a clean pushed commit")
    if git.get("commit") != expected_commit or git.get("snapshot_sha256") != expected_workflow_snapshot_sha256:
        raise ProvenanceError("formal manifest Git provenance does not match the requested commit/workflow snapshot")
    if git.get("source_manifest_sha256") != snapshot_manifest_sha256:
        raise ProvenanceError("formal manifest does not bind the exact source SNAPSHOT_MANIFEST.json")

    config = _require_mapping(formal.get("config"), "formal config")
    if config.get("path") != _EXPECTED_CONFIG_PATH:
        raise ProvenanceError(f"formal config path mismatch: {config.get('path')!r}")
    expected_config_sha256 = _require_sha256(config.get("sha256"), "formal config SHA-256")
    config_path = snapshot_root / _EXPECTED_CONFIG_PATH
    actual_config_sha256 = sha256_file(config_path)
    if actual_config_sha256 != expected_config_sha256:
        raise ProvenanceError(
            f"formal config checksum mismatch: expected={expected_config_sha256}, actual={actual_config_sha256}"
        )

    dataset_root = _canonical_root(dataset_root)
    source_root = _canonical_root(source_root)
    dataset_contract = _processed_dataset_contract(
        dataset_root,
        expected_source_root=source_root,
        verify_payload_sha256=False,
    )
    data = _require_mapping(formal.get("data"), "formal data")
    if data.get("root") != str(dataset_root):
        raise ProvenanceError(
            f"formal processed dataset root mismatch: formal={data.get('root')!r}, runtime={str(dataset_root)!r}"
        )
    if data.get("source_root") != str(source_root):
        raise ProvenanceError(
            f"formal raw source root mismatch: formal={data.get('source_root')!r}, runtime={str(source_root)!r}"
        )
    expected_success_sha256 = _require_sha256(data.get("success_sha256"), "formal data _SUCCESS SHA-256")
    actual_success_sha256 = dataset_contract["success_sha256"]
    if actual_success_sha256 != expected_success_sha256:
        raise ProvenanceError(
            f"dataset _SUCCESS checksum mismatch: expected={expected_success_sha256}, actual={actual_success_sha256}"
        )
    actual_dataset_manifest_sha256 = dataset_contract["manifest_sha256"]
    expected_dataset_manifest_sha256 = _require_sha256(
        data.get("manifest_sha256"),
        "formal data manifest SHA-256",
    )
    if actual_dataset_manifest_sha256 != expected_dataset_manifest_sha256:
        raise ProvenanceError(
            "dataset manifest does not match the formal manifest: "
            f"expected={expected_dataset_manifest_sha256}, actual={actual_dataset_manifest_sha256}"
        )
    if data.get("revision") != f"manifest-sha256:{actual_dataset_manifest_sha256}":
        raise ProvenanceError(f"formal data revision mismatch: {data.get('revision')!r}")
    if data.get("version") != dataset_contract["dataset_id"]:
        raise ProvenanceError(
            f"formal data version mismatch: formal={data.get('version')!r}, dataset={dataset_contract['dataset_id']!r}"
        )

    environment_contract = _require_mapping(formal.get("environment"), "formal environment")
    if environment_contract.get("container_image") != expected_container_image:
        raise ProvenanceError(
            "formal container image mismatch: "
            f"formal={environment_contract.get('container_image')!r}, expected={expected_container_image!r}"
        )
    artifacts = _require_mapping(formal.get("artifacts"), "formal artifacts")
    expected_artifact_paths = {
        "base_checkpoint": str(_canonical_root(base_checkpoint)),
        "wan_vae": str(wan_vae.expanduser().resolve(strict=True)),
        "qwen_tokenizer": str(_canonical_root(qwen_tokenizer)),
        "environment": str(_canonical_root(environment)),
        "processed_dataset": str(dataset_root),
        "raw_video_source": str(source_root),
    }
    declared_artifact_paths = _require_mapping(artifacts.get("paths"), "formal artifact paths")
    if declared_artifact_paths != expected_artifact_paths:
        raise ProvenanceError(
            "formal artifact paths do not exactly bind all runtime artifacts: "
            f"expected={expected_artifact_paths}, actual={declared_artifact_paths}"
        )
    inventory_path_value = artifacts.get("inventory_path")
    if not isinstance(inventory_path_value, str) or not inventory_path_value.startswith("/"):
        raise ProvenanceError(f"formal artifact inventory path must be absolute: {inventory_path_value!r}")
    inventory_sha256 = _require_sha256(artifacts.get("inventory_sha256"), "formal artifact inventory SHA-256")
    inventory_report = verify_artifact_inventory(
        inventory_path=Path(inventory_path_value),
        expected_inventory_sha256=inventory_sha256,
        base_checkpoint=base_checkpoint,
        wan_vae=wan_vae,
        qwen_tokenizer=qwen_tokenizer,
        environment=environment,
        dataset_root=dataset_root,
        source_root=source_root,
    )

    command = str(formal.get("command", ""))
    if "BLOCKED" in command or "run_cwam_cosmos3_nano_production_then_formal.sh" not in command:
        raise ProvenanceError("formal manifest command is stale or does not name the formal wrapper")
    hardware = str(formal.get("hardware", ""))
    if f"{nproc_per_node}x H200" not in hardware or "128 samples/rank" not in hardware:
        raise ProvenanceError(f"formal manifest hardware contract is incomplete: {hardware!r}")
    if "74000 packed-token cap" in hardware:
        raise ProvenanceError("formal manifest still claims the inactive 74000-token loader cap")

    return {
        "schema_version": 1,
        "status": "passed",
        "experiment_id": experiment_id,
        "snapshot_root": str(snapshot_root_canonical),
        "formal_manifest": str(formal_manifest_path.resolve(strict=True)),
        "formal_manifest_sha256": actual_formal_sha256,
        "source_manifest_sha256": actual_snapshot_manifest_sha256,
        "source_commit": expected_commit,
        "workflow_snapshot_sha256": expected_workflow_snapshot_sha256,
        "protocol": protocol,
        "storage": storage_contract,
        "rjob": rjob_contract,
        "config_sha256": actual_config_sha256,
        "dataset_root": str(dataset_root),
        "source_root": str(source_root),
        "dataset_success_sha256": actual_success_sha256,
        "dataset_manifest_sha256": actual_dataset_manifest_sha256,
        "artifact_inventory": inventory_report,
        "container_image": expected_container_image,
        "nproc_per_node": nproc_per_node,
    }


def _write_json_report(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("verify-snapshot")
    snapshot.add_argument("--snapshot-root", type=Path, required=True)
    snapshot.add_argument("--allowed-root", type=Path, required=True)
    snapshot.add_argument("--expected-manifest-sha256", required=True)
    snapshot.add_argument("--output-json", type=Path)

    create = subparsers.add_parser("create-artifact-inventory")
    create.add_argument("--base-checkpoint", type=Path, required=True)
    create.add_argument("--wan-vae", type=Path, required=True)
    create.add_argument("--qwen-tokenizer", type=Path, required=True)
    create.add_argument("--environment", type=Path, required=True)
    create.add_argument("--dataset-root", type=Path, required=True)
    create.add_argument("--source-root", type=Path, required=True)
    create.add_argument("--large-file-threshold-bytes", type=int, default=_DEFAULT_LARGE_FILE_THRESHOLD)
    create.add_argument("--sample-bytes", type=int, default=_DEFAULT_SAMPLE_BYTES)
    create.add_argument("--output-json", type=Path, required=True)

    inventory = subparsers.add_parser("verify-artifact-inventory")
    inventory.add_argument("--inventory", type=Path, required=True)
    inventory.add_argument("--inventory-sha256", required=True)
    inventory.add_argument("--base-checkpoint", type=Path, required=True)
    inventory.add_argument("--wan-vae", type=Path, required=True)
    inventory.add_argument("--qwen-tokenizer", type=Path, required=True)
    inventory.add_argument("--environment", type=Path, required=True)
    inventory.add_argument("--dataset-root", type=Path, required=True)
    inventory.add_argument("--source-root", type=Path, required=True)
    inventory.add_argument("--output-json", type=Path)

    formal = subparsers.add_parser("verify-formal-contract")
    formal.add_argument("--snapshot-root", type=Path, required=True)
    formal.add_argument("--snapshot-manifest-sha256", required=True)
    formal.add_argument("--formal-manifest", type=Path, required=True)
    formal.add_argument("--formal-manifest-sha256", required=True)
    formal.add_argument("--experiment-id", required=True)
    formal.add_argument("--expected-commit", required=True)
    formal.add_argument("--expected-workflow-snapshot-sha256", required=True)
    formal.add_argument("--nproc-per-node", type=int, required=True)
    formal.add_argument("--expected-container-image", required=True)
    formal.add_argument("--dataset-root", type=Path, required=True)
    formal.add_argument("--source-root", type=Path, required=True)
    formal.add_argument("--base-checkpoint", type=Path, required=True)
    formal.add_argument("--wan-vae", type=Path, required=True)
    formal.add_argument("--qwen-tokenizer", type=Path, required=True)
    formal.add_argument("--environment", type=Path, required=True)
    formal.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "verify-snapshot":
            report = verify_snapshot(
                snapshot_root=args.snapshot_root,
                allowed_root=args.allowed_root,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            _write_json_report(report, args.output_json)
            return
        if args.command == "create-artifact-inventory":
            payload = create_artifact_inventory(
                base_checkpoint=args.base_checkpoint,
                wan_vae=args.wan_vae,
                qwen_tokenizer=args.qwen_tokenizer,
                environment=args.environment,
                dataset_root=args.dataset_root,
                source_root=args.source_root,
                large_file_threshold=args.large_file_threshold_bytes,
                sample_bytes=args.sample_bytes,
            )
            inventory_sha256 = write_inventory(payload, args.output_json)
            sys.stdout.write(
                json.dumps(
                    {
                        "status": "created",
                        "inventory_path": str(args.output_json.resolve()),
                        "inventory_sha256": inventory_sha256,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            return
        if args.command == "verify-artifact-inventory":
            report = verify_artifact_inventory(
                inventory_path=args.inventory,
                expected_inventory_sha256=args.inventory_sha256,
                base_checkpoint=args.base_checkpoint,
                wan_vae=args.wan_vae,
                qwen_tokenizer=args.qwen_tokenizer,
                environment=args.environment,
                dataset_root=args.dataset_root,
                source_root=args.source_root,
            )
            _write_json_report(report, args.output_json)
            return
        if args.command == "verify-formal-contract":
            report = verify_formal_contract(
                snapshot_root=args.snapshot_root,
                snapshot_manifest_sha256=args.snapshot_manifest_sha256,
                formal_manifest_path=args.formal_manifest,
                formal_manifest_sha256=args.formal_manifest_sha256,
                experiment_id=args.experiment_id,
                expected_commit=args.expected_commit,
                expected_workflow_snapshot_sha256=args.expected_workflow_snapshot_sha256,
                nproc_per_node=args.nproc_per_node,
                expected_container_image=args.expected_container_image,
                dataset_root=args.dataset_root,
                source_root=args.source_root,
                base_checkpoint=args.base_checkpoint,
                wan_vae=args.wan_vae,
                qwen_tokenizer=args.qwen_tokenizer,
                environment=args.environment,
            )
            _write_json_report(report, args.output_json)
            return
    except (OSError, ValueError, json.JSONDecodeError, ProvenanceError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
