#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed input provenance for the Alibaba DSW RoboCasa365 evaluation.

The H-cluster serving manifest intentionally contains the H absolute model
path.  It is source provenance, not a portable manifest.  This helper verifies
the complete DSW model tree against that source manifest, then publishes a new
manifest bound to the canonical DSW destination.  It also issues immutable
receipts for the source snapshot, environments, closure bundle, and static
runtime inputs so the final evaluation evidence can bind both pre- and
post-rollout verification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import cwam_formal_provenance
    import cwam_serving_checkpoint_manifest
except ImportError:  # pragma: no cover - package-style invocation
    from examples.h_cluster import cwam_formal_provenance, cwam_serving_checkpoint_manifest


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_VERIFICATION_KIND = "cwam_dsw_source_snapshot_verification"
_RELOCATION_KIND = "cwam_dsw_serving_checkpoint_relocation"
_STATIC_INPUT_KIND = "cwam_dsw_static_runtime_inputs_verification"
_ENVIRONMENT_KIND = "cwam_dsw_environment_verification"
_CLOSURE_KIND = "cwam_dsw_asset_env_closure_provenance"

_EXPECTED_TARGET_SHARD_COUNT = 8
_EXPECTED_SOURCE_DCP_FILE_COUNT = 9
_EXPECTED_ROBOCASA_PACKAGE_SOURCE_NOTE = (
    "exact versions rebuilt with Alibaba PyPI mirror and PyTorch cu128 index; "
    "torchcodec==0.5 overridden by verified official PyPI cp311 wheel; "
    "torchcodec_relay_receipt_sha256="
    "1b16cc5419b7eb8558d5c0bd6dc4099493df629d50756ad3116b3c1da47f5ce4"
)


class DswInputProvenanceError(RuntimeError):
    """Raised when a formal DSW input or provenance binding is invalid."""


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DswInputProvenanceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise DswInputProvenanceError(f"{label} must be a lowercase 40-character Git commit")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise DswInputProvenanceError(
            f"{label} keys mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(str(key) for key in observed - expected)}"
        )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DswInputProvenanceError(f"JSON contains duplicate object key {key!r}")
        result[key] = value
    return result


def _canonical_directory(path: Path, label: str) -> Path:
    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        canonical = supplied.resolve(strict=True)
        metadata = supplied.lstat()
    except FileNotFoundError:
        raise DswInputProvenanceError(f"{label} does not exist: {supplied}") from None
    if supplied != canonical or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise DswInputProvenanceError(f"{label} must be a canonical non-symlink directory: {supplied}")
    return canonical


def _canonical_file(path: Path, label: str, *, expected_mode: int | None = None) -> Path:
    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        canonical = supplied.resolve(strict=True)
        metadata = supplied.lstat()
    except FileNotFoundError:
        raise DswInputProvenanceError(f"{label} does not exist: {supplied}") from None
    if supplied != canonical or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DswInputProvenanceError(f"{label} must be a canonical regular non-symlink file: {supplied}")
    if expected_mode is not None and stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise DswInputProvenanceError(
            f"{label} mode mismatch: expected={oct(expected_mode)}, actual={oct(stat.S_IMODE(metadata.st_mode))}"
        )
    return canonical


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def stable_file_descriptor(
    path: Path,
    *,
    label: str = "artifact",
    expected_mode: int | None = None,
) -> dict[str, Any]:
    """Hash one canonical regular file while detecting replacement or mutation."""

    canonical = _canonical_file(path, label, expected_mode=expected_mode)
    before = canonical.lstat()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(canonical, flags)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        opened = os.fstat(descriptor)
        if _identity(before) != _identity(opened):
            raise DswInputProvenanceError(f"{label} changed while opening: {canonical}")
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            size_bytes += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = canonical.lstat()
    if not (_identity(before) == _identity(opened) == _identity(after) == _identity(final)):
        raise DswInputProvenanceError(f"{label} changed while hashing: {canonical}")
    if size_bytes != before.st_size:
        raise DswInputProvenanceError(f"{label} size changed while hashing: {canonical}")
    return {
        "path": str(canonical),
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _read_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_mode: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = stable_file_descriptor(path, label=label, expected_mode=expected_mode)
    if expected_sha256 is not None:
        expected_sha256 = _require_sha256(expected_sha256, f"expected {label} SHA-256")
        if descriptor["sha256"] != expected_sha256:
            raise DswInputProvenanceError(
                f"{label} SHA-256 mismatch: expected={expected_sha256}, actual={descriptor['sha256']}"
            )
    payload = Path(descriptor["path"]).read_bytes()
    if len(payload) != descriptor["size_bytes"] or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise DswInputProvenanceError(f"{label} changed between hashing and reading")
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DswInputProvenanceError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise DswInputProvenanceError(f"{label} JSON root must be an object")
    if stable_file_descriptor(Path(descriptor["path"]), label=label, expected_mode=expected_mode) != descriptor:
        raise DswInputProvenanceError(f"{label} changed after reading")
    return value, descriptor


def _atomic_write_immutable(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    supplied = Path(os.path.abspath(path.expanduser()))
    parent = _canonical_directory(supplied.parent, "output parent")
    if supplied.parent != parent:
        raise DswInputProvenanceError(f"output parent must be canonical: {supplied.parent}")
    if supplied.exists() or supplied.is_symlink():
        raise DswInputProvenanceError(f"refusing to overwrite output: {supplied}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{supplied.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, supplied, follow_symlinks=False)
        published = True
        # Complete both directory mutations before the durability barrier.  In
        # particular, unlinking the temporary hard link changes the shared
        # inode ctime; hashing before that unlink has propagated can produce a
        # false replacement signal on CPFS.
        temporary.unlink()
        parent_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        if published:
            supplied.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    # CPFS can expose the completed hard-link ctime transition to lstat and
    # fstat a few seconds apart.  Keep the full identity check intact and
    # require one complete strict pass; retry only that transient mismatch.
    deadline = time.monotonic() + 10.0
    delay_seconds = 0.05
    while True:
        try:
            return stable_file_descriptor(
                supplied,
                label="published receipt",
                expected_mode=0o444,
            )
        except DswInputProvenanceError as error:
            message = str(error)
            if (
                "published receipt changed while opening:" not in message
                and "published receipt changed while hashing:" not in message
            ):
                raise
            if time.monotonic() >= deadline:
                raise DswInputProvenanceError(
                    "published receipt metadata did not stabilize within 10 seconds"
                ) from error
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 0.5)


def _validate_file_records(
    raw_files: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_files, list) or not raw_files:
        raise DswInputProvenanceError(f"{label} must be a non-empty list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, dict):
            raise DswInputProvenanceError(f"{label}[{index}] must be an object")
        _require_exact_keys(
            raw,
            {"relative_path", "size_bytes", "sha256"},
            f"{label}[{index}]",
        )
        relative = raw["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(component in ("", ".", "..") for component in relative.split("/"))
            or relative in seen
        ):
            raise DswInputProvenanceError(f"{label}[{index}].relative_path is unsafe or duplicated")
        seen.add(relative)
        size_bytes = raw["size_bytes"]
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise DswInputProvenanceError(f"{label}[{index}].size_bytes is invalid")
        records.append(
            {
                "relative_path": relative,
                "size_bytes": size_bytes,
                "sha256": _require_sha256(raw["sha256"], f"{label}[{index}].sha256"),
            }
        )
    if [record["relative_path"] for record in records] != sorted(seen):
        raise DswInputProvenanceError(f"{label} must be sorted by relative_path")
    return records


def _canonical_records_digest(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(records),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _live_distribution_summary() -> tuple[int, str]:
    """Return the canonical installed name/version inventory count and digest."""

    records: dict[str, dict[str, str]] = {}
    for index, distribution in enumerate(importlib.metadata.distributions()):
        raw_name = distribution.metadata.get("Name")
        raw_version = distribution.version
        if (
            not isinstance(raw_name, str)
            or not raw_name.strip()
            or not isinstance(raw_version, str)
            or not raw_version.strip()
        ):
            raise DswInputProvenanceError(f"installed distribution {index} lacks a valid name/version")
        name = raw_name.strip()
        version = raw_version.strip()
        canonical_name = re.sub(r"[-_.]+", "-", name).lower()
        if canonical_name in records:
            raise DswInputProvenanceError(
                f"installed distributions contain duplicate canonical name {canonical_name!r}"
            )
        records[canonical_name] = {"name": name, "version": version}
    ordered = {name: records[name] for name in sorted(records)}
    payload = json.dumps(
        ordered,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return len(ordered), hashlib.sha256(payload).hexdigest()


def _validate_source_dcp_receipt(
    receipt: Mapping[str, Any],
    *,
    export_receipt_sha256: str,
    target_manifest_sha256: str,
    target_tree_sha256: str,
) -> None:
    _require_exact_keys(
        receipt,
        {
            "completed_at",
            "derived_target_binding",
            "elapsed_seconds",
            "kind",
            "schema_version",
            "source",
            "started_at",
            "status",
            "verifier",
        },
        "source DCP verification receipt",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "cwam_source_dcp_full_verification"
        or receipt["status"] != "passed"
    ):
        raise DswInputProvenanceError("source DCP verification schema/kind/status mismatch")
    source = receipt["source"]
    binding = receipt["derived_target_binding"]
    verifier = receipt["verifier"]
    if not isinstance(source, dict) or not isinstance(binding, dict) or not isinstance(verifier, dict):
        raise DswInputProvenanceError("source DCP receipt nested fields must be objects")
    _require_exact_keys(
        source,
        {
            "file_count",
            "files",
            "full_file_sha256_readback",
            "model_path",
            "origin_manifest_path",
            "origin_manifest_sha256",
            "origin_tree_sha256",
            "total_bytes",
        },
        "source DCP receipt source",
    )
    _require_exact_keys(
        binding,
        {
            "export_receipt_path",
            "export_receipt_sha256",
            "target_manifest_path",
            "target_manifest_sha256",
            "target_tree_sha256",
        },
        "source DCP receipt derived_target_binding",
    )
    _require_exact_keys(
        verifier,
        {"python", "sha256", "source_path"},
        "source DCP receipt verifier",
    )
    records = _validate_file_records(source["files"], label="source DCP receipt source.files")
    if (
        source["file_count"] != _EXPECTED_SOURCE_DCP_FILE_COUNT
        or len(records) != _EXPECTED_SOURCE_DCP_FILE_COUNT
        or type(source["total_bytes"]) is not int
        or source["total_bytes"] <= 0
        or sum(record["size_bytes"] for record in records) != source["total_bytes"]
        or source["full_file_sha256_readback"] is not True
    ):
        raise DswInputProvenanceError("source DCP receipt does not prove the complete 9-file source")
    if _canonical_records_digest(records) != source["origin_tree_sha256"]:
        raise DswInputProvenanceError("source DCP receipt files do not reproduce origin_tree_sha256")
    for key, expected in (
        ("export_receipt_sha256", export_receipt_sha256),
        ("target_manifest_sha256", target_manifest_sha256),
        ("target_tree_sha256", target_tree_sha256),
    ):
        if binding.get(key) != expected:
            raise DswInputProvenanceError(f"source DCP receipt derived target binding mismatch for {key}")
    _require_sha256(source["origin_manifest_sha256"], "source origin manifest SHA-256")
    _require_sha256(source["origin_tree_sha256"], "source origin tree SHA-256")
    _require_sha256(verifier["sha256"], "source DCP verifier SHA-256")


def relocate_checkpoint(
    *,
    checkpoint_path: Path,
    h_manifest_path: Path,
    expected_h_manifest_sha256: str,
    expected_target_tree_sha256: str,
    export_receipt_path: Path,
    expected_export_receipt_sha256: str,
    key_manifest_path: Path,
    expected_key_manifest_sha256: str,
    validation_path: Path,
    expected_validation_sha256: str,
    source_dcp_receipt_path: Path,
    expected_source_dcp_receipt_sha256: str,
    training_config_path: Path,
    expected_training_config_sha256: str,
    expected_ema_key_count: int,
    expected_ema_chunk_count: int,
    destination_manifest_output: Path,
    relocation_receipt_output: Path,
) -> dict[str, Any]:
    """Verify H provenance and publish a DSW-path-bound checkpoint manifest."""

    checkpoint_path = _canonical_directory(checkpoint_path, "DSW checkpoint")
    expected_target_tree_sha256 = _require_sha256(expected_target_tree_sha256, "expected target tree SHA-256")
    if type(expected_ema_key_count) is not int or expected_ema_key_count <= 0:
        raise DswInputProvenanceError("expected EMA key count must be positive")
    if type(expected_ema_chunk_count) is not int or expected_ema_chunk_count <= 0:
        raise DswInputProvenanceError("expected EMA chunk count must be positive")
    h_manifest, h_manifest_descriptor = _read_json(
        h_manifest_path,
        label="H serving checkpoint manifest",
        expected_sha256=expected_h_manifest_sha256,
        expected_mode=0o444,
    )
    original_h_checkpoint = h_manifest.get("checkpoint_path")
    if not isinstance(original_h_checkpoint, str) or not original_h_checkpoint.startswith("/"):
        raise DswInputProvenanceError("H serving manifest lacks an absolute checkpoint_path")
    try:
        h_files, h_tree = cwam_serving_checkpoint_manifest._validated_manifest(
            h_manifest,
            checkpoint_path=original_h_checkpoint,
        )
    except cwam_serving_checkpoint_manifest.ServingCheckpointManifestError as error:
        raise DswInputProvenanceError(str(error)) from error
    if h_tree != expected_target_tree_sha256:
        raise DswInputProvenanceError("H manifest target tree differs from the pinned target tree")

    live_manifest = cwam_serving_checkpoint_manifest.create_serving_checkpoint_manifest(checkpoint_path)
    if live_manifest["files"] != h_files or live_manifest["tree_sha256"] != h_tree:
        raise DswInputProvenanceError("DSW destination model tree is not byte-identical to the H serving model tree")

    export_receipt, export_descriptor = _read_json(
        export_receipt_path,
        label="EMA BF16 export receipt",
        expected_sha256=expected_export_receipt_sha256,
        expected_mode=0o444,
    )
    key_manifest, key_descriptor = _read_json(
        key_manifest_path,
        label="EMA BF16 key manifest",
        expected_sha256=expected_key_manifest_sha256,
        expected_mode=0o444,
    )
    validation, validation_descriptor = _read_json(
        validation_path,
        label="EMA BF16 validation",
        expected_sha256=expected_validation_sha256,
        expected_mode=0o444,
    )
    source_receipt, source_descriptor = _read_json(
        source_dcp_receipt_path,
        label="source DCP full verification receipt",
        expected_sha256=expected_source_dcp_receipt_sha256,
        expected_mode=0o444,
    )
    config_descriptor = stable_file_descriptor(
        training_config_path,
        label="training config",
    )
    if config_descriptor["sha256"] != _require_sha256(
        expected_training_config_sha256, "expected training config SHA-256"
    ):
        raise DswInputProvenanceError("training config SHA-256 mismatch")

    if (
        export_receipt.get("schema_version") != 1
        or export_receipt.get("kind") != "cwam_serving_ema_bf16_export_receipt"
        or export_receipt.get("status") != "succeeded"
    ):
        raise DswInputProvenanceError("EMA export receipt schema/kind/status mismatch")
    selection = export_receipt.get("selection")
    target = export_receipt.get("target")
    source = export_receipt.get("source")
    publication = export_receipt.get("publication")
    if not all(isinstance(value, dict) for value in (selection, target, source, publication)):
        raise DswInputProvenanceError("EMA export receipt lacks required nested bindings")
    expected_selection = {
        "included_prefix": "net_ema.",
        "excluded_prefix": "net.",
        "key_names_preserved": True,
        "source_dtype": "torch.float32",
        "target_dtype": "torch.bfloat16",
        "key_count": expected_ema_key_count,
        "chunk_count": expected_ema_chunk_count,
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise DswInputProvenanceError(f"EMA export selection mismatch for {key}")
    if (
        target.get("shard_count") != _EXPECTED_TARGET_SHARD_COUNT
        or target.get("checkpoint_tree_sha256") != h_tree
        or target.get("serving_manifest_sha256") != h_manifest_descriptor["sha256"]
        or target.get("key_manifest_sha256") != key_descriptor["sha256"]
        or target.get("validation_sha256") != validation_descriptor["sha256"]
        or source.get("config_sha256") != config_descriptor["sha256"]
        or publication.get("final_model_path") != original_h_checkpoint
    ):
        raise DswInputProvenanceError("EMA export receipt source/target binding mismatch")

    if (
        key_manifest.get("schema_version") != 1
        or key_manifest.get("kind") != "cwam_serving_ema_bf16_key_manifest"
        or key_manifest.get("selection") != "net_ema.* only; key names preserved"
        or key_manifest.get("key_count") != expected_ema_key_count
        or key_manifest.get("target_model_path") != original_h_checkpoint
    ):
        raise DswInputProvenanceError("EMA key manifest header mismatch")
    raw_keys = key_manifest.get("keys")
    if not isinstance(raw_keys, list) or len(raw_keys) != expected_ema_key_count:
        raise DswInputProvenanceError(f"EMA key manifest must contain exactly {expected_ema_key_count} keys")
    observed_keys: list[str] = []
    observed_chunks = 0
    for index, record in enumerate(raw_keys):
        if not isinstance(record, dict):
            raise DswInputProvenanceError(f"EMA key manifest keys[{index}] must be an object")
        key = record.get("key")
        if not isinstance(key, str) or not key.startswith("net_ema."):
            raise DswInputProvenanceError(f"EMA key manifest contains a non-net_ema key: {key!r}")
        if record.get("source_dtype") != "torch.float32" or record.get("target_dtype") != "torch.bfloat16":
            raise DswInputProvenanceError(f"EMA key manifest dtype mismatch for {key}")
        chunk_count = record.get("chunk_count")
        chunks = record.get("chunks")
        if (
            isinstance(chunk_count, bool)
            or not isinstance(chunk_count, int)
            or chunk_count < 1
            or not isinstance(chunks, list)
            or len(chunks) != chunk_count
        ):
            raise DswInputProvenanceError(f"EMA key manifest chunk inventory mismatch for {key}")
        observed_keys.append(key)
        observed_chunks += chunk_count
    if observed_keys != sorted(set(observed_keys)) or observed_chunks != expected_ema_chunk_count:
        raise DswInputProvenanceError("EMA key manifest key ordering/count or chunk count mismatch")

    full_readback = validation.get("full_readback")
    samples = validation.get("custom_load_planner_samples")
    if (
        validation.get("schema_version") != 1
        or validation.get("kind") != "cwam_serving_ema_bf16_validation"
        or validation.get("status") != "passed"
        or not isinstance(full_readback, dict)
        or full_readback.get("status") != "passed"
        or full_readback.get("verified_key_count") != expected_ema_key_count
        or full_readback.get("verified_chunk_count") != expected_ema_chunk_count
        or full_readback.get("verified_shard_count") != _EXPECTED_TARGET_SHARD_COUNT
        or full_readback.get("checkpoint_tree_sha256") != h_tree
        or full_readback.get("every_serialized_chunk_sha256_recorded") is not True
        or full_readback.get("every_bf16_tensor_sha256_matched") is not True
        or full_readback.get("every_target_file_sha256_recorded") is not True
        or not isinstance(samples, list)
        or len(samples) < 3
    ):
        raise DswInputProvenanceError("EMA validation does not prove full readback and planner samples")
    for sample in samples:
        if (
            not isinstance(sample, dict)
            or sample.get("dtype") != "torch.bfloat16"
            or sample.get("explicit_target_matches_source_ema_cast") is not True
            or sample.get("custom_load_planner_matches_source_ema_cast") is not True
        ):
            raise DswInputProvenanceError("EMA validation planner sample mismatch")

    _validate_source_dcp_receipt(
        source_receipt,
        export_receipt_sha256=export_descriptor["sha256"],
        target_manifest_sha256=h_manifest_descriptor["sha256"],
        target_tree_sha256=h_tree,
    )
    source_dcp_binding = source_receipt["source"]
    if (
        source.get("model_path") != source_dcp_binding["model_path"]
        or source.get("origin_manifest_sha256") != source_dcp_binding["origin_manifest_sha256"]
        or source.get("origin_tree_sha256") != source_dcp_binding["origin_tree_sha256"]
    ):
        raise DswInputProvenanceError("source-DCP full receipt and EMA export source bindings differ")

    destination_manifest_descriptor = _atomic_write_immutable(
        destination_manifest_output,
        live_manifest,
    )
    destination_manifest, _ = _read_json(
        Path(destination_manifest_descriptor["path"]),
        label="DSW destination serving manifest",
        expected_sha256=destination_manifest_descriptor["sha256"],
        expected_mode=0o444,
    )
    try:
        verification = cwam_serving_checkpoint_manifest.verify_serving_checkpoint_manifest(
            destination_manifest,
            checkpoint_path,
        )
    except cwam_serving_checkpoint_manifest.ServingCheckpointManifestError as error:
        raise DswInputProvenanceError(str(error)) from error

    receipt = {
        "schema_version": 1,
        "kind": _RELOCATION_KIND,
        "status": "passed",
        "source_h_provenance": {
            "checkpoint_path": original_h_checkpoint,
            "serving_manifest": h_manifest_descriptor,
            "ema_export_receipt": export_descriptor,
            "ema_key_manifest": key_descriptor,
            "ema_validation": validation_descriptor,
            "source_dcp_full_verification": source_descriptor,
        },
        "destination": {
            "checkpoint_path": str(checkpoint_path),
            "serving_manifest": destination_manifest_descriptor,
            "file_count": verification["file_count"],
            "tree_sha256": verification["tree_sha256"],
        },
        "training_config": config_descriptor,
        "identity": {
            "same_relative_file_records": True,
            "same_tree_sha256": True,
            "source_h_tree_sha256": h_tree,
            "destination_tree_sha256": verification["tree_sha256"],
        },
        "ema_selection": {
            "selected_tree": "net_ema",
            "included_prefix": "net_ema.",
            "excluded_prefix": "net.",
            "key_count": expected_ema_key_count,
            "chunk_count": expected_ema_chunk_count,
            "source_dtype": "torch.float32",
            "target_dtype": "torch.bfloat16",
            "custom_load_planner": "CustomLoadPlanner(load_ema_to_reg=True)",
        },
        "validation": {
            "h_manifest_sha256_matches": True,
            "source_dcp_full_receipt_cross_binding_matches": True,
            "ema_export_receipt_cross_binding_matches": True,
            "ema_key_manifest_cross_binding_matches": True,
            "ema_validation_cross_binding_matches": True,
            "destination_full_tree_rehashed": True,
            "destination_manifest_is_path_bound": True,
            "destination_and_h_tree_are_byte_identical": True,
        },
    }
    _atomic_write_immutable(relocation_receipt_output, receipt)
    return receipt


def verify_static_inputs(
    *,
    phase: str,
    checkpoint_path: Path,
    destination_manifest_path: Path,
    expected_destination_manifest_sha256: str,
    training_config_path: Path,
    expected_training_config_sha256: str,
    action_stats_path: Path,
    expected_action_stats_sha256: str,
    wan2pt2_vae_path: Path,
    expected_wan2pt2_vae_sha256: str,
    output: Path,
) -> dict[str, Any]:
    if phase not in {"pre", "post"}:
        raise DswInputProvenanceError("static input verification phase must be pre or post")
    checkpoint_path = _canonical_directory(checkpoint_path, "DSW checkpoint")
    manifest, manifest_descriptor = _read_json(
        destination_manifest_path,
        label="DSW destination serving manifest",
        expected_sha256=expected_destination_manifest_sha256,
        expected_mode=0o444,
    )
    try:
        checkpoint_verification = cwam_serving_checkpoint_manifest.verify_serving_checkpoint_manifest(
            manifest,
            checkpoint_path,
        )
    except cwam_serving_checkpoint_manifest.ServingCheckpointManifestError as error:
        raise DswInputProvenanceError(str(error)) from error
    files: dict[str, dict[str, Any]] = {}
    for label, path, expected in (
        ("training_config", training_config_path, expected_training_config_sha256),
        ("action_stats", action_stats_path, expected_action_stats_sha256),
        ("wan2pt2_vae", wan2pt2_vae_path, expected_wan2pt2_vae_sha256),
    ):
        descriptor = stable_file_descriptor(path, label=label)
        if descriptor["sha256"] != _require_sha256(expected, f"expected {label} SHA-256"):
            raise DswInputProvenanceError(f"{label} SHA-256 mismatch")
        files[label] = descriptor
    receipt = {
        "schema_version": 1,
        "kind": _STATIC_INPUT_KIND,
        "status": "passed",
        "phase": phase,
        "checkpoint": {
            "path": str(checkpoint_path),
            "manifest": manifest_descriptor,
            "file_count": checkpoint_verification["file_count"],
            "tree_sha256": checkpoint_verification["tree_sha256"],
        },
        "files": files,
        "validation": {
            "checkpoint_manifest_path_matches_destination": True,
            "checkpoint_exact_file_set_sizes_and_sha256_match": True,
            "training_config_sha256_matches": True,
            "action_stats_sha256_matches": True,
            "wan2pt2_vae_sha256_matches": True,
        },
    }
    _atomic_write_immutable(output, receipt)
    return receipt


def verify_source_snapshot(
    *,
    phase: str,
    snapshot_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_source_commit: str,
    output: Path,
) -> dict[str, Any]:
    if phase not in {"pre", "post"}:
        raise DswInputProvenanceError("source verification phase must be pre or post")
    root = _canonical_directory(snapshot_root, "Cosmos source snapshot")
    manifest_path = _canonical_file(manifest_path, "Cosmos source manifest", expected_mode=0o444)
    if manifest_path != root / "SNAPSHOT_MANIFEST.json":
        raise DswInputProvenanceError("source manifest must be the exact SNAPSHOT_MANIFEST.json under the snapshot")
    expected_source_commit = _require_commit(expected_source_commit, "expected source commit")
    manifest, descriptor = _read_json(
        manifest_path,
        label="Cosmos source manifest",
        expected_sha256=expected_manifest_sha256,
        expected_mode=0o444,
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_head") != expected_source_commit
        or manifest.get("source_dirty") is not False
        or not isinstance(manifest.get("files"), list)
        or manifest.get("file_count") != len(manifest["files"])
    ):
        raise DswInputProvenanceError("source manifest schema, source_head, clean state, or file count mismatch")
    try:
        live = cwam_formal_provenance.verify_snapshot(
            snapshot_root=root,
            allowed_root=root.parent,
            expected_manifest_sha256=descriptor["sha256"],
        )
    except cwam_formal_provenance.ProvenanceError as error:
        raise DswInputProvenanceError(str(error)) from error
    receipt = {
        "schema_version": 1,
        "kind": _SOURCE_VERIFICATION_KIND,
        "status": "passed",
        "phase": phase,
        "snapshot_root": str(root),
        "source_commit": expected_source_commit,
        "source_manifest": descriptor,
        "file_count": live["file_count"],
        "file_bytes": live["file_bytes"],
        "validation": {
            "source_head_matches_expected_commit": True,
            "source_dirty_is_false": True,
            "snapshot_is_read_only": True,
            "exact_file_set_matches": True,
            "all_sizes_and_sha256_match": True,
        },
    }
    _atomic_write_immutable(output, receipt)
    return receipt


def verify_environment(
    *,
    phase: str,
    environment: Path,
    environment_type: str,
    installation_mode: str,
    marker_path: Path,
    expected_marker_sha256: str,
    expected_source_commit: str | None,
    expected_robocasa_commit: str | None,
    expected_robosuite_commit: str | None,
    import_probes: Sequence[str],
    expect_cuda_gpus: int,
    output: Path,
) -> dict[str, Any]:
    """Verify a relocated venv from inside that exact venv interpreter."""

    if phase not in {"pre", "post"}:
        raise DswInputProvenanceError("environment verification phase must be pre or post")
    if environment_type not in {"cosmos", "robocasa"}:
        raise DswInputProvenanceError("environment_type must be cosmos or robocasa")
    if installation_mode not in {"native_exact_rebuild", "portable_bundle_fallback"}:
        raise DswInputProvenanceError("installation_mode must be native_exact_rebuild or portable_bundle_fallback")
    environment = _canonical_directory(environment, f"{environment_type} environment")
    marker_path = _canonical_file(marker_path, f"{environment_type} environment marker", expected_mode=0o444)
    expected_marker_name = (
        ".cwam_env_ready.json"
        if installation_mode == "native_exact_rebuild" or environment_type == "cosmos"
        else ".cwam_bundle_manifest.json"
    )
    if marker_path != environment / expected_marker_name:
        raise DswInputProvenanceError(f"{environment_type} marker must be {environment / expected_marker_name}")
    marker, marker_descriptor = _read_json(
        marker_path,
        label=f"{environment_type} environment marker",
        expected_sha256=expected_marker_sha256,
        expected_mode=0o444,
    )
    if marker.get("schema_version") != 1:
        raise DswInputProvenanceError("environment marker schema_version must be 1")
    if environment_type == "cosmos":
        if installation_mode != "native_exact_rebuild":
            raise DswInputProvenanceError("Cosmos formal environment must use a native exact rebuild")
        _require_exact_keys(
            marker,
            {
                "cuda_available",
                "cudart_alias",
                "default_index",
                "packages",
                "pyproject_sha256",
                "python",
                "schema_version",
                "source_head",
                "status",
                "torch",
                "torch_cuda",
                "uv",
                "uv_lock_sha256",
            },
            "Cosmos native ready marker",
        )
        if marker.get("status") != "ready":
            raise DswInputProvenanceError("Cosmos environment marker status must be ready")
        if expected_source_commit is None:
            raise DswInputProvenanceError("Cosmos environment requires expected_source_commit")
        expected_source_commit = _require_commit(expected_source_commit, "expected Cosmos source commit")
        if marker.get("source_head") != expected_source_commit:
            raise DswInputProvenanceError("Cosmos environment marker source_head mismatch")
        if (
            marker.get("torch") != "2.10.0+cu128"
            or marker.get("torch_cuda") != "12.8"
            or marker.get("uv") != "0.11.28"
            or marker.get("cuda_available") is not True
        ):
            raise DswInputProvenanceError("Cosmos native ready marker runtime versions differ from the formal contract")
        _require_sha256(marker.get("pyproject_sha256"), "Cosmos marker pyproject_sha256")
        _require_sha256(marker.get("uv_lock_sha256"), "Cosmos marker uv_lock_sha256")
        if not isinstance(marker.get("packages"), dict) or not marker["packages"]:
            raise DswInputProvenanceError("Cosmos marker packages must be a non-empty object")
        for package_name, expected_version in marker["packages"].items():
            if (
                not isinstance(package_name, str)
                or not package_name
                or not isinstance(expected_version, str)
                or not expected_version
            ):
                raise DswInputProvenanceError("Cosmos marker packages must map names to non-empty versions")
            try:
                live_version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError as error:
                raise DswInputProvenanceError(f"Cosmos marker package is not installed: {package_name}") from error
            if live_version != expected_version:
                raise DswInputProvenanceError(f"Cosmos live package version differs for {package_name}")
    elif installation_mode == "native_exact_rebuild":
        _require_exact_keys(
            marker,
            {
                "cuda_available",
                "distribution_count",
                "distributions_sha256",
                "mujoco",
                "origin_environment_manifest_sha256",
                "origin_requirements_sha256",
                "package_source_note",
                "python",
                "relocated_requirements_sha256",
                "robocasa",
                "robocasa_commit",
                "robosuite",
                "robosuite_commit",
                "schema_version",
                "status",
                "torch",
                "torch_cuda",
            },
            "RoboCasa native ready marker",
        )
        if marker.get("status") != "ready":
            raise DswInputProvenanceError("RoboCasa native environment marker status must be ready")
        if (
            marker.get("python") != "3.11.14"
            or marker.get("torch") != "2.7.1+cu128"
            or marker.get("torch_cuda") != "12.8"
            or marker.get("mujoco") != "3.3.1"
            or marker.get("robocasa") != "1.0.1"
            or marker.get("robosuite") != "1.5.2"
            or marker.get("cuda_available") is not True
            or marker.get("package_source_note") != _EXPECTED_ROBOCASA_PACKAGE_SOURCE_NOTE
        ):
            raise DswInputProvenanceError(
                "RoboCasa native ready marker runtime versions differ from the formal contract"
            )
        expected_robocasa_commit = _require_commit(expected_robocasa_commit, "expected RoboCasa commit")
        expected_robosuite_commit = _require_commit(expected_robosuite_commit, "expected robosuite commit")
        if (
            marker.get("robocasa_commit") != expected_robocasa_commit
            or marker.get("robosuite_commit") != expected_robosuite_commit
        ):
            raise DswInputProvenanceError("RoboCasa native marker commit binding mismatch")
        distribution_count = marker.get("distribution_count")
        if isinstance(distribution_count, bool) or not isinstance(distribution_count, int) or distribution_count < 1:
            raise DswInputProvenanceError("RoboCasa marker distribution_count is invalid")
        for key in (
            "distributions_sha256",
            "origin_environment_manifest_sha256",
            "origin_requirements_sha256",
            "relocated_requirements_sha256",
        ):
            _require_sha256(marker.get(key), f"RoboCasa marker {key}")
    elif marker.get("kind") != "cwam_uv_venv_bundle":
        raise DswInputProvenanceError("RoboCasa environment bundle marker kind mismatch")

    if Path(sys.prefix).resolve(strict=True) != environment:
        raise DswInputProvenanceError(
            f"verification must run from the declared environment: "
            f"sys.prefix={sys.prefix!r}, expected={str(environment)!r}"
        )
    executable = Path(sys.executable)
    if not executable.is_absolute():
        raise DswInputProvenanceError("sys.executable must be absolute")
    try:
        executable.relative_to(environment)
    except ValueError as error:
        raise DswInputProvenanceError("live Python executable is outside the declared environment") from error

    probes: list[dict[str, Any]] = []
    for name in import_probes:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
            raise DswInputProvenanceError(f"unsafe import probe name: {name!r}")
        module = importlib.import_module(name)
        module_file = getattr(module, "__file__", None)
        module_version = getattr(module, "__version__", None)
        if module_version is None:
            try:
                module_version = importlib.metadata.version(name.split(".", 1)[0])
            except importlib.metadata.PackageNotFoundError:
                module_version = None
        probes.append(
            {
                "module": name,
                "module_file": str(Path(module_file).resolve()) if module_file else None,
                "version": str(module_version) if module_version is not None else None,
            }
        )

    marker_python = marker.get("python")
    live_python = platform.python_version()
    if not isinstance(marker_python, str) or not marker_python or marker_python.split()[0] != live_python:
        raise DswInputProvenanceError("live Python version differs from the native environment marker")

    cuda: dict[str, Any] | None = None
    if expect_cuda_gpus:
        if environment_type != "cosmos":
            raise DswInputProvenanceError("only the Cosmos environment may require CUDA GPUs")
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != expect_cuda_gpus:
            raise DswInputProvenanceError(
                f"expected {expect_cuda_gpus} live CUDA GPUs, got {torch.cuda.device_count()}"
            )
        gpus = []
        for index in range(expect_cuda_gpus):
            properties = torch.cuda.get_device_properties(index)
            record = {
                "visible_index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
            if "4090" not in record["name"] or record["compute_capability"] != "8.9":
                raise DswInputProvenanceError(f"unexpected formal-eval GPU: {record}")
            gpus.append(record)
        if marker.get("torch") != torch.__version__ or marker.get("torch_cuda") != torch.version.cuda:
            raise DswInputProvenanceError("live torch/CUDA versions differ from the Cosmos ready marker")
        cuda = {
            "available": True,
            "device_count": expect_cuda_gpus,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpus": gpus,
        }
    elif environment_type == "robocasa" and installation_mode == "native_exact_rebuild":
        import torch

        if (
            torch.__version__ != marker["torch"]
            or torch.version.cuda != marker["torch_cuda"]
            or torch.cuda.is_available() is not marker["cuda_available"]
        ):
            raise DswInputProvenanceError("live RoboCasa torch/CUDA runtime differs from the native marker")
        expected_probe_versions = {
            "mujoco": marker["mujoco"],
            "robocasa": marker["robocasa"],
            "robosuite": marker["robosuite"],
        }
        observed_probe_versions = {probe["module"]: probe["version"] for probe in probes}
        if observed_probe_versions != expected_probe_versions:
            raise DswInputProvenanceError("live RoboCasa package versions differ from the native marker")
        live_distribution_count, live_distributions_sha256 = _live_distribution_summary()
        if (
            live_distribution_count != marker["distribution_count"]
            or live_distributions_sha256 != marker["distributions_sha256"]
        ):
            raise DswInputProvenanceError("live RoboCasa distribution inventory differs from the native marker")
        cuda = {
            "available": marker["cuda_available"],
            "device_count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpus": None,
            "distribution_count": live_distribution_count,
            "distributions_sha256": live_distributions_sha256,
        }

    receipt = {
        "schema_version": 1,
        "kind": _ENVIRONMENT_KIND,
        "status": "passed",
        "phase": phase,
        "environment_type": environment_type,
        "installation_mode": installation_mode,
        "environment_path": str(environment),
        "marker": marker_descriptor,
        "live_runtime": {
            "sys_prefix": str(Path(sys.prefix).resolve(strict=True)),
            "sys_executable": str(executable),
            "python_version": sys.version,
            "import_probes": probes,
            "cuda": cuda,
        },
        "relocation": {
            "portable_bundle_used": installation_mode == "portable_bundle_fallback",
            "native_exact_rebuild_used": installation_mode == "native_exact_rebuild",
            "marker_embedded_source_paths_are_provenance_only": (installation_mode == "portable_bundle_fallback"),
            "live_sys_prefix_and_executable_are_dsw_path_bound": True,
        },
        "validation": {
            "marker_sha256_matches": True,
            "marker_schema_matches": True,
            "live_interpreter_matches_environment": True,
            "all_import_probes_succeeded": True,
            "live_python_matches_marker": True,
            "live_framework_versions_match_marker": True,
            (
                "live_distribution_inventory_matches_marker"
                if environment_type == "robocasa"
                else "live_marker_package_versions_match"
            ): True,
            "cuda_requirement_matches": True,
        },
    }
    _atomic_write_immutable(output, receipt)
    return receipt


def verify_closure_provenance(
    *,
    closure_manifest_path: Path,
    expected_closure_manifest_sha256: str,
    h_verification_receipt_path: Path,
    expected_h_verification_receipt_sha256: str,
    output: Path,
) -> dict[str, Any]:
    manifest, manifest_descriptor = _read_json(
        closure_manifest_path,
        label="asset/environment closure manifest",
        expected_sha256=expected_closure_manifest_sha256,
        expected_mode=0o444,
    )
    verification, verification_descriptor = _read_json(
        h_verification_receipt_path,
        label="H closure verification receipt",
        expected_sha256=expected_h_verification_receipt_sha256,
        expected_mode=0o444,
    )
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("kind"), str):
        raise DswInputProvenanceError("closure manifest schema/kind mismatch")
    if (
        verification.get("schema_version") != 1
        or verification.get("kind") != "cwam_dsw974_asset_env_closure_verification"
        or verification.get("status") != "passed"
    ):
        raise DswInputProvenanceError("H closure verification schema/kind/status mismatch")
    bound_manifest_sha = verification.get("manifest_sha256")
    if bound_manifest_sha is None and isinstance(verification.get("manifest"), dict):
        bound_manifest_sha = verification["manifest"].get("sha256")
    if bound_manifest_sha != manifest_descriptor["sha256"]:
        raise DswInputProvenanceError("H closure verification does not bind the exact closure manifest")
    receipt = {
        "schema_version": 1,
        "kind": _CLOSURE_KIND,
        "status": "passed",
        "closure_manifest": manifest_descriptor,
        "h_verification_receipt": verification_descriptor,
        "closure_kind": manifest["kind"],
        "validation": {
            "manifest_sha256_matches": True,
            "h_verification_receipt_sha256_matches": True,
            "h_verification_status_passed": True,
            "h_verification_binds_manifest": True,
        },
    }
    _atomic_write_immutable(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    relocate = subparsers.add_parser("relocate-checkpoint")
    relocate.add_argument("--checkpoint-path", type=Path, required=True)
    relocate.add_argument("--h-manifest-path", type=Path, required=True)
    relocate.add_argument("--expected-h-manifest-sha256", required=True)
    relocate.add_argument("--expected-target-tree-sha256", required=True)
    relocate.add_argument("--export-receipt-path", type=Path, required=True)
    relocate.add_argument("--expected-export-receipt-sha256", required=True)
    relocate.add_argument("--key-manifest-path", type=Path, required=True)
    relocate.add_argument("--expected-key-manifest-sha256", required=True)
    relocate.add_argument("--validation-path", type=Path, required=True)
    relocate.add_argument("--expected-validation-sha256", required=True)
    relocate.add_argument("--source-dcp-receipt-path", type=Path, required=True)
    relocate.add_argument("--expected-source-dcp-receipt-sha256", required=True)
    relocate.add_argument("--training-config-path", type=Path, required=True)
    relocate.add_argument("--expected-training-config-sha256", required=True)
    relocate.add_argument("--expected-ema-key-count", type=int, required=True)
    relocate.add_argument("--expected-ema-chunk-count", type=int, required=True)
    relocate.add_argument("--destination-manifest-output", type=Path, required=True)
    relocate.add_argument("--relocation-receipt-output", type=Path, required=True)

    static = subparsers.add_parser("verify-static-inputs")
    static.add_argument("--phase", choices=("pre", "post"), required=True)
    static.add_argument("--checkpoint-path", type=Path, required=True)
    static.add_argument("--destination-manifest-path", type=Path, required=True)
    static.add_argument("--expected-destination-manifest-sha256", required=True)
    static.add_argument("--training-config-path", type=Path, required=True)
    static.add_argument("--expected-training-config-sha256", required=True)
    static.add_argument("--action-stats-path", type=Path, required=True)
    static.add_argument("--expected-action-stats-sha256", required=True)
    static.add_argument("--wan2pt2-vae-path", type=Path, required=True)
    static.add_argument("--expected-wan2pt2-vae-sha256", required=True)
    static.add_argument("--output", type=Path, required=True)

    source = subparsers.add_parser("verify-source-snapshot")
    source.add_argument("--phase", choices=("pre", "post"), required=True)
    source.add_argument("--snapshot-root", type=Path, required=True)
    source.add_argument("--manifest-path", type=Path, required=True)
    source.add_argument("--expected-manifest-sha256", required=True)
    source.add_argument("--expected-source-commit", required=True)
    source.add_argument("--output", type=Path, required=True)

    environment = subparsers.add_parser("verify-environment")
    environment.add_argument("--phase", choices=("pre", "post"), required=True)
    environment.add_argument("--environment", type=Path, required=True)
    environment.add_argument("--environment-type", choices=("cosmos", "robocasa"), required=True)
    environment.add_argument(
        "--installation-mode",
        choices=("native_exact_rebuild", "portable_bundle_fallback"),
        required=True,
    )
    environment.add_argument("--marker-path", type=Path, required=True)
    environment.add_argument("--expected-marker-sha256", required=True)
    environment.add_argument("--expected-source-commit")
    environment.add_argument("--expected-robocasa-commit")
    environment.add_argument("--expected-robosuite-commit")
    environment.add_argument("--import-probe", action="append", default=[])
    environment.add_argument("--expect-cuda-gpus", type=int, default=0)
    environment.add_argument("--output", type=Path, required=True)

    closure = subparsers.add_parser("verify-closure-provenance")
    closure.add_argument("--closure-manifest-path", type=Path, required=True)
    closure.add_argument("--expected-closure-manifest-sha256", required=True)
    closure.add_argument("--h-verification-receipt-path", type=Path, required=True)
    closure.add_argument("--expected-h-verification-receipt-sha256", required=True)
    closure.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "relocate-checkpoint":
        result = relocate_checkpoint(
            checkpoint_path=args.checkpoint_path,
            h_manifest_path=args.h_manifest_path,
            expected_h_manifest_sha256=args.expected_h_manifest_sha256,
            expected_target_tree_sha256=args.expected_target_tree_sha256,
            export_receipt_path=args.export_receipt_path,
            expected_export_receipt_sha256=args.expected_export_receipt_sha256,
            key_manifest_path=args.key_manifest_path,
            expected_key_manifest_sha256=args.expected_key_manifest_sha256,
            validation_path=args.validation_path,
            expected_validation_sha256=args.expected_validation_sha256,
            source_dcp_receipt_path=args.source_dcp_receipt_path,
            expected_source_dcp_receipt_sha256=args.expected_source_dcp_receipt_sha256,
            training_config_path=args.training_config_path,
            expected_training_config_sha256=args.expected_training_config_sha256,
            expected_ema_key_count=args.expected_ema_key_count,
            expected_ema_chunk_count=args.expected_ema_chunk_count,
            destination_manifest_output=args.destination_manifest_output,
            relocation_receipt_output=args.relocation_receipt_output,
        )
    elif args.command == "verify-static-inputs":
        result = verify_static_inputs(
            phase=args.phase,
            checkpoint_path=args.checkpoint_path,
            destination_manifest_path=args.destination_manifest_path,
            expected_destination_manifest_sha256=args.expected_destination_manifest_sha256,
            training_config_path=args.training_config_path,
            expected_training_config_sha256=args.expected_training_config_sha256,
            action_stats_path=args.action_stats_path,
            expected_action_stats_sha256=args.expected_action_stats_sha256,
            wan2pt2_vae_path=args.wan2pt2_vae_path,
            expected_wan2pt2_vae_sha256=args.expected_wan2pt2_vae_sha256,
            output=args.output,
        )
    elif args.command == "verify-source-snapshot":
        result = verify_source_snapshot(
            phase=args.phase,
            snapshot_root=args.snapshot_root,
            manifest_path=args.manifest_path,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_source_commit=args.expected_source_commit,
            output=args.output,
        )
    elif args.command == "verify-environment":
        result = verify_environment(
            phase=args.phase,
            environment=args.environment,
            environment_type=args.environment_type,
            installation_mode=args.installation_mode,
            marker_path=args.marker_path,
            expected_marker_sha256=args.expected_marker_sha256,
            expected_source_commit=args.expected_source_commit,
            expected_robocasa_commit=args.expected_robocasa_commit,
            expected_robosuite_commit=args.expected_robosuite_commit,
            import_probes=args.import_probe,
            expect_cuda_gpus=args.expect_cuda_gpus,
            output=args.output,
        )
    else:
        result = verify_closure_provenance(
            closure_manifest_path=args.closure_manifest_path,
            expected_closure_manifest_sha256=args.expected_closure_manifest_sha256,
            h_verification_receipt_path=args.h_verification_receipt_path,
            expected_h_verification_receipt_sha256=args.expected_h_verification_receipt_sha256,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
