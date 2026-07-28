#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fail-closed input binding for the RoboCasa365 Cosmos3 formal continuation.

The continuation is deliberately derived from three immutable inputs:

* one published Wan latent cache;
* one passed throughput matrix and its ``best_observed_cell_id``; and
* the complete R6 ``iter_000001000`` model/optimizer/scheduler/trainer DCP.

This module validates those inputs, records their cryptographic identities, and
creates a fresh run directory whose only initial checkpoint is an absolute
symlink to the audited R6 checkpoint.  It never consumes the throughput probe's
fixed-index trace: that trace exists only to compare benchmark cells fairly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
CHECKPOINT_ITERATION = 1_000
FINAL_ITERATION = 10_000
CHECKPOINT_COMPONENTS = ("model", "optim", "scheduler", "trainer")
CHECKPOINT_WORLD_SIZE = 8
CHECKPOINT_FINGERPRINT_ALGORITHM = "sha256-full-dcp-content-inventory-v3"
EXPECTED_DCP_SHARD_NAMES = tuple(f"__{rank}_0.distcp" for rank in range(CHECKPOINT_WORLD_SIZE))
THROUGHPUT_WORKERS = (4, 8, 12, 16)
THROUGHPUT_TRACE_ALGORITHM = "sha256-canonical-ranked-microbatch-identities-text-token-contents-and-token-lengths-v1"
OFFICIAL_WAN22_VAE_SHA256 = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
CELL_ID_PATTERN = re.compile(
    r"^workers-(?P<workers>4|8|12|16)__storage-(?P<storage>gpfs|nvme)"
    r"__ac-(?P<activation_checkpointing>selective|none)$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
METADATA_ONLY_VIDEO_TOKENIZER_TARGET = (
    "cosmos_framework.model.generator.tokenizers.precomputed_wan2pt2.PrecomputedWan2pt2LatentTokenizer"
)
PINNED_FORMAL_OUTPUT_ROOT = Path(
    "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-formal"
)
PINNED_CONTAINER_IMAGE = (
    "registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
)
PINNED_MOUNTS = [
    "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs",
    "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public",
]
HOST_NVME_BINDING = {
    "annotation": "kubebrain.brainpp.cn/writeread-hostpath-mounts",
    "value": "/nvme:/mnt/nvme1n1,/cfs:/mnt/cfs",
    "container_target": "/nvme",
    "host_source": "/mnt/nvme1n1",
}


class ContinuationError(ValueError):
    """Raised when a formal-continuation input violates its immutable contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ContinuationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ContinuationError(f"expected a regular non-symlink file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ContinuationError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _load_json_and_sha256(path: Path, label: str) -> tuple[dict[str, Any], str]:
    digest = _sha256_file(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContinuationError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} must contain one JSON object: {path}")
    return value, digest


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} must be an object")
    return value


def _canonical_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    if path != absolute or path.is_symlink() or not path.is_dir():
        raise ContinuationError(f"{label} must be an absolute non-symlink directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ContinuationError(f"{label} must already be canonical: {path}")
    return resolved


def _require_readonly_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContinuationError(f"{label} must be a regular non-symlink file: {path}")
    if path.stat().st_mode & 0o777 != 0o444:
        raise ContinuationError(f"{label} must be immutable mode 0444: {path}")


def _require_path_absent(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise ContinuationError(f"{label} must be absent: {path}")


def _parse_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ContinuationError(f"{label} must be a non-empty timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContinuationError(f"{label} is not ISO-8601: {value!r}") from error
    if parsed.tzinfo is None:
        raise ContinuationError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _validate_submission_artifact(
    *,
    descriptor_value: object,
    expected_path: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = _require_mapping(descriptor_value, f"{label} descriptor")
    if set(descriptor) != {"path", "size_bytes", "sha256"}:
        raise ContinuationError(f"{label} descriptor has an unexpected schema")
    if descriptor.get("path") != str(expected_path):
        raise ContinuationError(f"{label} path mismatch: expected={expected_path}, actual={descriptor.get('path')!r}")
    _require_readonly_regular_file(expected_path, label)
    size_bytes = expected_path.stat().st_size
    if type(descriptor.get("size_bytes")) is not int or descriptor["size_bytes"] != size_bytes or size_bytes <= 0:
        raise ContinuationError(f"{label} size binding is invalid")
    expected_sha256 = _require_sha256(
        descriptor.get("sha256"),
        f"{label} descriptor SHA-256",
    )
    payload, observed_sha256 = (
        _load_json_and_sha256(
            expected_path,
            label,
        )
        if expected_path.suffix == ".json"
        else ({}, _sha256_file(expected_path))
    )
    if observed_sha256 != expected_sha256:
        raise ContinuationError(f"{label} SHA-256 binding mismatch")
    return payload, dict(descriptor)


def validate_submission_binding(
    *,
    reservation_path: Path,
    submission_nonce: str,
    expected_command_sha256: str,
    expected_cpu: int,
    experiment_id: str,
    cache: Mapping[str, Any],
    throughput: Mapping[str, Any],
    resume: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable submitter, CRD, and expanded Replica binding."""

    submission_nonce = _require_sha256(
        submission_nonce,
        "formal submission nonce",
    )
    expected_command_sha256 = _require_sha256(
        expected_command_sha256,
        "formal scheduler command SHA-256",
    )
    if expected_cpu not in {128, 160}:
        raise ContinuationError("formal scheduler CPU binding must be 128 or 160")
    expected_reservation = (
        PINNED_FORMAL_OUTPUT_ROOT / "formal_continuation_submission_reservations" / experiment_id / "submission.json"
    )
    if reservation_path != expected_reservation:
        raise ContinuationError("formal submission reservation must use the canonical continuation path")
    submission_unknown_path = reservation_path.with_name("submission-unknown.json")
    _require_path_absent(
        submission_unknown_path,
        "formal continuation submission-unknown sibling",
    )
    _require_readonly_regular_file(
        reservation_path,
        "formal continuation submission reservation",
    )
    if reservation_path.parent.stat().st_mode & 0o777 != 0o555:
        raise ContinuationError("formal continuation reservation directory must be sealed mode 0555")
    reservation, reservation_sha256 = _load_json_and_sha256(
        reservation_path,
        "formal continuation submission reservation",
    )
    expected_reservation_keys = {
        "schema_version",
        "kind",
        "status",
        "created_at",
        "accepted_at",
        "experiment_id",
        "run_name",
        "rjob_show_name",
        "cpu",
        "gpu",
        "memory_mib",
        "workers_per_rank",
        "storage",
        "activation_checkpointing",
        "best_observed_cell_id",
        "cache_success_sha256",
        "cache_manifest_sha256",
        "throughput_matrix_sha256",
        "r6_checkpoint_fingerprint_sha256",
        "source_commit",
        "snapshot_manifest_sha256",
        "container_image",
        "command_sha256",
        "store_host_nvme",
        "submission_nonce",
        "submission_reservation_path",
        "mounts",
        "scheduler_rjob_name",
        "scheduler_rjob_uid",
        "scheduler_replica_name",
        "scheduler_pod_name",
        "artifacts",
    }
    if set(reservation) != expected_reservation_keys:
        raise ContinuationError(
            "formal submission reservation schema mismatch: "
            f"missing={sorted(expected_reservation_keys - set(reservation))}, "
            f"extra={sorted(set(reservation) - expected_reservation_keys)}"
        )
    for key, expected in {
        "schema_version": 2,
        "kind": "cwam_cosmos3_formal_continuation_submission_reservation",
        "status": "accepted",
        "experiment_id": experiment_id,
        "run_name": experiment_id,
        "cpu": expected_cpu,
        "gpu": 8,
        "memory_mib": 1_500_000,
        "workers_per_rank": throughput["workers_per_rank"],
        "storage": throughput["storage"],
        "activation_checkpointing": throughput["activation_checkpointing"],
        "best_observed_cell_id": throughput["best_observed_cell_id"],
        "cache_success_sha256": cache["success_sha256"],
        "cache_manifest_sha256": cache["manifest_sha256"],
        "throughput_matrix_sha256": throughput["matrix_summary_sha256"],
        "r6_checkpoint_fingerprint_sha256": resume["fingerprint_sha256"],
        "container_image": PINNED_CONTAINER_IMAGE,
        "command_sha256": expected_command_sha256,
        "store_host_nvme": throughput["storage"] == "nvme",
        "submission_nonce": submission_nonce,
        "submission_reservation_path": str(reservation_path),
        "mounts": PINNED_MOUNTS,
    }.items():
        if reservation.get(key) != expected:
            raise ContinuationError(
                f"formal submission reservation {key} mismatch: expected={expected!r}, actual={reservation.get(key)!r}"
            )
    created_at = _parse_timestamp(
        reservation.get("created_at"),
        "formal submission created_at",
    )
    accepted_at = _parse_timestamp(
        reservation.get("accepted_at"),
        "formal submission accepted_at",
    )
    if accepted_at < created_at:
        raise ContinuationError("formal submission accepted_at precedes reservation creation")
    if (
        not isinstance(reservation.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", reservation["source_commit"]) is None
    ):
        raise ContinuationError("formal submission source commit is invalid")
    _require_sha256(
        reservation.get("snapshot_manifest_sha256"),
        "formal submission snapshot manifest SHA-256",
    )
    scheduler_rjob_name = reservation.get("scheduler_rjob_name")
    showname = reservation.get("rjob_show_name")
    if (
        not isinstance(showname, str)
        or not isinstance(scheduler_rjob_name, str)
        or not scheduler_rjob_name.startswith(showname + "-")
        or re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", scheduler_rjob_name) is None
    ):
        raise ContinuationError("formal submission scheduler RJob identity is invalid")
    artifacts = _require_mapping(
        reservation.get("artifacts"),
        "formal submission artifacts",
    )
    expected_artifact_names = {
        "submission_log": "rjob-submit.log",
        "scheduler_expected_contract": "rjob-expected-contract.json",
        "scheduler_capture_log": "rjob-scheduler-capture.log",
        "scheduler_structured_receipt": "rjob-scheduler-receipt.json",
        "scheduler_text_readback": "rjob-get.log",
    }
    if set(artifacts) != set(expected_artifact_names):
        raise ContinuationError("formal submission artifact set mismatch")
    artifact_payloads: dict[str, dict[str, Any]] = {}
    normalized_descriptors: dict[str, dict[str, Any]] = {}
    for key, filename in expected_artifact_names.items():
        payload, descriptor = _validate_submission_artifact(
            descriptor_value=artifacts[key],
            expected_path=reservation_path.parent / filename,
            label=f"formal submission {key}",
        )
        artifact_payloads[key] = payload
        normalized_descriptors[key] = descriptor

    expected = artifact_payloads["scheduler_expected_contract"]
    expected_keys = {
        "schema_version",
        "scheduler_rjob_name",
        "showname",
        "quota_group",
        "task_type",
        "priority",
        "backoff_limit",
        "task_name",
        "replicas",
        "image",
        "cpu",
        "memory_mib",
        "gpu",
        "negative_tags",
        "private_machine",
        "restart_policy",
        "termination_grace_period_seconds",
        "mounts",
        "snapshot_root",
        "wrapper_path",
        "experiment_id",
        "command_sha256",
        "submission_nonce",
        "submission_reservation_path",
        "store_host_nvme",
    }
    if set(expected) != expected_keys:
        raise ContinuationError("scheduler expected-contract schema mismatch")
    for key, value in {
        "schema_version": 2,
        "scheduler_rjob_name": scheduler_rjob_name,
        "showname": showname,
        "quota_group": "eailabagent_gpu",
        "task_type": "normal",
        "priority": 9,
        "backoff_limit": 1,
        "task_name": "t0",
        "replicas": 1,
        "image": PINNED_CONTAINER_IMAGE,
        "cpu": expected_cpu,
        "memory_mib": 1_500_000,
        "gpu": 8,
        "negative_tags": ["node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"],
        "private_machine": "group",
        "restart_policy": "Never",
        "termination_grace_period_seconds": 60,
        "mounts": PINNED_MOUNTS,
        "experiment_id": experiment_id,
        "command_sha256": expected_command_sha256,
        "submission_nonce": submission_nonce,
        "submission_reservation_path": str(reservation_path),
        "store_host_nvme": throughput["storage"] == "nvme",
    }.items():
        if expected.get(key) != value:
            raise ContinuationError(f"scheduler expected-contract {key} mismatch")
    wrapper_path = Path(str(expected.get("wrapper_path")))
    snapshot_root = Path(str(expected.get("snapshot_root")))
    if wrapper_path != (snapshot_root / "examples/h_cluster/run_cwam_cosmos3_nano_formal_continuation.sh"):
        raise ContinuationError("scheduler wrapper is outside its snapshot root")

    receipt = artifact_payloads["scheduler_structured_receipt"]
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "cwam_formal_rjob_scheduler_receipt"
        or receipt.get("validation")
        != {
            "status": "passed",
            "expected_contract_schema_version": 2,
        }
    ):
        raise ContinuationError("structured scheduler receipt is not passed v2")
    metadata = _require_mapping(
        receipt.get("metadata"),
        "structured scheduler metadata",
    )
    for key, value in {
        "name": scheduler_rjob_name,
        "showname": showname,
        "uid": reservation.get("scheduler_rjob_uid"),
    }.items():
        if metadata.get(key) != value:
            raise ContinuationError(f"structured scheduler metadata {key} mismatch")
    scheduler_spec = _require_mapping(
        receipt.get("spec"),
        "structured scheduler spec",
    )
    for key, value in {
        "quota_group": "eailabagent_gpu",
        "task_type": "normal",
        "priority": 9,
        "backoff_limit": 1,
    }.items():
        if scheduler_spec.get(key) != value:
            raise ContinuationError(f"structured scheduler spec {key} mismatch")
    task = _require_mapping(
        scheduler_spec.get("task"),
        "structured scheduler task",
    )
    host_nvme = {
        "enabled": throughput["storage"] == "nvme",
        "annotation": HOST_NVME_BINDING["annotation"],
        "value": (HOST_NVME_BINDING["value"] if throughput["storage"] == "nvme" else None),
        "container_target": (HOST_NVME_BINDING["container_target"] if throughput["storage"] == "nvme" else None),
        "host_source": (HOST_NVME_BINDING["host_source"] if throughput["storage"] == "nvme" else None),
    }
    for key, value in {
        "name": "t0",
        "replicas": 1,
        "negative_tags": expected["negative_tags"],
        "private_machine": "group",
        "restart_policy": "Never",
        "termination_grace_period_seconds": 60,
        "mounts": PINNED_MOUNTS,
        "store_host_nvme": host_nvme,
        "submission_nonce": submission_nonce,
        "submission_reservation_path": str(reservation_path),
    }.items():
        if task.get(key) != value:
            raise ContinuationError(f"structured scheduler task {key} mismatch")
    container = _require_mapping(
        task.get("container"),
        "structured scheduler container",
    )
    expected_resources = {
        "requests_equal_core_limits": True,
        "cpu": expected_cpu,
        "memory_mib": 1_500_000,
        "gpu": 8,
        "platform_injected_limit_keys": [],
    }
    if (
        container.get("image") != PINNED_CONTAINER_IMAGE
        or container.get("resources") != expected_resources
        or container.get("snapshot_root") != str(snapshot_root)
        or container.get("wrapper_path") != str(wrapper_path)
        or container.get("experiment_id") != experiment_id
        or container.get("command_sha256") != expected_command_sha256
    ):
        raise ContinuationError("structured scheduler container binding mismatch")
    command = container.get("command")
    if (
        not isinstance(command, list)
        or len(command) != 3
        or command[:2] != ["bash", "-lc"]
        or not isinstance(command[2], str)
        or hashlib.sha256(command[2].encode()).hexdigest() != expected_command_sha256
    ):
        raise ContinuationError("structured scheduler command binding mismatch")
    replicas = receipt.get("replicas")
    if not isinstance(replicas, list) or len(replicas) != 1:
        raise ContinuationError("structured scheduler receipt must bind exactly one Replica")
    replica = _require_mapping(replicas[0], "structured scheduler Replica")
    for key, value in {
        "replica_name": reservation.get("scheduler_replica_name"),
        "pod_name": reservation.get("scheduler_pod_name"),
    }.items():
        if replica.get(key) != value:
            raise ContinuationError(f"structured scheduler Replica {key} mismatch")
    worker = _require_mapping(
        replica.get("worker_spec"),
        "structured scheduler expanded worker",
    )
    if (
        worker.get("image") != PINNED_CONTAINER_IMAGE
        or worker.get("mounts") != PINNED_MOUNTS
        or worker.get("store_host_nvme") != host_nvme
        or worker.get("restart_policy") != "Never"
        or worker.get("termination_grace_period_seconds") != 60
    ):
        raise ContinuationError("structured scheduler expanded worker binding mismatch")
    worker_resources = _require_mapping(
        worker.get("resources"),
        "structured scheduler expanded worker resources",
    )
    injected = worker_resources.get("platform_injected_limit_keys")
    if (
        not isinstance(injected, list)
        or not set(injected).issubset({"mellanox.com/mlnx_rdma", "rdma/mlnx_shared"})
        or worker_resources
        != {
            **expected_resources,
            "platform_injected_limit_keys": injected,
        }
    ):
        raise ContinuationError("structured scheduler expanded worker resources mismatch")
    if worker.get("command") != worker.get("worker_init_prefix", []) + command:
        raise ContinuationError("structured scheduler expanded worker command mismatch")
    source_objects = _require_mapping(
        receipt.get("source_objects"),
        "structured scheduler source-object hashes",
    )
    if set(source_objects) != {
        "rjob_raw_sha256",
        "replicas_raw_sha256",
    }:
        raise ContinuationError("structured scheduler raw-object hash set mismatch")
    for key, value in source_objects.items():
        _require_sha256(value, f"structured scheduler {key}")

    _require_path_absent(
        submission_unknown_path,
        "formal continuation submission-unknown sibling",
    )
    return {
        "reservation_path": str(reservation_path),
        "reservation_sha256": reservation_sha256,
        "submission_nonce": submission_nonce,
        "command_sha256": expected_command_sha256,
        "container_image": PINNED_CONTAINER_IMAGE,
        "resources": {
            "cpu": expected_cpu,
            "memory_mib": 1_500_000,
            "gpu": 8,
        },
        "mounts": list(PINNED_MOUNTS),
        "store_host_nvme": host_nvme,
        "scheduler_rjob_name": scheduler_rjob_name,
        "scheduler_rjob_uid": reservation["scheduler_rjob_uid"],
        "scheduler_replica_name": reservation["scheduler_replica_name"],
        "scheduler_pod_name": reservation["scheduler_pod_name"],
        "artifacts": normalized_descriptors,
        "scheduler_source_objects": dict(source_objects),
    }


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContinuationError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != value:
        raise ContinuationError(f"{label} is not a safe normalized relative path: {value!r}")
    return value


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mode": stat.st_mode,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _validate_torch_dcp_metadata(
    component_dir: Path,
    *,
    nonempty_shards: set[str],
) -> dict[str, Any]:
    try:
        from torch.distributed.checkpoint import FileSystemReader
    except ImportError as error:
        raise ContinuationError("Torch DCP metadata validation was required but torch is unavailable") from error

    try:
        metadata = FileSystemReader(component_dir).read_metadata()
        state_dict_metadata = metadata.state_dict_metadata
        storage_data = metadata.storage_data
    except (
        AttributeError,
        EOFError,
        OSError,
        pickle.PickleError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise ContinuationError(f"cannot read Torch DCP metadata for {component_dir}: {error}") from error
    if not state_dict_metadata or not storage_data:
        raise ContinuationError(f"Torch DCP metadata has no state or storage entries: {component_dir}")
    referenced_shards: set[str] = set()
    for storage_info in storage_data.values():
        relative_path = getattr(storage_info, "relative_path", None)
        if not isinstance(relative_path, str):
            raise ContinuationError(f"Torch DCP storage entry has no relative path: {component_dir}")
        relative_path = _safe_relative_path(
            relative_path,
            f"Torch DCP storage path in {component_dir}",
        )
        if relative_path not in EXPECTED_DCP_SHARD_NAMES:
            raise ContinuationError(
                f"Torch DCP metadata references an unexpected shard: {component_dir / relative_path}"
            )
        referenced_shards.add(relative_path)
    if referenced_shards != nonempty_shards:
        raise ContinuationError(
            "Torch DCP metadata/payload mismatch for "
            f"{component_dir}: referenced={sorted(referenced_shards)}, "
            f"nonempty={sorted(nonempty_shards)}"
        )
    return {
        "status": "passed",
        "state_dict_entry_count": len(state_dict_metadata),
        "storage_entry_count": len(storage_data),
        "referenced_shards": sorted(referenced_shards),
    }


def _checkpoint_inventory(
    path: Path,
    *,
    hash_files: bool,
    require_torch_metadata: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    children = sorted(child.name for child in path.iterdir())
    if children != sorted(CHECKPOINT_COMPONENTS):
        raise ContinuationError(
            f"R6 checkpoint component set mismatch: expected={sorted(CHECKPOINT_COMPONENTS)}, actual={children}"
        )

    files: list[dict[str, Any]] = []
    component_reports: dict[str, Any] = {}
    expected_names = sorted((".metadata", *EXPECTED_DCP_SHARD_NAMES))
    for component in CHECKPOINT_COMPONENTS:
        component_dir = path / component
        if component_dir.is_symlink() or not component_dir.is_dir():
            raise ContinuationError(f"R6 checkpoint component is unsafe: {component_dir}")
        if component_dir.stat().st_mode & 0o022:
            raise ContinuationError(f"R6 checkpoint component must not be group/other writable: {component_dir}")
        component_files = sorted(component_dir.iterdir(), key=lambda item: item.name)
        names = [item.name for item in component_files]
        if names != expected_names:
            raise ContinuationError(
                f"R6 checkpoint component file set mismatch for {component_dir}: "
                f"expected={expected_names}, actual={names}"
            )

        nonempty_shards: set[str] = set()
        zero_byte_shards: list[str] = []
        for file_path in component_files:
            if file_path.is_symlink() or not file_path.is_file():
                raise ContinuationError(f"R6 checkpoint contains an unsafe entry: {file_path}")
            identity = _stat_identity(file_path)
            if identity["mode"] & 0o022:
                raise ContinuationError(f"R6 checkpoint file must not be group/other writable: {file_path}")
            if file_path.name == ".metadata":
                if identity["size_bytes"] <= 0:
                    raise ContinuationError(f"R6 checkpoint metadata must be non-empty: {file_path}")
            elif file_path.name in EXPECTED_DCP_SHARD_NAMES:
                if identity["size_bytes"] > 0:
                    nonempty_shards.add(file_path.name)
                else:
                    zero_byte_shards.append(file_path.name)
            else:  # The exact-name comparison above makes this defensive.
                raise ContinuationError(f"R6 checkpoint contains a non-DCP entry: {file_path}")
            item: dict[str, Any] = {
                "path": file_path.relative_to(path).as_posix(),
                **identity,
            }
            if hash_files:
                item["sha256"] = _sha256_file(file_path)
                if _stat_identity(file_path) != identity:
                    raise ContinuationError(f"R6 checkpoint entry changed while fingerprinting: {file_path}")
            files.append(item)
        if not nonempty_shards:
            raise ContinuationError(f"R6 checkpoint component has no non-empty DCP payload shard: {component_dir}")
        torch_metadata = (
            _validate_torch_dcp_metadata(
                component_dir,
                nonempty_shards=nonempty_shards,
            )
            if require_torch_metadata
            else {"status": "not_requested"}
        )
        component_reports[component] = {
            "nonempty_payload_shards": sorted(nonempty_shards),
            "zero_byte_rank_shards": zero_byte_shards,
            "torch_dcp_metadata": torch_metadata,
        }
    return files, component_reports


def validate_cache_binding(
    root: Path,
    *,
    expected_success_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind the exact completion marker and manifest selected for training."""

    root = _canonical_directory(root, "latent-cache root")
    expected_success_sha256 = _require_sha256(
        expected_success_sha256,
        "expected cache _SUCCESS SHA-256",
    )
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        "expected cache manifest SHA-256",
    )
    success_path = root / "_SUCCESS.json"
    manifest_path = root / "manifest.json"
    success, success_sha256 = _load_json_and_sha256(success_path, "cache _SUCCESS")
    manifest, manifest_sha256 = _load_json_and_sha256(manifest_path, "cache manifest")
    if success_sha256 != expected_success_sha256:
        raise ContinuationError(
            f"cache _SUCCESS SHA-256 mismatch: expected={expected_success_sha256}, actual={success_sha256}"
        )
    if manifest_sha256 != expected_manifest_sha256:
        raise ContinuationError(
            f"cache manifest SHA-256 mismatch: expected={expected_manifest_sha256}, actual={manifest_sha256}"
        )
    if success.get("schema_version") != 1 or success.get("status") != "complete":
        raise ContinuationError("cache _SUCCESS must be schema-v1 and complete")
    if _safe_relative_path(success.get("manifest_path"), "_SUCCESS.manifest_path") != "manifest.json":
        raise ContinuationError("cache _SUCCESS must point exactly to manifest.json")
    if success.get("manifest_sha256") != manifest_sha256:
        raise ContinuationError("cache _SUCCESS does not bind the selected manifest")
    if manifest.get("schema_version") != 1:
        raise ContinuationError("cache manifest must use schema_version=1")
    cache_id = manifest.get("cache_id")
    if not isinstance(cache_id, str) or not cache_id or success.get("cache_id") != cache_id:
        raise ContinuationError("cache ID differs between _SUCCESS and manifest")
    equivalence = _require_mapping(manifest.get("equivalence"), "cache manifest equivalence")
    if equivalence.get("status") != "passed" or equivalence.get("torch_equal") is not True:
        raise ContinuationError("cache has no passed bitwise equivalence gate")
    if not isinstance(manifest.get("shards"), list) or not manifest["shards"]:
        raise ContinuationError("cache manifest has no shards")
    if not isinstance(manifest.get("splits"), dict) or not manifest["splits"]:
        raise ContinuationError("cache manifest has no split mappings")
    return {
        "root": str(root),
        "cache_id": cache_id,
        "success_path": str(success_path),
        "success_sha256": success_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
    }


def _validate_throughput_artifact_binding(
    *,
    matrix_path: Path,
    matrix: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    run_root_value = matrix.get("run_root")
    if not isinstance(run_root_value, str):
        raise ContinuationError("throughput matrix has no run_root")
    run_root = _canonical_directory(Path(run_root_value), "throughput run root")
    if matrix_path != run_root / "matrix-summary.json":
        raise ContinuationError("throughput matrix must be the canonical matrix-summary.json under run_root")

    run_exit_path = run_root / "run-exit.json"
    run_exit, run_exit_sha256 = _load_json_and_sha256(
        run_exit_path,
        "throughput run exit receipt",
    )
    expected_run_exit = {
        "schema_version": 1,
        "status": "completed",
        "exit_code": 0,
        "last_phase": "complete",
    }
    if run_exit != expected_run_exit:
        raise ContinuationError(f"throughput run has no successful terminal receipt: {run_exit!r}")

    shared_contract: dict[str, Any] | None = None
    cell_artifacts: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = str(cell.get("cell_id"))
        match = CELL_ID_PATTERN.fullmatch(cell_id)
        if match is None:
            raise ContinuationError(f"invalid throughput artifact cell ID: {cell_id!r}")
        cell_root = _canonical_directory(
            run_root / "cells" / cell_id,
            f"throughput cell root for {cell_id}",
        )
        metrics_path = cell_root / "metrics.json"
        metrics, metrics_sha256 = _load_json_and_sha256(
            metrics_path,
            f"throughput metrics for {cell_id}",
        )
        if metrics != cell:
            raise ContinuationError(f"throughput matrix cell does not match its metrics artifact: {cell_id}")
        contract_path = cell_root / "contract.json"
        contract, contract_sha256 = _load_json_and_sha256(
            contract_path,
            f"throughput contract for {cell_id}",
        )
        expected_contract_fields = {
            "schema_version": 1,
            "status": "planned",
            "run_id": run_root.name,
            "cell_id": cell_id,
            "world_size": CHECKPOINT_WORLD_SIZE,
            "workers_per_rank": int(match.group("workers")),
            "node_total_workers": int(match.group("workers")) * CHECKPOINT_WORLD_SIZE,
            "storage": match.group("storage"),
            "activation_checkpointing": match.group("activation_checkpointing"),
            "warmup_steps": 10,
            "measure_steps": 50,
            "total_steps": 60,
            "resume_mode": "r6_iter1000_model_optimizer",
            "data_stream_resume": "restarted_not_cursor_resumed",
        }
        mismatches = {
            key: {"expected": expected, "actual": contract.get(key)}
            for key, expected in expected_contract_fields.items()
            if contract.get(key) != expected
        }
        if mismatches:
            raise ContinuationError(f"throughput cell contract mismatch for {cell_id}: {mismatches}")
        tokenizer = _require_mapping(
            contract.get("video_tokenizer"),
            f"throughput video tokenizer for {cell_id}",
        )
        if tokenizer != {
            "target": METADATA_ONLY_VIDEO_TOKENIZER_TARGET,
            "metadata_only": True,
        }:
            raise ContinuationError(f"throughput cell did not use the metadata-only tokenizer: {cell_id}")
        timeout = contract.get("cell_timeout_seconds")
        if (match.group("activation_checkpointing") == "selective" and timeout != 0) or (
            match.group("activation_checkpointing") == "none" and (not isinstance(timeout, int) or timeout < 600)
        ):
            raise ContinuationError(f"throughput cell timeout contract is invalid: {cell_id}")

        latent_cache_root = contract.get("latent_cache_root")
        if not isinstance(latent_cache_root, str):
            raise ContinuationError(f"throughput cell has no latent-cache root: {cell_id}")
        if match.group("storage") == "gpfs":
            gpfs_cache_root = str(
                _canonical_directory(
                    Path(latent_cache_root),
                    f"throughput GPFS cache root for {cell_id}",
                )
            )
        else:
            expected_nvme_root = Path("/nvme/.cwam-cosmos3-latent-cache") / run_root.name / "cache"
            if Path(latent_cache_root) != expected_nvme_root:
                raise ContinuationError(
                    f"throughput NVMe cache root mismatch for {cell_id}: "
                    f"expected={expected_nvme_root}, actual={latent_cache_root}"
                )
            gpfs_cache_root = None

        fixed_trace_path = contract.get("fixed_index_trace_path")
        resume_checkpoint = contract.get("resume_checkpoint")
        node_evidence = contract.get("node_evidence")
        for label, value in (
            ("fixed-index trace path", fixed_trace_path),
            ("resume checkpoint", resume_checkpoint),
            ("node evidence", node_evidence),
        ):
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ContinuationError(f"throughput {label} is not an absolute path for {cell_id}")
        shared = {
            "experiment_id": contract.get("experiment_id"),
            "run_id": contract.get("run_id"),
            "latent_cache_manifest_sha256": _require_sha256(
                contract.get("latent_cache_manifest_sha256"),
                f"throughput cache manifest SHA-256 for {cell_id}",
            ),
            "fixed_index_trace_path": fixed_trace_path,
            "fixed_index_trace_sha256": _require_sha256(
                contract.get("fixed_index_trace_sha256"),
                f"throughput fixed trace SHA-256 for {cell_id}",
            ),
            "fixed_index_trace_manifest_sha256": _require_sha256(
                contract.get("fixed_index_trace_manifest_sha256"),
                f"throughput fixed trace manifest SHA-256 for {cell_id}",
            ),
            "wan_vae_path": contract.get("wan_vae_path"),
            "wan_vae_sha256": _require_sha256(
                contract.get("wan_vae_sha256"),
                f"throughput Wan VAE SHA-256 for {cell_id}",
            ),
            "resume_checkpoint": resume_checkpoint,
            "source_root": contract.get("source_root"),
            "source_commit": contract.get("source_commit"),
            "snapshot_manifest_sha256": _require_sha256(
                contract.get("snapshot_manifest_sha256"),
                f"throughput snapshot manifest SHA-256 for {cell_id}",
            ),
            "runner_sha256": _require_sha256(
                contract.get("runner_sha256"),
                f"throughput runner SHA-256 for {cell_id}",
            ),
            "helper_sha256": _require_sha256(
                contract.get("helper_sha256"),
                f"throughput helper SHA-256 for {cell_id}",
            ),
            "node_evidence": node_evidence,
        }
        if (
            not isinstance(shared["experiment_id"], str)
            or not shared["experiment_id"].startswith("CWAM-E35-")
            or shared["run_id"] != run_root.name
            or not isinstance(shared["source_root"], str)
            or not Path(shared["source_root"]).is_absolute()
            or not isinstance(shared["source_commit"], str)
            or re.fullmatch(r"[0-9a-f]{40}", shared["source_commit"]) is None
            or not isinstance(shared["wan_vae_path"], str)
            or not Path(shared["wan_vae_path"]).is_absolute()
        ):
            raise ContinuationError(f"throughput shared provenance contract is invalid: {cell_id}")
        if shared_contract is None:
            shared_contract = shared
        elif shared != shared_contract:
            raise ContinuationError(f"throughput cells do not share one benchmark input contract: {cell_id}")
        cell_artifacts.append(
            {
                "cell_id": cell_id,
                "contract_path": str(contract_path),
                "contract_sha256": contract_sha256,
                "metrics_path": str(metrics_path),
                "metrics_sha256": metrics_sha256,
                "gpfs_latent_cache_root": gpfs_cache_root,
            }
        )
    if shared_contract is None:
        raise ContinuationError("throughput matrix contains no artifact-bound cells")

    gpfs_roots = {
        item["gpfs_latent_cache_root"] for item in cell_artifacts if item["gpfs_latent_cache_root"] is not None
    }
    if len(gpfs_roots) != 1:
        raise ContinuationError(f"throughput matrix must bind one GPFS cache root: {sorted(gpfs_roots)}")
    fixed_trace_path = Path(shared_contract["fixed_index_trace_path"])
    if (
        fixed_trace_path.is_symlink()
        or not fixed_trace_path.is_file()
        or fixed_trace_path.resolve(strict=True) != fixed_trace_path
    ):
        raise ContinuationError(f"throughput fixed-index trace is not canonical and regular: {fixed_trace_path}")
    fixed_trace_sha256 = _sha256_file(fixed_trace_path)
    if fixed_trace_sha256 != shared_contract["fixed_index_trace_sha256"]:
        raise ContinuationError("throughput fixed-index trace SHA-256 drifted")
    fixed_trace_manifest_path = fixed_trace_path.with_suffix(".manifest.json")
    fixed_trace_manifest_sha256 = _sha256_file(fixed_trace_manifest_path)
    if fixed_trace_manifest_sha256 != shared_contract["fixed_index_trace_manifest_sha256"]:
        raise ContinuationError("throughput fixed-index trace manifest SHA-256 drifted")

    node_evidence_path = Path(shared_contract["node_evidence"])
    node_evidence, node_evidence_sha256 = _load_json_and_sha256(
        node_evidence_path,
        "throughput eight-H200 node evidence",
    )
    for key, expected in {
        "schema_version": 1,
        "status": "passed",
        "expected_gpu_count": CHECKPOINT_WORLD_SIZE,
        "visible_gpu_count": CHECKPOINT_WORLD_SIZE,
        "training_rank_count": CHECKPOINT_WORLD_SIZE,
    }.items():
        if node_evidence.get(key) != expected:
            raise ContinuationError(
                f"throughput node evidence {key} mismatch: expected={expected!r}, actual={node_evidence.get(key)!r}"
            )
    if (
        not isinstance(node_evidence.get("gpus"), list)
        or len(node_evidence["gpus"]) != CHECKPOINT_WORLD_SIZE
        or any(not isinstance(gpu, dict) or gpu.get("name") != "NVIDIA H200" for gpu in node_evidence["gpus"])
    ):
        raise ContinuationError("throughput node evidence is not exactly eight H200 GPUs")

    return {
        "run_root": str(run_root),
        "run_exit_path": str(run_exit_path),
        "run_exit_sha256": run_exit_sha256,
        "experiment_id": shared_contract["experiment_id"],
        "run_id": shared_contract["run_id"],
        "gpfs_latent_cache_root": next(iter(gpfs_roots)),
        "latent_cache_manifest_sha256": shared_contract["latent_cache_manifest_sha256"],
        "fixed_index_trace_path": str(fixed_trace_path),
        "fixed_index_trace_sha256": fixed_trace_sha256,
        "fixed_index_trace_manifest_path": str(fixed_trace_manifest_path),
        "fixed_index_trace_manifest_sha256": fixed_trace_manifest_sha256,
        "wan_vae_path": shared_contract["wan_vae_path"],
        "wan_vae_sha256": shared_contract["wan_vae_sha256"],
        "resume_checkpoint": shared_contract["resume_checkpoint"],
        "source_root": shared_contract["source_root"],
        "source_commit": shared_contract["source_commit"],
        "snapshot_manifest_sha256": shared_contract["snapshot_manifest_sha256"],
        "runner_sha256": shared_contract["runner_sha256"],
        "helper_sha256": shared_contract["helper_sha256"],
        "node_evidence_path": str(node_evidence_path),
        "node_evidence_sha256": node_evidence_sha256,
        "cells": cell_artifacts,
    }


def _require_finite_float(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContinuationError(f"{label} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ContinuationError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ContinuationError(f"{label} must be at most {maximum}")
    return result


def _validate_passed_throughput_cell(
    cell: Mapping[str, Any],
    *,
    cell_id: str,
) -> dict[str, Any]:
    if cell.get("schema_version") != 1:
        raise ContinuationError(f"passed throughput cell {cell_id!r} must use schema_version=1")
    if cell.get("status") != "passed" or cell.get("training_exit_code") != 0:
        raise ContinuationError(f"passed throughput cell {cell_id!r} has invalid exit semantics")
    _parse_timestamp(
        cell.get("summarized_at"),
        f"passed throughput cell {cell_id!r} summarized_at",
    )
    for key in ("log_sha256", "optimizer_step_ndjson_sha256"):
        _require_sha256(
            cell.get(key),
            f"passed throughput cell {cell_id!r} {key}",
        )

    timing = _require_mapping(
        cell.get("timing"),
        f"passed throughput cell {cell_id!r} timing",
    )
    timing_contract = {
        "observed_optimizer_steps": 60,
        "contiguous": True,
        "warmup_steps": 10,
        "measure_steps": 50,
        "first_measured_iteration": 1_011,
        "last_measured_iteration": 1_060,
    }
    timing_mismatches = {
        key: {"expected": expected, "actual": timing.get(key)}
        for key, expected in timing_contract.items()
        if timing.get(key) != expected
    }
    if timing_mismatches:
        raise ContinuationError(
            f"passed throughput cell {cell_id!r} optimizer-step contract mismatch: {timing_mismatches}"
        )
    measured_start = timing.get("measured_start_unix_ns")
    measured_end = timing.get("measured_end_unix_ns")
    if (
        type(measured_start) is not int
        or type(measured_end) is not int
        or measured_start <= 0
        or measured_end <= measured_start
    ):
        raise ContinuationError(f"passed throughput cell {cell_id!r} has invalid measured timestamps")
    positive_timing_fields = (
        "mean_step_seconds",
        "median_step_seconds",
        "p90_step_seconds",
        "p95_step_seconds",
        "min_step_seconds",
        "max_step_seconds",
        "optimizer_steps_per_second",
        "mean_timer_dataloader_train_seconds",
        "p95_timer_dataloader_train_seconds",
        "mean_callback_loader_wait_seconds",
        "mean_timer_dataloader_fraction",
    )
    timing_values = {
        key: _require_finite_float(
            timing.get(key),
            f"passed throughput cell {cell_id!r} timing.{key}",
            minimum=0.0,
        )
        for key in positive_timing_fields
    }
    if timing_values["mean_step_seconds"] <= 0:
        raise ContinuationError(f"passed throughput cell {cell_id!r} mean step time is not positive")
    ordered_timing = [
        timing_values["min_step_seconds"],
        timing_values["median_step_seconds"],
        timing_values["p90_step_seconds"],
        timing_values["p95_step_seconds"],
        timing_values["max_step_seconds"],
    ]
    if ordered_timing != sorted(ordered_timing):
        raise ContinuationError(f"passed throughput cell {cell_id!r} timing quantiles are inconsistent")
    if not (
        timing_values["min_step_seconds"] <= timing_values["mean_step_seconds"] <= timing_values["max_step_seconds"]
    ):
        raise ContinuationError(f"passed throughput cell {cell_id!r} mean timing is out of range")
    if not math.isclose(
        timing_values["optimizer_steps_per_second"],
        1.0 / timing_values["mean_step_seconds"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ContinuationError(f"passed throughput cell {cell_id!r} optimizer throughput is inconsistent")
    if not math.isclose(
        timing_values["mean_timer_dataloader_fraction"],
        timing_values["mean_timer_dataloader_train_seconds"] / timing_values["mean_step_seconds"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ContinuationError(f"passed throughput cell {cell_id!r} loader fraction is inconsistent")

    gpu = _require_mapping(
        cell.get("gpu"),
        f"passed throughput cell {cell_id!r} GPU summary",
    )
    gpu_uuids = gpu.get("gpu_uuids")
    if (
        gpu.get("gpu_count") != CHECKPOINT_WORLD_SIZE
        or type(gpu.get("sample_batches")) is not int
        or gpu["sample_batches"] <= 0
        or type(gpu.get("sample_rows")) is not int
        or gpu["sample_rows"] < CHECKPOINT_WORLD_SIZE
        or not isinstance(gpu_uuids, list)
        or len(gpu_uuids) != CHECKPOINT_WORLD_SIZE
        or len(set(gpu_uuids)) != CHECKPOINT_WORLD_SIZE
        or any(not isinstance(value, str) or not value for value in gpu_uuids)
        or gpu.get("window_start_unix_ns") != measured_start
        or gpu.get("window_end_unix_ns") != measured_end
    ):
        raise ContinuationError(f"passed throughput cell {cell_id!r} has incomplete eight-GPU telemetry")
    average_utilization = _require_finite_float(
        gpu.get("average_utilization_percent"),
        f"passed throughput cell {cell_id!r} GPU utilization",
        minimum=0.0,
        maximum=100.0,
    )
    _require_finite_float(
        gpu.get("p50_utilization_percent"),
        f"passed throughput cell {cell_id!r} p50 GPU utilization",
        minimum=0.0,
        maximum=100.0,
    )
    _require_finite_float(
        gpu.get("average_power_draw_w"),
        f"passed throughput cell {cell_id!r} GPU power",
        minimum=0.0,
    )
    _require_finite_float(
        gpu.get("peak_memory_used_mib"),
        f"passed throughput cell {cell_id!r} GPU memory",
        minimum=0.0,
    )
    peak_memory_fraction = _require_finite_float(
        gpu.get("peak_memory_used_fraction"),
        f"passed throughput cell {cell_id!r} GPU memory fraction",
        minimum=0.0,
        maximum=1.0,
    )

    memory = _require_mapping(
        cell.get("production_batch_memory_gate"),
        f"passed throughput cell {cell_id!r} production memory gate",
    )
    memory_contract = {
        "status": "passed",
        "start_iteration": 1_000,
        "observed_iteration": 1_001,
        "world_size": CHECKPOINT_WORLD_SIZE,
        "thresholds": {
            "max_peak_reserved_fraction": 0.94,
            "max_driver_used_fraction": 0.94,
        },
        "production_batch_contract": {
            "expected_samples_per_microbatch": 128,
            "expected_grad_accum_iter": 2,
            "expected_action_tokens_per_sample": 16,
            "allowed_world_sizes": [CHECKPOINT_WORLD_SIZE],
        },
        "violations": [],
    }
    for key, expected in memory_contract.items():
        if memory.get(key) != expected:
            raise ContinuationError(
                f"passed throughput cell {cell_id!r} memory gate {key} "
                f"mismatch: expected={expected!r}, actual={memory.get(key)!r}"
            )
    memory_sha256 = _require_sha256(
        memory.get("sha256"),
        f"passed throughput cell {cell_id!r} memory gate SHA-256",
    )
    rows = memory.get("per_rank")
    if (
        not isinstance(rows, list)
        or len(rows) != CHECKPOINT_WORLD_SIZE
        or sorted(row.get("rank", -1) if isinstance(row, dict) else -1 for row in rows)
        != list(range(CHECKPOINT_WORLD_SIZE))
    ):
        raise ContinuationError(f"passed throughput cell {cell_id!r} memory gate lacks eight ranks")
    row_fractions: dict[str, list[float]] = {
        "peak_allocated_fraction": [],
        "peak_reserved_fraction": [],
        "driver_used_fraction": [],
    }
    for row_index, raw_row in enumerate(rows):
        row = _require_mapping(
            raw_row,
            f"passed throughput cell {cell_id!r} memory row {row_index}",
        )
        for key in row_fractions:
            row_fractions[key].append(
                _require_finite_float(
                    row.get(key),
                    f"passed throughput cell {cell_id!r} memory row {row_index}.{key}",
                    minimum=0.0,
                    maximum=1.0,
                )
            )
    if max(row_fractions["peak_reserved_fraction"]) >= 0.94 or max(row_fractions["driver_used_fraction"]) >= 0.94:
        raise ContinuationError(f"passed throughput cell {cell_id!r} violates the memory headroom gate")
    for summary_key, row_key in (
        ("max_peak_allocated_fraction", "peak_allocated_fraction"),
        ("max_peak_reserved_fraction", "peak_reserved_fraction"),
        ("max_driver_used_fraction", "driver_used_fraction"),
    ):
        if not math.isclose(
            _require_finite_float(
                memory.get(summary_key),
                f"passed throughput cell {cell_id!r} {summary_key}",
            ),
            max(row_fractions[row_key]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ContinuationError(f"passed throughput cell {cell_id!r} memory maximum is inconsistent")

    trace = _require_mapping(
        cell.get("sample_trace"),
        f"passed throughput cell {cell_id!r} sample trace",
    )
    per_step = trace.get("per_step")
    if (
        trace.get("algorithm") != THROUGHPUT_TRACE_ALGORITHM
        or trace.get("step_count") != 60
        or not isinstance(per_step, list)
        or len(per_step) != 60
    ):
        raise ContinuationError(f"passed throughput cell {cell_id!r} sample trace is incomplete")
    for index, raw_step in enumerate(per_step, start=1):
        step = _require_mapping(
            raw_step,
            f"passed throughput cell {cell_id!r} trace step {index}",
        )
        if (
            set(step)
            != {
                "relative_optimizer_step",
                "completed_iteration",
                "identity_count",
                "identity_sha256",
            }
            or step.get("relative_optimizer_step") != index
            or step.get("completed_iteration") != 1_000 + index
            or type(step.get("identity_count")) is not int
            or step["identity_count"] <= 0
        ):
            raise ContinuationError(f"passed throughput cell {cell_id!r} trace step {index} is invalid")
        _require_sha256(
            step.get("identity_sha256"),
            f"passed throughput cell {cell_id!r} trace step {index} SHA-256",
        )
    recomputed_trace_sha256 = hashlib.sha256(
        json.dumps(
            per_step,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if trace.get("aggregate_sha256") != recomputed_trace_sha256:
        raise ContinuationError(f"passed throughput cell {cell_id!r} sample-trace digest mismatch")
    return {
        "mean_step_seconds": timing_values["mean_step_seconds"],
        "average_gpu_utilization_percent": average_utilization,
        "peak_memory_used_fraction": peak_memory_fraction,
        "sample_trace_aggregate_sha256": recomputed_trace_sha256,
        "memory_gate_sha256": memory_sha256,
    }


def _validate_infeasible_throughput_cell(
    cell: Mapping[str, Any],
    *,
    cell_id: str,
) -> None:
    status = cell.get("status")
    if (
        cell.get("schema_version") != 1
        or status not in {"infeasible_oom", "infeasible_memory_headroom"}
        or cell.get("activation_checkpointing") != "none"
        or type(cell.get("training_exit_code")) is not int
        or cell["training_exit_code"] == 0
        or cell.get("evidence_parse_errors") != []
    ):
        raise ContinuationError(f"throughput cell {cell_id!r} has invalid infeasible semantics")
    _parse_timestamp(
        cell.get("summarized_at"),
        f"infeasible throughput cell {cell_id!r} summarized_at",
    )
    for key in (
        "log_sha256",
        "optimizer_step_ndjson_sha256",
        "gpu_samples_sha256",
    ):
        _require_sha256(
            cell.get(key),
            f"infeasible throughput cell {cell_id!r} {key}",
        )
    trace = _require_mapping(
        cell.get("sample_trace"),
        f"infeasible throughput cell {cell_id!r} sample trace",
    )
    per_step = trace.get("per_step")
    if (
        trace.get("algorithm") != THROUGHPUT_TRACE_ALGORITHM
        or type(trace.get("step_count")) is not int
        or trace["step_count"] < 0
        or not isinstance(per_step, list)
        or len(per_step) != trace["step_count"]
    ):
        raise ContinuationError(f"infeasible throughput cell {cell_id!r} has invalid partial trace")
    expected_trace_digest = (
        hashlib.sha256(
            json.dumps(
                per_step,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if per_step
        else None
    )
    if trace.get("aggregate_sha256") != expected_trace_digest:
        raise ContinuationError(f"infeasible throughput cell {cell_id!r} partial trace digest mismatch")
    timing = _require_mapping(
        cell.get("timing"),
        f"infeasible throughput cell {cell_id!r} timing",
    )
    if timing.get("observed_optimizer_steps") != len(per_step):
        raise ContinuationError(f"infeasible throughput cell {cell_id!r} optimizer-step count mismatch")
    memory = _require_mapping(
        cell.get("production_batch_memory_gate"),
        f"infeasible throughput cell {cell_id!r} memory gate",
    )
    if status == "infeasible_oom":
        if (
            cell.get("infeasibility_reason") != "no_ac_cuda_oom"
            or cell.get("canonical_marker")
            not in {
                "torch.cuda.OutOfMemoryError",
                "CUDA out of memory",
                "CUDA error: out of memory",
            }
            or memory.get("status") not in {"missing", "invalid", "passed"}
        ):
            raise ContinuationError(f"infeasible throughput cell {cell_id!r} lacks exact OOM evidence")
        return
    if (
        cell.get("infeasibility_reason") != "no_ac_exceeded_eight_gpu_memory_gate"
        or cell.get("canonical_marker") != "production_batch_memory_gate_threshold_violation"
        or memory.get("status") != "failed"
        or memory.get("world_size") != CHECKPOINT_WORLD_SIZE
        or not isinstance(memory.get("violations"), list)
        or not memory["violations"]
    ):
        raise ContinuationError(f"infeasible throughput cell {cell_id!r} lacks exact memory-gate evidence")
    peak_reserved = _require_finite_float(
        memory.get("max_peak_reserved_fraction"),
        f"infeasible throughput cell {cell_id!r} peak reserved fraction",
    )
    driver_used = _require_finite_float(
        memory.get("max_driver_used_fraction"),
        f"infeasible throughput cell {cell_id!r} driver used fraction",
    )
    if peak_reserved < 0.94 and driver_used < 0.94:
        raise ContinuationError(f"infeasible throughput cell {cell_id!r} did not exceed memory limits")


def _validate_throughput_matrix_protocol(
    matrix: Mapping[str, Any],
    *,
    cells: Sequence[Mapping[str, Any]],
    passed_summaries: Mapping[str, Mapping[str, Any]],
) -> str:
    cell_ids = [str(cell.get("cell_id")) for cell in cells]
    if len(set(cell_ids)) != len(cell_ids):
        raise ContinuationError("throughput matrix cell IDs are not unique")
    actual_cell_ids = set(cell_ids)
    full16_ids = {
        f"workers-{workers}__storage-{storage}__ac-{activation}"
        for workers in THROUGHPUT_WORKERS
        for storage in ("gpfs", "nvme")
        for activation in ("selective", "none")
    }
    gpfs_selective_ids = {f"workers-{workers}__storage-gpfs__ac-selective" for workers in THROUGHPUT_WORKERS}
    if len(cells) == 16:
        if actual_cell_ids != full16_ids:
            raise ContinuationError("full16 throughput matrix does not contain the exact factorial cell set")
        protocol_mode = "full16"
    elif len(cells) == 6:
        if not gpfs_selective_ids.issubset(actual_cell_ids):
            raise ContinuationError("adaptive6 throughput matrix lacks the four GPFS/selective worker cells")
        fastest_worker_cell = min(
            gpfs_selective_ids,
            key=lambda cell_id: (
                float(passed_summaries[cell_id]["mean_step_seconds"]),
                cell_id,
            ),
        )
        fastest_workers = int(
            CELL_ID_PATTERN.fullmatch(fastest_worker_cell).group("workers")  # type: ignore[union-attr]
        )
        nvme_selective = f"workers-{fastest_workers}__storage-nvme__ac-selective"
        if nvme_selective not in actual_cell_ids:
            raise ContinuationError("adaptive6 throughput matrix lacks the selected-worker NVMe cell")
        selected_storage_cell = min(
            (fastest_worker_cell, nvme_selective),
            key=lambda cell_id: (
                float(passed_summaries[cell_id]["mean_step_seconds"]),
                cell_id,
            ),
        )
        selected_storage = (
            CELL_ID_PATTERN.fullmatch(selected_storage_cell).group("storage")  # type: ignore[union-attr]
        )
        no_ac_cell = f"workers-{fastest_workers}__storage-{selected_storage}__ac-none"
        expected_adaptive_ids = {
            *gpfs_selective_ids,
            nvme_selective,
            no_ac_cell,
        }
        if actual_cell_ids != expected_adaptive_ids:
            raise ContinuationError("adaptive6 throughput matrix does not match the adaptive selection path")
        protocol_mode = "adaptive6"
    else:
        raise ContinuationError("throughput matrix must contain exactly 6 adaptive or 16 factorial cells")

    passed = [cell for cell in cells if cell.get("status") == "passed"]
    infeasible = [cell for cell in cells if cell.get("status") in {"infeasible_oom", "infeasible_memory_headroom"}]
    if len(passed) + len(infeasible) != len(cells):
        raise ContinuationError("passed throughput matrix contains a failed or unknown cell outcome")
    if any(cell.get("activation_checkpointing") == "selective" and cell.get("status") != "passed" for cell in cells):
        raise ContinuationError("throughput matrix has a non-passed selective fallback cell")
    expected_status = "passed_with_no_ac_fallback" if infeasible else "passed"
    expected_outcome = "optional_no_ac_infeasible_selective_fallback" if infeasible else "all_cells_passed"
    counter_contract = {
        "status": expected_status,
        "outcome": expected_outcome,
        "expected_cell_count": len(cells),
        "observed_cell_count": len(cells),
        "passed_cell_count": len(passed),
        "infeasible_cell_count": len(infeasible),
        "infeasible_cell_ids": [cell.get("cell_id") for cell in infeasible],
        "failed_cell_count": 0,
        "invalid_infeasible_cell_ids": [],
        "missing_selective_fallback_cell_ids": [],
        "nonpassed_selective_cell_ids": [],
        "missing_cell_ids": [],
        "cell_identity_mismatches": [],
    }
    mismatches = {
        key: {"expected": expected, "actual": matrix.get(key)}
        for key, expected in counter_contract.items()
        if matrix.get(key) != expected
    }
    if mismatches:
        raise ContinuationError(f"throughput matrix outcome/counter mismatch: {mismatches}")
    passed_trace_digests = {str(summary["sample_trace_aggregate_sha256"]) for summary in passed_summaries.values()}
    trace_consistent = bool(passed_trace_digests) and len(passed_trace_digests) == 1
    trace_digest = next(iter(passed_trace_digests)) if trace_consistent else None
    if (
        matrix.get("sample_trace_consistent_across_cells") is not trace_consistent
        or matrix.get("sample_trace_aggregate_sha256") != trace_digest
        or not trace_consistent
    ):
        raise ContinuationError("throughput matrix sample-trace consistency claim is invalid")
    ranked = sorted(
        passed,
        key=lambda cell: (
            float(passed_summaries[str(cell["cell_id"])]["mean_step_seconds"]),
            str(cell["cell_id"]),
        ),
    )
    expected_ranking = [
        {
            "cell_id": cell["cell_id"],
            "mean_step_seconds": passed_summaries[str(cell["cell_id"])]["mean_step_seconds"],
            "average_gpu_utilization_percent": passed_summaries[str(cell["cell_id"])][
                "average_gpu_utilization_percent"
            ],
            "peak_memory_used_fraction": passed_summaries[str(cell["cell_id"])]["peak_memory_used_fraction"],
        }
        for cell in ranked
    ]
    if (
        matrix.get("ranking_by_mean_step_seconds") != expected_ranking
        or matrix.get("best_observed_cell_id") != expected_ranking[0]["cell_id"]
    ):
        raise ContinuationError("throughput matrix best cell/ranking does not match validated evidence")
    return protocol_mode


def validate_throughput_matrix(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate and materialize the one matrix-selected training configuration."""

    absolute = Path(os.path.abspath(path))
    if path != absolute or path.is_symlink() or path.resolve(strict=True) != absolute:
        raise ContinuationError("throughput matrix path must be canonical, absolute, and non-symlink")
    expected_sha256 = _require_sha256(expected_sha256, "expected throughput matrix SHA-256")
    matrix, matrix_sha256 = _load_json_and_sha256(path, "throughput matrix")
    if matrix_sha256 != expected_sha256:
        raise ContinuationError(
            f"throughput matrix SHA-256 mismatch: expected={expected_sha256}, actual={matrix_sha256}"
        )
    if matrix.get("schema_version") != 1:
        raise ContinuationError("throughput matrix must use schema_version=1")
    if matrix.get("status") not in {"passed", "passed_with_no_ac_fallback"}:
        raise ContinuationError(f"throughput matrix did not pass: {matrix.get('status')!r}")
    raw_cells = matrix.get("cells")
    if not isinstance(raw_cells, list):
        raise ContinuationError("throughput matrix cells must be a list")
    validated_cells: list[Mapping[str, Any]] = []
    passed_summaries: dict[str, Mapping[str, Any]] = {}
    for index, cell in enumerate(raw_cells):
        if not isinstance(cell, dict):
            raise ContinuationError(f"throughput matrix cell {index} must be an object")
        cell_id = cell.get("cell_id")
        cell_match = CELL_ID_PATTERN.fullmatch(str(cell_id))
        if cell_match is None:
            raise ContinuationError(f"invalid throughput matrix cell_id: {cell_id!r}")
        expected_cell_identity = {
            "workers_per_rank": int(cell_match.group("workers")),
            "node_total_workers": int(cell_match.group("workers")) * 8,
            "storage": cell_match.group("storage"),
            "activation_checkpointing": cell_match.group("activation_checkpointing"),
        }
        for key, expected in expected_cell_identity.items():
            if cell.get(key) != expected:
                raise ContinuationError(
                    f"throughput cell {cell_id!r} {key} mismatch: expected={expected!r}, actual={cell.get(key)!r}"
                )
        if cell.get("status") == "passed":
            passed_summaries[str(cell_id)] = _validate_passed_throughput_cell(
                cell,
                cell_id=str(cell_id),
            )
        elif cell.get("status") in {
            "infeasible_oom",
            "infeasible_memory_headroom",
        }:
            _validate_infeasible_throughput_cell(
                cell,
                cell_id=str(cell_id),
            )
        else:
            raise ContinuationError(
                f"throughput matrix cell {cell_id!r} did not pass or prove an optional no-AC infeasibility"
            )
        validated_cells.append(cell)
    protocol_mode = _validate_throughput_matrix_protocol(
        matrix,
        cells=validated_cells,
        passed_summaries=passed_summaries,
    )
    best_cell_id = matrix.get("best_observed_cell_id")
    match = CELL_ID_PATTERN.fullmatch(str(best_cell_id))
    if match is None or str(best_cell_id) not in passed_summaries:
        raise ContinuationError(f"invalid best_observed_cell_id: {best_cell_id!r}")
    mean_step_seconds = passed_summaries[str(best_cell_id)]["mean_step_seconds"]
    artifact_binding = _validate_throughput_artifact_binding(
        matrix_path=path,
        matrix=matrix,
        cells=validated_cells,
    )
    return {
        "matrix_summary_path": str(path),
        "matrix_summary_sha256": matrix_sha256,
        "matrix_protocol_mode": protocol_mode,
        "matrix_status": matrix["status"],
        "best_observed_cell_id": str(best_cell_id),
        "workers_per_rank": int(match.group("workers")),
        "node_total_workers": int(match.group("workers")) * 8,
        "storage": match.group("storage"),
        "activation_checkpointing": match.group("activation_checkpointing"),
        "mean_step_seconds": float(mean_step_seconds),
        "benchmark_artifacts": artifact_binding,
    }


def fingerprint_checkpoint(
    path: Path,
    *,
    require_torch_metadata: bool = False,
) -> dict[str, Any]:
    """Hash every R6 DCP byte under an exact component/file-layout contract."""

    path = _canonical_directory(path, "R6 checkpoint")
    if path.name != "iter_000001000":
        raise ContinuationError("resume checkpoint must be named iter_000001000")
    files, component_reports = _checkpoint_inventory(
        path,
        hash_files=True,
        require_torch_metadata=require_torch_metadata,
    )
    portable_files = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in files
    ]
    canonical_inventory = {
        "algorithm": CHECKPOINT_FINGERPRINT_ALGORITHM,
        "checkpoint_iteration": CHECKPOINT_ITERATION,
        "files": portable_files,
    }
    encoded = json.dumps(
        canonical_inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "path": str(path),
        "iteration": CHECKPOINT_ITERATION,
        "components": list(CHECKPOINT_COMPONENTS),
        "fingerprint_algorithm": CHECKPOINT_FINGERPRINT_ALGORITHM,
        "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(files),
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "zero_byte_distcp_file_count": sum(
            len(report["zero_byte_rank_shards"]) for report in component_reports.values()
        ),
        "component_reports": component_reports,
        "files": files,
    }


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.world_size != 8:
        raise ContinuationError("formal continuation requires exactly eight ranks")
    if args.run_name != args.experiment_id:
        raise ContinuationError("formal continuation run name must equal experiment ID")
    if not re.fullmatch(r"CWAM-E35-[A-Za-z0-9._-]+", args.experiment_id):
        raise ContinuationError("formal continuation experiment ID must use the CWAM-E35 family")
    cache = validate_cache_binding(
        args.cache_root,
        expected_success_sha256=args.cache_success_sha256,
        expected_manifest_sha256=args.cache_manifest_sha256,
    )
    throughput = validate_throughput_matrix(
        args.matrix_summary,
        expected_sha256=args.matrix_summary_sha256,
    )
    resume = fingerprint_checkpoint(
        args.resume_checkpoint,
        require_torch_metadata=bool(getattr(args, "require_torch_dcp_metadata", False)),
    )
    expected_fingerprint = _require_sha256(
        args.resume_checkpoint_fingerprint_sha256,
        "expected R6 checkpoint fingerprint",
    )
    if resume["fingerprint_sha256"] != expected_fingerprint:
        raise ContinuationError(
            "R6 checkpoint fingerprint mismatch: "
            f"expected={expected_fingerprint}, actual={resume['fingerprint_sha256']}"
        )
    benchmark_artifacts = _require_mapping(
        throughput.get("benchmark_artifacts"),
        "throughput benchmark artifacts",
    )
    if benchmark_artifacts.get("gpfs_latent_cache_root") != cache["root"]:
        raise ContinuationError("throughput matrix GPFS cache root does not match the continuation cache")
    if benchmark_artifacts.get("latent_cache_manifest_sha256") != cache["manifest_sha256"]:
        raise ContinuationError("throughput matrix cache manifest does not match the continuation cache")
    if benchmark_artifacts.get("resume_checkpoint") != resume["path"]:
        raise ContinuationError("throughput matrix R6 checkpoint does not match the continuation checkpoint")
    expected_benchmark_source_root = _canonical_directory(
        args.expected_benchmark_source_root,
        "expected benchmark source root",
    )
    expected_benchmark_source_commit = str(args.expected_benchmark_source_commit)
    if (
        re.fullmatch(
            r"[0-9a-f]{40}",
            expected_benchmark_source_commit,
        )
        is None
    ):
        raise ContinuationError("expected benchmark source commit must be a lowercase 40-hex SHA")
    expected_benchmark_snapshot_sha256 = _require_sha256(
        args.expected_benchmark_snapshot_manifest_sha256,
        "expected benchmark snapshot manifest SHA-256",
    )
    expected_wan_vae_sha256 = _require_sha256(
        args.expected_wan_vae_sha256,
        "expected Wan VAE SHA-256",
    )
    if expected_wan_vae_sha256 != OFFICIAL_WAN22_VAE_SHA256:
        raise ContinuationError("formal continuation requires the audited official Wan2.2 VAE digest")
    wan_vae_path = Path(args.wan_vae_path)
    if (
        wan_vae_path != Path(os.path.abspath(wan_vae_path))
        or wan_vae_path.is_symlink()
        or not wan_vae_path.is_file()
        or wan_vae_path.resolve(strict=True) != wan_vae_path
    ):
        raise ContinuationError(f"Wan VAE path must be canonical and non-symlink: {wan_vae_path}")
    benchmark_provenance_expected = {
        "source_root": str(expected_benchmark_source_root),
        "source_commit": expected_benchmark_source_commit,
        "snapshot_manifest_sha256": expected_benchmark_snapshot_sha256,
        "wan_vae_path": str(wan_vae_path),
        "wan_vae_sha256": expected_wan_vae_sha256,
    }
    benchmark_provenance_mismatches = {
        key: {"expected": expected, "actual": benchmark_artifacts.get(key)}
        for key, expected in benchmark_provenance_expected.items()
        if benchmark_artifacts.get(key) != expected
    }
    if benchmark_provenance_mismatches:
        raise ContinuationError(
            "throughput matrix provenance does not match the predeclared "
            f"benchmark inputs: {benchmark_provenance_mismatches}"
        )
    submission_args = {
        "reservation_path": getattr(args, "submission_reservation", None),
        "submission_nonce": getattr(args, "submission_nonce", None),
        "expected_command_sha256": getattr(
            args,
            "expected_scheduler_command_sha256",
            None,
        ),
        "expected_cpu": getattr(args, "expected_rjob_cpu", None),
    }
    populated_submission_args = {key: value for key, value in submission_args.items() if value is not None}
    if populated_submission_args and len(populated_submission_args) != len(submission_args):
        missing = sorted(set(submission_args) - set(populated_submission_args))
        raise ContinuationError("submission binding is incomplete; missing " + ", ".join(missing))
    submission = (
        validate_submission_binding(
            reservation_path=submission_args["reservation_path"],
            submission_nonce=submission_args["submission_nonce"],
            expected_command_sha256=submission_args["expected_command_sha256"],
            expected_cpu=submission_args["expected_cpu"],
            experiment_id=args.experiment_id,
            cache=cache,
            throughput=throughput,
            resume=resume,
        )
        if populated_submission_args
        else None
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "kind": "cwam_cosmos3_formal_continuation_inputs",
        "validated_at": _utc_now(),
        "experiment_id": args.experiment_id,
        "run_name": args.run_name,
        "world_size": args.world_size,
        "cache": cache,
        "throughput": throughput,
        "resume": resume,
        "protocol": {
            "initial_iteration": CHECKPOINT_ITERATION,
            "max_iter": FINAL_ITERATION,
            "resume_existing_run": True,
            "allow_data_stream_replay": True,
            "data_stream_resume": "restarted_from_beginning_not_exact_loader_cursor",
            "checkpoint_load_training_state": True,
            "checkpoint_keys_to_skip_loading": [],
            "checkpoint_keys_not_to_resume": ["dataloader"],
            "checkpoint_strict_resume": True,
            "fixed_index_trace_path": None,
        },
    }
    if submission is not None:
        payload["submission"] = submission
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_run(
    *,
    output_root: Path,
    run_name: str,
    contract_path: Path,
    expected_contract_sha256: str,
    require_torch_metadata: bool = False,
) -> dict[str, Any]:
    """Atomically create one new run with an exact R6 checkpoint symlink."""

    output_root = _canonical_directory(output_root, "formal output root")
    expected_contract_sha256 = _require_sha256(
        expected_contract_sha256,
        "expected continuation-contract SHA-256",
    )
    contract, contract_sha256 = _load_json_and_sha256(
        contract_path,
        "formal continuation contract",
    )
    if contract_sha256 != expected_contract_sha256:
        raise ContinuationError(
            f"continuation-contract SHA-256 mismatch: expected={expected_contract_sha256}, actual={contract_sha256}"
        )
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "passed"
        or contract.get("kind") != "cwam_cosmos3_formal_continuation_inputs"
        or contract.get("run_name") != run_name
    ):
        raise ContinuationError("continuation contract is not a passed binding for this run")
    submission = contract.get("submission")
    if submission is not None:
        submission = _require_mapping(
            submission,
            "continuation contract submission binding",
        )
        reservation_path = Path(str(submission.get("reservation_path")))
        _require_readonly_regular_file(
            reservation_path,
            "continuation submission reservation",
        )
        if _sha256_file(reservation_path) != submission.get("reservation_sha256"):
            raise ContinuationError("continuation submission reservation changed after input validation")
        artifacts = _require_mapping(
            submission.get("artifacts"),
            "continuation submission artifact bindings",
        )
        for label, descriptor_value in artifacts.items():
            descriptor = _require_mapping(
                descriptor_value,
                f"continuation submission artifact {label}",
            )
            artifact_path = Path(str(descriptor.get("path")))
            _require_readonly_regular_file(
                artifact_path,
                f"continuation submission artifact {label}",
            )
            if artifact_path.stat().st_size != descriptor.get("size_bytes") or _sha256_file(
                artifact_path
            ) != descriptor.get("sha256"):
                raise ContinuationError(f"continuation submission artifact changed after input validation: {label}")
    resume = _require_mapping(contract.get("resume"), "continuation contract resume")
    resume_path = Path(str(resume.get("path")))
    resume_path = _canonical_directory(resume_path, "R6 checkpoint")
    if resume_path.name != "iter_000001000":
        raise ContinuationError("continuation contract no longer points to iter_000001000")
    declared_files = resume.get("files")
    if not isinstance(declared_files, list) or not declared_files:
        raise ContinuationError("continuation contract has no R6 checkpoint file inventory")
    observed_files, _ = _checkpoint_inventory(
        resume_path,
        hash_files=False,
        require_torch_metadata=require_torch_metadata,
    )
    stat_keys = ("device", "inode", "mode", "size_bytes", "mtime_ns", "ctime_ns")
    observed_inventory = [(str(item["path"]), *(int(item[key]) for key in stat_keys)) for item in observed_files]
    declared_inventory = [
        (
            str(item.get("path")),
            *(int(item.get(key, -1)) for key in stat_keys),
        )
        for item in declared_files
        if isinstance(item, dict)
    ]
    if observed_inventory != declared_inventory:
        raise ContinuationError("R6 checkpoint file/stat inventory changed after contract validation")
    canonical_root = output_root / "cosmos3_action_robocasa365" / "action_sft"
    canonical_root.mkdir(parents=True, exist_ok=True)
    canonical_root = _canonical_directory(canonical_root, "formal run parent")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        raise ContinuationError(f"unsupported formal run name: {run_name!r}")
    run_dir = canonical_root / run_name
    if run_dir.exists() or run_dir.is_symlink():
        raise ContinuationError(f"formal continuation run already exists: {run_dir}")
    stage = canonical_root / f".{run_name}.incomplete-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o755)
    checkpoints = stage / "checkpoints"
    checkpoints.mkdir(mode=0o755)
    seed_name = f"iter_{CHECKPOINT_ITERATION:09d}"
    (checkpoints / seed_name).symlink_to(resume_path, target_is_directory=True)
    (checkpoints / "latest_checkpoint.txt").write_text(f"{seed_name}\n")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "prepared_at": _utc_now(),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "seed_checkpoint_link": str(run_dir / "checkpoints" / seed_name),
        "seed_checkpoint_target": str(resume_path),
        "seed_checkpoint_fingerprint_algorithm": resume.get("fingerprint_algorithm"),
        "seed_checkpoint_fingerprint_sha256": resume.get("fingerprint_sha256"),
        "continuation_contract": str(contract_path),
        "continuation_contract_sha256": contract_sha256,
    }
    if submission is not None:
        receipt.update(
            {
                "submission_reservation": submission.get("reservation_path"),
                "submission_reservation_sha256": submission.get("reservation_sha256"),
                "submission_nonce": submission.get("submission_nonce"),
                "scheduler_rjob_name": submission.get("scheduler_rjob_name"),
                "scheduler_rjob_uid": submission.get("scheduler_rjob_uid"),
                "scheduler_replica_name": submission.get("scheduler_replica_name"),
                "scheduler_pod_name": submission.get("scheduler_pod_name"),
            }
        )
    _write_json_atomic(stage / "continuation-seed.json", receipt, exclusive=True)
    os.replace(stage, run_dir)
    return receipt


def _print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fingerprint = commands.add_parser("fingerprint-checkpoint")
    fingerprint.add_argument("--checkpoint", type=Path, required=True)
    fingerprint.add_argument("--require-torch-dcp-metadata", action="store_true")

    validate = commands.add_parser("validate-inputs")
    validate.add_argument("--experiment-id", required=True)
    validate.add_argument("--run-name", required=True)
    validate.add_argument("--world-size", type=int, default=8)
    validate.add_argument("--cache-root", type=Path, required=True)
    validate.add_argument("--cache-success-sha256", required=True)
    validate.add_argument("--cache-manifest-sha256", required=True)
    validate.add_argument("--matrix-summary", type=Path, required=True)
    validate.add_argument("--matrix-summary-sha256", required=True)
    validate.add_argument("--resume-checkpoint", type=Path, required=True)
    validate.add_argument("--resume-checkpoint-fingerprint-sha256", required=True)
    validate.add_argument(
        "--expected-benchmark-source-root",
        type=Path,
        required=True,
    )
    validate.add_argument("--expected-benchmark-source-commit", required=True)
    validate.add_argument(
        "--expected-benchmark-snapshot-manifest-sha256",
        required=True,
    )
    validate.add_argument("--wan-vae-path", type=Path, required=True)
    validate.add_argument("--expected-wan-vae-sha256", required=True)
    validate.add_argument("--require-torch-dcp-metadata", action="store_true")
    validate.add_argument("--submission-reservation", type=Path)
    validate.add_argument("--submission-nonce")
    validate.add_argument("--expected-scheduler-command-sha256")
    validate.add_argument("--expected-rjob-cpu", type=int)
    validate.add_argument("--output", type=Path)

    prepare = commands.add_parser("prepare-run")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--contract", type=Path, required=True)
    prepare.add_argument("--contract-sha256", required=True)
    prepare.add_argument("--require-torch-dcp-metadata", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "fingerprint-checkpoint":
        payload = fingerprint_checkpoint(
            args.checkpoint,
            require_torch_metadata=args.require_torch_dcp_metadata,
        )
    elif args.command == "validate-inputs":
        payload = build_contract(args)
        if args.output is not None:
            _write_json_atomic(args.output, payload, exclusive=True)
    else:
        payload = prepare_run(
            output_root=args.output_root,
            run_name=args.run_name,
            contract_path=args.contract,
            expected_contract_sha256=args.contract_sha256,
            require_torch_metadata=args.require_torch_dcp_metadata,
        )
    _print(payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContinuationError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
