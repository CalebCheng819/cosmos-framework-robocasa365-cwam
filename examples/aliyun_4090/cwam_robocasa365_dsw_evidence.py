#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Capture Alibaba DSW runtime evidence and aggregate RoboCasa365 results.

This module is intentionally separate from the H-cluster evidence contract.
The DSW execution has one physical policy server backed by four local FSDP
ranks, while the evaluation protocol retains eight logical task shards.

Workflow iteration: ITER-2026-07-31-CWAM-DSW974-FSDP4-EVAL-ADAPTER.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import socket
import stat
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import cwam_robocasa365_eval_evidence as existing_aggregator
import cwam_serving_checkpoint_manifest

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_KIND = "cwam_robocasa365_alibaba_dsw_runtime_receipt"
_EVIDENCE_KIND = "cwam_robocasa365_alibaba_dsw_formal_eval_evidence"
_SEALED_KIND = "cwam_robocasa365_alibaba_dsw_sealed_evidence_verification"
_SHUTDOWN_KIND = "cwam_robocasa365_alibaba_dsw_controlled_shutdown_receipt"
_TERMINAL_KIND = "cwam_robocasa365_alibaba_dsw_server_terminal_receipt"
_GATE_KIND = "cwam_robocasa365_alibaba_dsw_one_episode_gate"
_QWEN_MANIFEST_KIND = "cwam_qwen_tokenizer_manifest"
_QWEN_VERIFICATION_KIND = "cwam_qwen_tokenizer_manifest_verification"
_QWEN_REQUIRED_FILES = frozenset(
    (
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
)
_PLATFORM = {
    "provider": "Alibaba Cloud",
    "service": "PAI DSW",
    "region": "cn-beijing",
}
_EXPECTED_PARALLELISM = {
    "dp_shard_size": 4,
    "dp_replicate_size": 1,
    "cp_size": 1,
    "cfgp_size": 1,
}
_EXPECTED_TASK_COUNT = 17
_EXPECTED_EPISODES_PER_TASK = 20
_EXPECTED_EPISODE_COUNT = _EXPECTED_TASK_COUNT * _EXPECTED_EPISODES_PER_TASK
_EXPECTED_LOGICAL_NUM_SHARDS = 8
_EXPECTED_PHYSICAL_GPU_COUNT = 4
_EXPECTED_PHYSICAL_SERVER_COUNT = 1
_EXPECTED_CPU_COUNT = 128
_EXPECTED_MEMORY_GIB = 400
_REQUIRED_RUNTIME_PROVENANCE = frozenset(
    {
        "asset_env_closure",
        "checkpoint_atomic_publication",
        "checkpoint_relocation",
        "cosmos_environment_pre",
        "ema_export_receipt",
        "ema_key_manifest",
        "ema_source_dcp_full",
        "ema_transfer_manifest",
        "ema_validation",
        "h_checkpoint_manifest",
        "qwen_tokenizer_pre",
        "robocasa_assets_pre",
        "robocasa_environment_pre",
        "robocasa_torchcodec_relay",
        "source_snapshot_pre",
        "static_inputs_pre",
    }
)
_REQUIRED_POST_PROVENANCE = frozenset(
    {
        "cosmos_environment_post",
        "qwen_tokenizer_post",
        "robocasa_assets_post",
        "robocasa_environment_post",
        "source_snapshot_post",
        "static_inputs_post",
        "one_episode_gate",
    }
)
_NCCL_WATCHDOG_ENV = {
    "NCCL_DEBUG": "INFO",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_DESYNC_DEBUG": "1",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_ENABLE_MONITORING": "1",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "900",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "1048576",
}


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: str, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 40-character Git commit")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(str(key) for key in observed - expected)}"
        )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON contains duplicate object key: {key!r}")
        value[key] = item
    return value


def _stable_sha256(path: Path, *, expected_mode: int | None = None) -> dict[str, Any]:
    supplied = Path(os.path.abspath(path.expanduser()))
    canonical = supplied.resolve(strict=True)
    before = supplied.lstat()
    if supplied != canonical or stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"artifact must be a canonical regular non-symlink file: {supplied}")
    if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
        raise ValueError(
            f"artifact mode mismatch for {supplied}: "
            f"expected={oct(expected_mode)} actual={oct(stat.S_IMODE(before.st_mode))}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(supplied, flags)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError(f"artifact was replaced while opening: {supplied}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size_bytes += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = supplied.lstat()

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns)

    if (
        identity(before) != identity(opened)
        or identity(opened) != identity(after)
        or identity(after) != identity(final)
    ):
        raise RuntimeError(f"artifact changed while hashing: {supplied}")
    if size_bytes != before.st_size:
        raise RuntimeError(f"artifact size changed while hashing: {supplied}")
    return {"path": str(supplied), "size_bytes": size_bytes, "sha256": digest.hexdigest()}


def _atomic_write_exclusive(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    supplied = Path(os.path.abspath(path.expanduser()))
    parent = supplied.parent.resolve(strict=True)
    if supplied.parent != parent or parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"output parent must be an existing canonical directory: {supplied.parent}")
    if supplied.exists() or supplied.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {supplied}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{supplied.name}.", suffix=".tmp", dir=parent)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, supplied, follow_symlinks=False)
        published = True
        # Include removal of the temporary hard link in the directory
        # durability barrier before performing the immutable readback.
        os.unlink(temporary)
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
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return _stable_sha256(supplied, expected_mode=0o444)


def _read_json(path: Path, *, expected_mode: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = _stable_sha256(path, expected_mode=expected_mode)
    encoded = Path(descriptor["path"]).read_bytes()
    if len(encoded) != descriptor["size_bytes"] or hashlib.sha256(encoded).hexdigest() != descriptor["sha256"]:
        raise RuntimeError(f"JSON artifact changed between hashing and reading: {path}")
    if _stable_sha256(path, expected_mode=expected_mode) != descriptor:
        raise RuntimeError(f"JSON artifact changed after reading: {path}")
    value = json.loads(encoded, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return value, descriptor


def _canonical_non_symlink_directory(path: Path) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise ValueError(f"directory must be an absolute canonical path: {supplied}")
    try:
        canonical = supplied.resolve(strict=True)
        metadata = supplied.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"directory does not exist: {supplied}") from None
    if supplied != canonical or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"directory must be canonical, non-symlink, and a directory: {supplied}")
    return canonical


def _require_canonical_absolute_path_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{label} must be a non-empty absolute path")
    if os.path.normpath(value) != value or str(Path(value)) != value:
        raise ValueError(f"{label} must be a canonical absolute path")
    return value


def _descriptor_for_expected_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    expected_mode: int | None = None,
) -> dict[str, Any]:
    descriptor = _stable_sha256(path, expected_mode=expected_mode)
    if descriptor["sha256"] != _require_sha256(expected_sha256, f"{label} expected SHA-256"):
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected_sha256}, "
            f"actual={descriptor['sha256']}"
        )
    return descriptor


def _nvidia_smi_gpu_identity() -> list[dict[str, Any]]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    inventory: list[dict[str, Any]] = []
    for raw_line in process.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"unexpected nvidia-smi GPU identity row: {raw_line!r}")
        index_text, uuid, name, pci_bus_id = fields
        try:
            index = int(index_text)
        except ValueError as error:
            raise ValueError(
                f"invalid nvidia-smi physical GPU index: {index_text!r}"
            ) from error
        inventory.append(
            {
                "physical_index": index,
                "uuid": uuid,
                "name": name,
                "pci_bus_id": pci_bus_id,
            }
        )
    if [row["physical_index"] for row in inventory] != list(
        range(_EXPECTED_PHYSICAL_GPU_COUNT)
    ):
        raise ValueError(f"nvidia-smi GPU identity inventory is incomplete: {inventory}")
    if any(
        not row["uuid"].startswith("GPU-") or "4090" not in row["name"]
        for row in inventory
    ):
        raise ValueError(f"nvidia-smi GPU identity inventory is unexpected: {inventory}")
    return inventory


def _verify_mode_bound_descriptor(
    value: Any,
    *,
    expected_mode: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an artifact descriptor")
    _require_exact_keys(
        value,
        frozenset({"mode", "path", "sha256", "size_bytes"}),
        label,
    )
    expected = {
        "path": _require_canonical_absolute_path_string(
            value["path"], f"{label}.path"
        ),
        "size_bytes": value["size_bytes"],
        "sha256": _require_sha256(value["sha256"], f"{label}.sha256"),
        "mode": f"{expected_mode:04o}",
    }
    if (
        isinstance(expected["size_bytes"], bool)
        or not isinstance(expected["size_bytes"], int)
        or expected["size_bytes"] < 0
        or value != expected
    ):
        raise ValueError(f"{label} descriptor fields are invalid")
    actual = _stable_sha256(Path(expected["path"]), expected_mode=expected_mode)
    if actual != {key: expected[key] for key in ("path", "size_bytes", "sha256")}:
        raise ValueError(f"{label} changed after the EGL probe")
    return dict(value)


def _read_mode_bound_evidence(
    value: Any,
    *,
    expected_mode: int,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    """Reopen one receipt-bound artifact and return stable, non-empty bytes."""

    verified = _verify_mode_bound_descriptor(
        value,
        expected_mode=expected_mode,
        label=label,
    )
    path = Path(verified["path"])
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"{label} must be non-empty")
    if (
        len(payload) != verified["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != verified["sha256"]
    ):
        raise ValueError(f"{label} changed while reading")
    if _stable_sha256(path, expected_mode=expected_mode) != {
        key: verified[key] for key in ("path", "size_bytes", "sha256")
    }:
        raise ValueError(f"{label} changed after reading")
    return verified, payload


def _decode_evidence_text(payload: bytes, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 text") from error


def _parse_egl_probe_log(
    payload: bytes,
    *,
    row: Mapping[str, Any],
    label: str,
) -> int:
    """Parse the exact two-line NDJSON emitted by the pinned MuJoCo probe."""

    text = _decode_evidence_text(payload, label)
    lines = text.splitlines()
    if len(lines) != 2 or any(not line.strip() for line in lines):
        raise ValueError(f"{label} must contain exactly two non-empty NDJSON lines")
    try:
        ready = json.loads(lines[0], object_pairs_hook=_reject_duplicate_json_keys)
        closed = json.loads(lines[1], object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} contains invalid NDJSON: {error}") from error
    if not isinstance(ready, dict) or not isinstance(closed, dict):
        raise ValueError(f"{label} NDJSON lines must be JSON objects")
    _require_exact_keys(
        ready,
        frozenset(
            {
                "cuda_visible_devices",
                "event",
                "frame_sha256",
                "frame_shape",
                "mujoco",
                "mujoco_egl_device_id",
                "mujoco_gl",
                "pid",
                "python",
            }
        ),
        f"{label} egl_ready",
    )
    _require_exact_keys(
        closed,
        frozenset({"event", "frames"}),
        f"{label} egl_closed",
    )
    probe_pid = ready.get("pid")
    if (
        ready.get("event") != "egl_ready"
        or type(probe_pid) is not int
        or probe_pid <= 0
        or ready.get("python") != "3.11.14"
        or ready.get("mujoco") != "3.3.1"
        or ready.get("mujoco_gl") != "egl"
        or ready.get("cuda_visible_devices") != row.get("cuda_visible_devices")
        or ready.get("mujoco_egl_device_id") != row.get("mujoco_egl_device_id")
        or ready.get("frame_shape") != [64, 64, 3]
    ):
        raise ValueError(f"{label} egl_ready fields do not match the receipt row")
    _require_sha256(ready.get("frame_sha256"), f"{label} frame_sha256")
    if (
        closed.get("event") != "egl_closed"
        or type(closed.get("frames")) is not int
        or closed["frames"] < 2
    ):
        raise ValueError(f"{label} egl_closed fields are invalid")
    return probe_pid


def _parse_nvidia_smi_pmon(
    payload: bytes,
    *,
    probe_pid: int,
    live_gpu_identity: Sequence[Mapping[str, Any]],
    label: str,
) -> int:
    """Resolve the probe PID to one physical GPU from raw ``nvidia-smi pmon``."""

    text = _decode_evidence_text(payload, label)
    lines = text.splitlines()
    if not any(line.lstrip().startswith("# gpu") for line in lines) or not any(
        line.lstrip().startswith("# Idx") for line in lines
    ):
        raise ValueError(f"{label} is missing the nvidia-smi pmon headers")
    valid_indices = {
        int(identity["physical_index"]) for identity in live_gpu_identity
    }
    sample_counts = {index: 0 for index in valid_indices}
    target_indices: list[int] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 3:
            raise ValueError(f"{label} contains a malformed pmon row: {line!r}")
        try:
            physical_index = int(fields[0])
        except ValueError as error:
            raise ValueError(f"{label} contains a non-numeric GPU index") from error
        if physical_index not in valid_indices:
            raise ValueError(f"{label} contains an unknown physical GPU index")
        sample_counts[physical_index] += 1
        if fields[1] == "-":
            continue
        try:
            pid = int(fields[1])
        except ValueError as error:
            raise ValueError(f"{label} contains a non-numeric process PID") from error
        if pid <= 0:
            raise ValueError(f"{label} contains a non-positive process PID")
        if pid == probe_pid:
            if fields[2] != "G":
                raise ValueError(f"{label} probe PID is not an EGL graphics process")
            target_indices.append(physical_index)
    if any(count < 2 for count in sample_counts.values()):
        raise ValueError(f"{label} must contain two pmon samples for all four GPUs")
    if len(target_indices) < 2 or len(set(target_indices)) != 1:
        raise ValueError(
            f"{label} must observe the probe PID in two samples on one physical GPU"
        )
    return target_indices[0]


_NVIDIA_SMI_GPU_ROW_RE = re.compile(
    r"^\|\s*(?P<gpu>\d+)\s+"
    r"(?P<name>NVIDIA GeForce RTX 4090)\s+\S+\s+\|\s+"
    r"(?P<bus>[0-9A-Fa-f:.]+)\s+\S+\s+\|"
)
_NVIDIA_SMI_PROCESS_ROW_RE = re.compile(
    r"^\|\s*(?P<gpu>\d+)\s+"
    r"(?:N/A|\d+)\s+(?:N/A|\d+)\s+"
    r"(?P<pid>\d+)\s+(?P<type>\S+)\s+.+\|\s*$"
)


def _parse_nvidia_smi_snapshot(
    payload: bytes,
    *,
    probe_pid: int,
    live_gpu_identity: Sequence[Mapping[str, Any]],
    label: str,
) -> int:
    """Resolve the probe PID from a raw full ``nvidia-smi`` snapshot."""

    text = _decode_evidence_text(payload, label)
    if "NVIDIA-SMI" not in text or "| Processes:" not in text:
        raise ValueError(f"{label} is not a full nvidia-smi snapshot")
    expected_by_index = {
        int(identity["physical_index"]): dict(identity)
        for identity in live_gpu_identity
    }
    observed_inventory: dict[int, dict[str, str]] = {}
    target_indices: list[int] = []
    for line in text.splitlines():
        gpu_match = _NVIDIA_SMI_GPU_ROW_RE.match(line)
        if gpu_match is not None:
            physical_index = int(gpu_match.group("gpu"))
            if physical_index in observed_inventory:
                raise ValueError(f"{label} repeats a GPU inventory row")
            observed_inventory[physical_index] = {
                "name": gpu_match.group("name"),
                "pci_bus_id": gpu_match.group("bus"),
            }
            continue
        process_match = _NVIDIA_SMI_PROCESS_ROW_RE.match(line)
        if process_match is None:
            continue
        physical_index = int(process_match.group("gpu"))
        pid = int(process_match.group("pid"))
        if physical_index not in expected_by_index or pid <= 0:
            raise ValueError(f"{label} contains an invalid process row")
        if pid == probe_pid:
            if "G" not in process_match.group("type"):
                raise ValueError(
                    f"{label} probe PID is not an EGL graphics process"
                )
            target_indices.append(physical_index)
    if set(observed_inventory) != set(expected_by_index):
        raise ValueError(f"{label} does not contain the exact four-GPU inventory")
    for physical_index, expected in expected_by_index.items():
        observed = observed_inventory[physical_index]
        if (
            observed["name"] != expected["name"]
            or observed["pci_bus_id"] != expected["pci_bus_id"]
        ):
            raise ValueError(
                f"{label} GPU {physical_index} identity differs from the live inventory"
            )
    if not target_indices or len(set(target_indices)) != 1:
        raise ValueError(
            f"{label} must observe the probe PID on exactly one physical GPU"
        )
    return target_indices[0]


def _expected_egl_scheme_environment(
    scheme_name: str,
    physical_index: int,
) -> tuple[str, str]:
    if scheme_name == "visible_all_physical_index":
        return "0,1,2,3", str(physical_index)
    if scheme_name == "isolated_relative_zero":
        return str(physical_index), "0"
    if scheme_name == "isolated_physical_identity":
        return str(physical_index), str(physical_index)
    raise ValueError(f"unsupported EGL mapping scheme: {scheme_name}")


def _validate_egl_mapping_receipt(
    value: Mapping[str, Any],
    *,
    robocasa_environment: Path,
    robocasa_ready_marker_sha256: str,
    egl_vendor_descriptor: Mapping[str, Any],
    live_gpu_identity: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require_exact_keys(
        value,
        frozenset(
            {
                "created_at_utc",
                "environment",
                "host",
                "kind",
                "probe",
                "recommendation",
                "schema_version",
                "schemes",
                "status",
            }
        ),
        "EGL mapping receipt",
    )
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "cwam_dsw974_robocasa_egl_device_mapping"
        or value.get("status") != "passed"
    ):
        raise ValueError("EGL mapping receipt schema/kind/status mismatch")
    created_at = value.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at.endswith("Z"):
        raise ValueError("EGL mapping receipt created_at_utc must be a UTC timestamp")
    try:
        parsed_created_at = dt.datetime.fromisoformat(
            created_at.removesuffix("Z") + "+00:00"
        )
    except ValueError as error:
        raise ValueError("EGL mapping receipt created_at_utc is invalid") from error
    if parsed_created_at.tzinfo is None:
        raise ValueError("EGL mapping receipt created_at_utc must be timezone-aware")

    host = value.get("host")
    if not isinstance(host, dict):
        raise TypeError("EGL mapping receipt host must be an object")
    _require_exact_keys(
        host,
        frozenset({"gpu_count", "gpu_inventory", "hostname"}),
        "EGL mapping receipt host",
    )
    expected_gpu_identity = [dict(row) for row in live_gpu_identity]
    if (
        host.get("hostname") != socket.gethostname()
        or host.get("gpu_count") != _EXPECTED_PHYSICAL_GPU_COUNT
        or host.get("gpu_inventory") != expected_gpu_identity
    ):
        raise ValueError("EGL mapping receipt host/GPU identity differs from the live DSW")

    environment = value.get("environment")
    if not isinstance(environment, dict):
        raise TypeError("EGL mapping receipt environment must be an object")
    _require_exact_keys(
        environment,
        frozenset(
            {
                "egl_vendor_json",
                "mujoco",
                "path",
                "python",
                "ready_marker_sha256",
            }
        ),
        "EGL mapping receipt environment",
    )
    expected_vendor = {
        **dict(egl_vendor_descriptor),
        "mode": "0644",
    }
    if (
        environment.get("path") != str(robocasa_environment)
        or environment.get("ready_marker_sha256")
        != _require_sha256(
            robocasa_ready_marker_sha256,
            "RoboCasa ready marker SHA-256 for EGL mapping",
        )
        or environment.get("python") != "3.11.14"
        or environment.get("mujoco") != "3.3.1"
        or environment.get("egl_vendor_json") != expected_vendor
    ):
        raise ValueError("EGL mapping receipt environment binding mismatch")

    probe = value.get("probe")
    if not isinstance(probe, dict):
        raise TypeError("EGL mapping receipt probe must be an object")
    _require_exact_keys(
        probe,
        frozenset({"evidence_root", "hold_seconds", "script"}),
        "EGL mapping receipt probe",
    )
    _verify_mode_bound_descriptor(
        probe.get("script"),
        expected_mode=0o555,
        label="EGL mapping probe script",
    )
    hold_seconds = probe.get("hold_seconds")
    if (
        isinstance(hold_seconds, bool)
        or not isinstance(hold_seconds, (int, float))
        or not math.isfinite(float(hold_seconds))
        or float(hold_seconds) < 5.0
    ):
        raise ValueError("EGL mapping probe hold_seconds must be finite and at least 5")
    evidence_root = _canonical_non_symlink_directory(
        Path(
            _require_canonical_absolute_path_string(
                probe.get("evidence_root"),
                "EGL mapping probe evidence_root",
            )
        )
    )

    schemes = value.get("schemes")
    if not isinstance(schemes, dict):
        raise TypeError("EGL mapping receipt schemes must be an object")
    _require_exact_keys(
        schemes,
        frozenset(
            {
                "isolated_physical_identity",
                "isolated_relative_zero",
                "visible_all_physical_index",
            }
        ),
        "EGL mapping receipt schemes",
    )
    evidence_paths: set[Path] = set()
    for scheme_name, scheme in schemes.items():
        if not isinstance(scheme, dict):
            raise TypeError(f"EGL mapping scheme {scheme_name} must be an object")
        _require_exact_keys(
            scheme,
            frozenset({"rows", "status"}),
            f"EGL mapping scheme {scheme_name}",
        )
        if scheme.get("status") not in {"passed", "failed"}:
            raise ValueError(f"EGL mapping scheme {scheme_name} status is invalid")
        rows = scheme.get("rows")
        if not isinstance(rows, list) or len(rows) != _EXPECTED_PHYSICAL_GPU_COUNT:
            raise ValueError(f"EGL mapping scheme {scheme_name} must contain four rows")
        computed_mapping_results: list[bool] = []
        for physical_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(
                    f"EGL mapping scheme {scheme_name} row must be an object"
                )
            _require_exact_keys(
                row,
                frozenset(
                    {
                        "cuda_visible_devices",
                        "evidence",
                        "expected_gpu_uuid",
                        "mapping_matches_expected",
                        "mujoco_egl_device_id",
                        "observed_gpu_uuid",
                        "observed_physical_index",
                        "physical_index",
                        "pmon_matched_pid",
                        "probe_pid",
                        "renderer_status",
                    }
                ),
                f"EGL mapping scheme {scheme_name} row {physical_index}",
            )
            identity = expected_gpu_identity[physical_index]
            expected_cuda_visible_devices, expected_egl_device_id = (
                _expected_egl_scheme_environment(scheme_name, physical_index)
            )
            if (
                row.get("physical_index") != physical_index
                or row.get("expected_gpu_uuid") != identity["uuid"]
                or type(row.get("probe_pid")) is not int
                or row["probe_pid"] <= 0
                or row.get("renderer_status") != "passed"
                or row.get("cuda_visible_devices")
                != expected_cuda_visible_devices
                or row.get("mujoco_egl_device_id") != expected_egl_device_id
            ):
                raise ValueError(
                    f"EGL mapping scheme {scheme_name} row identity/status/environment mismatch"
                )
            evidence = row.get("evidence")
            if not isinstance(evidence, dict):
                raise TypeError(
                    f"EGL mapping scheme {scheme_name} row evidence must be an object"
                )
            _require_exact_keys(
                evidence,
                frozenset(
                    {"nvidia_smi_pmon", "nvidia_smi_snapshot", "probe_log"}
                ),
                f"EGL mapping scheme {scheme_name} row evidence",
            )
            evidence_payloads: dict[str, bytes] = {}
            for evidence_name in (
                "probe_log",
                "nvidia_smi_pmon",
                "nvidia_smi_snapshot",
            ):
                verified, payload = _read_mode_bound_evidence(
                    evidence[evidence_name],
                    expected_mode=0o444,
                    label=(
                        f"EGL mapping scheme {scheme_name} row "
                        f"{physical_index} {evidence_name}"
                    ),
                )
                try:
                    Path(verified["path"]).relative_to(evidence_root)
                except ValueError as error:
                    raise ValueError(
                        "EGL mapping evidence artifact escaped evidence_root"
                    ) from error
                verified_path = Path(verified["path"])
                if verified_path in evidence_paths:
                    raise ValueError(
                        "every EGL mapping evidence artifact path must be unique"
                    )
                evidence_paths.add(verified_path)
                evidence_payloads[evidence_name] = payload

            evidence_label = (
                f"EGL mapping scheme {scheme_name} row {physical_index}"
            )
            probe_pid = _parse_egl_probe_log(
                evidence_payloads["probe_log"],
                row=row,
                label=f"{evidence_label} probe_log",
            )
            pmon_physical_index = _parse_nvidia_smi_pmon(
                evidence_payloads["nvidia_smi_pmon"],
                probe_pid=probe_pid,
                live_gpu_identity=expected_gpu_identity,
                label=f"{evidence_label} nvidia_smi_pmon",
            )
            snapshot_physical_index = _parse_nvidia_smi_snapshot(
                evidence_payloads["nvidia_smi_snapshot"],
                probe_pid=probe_pid,
                live_gpu_identity=expected_gpu_identity,
                label=f"{evidence_label} nvidia_smi_snapshot",
            )
            if pmon_physical_index != snapshot_physical_index:
                raise ValueError(
                    f"{evidence_label} pmon and snapshot physical GPU differ"
                )
            observed_identity = expected_gpu_identity[pmon_physical_index]
            mapping_matches_expected = (
                pmon_physical_index == physical_index
                and observed_identity["uuid"] == identity["uuid"]
            )
            if (
                row.get("probe_pid") != probe_pid
                or row.get("pmon_matched_pid") is not True
                or row.get("observed_physical_index") != pmon_physical_index
                or row.get("observed_gpu_uuid") != observed_identity["uuid"]
                or row.get("mapping_matches_expected")
                is not mapping_matches_expected
            ):
                raise ValueError(
                    f"{evidence_label} claims differ from parsed raw evidence"
                )
            computed_mapping_results.append(mapping_matches_expected)
        expected_scheme_status = (
            "passed" if all(computed_mapping_results) else "failed"
        )
        if scheme.get("status") != expected_scheme_status:
            raise ValueError(
                f"EGL mapping scheme {scheme_name} status differs from raw evidence"
            )
    if len(evidence_paths) != 36:
        raise ValueError("EGL mapping receipt must bind 36 unique evidence artifacts")

    recommended_rows = schemes["isolated_physical_identity"]["rows"]
    if schemes["isolated_physical_identity"]["status"] != "passed":
        raise ValueError("recommended isolated EGL mapping scheme did not pass")
    for physical_index, row in enumerate(recommended_rows):
        identity = expected_gpu_identity[physical_index]
        if (
            row.get("cuda_visible_devices") != str(physical_index)
            or row.get("mujoco_egl_device_id") != str(physical_index)
            or row.get("observed_physical_index") != physical_index
            or row.get("observed_gpu_uuid") != identity["uuid"]
            or row.get("renderer_status") != "passed"
            or row.get("pmon_matched_pid") is not True
            or row.get("mapping_matches_expected") is not True
        ):
            raise ValueError(
                f"recommended EGL mapping failed for physical GPU {physical_index}"
            )

    recommendation = value.get("recommendation")
    expected_recommendation = {
        "scheme": "isolated_physical_identity",
        "all_four_devices_passed": True,
        "cuda_visible_devices_expression": "${physical_index}",
        "mujoco_egl_device_id_expression": "${physical_index}",
        "mujoco_gl": "egl",
        "pyopengl_platform": "egl",
        "reason": (
            "isolates each evaluator while retaining verified physical EGL "
            "identity mapping"
        ),
    }
    if recommendation != expected_recommendation:
        raise ValueError("EGL mapping recommendation differs from the formal runner")
    return dict(value)


def _reverify_runtime_egl_closure(
    runtime: Mapping[str, Any],
    *,
    live_gpu_identity: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Reopen the complete receipt-bound EGL closure from a runtime receipt."""

    bindings = runtime.get("bindings")
    if not isinstance(bindings, dict):
        raise TypeError("DSW runtime receipt bindings must be an object")
    mapping_expected = bindings.get("egl_mapping_receipt")
    vendor_expected = bindings.get("egl_vendor_json")
    ready_marker = bindings.get("robocasa_ready_marker")
    if not isinstance(mapping_expected, dict):
        raise TypeError("DSW runtime EGL mapping receipt must be a descriptor")
    if not isinstance(vendor_expected, dict):
        raise TypeError("DSW runtime EGL vendor JSON must be a descriptor")
    if not isinstance(ready_marker, dict):
        raise TypeError("DSW runtime RoboCasa ready marker must be a descriptor")

    mapping_path = Path(
        _require_canonical_absolute_path_string(
            mapping_expected.get("path"),
            "DSW runtime EGL mapping receipt path",
        )
    )
    mapping_value, mapping_observed = _read_json(
        mapping_path,
        expected_mode=0o444,
    )
    if mapping_observed != mapping_expected:
        raise ValueError("DSW runtime EGL mapping receipt descriptor changed")

    vendor_path = Path(
        _require_canonical_absolute_path_string(
            vendor_expected.get("path"),
            "DSW runtime EGL vendor JSON path",
        )
    )
    vendor_observed = _stable_sha256(vendor_path, expected_mode=0o644)
    if vendor_observed != vendor_expected:
        raise ValueError("DSW runtime EGL vendor JSON descriptor changed")

    environment_path = Path(
        _require_canonical_absolute_path_string(
            bindings.get("robocasa_environment_path"),
            "DSW runtime RoboCasa environment path",
        )
    )
    if live_gpu_identity is None:
        live_gpu_identity = _nvidia_smi_gpu_identity()
    _validate_egl_mapping_receipt(
        mapping_value,
        robocasa_environment=environment_path,
        robocasa_ready_marker_sha256=_require_sha256(
            ready_marker.get("sha256"),
            "DSW runtime RoboCasa ready marker SHA-256",
        ),
        egl_vendor_descriptor=vendor_observed,
        live_gpu_identity=live_gpu_identity,
    )


def _validate_checkpoint_atomic_publication_receipt(
    value: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    expected_tree_sha256: str,
) -> dict[str, Any]:
    """Validate the committed hidden-stage to canonical-checkpoint transition."""

    _require_exact_keys(
        value,
        frozenset(
            {
                "completed_at",
                "final_root",
                "final_root_absent_before_publication",
                "kind",
                "mode",
                "postpublication",
                "prepublication",
                "publication",
                "recovered_uncommitted_final_before_publication",
                "schema_version",
                "stage_root",
                "status",
            }
        ),
        "checkpoint atomic publication receipt",
    )
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "cwam_dsw_ema_checkpoint_publication"
        or value.get("status") != "published"
        or value.get("mode") != "atomic_publish"
        or value.get("final_root_absent_before_publication") is not True
        or type(value.get("recovered_uncommitted_final_before_publication"))
        is not bool
    ):
        raise ValueError(
            "checkpoint atomic publication schema/kind/status/mode mismatch"
        )
    completed_at = value.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at.endswith("Z"):
        raise ValueError("checkpoint atomic publication completed_at must be UTC")
    try:
        completed = dt.datetime.fromisoformat(
            completed_at.removesuffix("Z") + "+00:00"
        )
    except ValueError as error:
        raise ValueError(
            "checkpoint atomic publication completed_at is invalid"
        ) from error
    if completed.tzinfo is None:
        raise ValueError("checkpoint atomic publication completed_at must be aware")

    checkpoint_path = _canonical_non_symlink_directory(checkpoint_path)
    expected_final_root = str(checkpoint_path.parent)
    final_root = _require_canonical_absolute_path_string(
        value.get("final_root"),
        "checkpoint atomic publication final_root",
    )
    stage_root = _require_canonical_absolute_path_string(
        value.get("stage_root"),
        "checkpoint atomic publication stage_root",
    )
    if (
        final_root != expected_final_root
        or Path(stage_root).parent != Path(final_root).parent
        or Path(stage_root).name
        != f".{Path(final_root).name}.incomplete-transfer-20260731"
    ):
        raise ValueError(
            "checkpoint atomic publication stage/final paths do not bind the serving checkpoint"
        )

    publication = value.get("publication")
    expected_publication = {
        "primitive": "renameat2(RENAME_NOREPLACE)",
        "same_parent_same_filesystem": True,
        "parent_fsync_before_and_after": True,
        "stage_absent_after": True,
        "final_present_after": True,
        "receipt_required_for_success": True,
        "rollback_on_postpublication_failure": True,
    }
    if publication != expected_publication:
        raise ValueError("checkpoint atomic publication guarantees are incomplete")

    pre = value.get("prepublication")
    post = value.get("postpublication")
    if not isinstance(pre, dict) or not isinstance(post, dict):
        raise TypeError("checkpoint pre/post publication audits must be objects")
    expected_audit_keys = frozenset(
        {
            "model",
            "root",
            "top_level_entries",
            "top_level_files",
            "validation",
        }
    )
    _require_exact_keys(pre, expected_audit_keys, "checkpoint prepublication")
    _require_exact_keys(post, expected_audit_keys, "checkpoint postpublication")
    pre_model = pre.get("model")
    post_model = post.get("model")
    if not isinstance(pre_model, dict) or not isinstance(post_model, dict):
        raise TypeError("checkpoint pre/post publication model audits must be objects")
    expected_model_keys = frozenset(
        {"file_count", "files", "path", "total_bytes", "tree_sha256"}
    )
    _require_exact_keys(
        pre_model,
        expected_model_keys,
        "checkpoint prepublication model",
    )
    _require_exact_keys(
        post_model,
        expected_model_keys,
        "checkpoint postpublication model",
    )
    expected_tree_sha256 = _require_sha256(
        expected_tree_sha256,
        "checkpoint atomic publication expected tree SHA-256",
    )
    files = pre_model.get("files")
    if (
        pre.get("root") != stage_root
        or post.get("root") != final_root
        or pre_model.get("path") != f"{stage_root}/model"
        or post_model.get("path") != str(checkpoint_path)
        or pre_model.get("tree_sha256") != expected_tree_sha256
        or post_model.get("tree_sha256") != expected_tree_sha256
        or not isinstance(files, list)
        or len(files) != 9
        or post_model.get("files") != files
        or pre_model.get("file_count") != 9
        or post_model.get("file_count") != 9
        or pre_model.get("total_bytes") != post_model.get("total_bytes")
        or pre.get("top_level_entries") != post.get("top_level_entries")
        or pre.get("top_level_files") != post.get("top_level_files")
    ):
        raise ValueError(
            "checkpoint pre/post publication file or tree audit differs"
        )
    expected_top_level_entries = {
        "H-SHA256SUMS",
        "config.yaml",
        "ema-bf16-export-receipt.json",
        "ema-bf16-key-manifest.json",
        "ema-bf16-validation.json",
        "ema-serving-dcp-transfer-manifest.json",
        "h-serving-checkpoint-manifest.json",
        "model",
        "source-dcp-full-verification.json",
    }
    if set(pre["top_level_entries"]) != expected_top_level_entries:
        raise ValueError("checkpoint publication top-level closure is incomplete")
    top_level_files = pre["top_level_files"]
    if (
        not isinstance(top_level_files, list)
        or len(top_level_files) != 8
        or {
            item.get("relative_path")
            for item in top_level_files
            if isinstance(item, dict)
        }
        != expected_top_level_entries - {"model"}
    ):
        raise ValueError("checkpoint publication top-level file closure is invalid")
    expected_model_names = {
        ".metadata",
        *(f"__{index}_0.distcp" for index in range(8)),
    }
    observed_model_names: set[str] = set()
    for file_index, record in enumerate(files):
        if not isinstance(record, dict):
            raise TypeError("checkpoint publication model file must be an object")
        _require_exact_keys(
            record,
            frozenset({"relative_path", "sha256", "size_bytes"}),
            f"checkpoint publication model file {file_index}",
        )
        relative_path = record.get("relative_path")
        if (
            not isinstance(relative_path, str)
            or PurePosixPath(relative_path).name != relative_path
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
        ):
            raise ValueError("checkpoint publication model file fields are invalid")
        _require_sha256(
            record.get("sha256"),
            f"checkpoint publication model file {relative_path} SHA-256",
        )
        observed_model_names.add(relative_path)
    if observed_model_names != expected_model_names:
        raise ValueError("checkpoint publication model file closure is incomplete")
    for phase, audit in (("pre", pre), ("post", post)):
        validation = audit.get("validation")
        if (
            not isinstance(validation, dict)
            or not validation
            or any(flag is not True for flag in validation.values())
        ):
            raise ValueError(
                f"checkpoint {phase}publication validation flags did not all pass"
            )
    return dict(value)


def _load_receipt_descriptors(
    receipts: Mapping[str, Path],
    *,
    expected_labels: frozenset[str],
    label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if set(receipts) != expected_labels:
        raise ValueError(
            f"{label} labels mismatch: missing={sorted(expected_labels - set(receipts))}, "
            f"extra={sorted(set(receipts) - expected_labels)}"
        )
    values: dict[str, dict[str, Any]] = {}
    descriptors: dict[str, dict[str, Any]] = {}
    for name in sorted(receipts):
        value, descriptor = _read_json(receipts[name], expected_mode=0o444)
        values[name] = value
        descriptors[name] = descriptor
    return values, descriptors


def _verify_bound_descriptors(
    descriptors: Mapping[str, Any],
    *,
    expected_labels: frozenset[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    if set(descriptors) != expected_labels:
        raise ValueError(
            f"{label} descriptor labels mismatch: "
            f"missing={sorted(expected_labels - set(descriptors))}, "
            f"extra={sorted(set(descriptors) - expected_labels)}"
        )
    verified: dict[str, dict[str, Any]] = {}
    for name in sorted(descriptors):
        expected = descriptors[name]
        if not isinstance(expected, dict):
            raise TypeError(f"{label}.{name} descriptor must be an object")
        _require_exact_keys(
            expected,
            frozenset({"path", "size_bytes", "sha256"}),
            f"{label}.{name}",
        )
        path = Path(_require_canonical_absolute_path_string(expected["path"], f"{label}.{name}.path"))
        actual = _stable_sha256(path, expected_mode=0o444)
        if actual != expected:
            raise ValueError(f"{label}.{name} descriptor changed")
        verified[name] = actual
    return verified


def _parse_named_paths(values: Sequence[str], label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw in values:
        name, separator, path_text = raw.partition("=")
        if (
            not separator
            or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
            or not path_text
            or name in parsed
        ):
            raise ValueError(f"{label} must use unique lowercase LABEL=/absolute/path entries")
        parsed[name] = Path(path_text)
    return parsed


def _descriptor_semantics(value: Mapping[str, Any]) -> tuple[int, str]:
    _require_exact_keys(
        value,
        frozenset({"path", "size_bytes", "sha256"}),
        "artifact descriptor",
    )
    size_bytes = value["size_bytes"]
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError("artifact descriptor size_bytes is invalid")
    return size_bytes, _require_sha256(value["sha256"], "artifact descriptor sha256")


def _canonical_manifest_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Qwen manifest relative_path must be a non-empty POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ValueError(f"Qwen manifest relative_path is not canonical: {value!r}")
    return value


def _qwen_directory_inventory(root: Path) -> dict[str, Path]:
    inventory: dict[str, Path] = {}
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path.resolve(strict=True) != current_path:
            raise ValueError(f"Qwen tokenizer directory contains a non-canonical directory: {current_path}")
        for name in directories:
            directory = current_path / name
            metadata = directory.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"Qwen tokenizer directory contains a symlink or non-directory: {directory}")
        for name in filenames:
            path = current_path / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"Qwen tokenizer directory contains a symlink or non-regular file: {path}")
            relative_path = path.relative_to(root).as_posix()
            if relative_path in inventory:
                raise ValueError(f"duplicate Qwen tokenizer file inventory entry: {relative_path}")
            inventory[relative_path] = path
    return inventory


def verify_qwen_tokenizer_manifest(
    *,
    tokenizer_path: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Verify a complete local Qwen tokenizer tree against a simple manifest."""
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        "expected_manifest_sha256",
    )
    tokenizer_path = _canonical_non_symlink_directory(tokenizer_path)
    manifest, manifest_descriptor = _read_json(manifest_path)
    if manifest_descriptor["sha256"] != expected_manifest_sha256:
        raise ValueError(
            "Qwen tokenizer manifest SHA-256 mismatch: "
            f"expected={expected_manifest_sha256} actual={manifest_descriptor['sha256']}"
        )
    manifest_file = Path(manifest_descriptor["path"])
    if manifest_file == tokenizer_path or tokenizer_path in manifest_file.parents:
        raise ValueError("Qwen tokenizer manifest must be outside the tokenizer directory")
    if set(manifest) != {"schema_version", "kind", "files"}:
        raise ValueError("Qwen tokenizer manifest top-level keys must be exactly schema_version, kind, files")
    if manifest["schema_version"] != 1 or manifest["kind"] != _QWEN_MANIFEST_KIND:
        raise ValueError("unsupported Qwen tokenizer manifest schema/kind")
    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Qwen tokenizer manifest files must be a non-empty list")

    expected_files: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"relative_path", "size_bytes", "sha256"}:
            raise ValueError(
                "each Qwen tokenizer manifest file entry must contain only relative_path, size_bytes, sha256"
            )
        relative_path = _canonical_manifest_relative_path(entry["relative_path"])
        size_bytes = entry["size_bytes"]
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError(f"Qwen tokenizer manifest size_bytes is invalid: {relative_path}")
        digest = _require_sha256(entry["sha256"], f"Qwen tokenizer file {relative_path} sha256")
        if relative_path in expected_files:
            raise ValueError(f"duplicate Qwen tokenizer manifest relative_path: {relative_path}")
        expected_files[relative_path] = {
            "relative_path": relative_path,
            "size_bytes": size_bytes,
            "sha256": digest,
        }

    missing_required = sorted(_QWEN_REQUIRED_FILES - set(expected_files))
    if missing_required:
        raise ValueError(f"Qwen tokenizer manifest is missing required files: {missing_required}")
    actual_files = _qwen_directory_inventory(tokenizer_path)
    if set(actual_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(actual_files))
        unexpected = sorted(set(actual_files) - set(expected_files))
        raise ValueError(
            "Qwen tokenizer directory file set does not exactly match its manifest: "
            f"missing={missing} unexpected={unexpected}"
        )

    verified_files: list[dict[str, Any]] = []
    for relative_path in sorted(expected_files):
        descriptor = _stable_sha256(actual_files[relative_path])
        expected = expected_files[relative_path]
        if descriptor["size_bytes"] != expected["size_bytes"] or descriptor["sha256"] != expected["sha256"]:
            raise ValueError(f"Qwen tokenizer file size/hash mismatch: {relative_path}")
        verified_files.append(expected)

    receipt = {
        "schema_version": 1,
        "kind": _QWEN_VERIFICATION_KIND,
        "tokenizer_path": str(tokenizer_path),
        "manifest": manifest_descriptor,
        "required_files": sorted(_QWEN_REQUIRED_FILES),
        "file_count": len(verified_files),
        "files": verified_files,
        "validation": {
            "manifest_schema_and_kind_match": True,
            "manifest_entries_have_exact_keys": True,
            "relative_paths_are_canonical": True,
            "required_files_are_present": True,
            "directory_contains_no_symlinks_or_special_files": True,
            "directory_file_set_matches_manifest_exactly": True,
            "all_file_sizes_and_sha256_match": True,
        },
    }
    _atomic_write_exclusive(output, receipt)
    return receipt


def verify_one_episode_gate(
    *,
    gate_results_dir: Path,
    checkpoint_path: Path,
    run_name: str,
    task: str,
    runtime_receipt: Path,
    output: Path,
) -> dict[str, Any]:
    """Verify one separate closed-loop episode before admitting the 340 run."""

    if task != "CloseBlenderLid":
        raise ValueError("formal DSW one-episode gate task must be CloseBlenderLid")
    gate_results_dir = _canonical_non_symlink_directory(gate_results_dir)
    checkpoint_path = _canonical_non_symlink_directory(checkpoint_path)
    runtime, runtime_descriptor = _load_platform_receipt(runtime_receipt)
    if runtime["bindings"]["checkpoint"]["path"] != str(checkpoint_path):
        raise ValueError("one-episode gate checkpoint differs from runtime")
    args_value, args_descriptor = _read_json(
        gate_results_dir / "args.json",
        expected_mode=0o444,
    )
    results_value, results_descriptor = _read_json(
        gate_results_dir / "results_shard0.json",
        expected_mode=0o444,
    )
    expected_args = {
        "task_shard": 0,
        "num_shards": 1,
        "episodes": 1,
        "episodes_start": 0,
        "max_steps": 400,
        "open_loop_horizon": 12,
        "seed": 42,
        "checkpoint": str(checkpoint_path),
        "run_name": run_name,
        "split": "atomic",
        "headless": True,
        "save_video": False,
    }
    for key, expected in expected_args.items():
        if args_value.get(key) != expected:
            raise ValueError(f"one-episode gate args mismatch for {key}")
    tasks = args_value.get("tasks")
    if tasks != [task]:
        raise ValueError("one-episode gate args must select exactly CloseBlenderLid")
    if set(results_value) != {task}:
        raise ValueError("one-episode gate results must contain exactly the selected task")
    task_result = results_value[task]
    if not isinstance(task_result, dict):
        raise TypeError("one-episode gate task result must be an object")
    if (
        task_result.get("complete") is not True
        or task_result.get("split") != "atomic"
        or task_result.get("episodes") != 1
        or "error" in task_result
    ):
        raise ValueError("one-episode gate task result is incomplete or erroneous")
    episodes = task_result.get("episode_results")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise ValueError("one-episode gate must contain exactly one episode result")
    episode = episodes[0]
    if not isinstance(episode, dict):
        raise TypeError("one-episode gate episode result must be an object")
    if type(episode.get("success")) is not bool or type(episode.get("place_anytime")) is not bool:
        raise TypeError("one-episode gate success fields must be booleans")
    steps = episode.get("steps")
    actions = episode.get("action_log")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 1 <= steps <= 400
        or not isinstance(actions, list)
        or len(actions) != steps
    ):
        raise ValueError("one-episode gate steps/action log mismatch")
    for step_index, action in enumerate(actions):
        if (
            not isinstance(action, list)
            or len(action) != 7
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in action
            )
        ):
            raise ValueError(
                f"one-episode gate action {step_index} is not finite width-7"
            )
    receipt = {
        "schema_version": 1,
        "kind": _GATE_KIND,
        "status": "passed",
        "role": "preflight_only_not_part_of_340_primary_metric",
        "task": task,
        "episode_count": 1,
        "steps": steps,
        "success": episode["success"],
        "checkpoint_path": str(checkpoint_path),
        "run_name": run_name,
        "runtime_receipt": runtime_descriptor,
        "args": args_descriptor,
        "results": results_descriptor,
        "validation": {
            "exactly_one_selected_task": True,
            "exactly_one_complete_episode": True,
            "at_least_one_policy_action_observed": True,
            "every_action_is_finite_width_7": True,
            "checkpoint_matches_runtime": True,
            "excluded_from_primary_340_episode_metric": True,
        },
    }
    _atomic_write_exclusive(output, receipt)
    return receipt


def _effective_cpu_ids() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Linux sched_getaffinity is required for a formal DSW receipt")
    return sorted(os.sched_getaffinity(0))


def _parse_positive_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    # cgroup v1 commonly uses a near-int64 maximum as the "unlimited" marker.
    return value if 0 < value < (1 << 60) else None


def _effective_memory_limit() -> tuple[int, str]:
    candidates = [
        (Path("/sys/fs/cgroup/memory.max"), "cgroup_v2_memory.max"),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            "cgroup_v1_memory.limit_in_bytes",
        ),
    ]
    finite = [(value, label) for path, label in candidates if (value := _parse_positive_int(path)) is not None]
    if finite:
        return min(finite)
    mem_total_kib: int | None = None
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                mem_total_kib = int(fields[1])
                break
    if mem_total_kib is None:
        raise RuntimeError("could not determine the DSW memory allocation")
    return mem_total_kib * 1024, "proc_meminfo_MemTotal"


def _process_is_descendant(pid: int, ancestor: int) -> bool:
    if pid <= 0 or ancestor <= 0:
        return False
    current = pid
    seen: set[int] = set()
    while current > 1 and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        try:
            stat_text = Path(f"/proc/{current}/stat").read_text(encoding="utf-8")
        except OSError:
            return False
        _head, separator, tail = stat_text.rpartition(")")
        fields = tail.split()
        if not separator or len(fields) < 2:
            return False
        # Fields after the final ')' start with state then ppid.
        current = int(fields[1])
    return current == ancestor


def _read_process_environment(pid: int, keys: Iterable[str]) -> dict[str, str]:
    try:
        payload = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as error:
        raise RuntimeError(f"could not read environment for server PID {pid}: {error}") from error
    values: dict[str, str] = {}
    for entry in payload.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        raw_key, raw_value = entry.split(b"=", 1)
        try:
            key = raw_key.decode("utf-8")
            value = raw_value.decode("utf-8")
        except UnicodeError as error:
            raise RuntimeError(f"server process environment is not UTF-8: {error}") from error
        if key in keys:
            values[key] = value
    return values


def _gpu_inventory() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    inventory = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        inventory.append(
            {
                "visible_index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "sm": f"sm{properties.major}{properties.minor}",
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return inventory


def validate_runtime_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        value,
        frozenset(
            {
                "bindings",
                "captured_at_utc",
                "kind",
                "platform",
                "policy_server",
                "provenance",
                "resources",
                "schema_version",
                "topology",
                "validation",
                "watchdogs",
            }
        ),
        "DSW runtime receipt",
    )
    if value.get("schema_version") != 2 or value.get("kind") != _RUNTIME_KIND:
        raise ValueError("unsupported DSW runtime receipt schema/kind")
    if value.get("platform") != _PLATFORM:
        raise ValueError("DSW runtime receipt platform identity mismatch")
    topology = value.get("topology")
    if not isinstance(topology, dict):
        raise TypeError("DSW runtime receipt topology must be an object")
    expected_topology = {
        "physical_server_count": _EXPECTED_PHYSICAL_SERVER_COUNT,
        "physical_policy_server_count": 1,
        "physical_gpu_count": _EXPECTED_PHYSICAL_GPU_COUNT,
        "distributed_world_size": _EXPECTED_PHYSICAL_GPU_COUNT,
        "logical_num_shards": _EXPECTED_LOGICAL_NUM_SHARDS,
        "evaluator_schedule": "sequential_shards_0_through_7",
    }
    for key, expected in expected_topology.items():
        if topology.get(key) != expected:
            raise ValueError(f"DSW runtime receipt topology {key} mismatch")
    policy_uri = topology.get("policy_server_uri")
    if policy_uri != "ws://127.0.0.1:16000":
        raise ValueError("DSW runtime receipt must use the fixed loopback policy endpoint")

    resources = value.get("resources")
    if not isinstance(resources, dict):
        raise TypeError("DSW runtime receipt resources must be an object")
    if resources.get("cpu_affinity_count") != _EXPECTED_CPU_COUNT:
        raise ValueError("DSW runtime receipt must record exactly 128 effective CPUs")
    if resources.get("memory_limit_bytes") != _EXPECTED_MEMORY_GIB * (1024**3):
        raise ValueError("DSW runtime receipt must record exactly 400 GiB")
    gpus = resources.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != _EXPECTED_PHYSICAL_GPU_COUNT:
        raise ValueError("DSW runtime receipt must contain exactly four GPUs")
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, dict) or gpu.get("visible_index") != index:
            raise ValueError("DSW GPU inventory indices are incomplete")
        if "4090" not in str(gpu.get("name", "")):
            raise ValueError("every DSW GPU must be a 4090")
        if gpu.get("compute_capability") != "8.9" or gpu.get("sm") != "sm89":
            raise ValueError("every DSW GPU must report compute capability 8.9 / sm89")

    server = value.get("policy_server")
    if not isinstance(server, dict):
        raise TypeError("DSW policy_server receipt must be an object")
    if server.get("distributed_world_size") != 4:
        raise ValueError("DSW policy server must report distributed_world_size=4")
    if server.get("parallelism") != _EXPECTED_PARALLELISM:
        raise ValueError("DSW policy server parallelism does not match FSDP4")
    expected_policy_values = {
        "sampler": "unipc",
        "policy_seed": 42,
        "guidance": 1.0,
        "num_steps": 30,
        "use_ema_weights": True,
        "guardrails": False,
        "vision_tokenizer": (
            "cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16."
            "Wan2pt2VAEInterface"
        ),
        "online_pixel_encoding": True,
        "reuse_policy_seed_per_request": True,
        "control_protocol": {
            "shutdown": "nonce_bound_websocket_control_v1",
            "requires_loopback_endpoint": True,
        },
    }
    for key, expected in expected_policy_values.items():
        if server.get(key) != expected:
            raise ValueError(f"DSW policy server formal setting mismatch for {key}")

    bindings = value.get("bindings")
    if not isinstance(bindings, dict):
        raise TypeError("DSW runtime receipt bindings must be an object")
    _require_exact_keys(
        bindings,
        frozenset(
            {
                "action_stats",
                "checkpoint",
                "demian_commit",
                "demian_openpi_server_source",
                "egl_mapping_receipt",
                "egl_vendor_json",
                "launch_nonce",
                "qwen_tokenizer_manifest",
                "qwen_tokenizer_path",
                "rank0_server_pid",
                "robocasa_environment_path",
                "robocasa_ready_marker",
                "source_commit",
                "source_manifest",
                "torchrun_launcher_pid",
                "training_config",
                "wan2pt2_vae",
            }
        ),
        "DSW runtime receipt bindings",
    )
    _require_sha256(bindings.get("launch_nonce"), "DSW runtime receipt launch_nonce")
    _require_commit(bindings.get("demian_commit"), "DSW runtime receipt demian_commit")
    _require_commit(bindings.get("source_commit"), "DSW runtime receipt source_commit")
    demian_openpi_source = bindings.get("demian_openpi_server_source")
    if not isinstance(demian_openpi_source, dict):
        raise TypeError(
            "DSW runtime demian_openpi_server_source must be a descriptor"
        )
    _descriptor_semantics(demian_openpi_source)
    _require_canonical_absolute_path_string(
        demian_openpi_source.get("path"),
        "DSW runtime demian_openpi_server_source path",
    )
    websocket_server = server.get("websocket_server")
    if not isinstance(websocket_server, dict):
        raise TypeError("DSW policy server websocket_server must be an object")
    _require_exact_keys(
        websocket_server,
        frozenset({"module", "qualname", "source_path", "source_sha256"}),
        "DSW policy server websocket_server",
    )
    if (
        websocket_server.get("module")
        != "openpi.serving.websocket_policy_server"
        or websocket_server.get("qualname") != "WebsocketPolicyServer"
        or websocket_server.get("source_path") != demian_openpi_source["path"]
        or websocket_server.get("source_sha256") != demian_openpi_source["sha256"]
    ):
        raise ValueError(
            "DSW policy server is not bound to the pinned DeMiAn OpenPI source"
        )
    checkpoint = bindings.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError("DSW runtime checkpoint binding must be an object")
    _require_exact_keys(
        checkpoint,
        frozenset({"manifest", "path", "tree_sha256"}),
        "DSW runtime checkpoint binding",
    )
    checkpoint_path = _require_canonical_absolute_path_string(
        checkpoint.get("path"), "DSW runtime checkpoint path"
    )
    _require_sha256(checkpoint.get("tree_sha256"), "DSW runtime checkpoint tree_sha256")
    for name in (
        "manifest",
        "training_config",
        "action_stats",
        "wan2pt2_vae",
        "qwen_tokenizer_manifest",
        "egl_mapping_receipt",
        "egl_vendor_json",
        "robocasa_ready_marker",
        "source_manifest",
    ):
        descriptor = checkpoint["manifest"] if name == "manifest" else bindings[name]
        if not isinstance(descriptor, dict):
            raise TypeError(f"DSW runtime {name} descriptor must be an object")
        _descriptor_semantics(descriptor)
        _require_canonical_absolute_path_string(
            descriptor.get("path"), f"DSW runtime {name} descriptor path"
        )
    qwen_path = _require_canonical_absolute_path_string(
        bindings.get("qwen_tokenizer_path"),
        "DSW runtime qwen_tokenizer_path",
    )
    robocasa_environment_path = _require_canonical_absolute_path_string(
        bindings.get("robocasa_environment_path"),
        "DSW runtime robocasa_environment_path",
    )
    if (
        bindings["robocasa_ready_marker"]["path"]
        != f"{robocasa_environment_path}/.cwam_env_ready.json"
    ):
        raise ValueError(
            "DSW runtime RoboCasa ready marker is outside the bound environment"
        )
    for name in ("torchrun_launcher_pid", "rank0_server_pid"):
        if type(bindings.get(name)) is not int or bindings[name] <= 0:
            raise ValueError(f"DSW runtime receipt {name} must be positive")

    if server.get("checkpoint") != checkpoint_path:
        raise ValueError("DSW policy server checkpoint path differs from the runtime binding")
    if server.get("config_file") != bindings["training_config"]["path"]:
        raise ValueError("DSW policy server config_file path differs from the runtime binding")
    if server.get("config_file_sha256") != bindings["training_config"]["sha256"]:
        raise ValueError("DSW policy server config_file SHA-256 differs from the runtime binding")
    if server.get("action_stats_path") != bindings["action_stats"]["path"]:
        raise ValueError("DSW policy server action_stats path differs from the runtime binding")
    if server.get("action_stats_sha256") != bindings["action_stats"]["sha256"]:
        raise ValueError("DSW policy server action_stats SHA-256 differs from the runtime binding")
    if server.get("wan2pt2_vae_path") != bindings["wan2pt2_vae"]["path"]:
        raise ValueError("DSW policy server Wan2.2 VAE path differs from the runtime binding")
    if server.get("qwen_tokenizer_path") != qwen_path:
        raise ValueError("DSW policy server Qwen tokenizer path differs from the runtime binding")
    weight = server.get("checkpoint_weight_receipt")
    expected_weight = {
        "selected_weight_tree": "net_ema",
        "selected_source_prefix": "net_ema.",
        "selected_key_count": 809,
        "selected_source_dtype_census": {"torch.bfloat16": 809},
        "regular_key_count": 0,
        "ema_key_count": 809,
        "other_key_count": 0,
        "serving_only_selected_tree": True,
        "load_planner": "CustomLoadPlanner(load_ema_to_reg=True)",
    }
    if not isinstance(weight, dict):
        raise TypeError("DSW policy server checkpoint_weight_receipt must be an object")
    for key, expected in expected_weight.items():
        if weight.get(key) != expected:
            raise ValueError(f"DSW policy server selected EMA tree mismatch for {key}")
    if (
        server.get("checkpoint_weight_tree") != "net_ema"
        or weight.get("checkpoint_path") != checkpoint_path
        or server.get("checkpoint_metadata_sha256")
        != weight.get("checkpoint_metadata_sha256")
    ):
        raise ValueError("DSW policy server selected checkpoint identity mismatch")
    metadata_record = next(
        (
            record
            for record in checkpoint["manifest"].get("_files", [])
            if isinstance(record, dict) and record.get("relative_path") == ".metadata"
        ),
        None,
    )
    # `_files` is optional in the compact runtime descriptor. The live server
    # still reports and validates its metadata hash via its weight receipt.
    if metadata_record is not None and server.get("checkpoint_metadata_sha256") != metadata_record.get("sha256"):
        raise ValueError("DSW policy server checkpoint metadata SHA-256 mismatch")

    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("DSW runtime provenance must be an object")
    _require_exact_keys(
        provenance,
        frozenset({"receipts"}),
        "DSW runtime provenance",
    )
    receipts = provenance["receipts"]
    if not isinstance(receipts, dict) or set(receipts) != _REQUIRED_RUNTIME_PROVENANCE:
        raise ValueError("DSW runtime provenance receipt labels are incomplete")
    for name, descriptor in receipts.items():
        if not isinstance(descriptor, dict):
            raise TypeError(f"DSW runtime provenance receipt {name} must be a descriptor")
        _descriptor_semantics(descriptor)

    watchdogs = value.get("watchdogs")
    if not isinstance(watchdogs, dict):
        raise TypeError("DSW runtime watchdogs must be an object")
    _require_exact_keys(
        watchdogs,
        frozenset(
            {
                "evaluator_timeout_seconds",
                "nccl_server_environment",
                "one_episode_gate_timeout_seconds",
                "server_exit_timeout_seconds",
                "server_ready_timeout_seconds",
            }
        ),
        "DSW runtime watchdogs",
    )
    if watchdogs.get("nccl_server_environment") != _NCCL_WATCHDOG_ENV:
        raise ValueError("DSW runtime NCCL watchdog environment mismatch")
    expected_timeouts = {
        "evaluator_timeout_seconds": 43200,
        "one_episode_gate_timeout_seconds": 3600,
        "server_exit_timeout_seconds": 180,
        "server_ready_timeout_seconds": 1800,
    }
    for key, expected in expected_timeouts.items():
        if watchdogs.get(key) != expected:
            raise ValueError(f"DSW runtime watchdog timeout mismatch for {key}")

    validation = value.get("validation")
    if not isinstance(validation, dict) or not validation or any(flag is not True for flag in validation.values()):
        raise ValueError("DSW runtime receipt validation flags did not all pass")
    return dict(value)


def capture_runtime(
    *,
    uri: str,
    output: Path,
    timeout_seconds: float,
    launcher_pid: int,
    expected_launch_nonce: str,
    checkpoint_path: Path,
    checkpoint_manifest_path: Path,
    checkpoint_manifest_sha256: str,
    checkpoint_tree_sha256: str,
    training_config_path: Path,
    training_config_sha256: str,
    action_stats_path: Path,
    action_stats_sha256: str,
    wan2pt2_vae_path: Path,
    wan2pt2_vae_sha256: str,
    qwen_tokenizer_path: Path,
    qwen_tokenizer_manifest_path: Path,
    qwen_tokenizer_manifest_sha256: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    source_commit: str,
    demian_commit: str,
    demian_openpi_server_source_path: Path,
    robocasa_environment_path: Path,
    robocasa_ready_marker_path: Path,
    robocasa_ready_marker_sha256: str,
    egl_vendor_json_path: Path,
    egl_vendor_json_sha256: str,
    egl_mapping_receipt_path: Path,
    egl_mapping_receipt_sha256: str,
    provenance_receipts: Mapping[str, Path],
    evaluator_timeout_seconds: int,
    one_episode_gate_timeout_seconds: int,
    server_exit_timeout_seconds: int,
) -> dict[str, Any]:
    import torch
    from capture_robocasa365_server_receipt import _connect_once, _validate_metadata

    if uri != "ws://127.0.0.1:16000":
        raise ValueError("formal DSW runtime must use ws://127.0.0.1:16000")
    _require_sha256(expected_launch_nonce, "expected_launch_nonce")
    source_commit = _require_commit(source_commit, "source_commit")
    demian_commit = _require_commit(demian_commit, "demian_commit")
    demian_openpi_source_descriptor = _stable_sha256(
        demian_openpi_server_source_path
    )
    robocasa_environment_path = _canonical_non_symlink_directory(
        robocasa_environment_path
    )
    robocasa_ready_marker_descriptor = _descriptor_for_expected_file(
        robocasa_ready_marker_path,
        expected_sha256=robocasa_ready_marker_sha256,
        label="RoboCasa native ready marker",
        expected_mode=0o444,
    )
    if (
        Path(robocasa_ready_marker_descriptor["path"])
        != robocasa_environment_path / ".cwam_env_ready.json"
    ):
        raise ValueError("RoboCasa native ready marker is outside the declared environment")
    egl_vendor_descriptor = _descriptor_for_expected_file(
        egl_vendor_json_path,
        expected_sha256=egl_vendor_json_sha256,
        label="NVIDIA EGL vendor JSON",
        expected_mode=0o644,
    )
    egl_mapping_receipt, egl_mapping_descriptor = _read_json(
        egl_mapping_receipt_path,
        expected_mode=0o444,
    )
    if egl_mapping_descriptor["sha256"] != _require_sha256(
        egl_mapping_receipt_sha256,
        "EGL mapping receipt expected SHA-256",
    ):
        raise ValueError("EGL mapping receipt SHA-256 mismatch")
    checkpoint_path = _canonical_non_symlink_directory(checkpoint_path)
    checkpoint_manifest, checkpoint_manifest_descriptor = _read_json(
        checkpoint_manifest_path,
        expected_mode=0o444,
    )
    if checkpoint_manifest_descriptor["sha256"] != _require_sha256(
        checkpoint_manifest_sha256, "checkpoint_manifest_sha256"
    ):
        raise ValueError("DSW destination checkpoint manifest SHA-256 mismatch")
    checkpoint_verification = (
        cwam_serving_checkpoint_manifest.verify_serving_checkpoint_manifest(
            checkpoint_manifest,
            checkpoint_path,
        )
    )
    if checkpoint_verification["tree_sha256"] != _require_sha256(
        checkpoint_tree_sha256, "checkpoint_tree_sha256"
    ):
        raise ValueError("DSW destination checkpoint tree SHA-256 mismatch")
    training_config_descriptor = _descriptor_for_expected_file(
        training_config_path,
        expected_sha256=training_config_sha256,
        label="training config",
    )
    action_stats_descriptor = _descriptor_for_expected_file(
        action_stats_path,
        expected_sha256=action_stats_sha256,
        label="action stats",
    )
    vae_descriptor = _descriptor_for_expected_file(
        wan2pt2_vae_path,
        expected_sha256=wan2pt2_vae_sha256,
        label="Wan2.2 VAE",
    )
    qwen_manifest_descriptor = _descriptor_for_expected_file(
        qwen_tokenizer_manifest_path,
        expected_sha256=qwen_tokenizer_manifest_sha256,
        label="Qwen tokenizer manifest",
    )
    source_manifest_descriptor = _descriptor_for_expected_file(
        source_manifest_path,
        expected_sha256=source_manifest_sha256,
        label="source manifest",
        expected_mode=0o444,
    )
    qwen_tokenizer_path = _canonical_non_symlink_directory(qwen_tokenizer_path)
    provenance_values, provenance_descriptors = _load_receipt_descriptors(
        provenance_receipts,
        expected_labels=_REQUIRED_RUNTIME_PROVENANCE,
        label="runtime provenance",
    )
    _validate_checkpoint_atomic_publication_receipt(
        provenance_values["checkpoint_atomic_publication"],
        checkpoint_path=checkpoint_path,
        expected_tree_sha256=checkpoint_tree_sha256,
    )
    if launcher_pid <= 0:
        raise ValueError("launcher_pid must be positive")

    deadline = time.monotonic() + timeout_seconds
    metadata: dict[str, Any] | None = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            os.kill(launcher_pid, 0)
            metadata = _connect_once(uri, min(10.0, max(0.1, deadline - time.monotonic())))
            break
        except ProcessLookupError:
            raise RuntimeError(f"torchrun launcher {launcher_pid} exited before server readiness") from None
        except Exception as error:  # noqa: BLE001 - connection failures vary by runtime
            last_error = error
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    if metadata is None:
        raise TimeoutError(f"timed out waiting for the DSW policy server at {uri}: {last_error}")
    _validate_metadata(metadata)
    if metadata.get("launch_nonce") != expected_launch_nonce:
        raise ValueError("DSW policy server launch nonce mismatch")
    server_pid = metadata.get("server_pid")
    if type(server_pid) is not int or not _process_is_descendant(server_pid, launcher_pid):
        raise ValueError("rank-0 server PID is not a live descendant of the torchrun launcher")
    server_watchdog_environment = _read_process_environment(
        server_pid,
        _NCCL_WATCHDOG_ENV,
    )
    if server_watchdog_environment != _NCCL_WATCHDOG_ENV:
        raise ValueError(
            "rank-0 server did not inherit the exact NCCL watchdog environment"
        )
    if metadata.get("distributed_world_size") != 4 or metadata.get("parallelism") != _EXPECTED_PARALLELISM:
        raise ValueError("rank-0 server did not report the required FSDP4 topology")
    exact_metadata = {
        "checkpoint": str(checkpoint_path),
        "config_file": training_config_descriptor["path"],
        "config_file_sha256": training_config_descriptor["sha256"],
        "action_stats_path": action_stats_descriptor["path"],
        "action_stats_sha256": action_stats_descriptor["sha256"],
        "wan2pt2_vae_path": vae_descriptor["path"],
        "qwen_tokenizer_path": str(qwen_tokenizer_path),
    }
    for key, expected in exact_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"rank-0 server metadata {key} differs from the exact verified CLI input"
            )
    expected_websocket_server = {
        "module": "openpi.serving.websocket_policy_server",
        "qualname": "WebsocketPolicyServer",
        "source_path": demian_openpi_source_descriptor["path"],
        "source_sha256": demian_openpi_source_descriptor["sha256"],
    }
    if metadata.get("websocket_server") != expected_websocket_server:
        raise ValueError(
            "rank-0 server did not load the exact pinned DeMiAn OpenPI "
            "WebSocket implementation"
        )
    weight_receipt = metadata.get("checkpoint_weight_receipt")
    if not isinstance(weight_receipt, dict):
        raise ValueError("rank-0 server omitted checkpoint_weight_receipt")
    metadata_records = {
        str(record["relative_path"]): record
        for record in checkpoint_manifest["files"]
        if isinstance(record, dict)
    }
    if (
        metadata.get("checkpoint_metadata_sha256")
        != metadata_records.get(".metadata", {}).get("sha256")
        or weight_receipt.get("checkpoint_metadata_sha256")
        != metadata_records.get(".metadata", {}).get("sha256")
    ):
        raise ValueError("rank-0 server loaded checkpoint metadata differs from destination manifest")

    cpu_ids = _effective_cpu_ids()
    memory_limit_bytes, memory_limit_source = _effective_memory_limit()
    gpus = _gpu_inventory()
    live_gpu_identity = _nvidia_smi_gpu_identity()
    if len(cpu_ids) != _EXPECTED_CPU_COUNT:
        raise ValueError(f"effective CPU count mismatch: expected=128 actual={len(cpu_ids)}")
    if memory_limit_bytes != _EXPECTED_MEMORY_GIB * (1024**3):
        raise ValueError(
            f"effective memory limit mismatch: expected={_EXPECTED_MEMORY_GIB * (1024**3)} actual={memory_limit_bytes}"
        )
    if len(gpus) != 4 or any("4090" not in gpu["name"] for gpu in gpus):
        raise ValueError("exactly four visible 4090 GPUs are required")
    if any(gpu["compute_capability"] != "8.9" for gpu in gpus):
        raise ValueError("all visible GPUs must report compute capability 8.9")
    _validate_egl_mapping_receipt(
        egl_mapping_receipt,
        robocasa_environment=robocasa_environment_path,
        robocasa_ready_marker_sha256=robocasa_ready_marker_descriptor["sha256"],
        egl_vendor_descriptor=egl_vendor_descriptor,
        live_gpu_identity=live_gpu_identity,
    )

    receipt = {
        "schema_version": 2,
        "kind": _RUNTIME_KIND,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": dict(_PLATFORM),
        "topology": {
            "hostname": socket.gethostname(),
            "physical_server_count": 1,
            "physical_policy_server_count": 1,
            "physical_gpu_count": 4,
            "distributed_world_size": 4,
            "logical_num_shards": 8,
            "evaluator_schedule": "sequential_shards_0_through_7",
            "policy_server_uri": uri,
        },
        "resources": {
            "cpu_affinity_count": len(cpu_ids),
            "cpu_affinity_ids": cpu_ids,
            "memory_limit_bytes": memory_limit_bytes,
            "memory_limit_gib": memory_limit_bytes / float(1024**3),
            "memory_limit_source": memory_limit_source,
            "gpus": gpus,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
        "policy_server": metadata,
        "bindings": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "manifest": checkpoint_manifest_descriptor,
                "tree_sha256": checkpoint_verification["tree_sha256"],
            },
            "training_config": training_config_descriptor,
            "action_stats": action_stats_descriptor,
            "wan2pt2_vae": vae_descriptor,
            "qwen_tokenizer_path": str(qwen_tokenizer_path),
            "qwen_tokenizer_manifest": qwen_manifest_descriptor,
            "source_manifest": source_manifest_descriptor,
            "source_commit": source_commit,
            "demian_commit": demian_commit,
            "demian_openpi_server_source": demian_openpi_source_descriptor,
            "egl_mapping_receipt": egl_mapping_descriptor,
            "egl_vendor_json": egl_vendor_descriptor,
            "launch_nonce": expected_launch_nonce,
            "robocasa_environment_path": str(robocasa_environment_path),
            "robocasa_ready_marker": robocasa_ready_marker_descriptor,
            "torchrun_launcher_pid": launcher_pid,
            "rank0_server_pid": server_pid,
        },
        "provenance": {"receipts": provenance_descriptors},
        "watchdogs": {
            "nccl_server_environment": server_watchdog_environment,
            "server_ready_timeout_seconds": int(timeout_seconds),
            "one_episode_gate_timeout_seconds": one_episode_gate_timeout_seconds,
            "evaluator_timeout_seconds": evaluator_timeout_seconds,
            "server_exit_timeout_seconds": server_exit_timeout_seconds,
        },
        "validation": {
            "single_physical_server": True,
            "one_rank0_policy_endpoint": True,
            "four_local_distributed_ranks": True,
            "four_visible_4090_gpus": True,
            "all_gpus_compute_capability_sm89": True,
            "effective_cpu_affinity_is_128": True,
            "effective_memory_limit_is_400_gib": True,
            "logical_num_shards_is_8": True,
            "parallelism_is_dp_shard4_only": True,
            "rank0_server_is_torchrun_descendant": True,
            "server_checkpoint_path_and_metadata_hash_match": True,
            "server_config_canonical_path_and_sha256_match": True,
            "server_action_stats_canonical_path_and_sha256_match": True,
            "server_wan2pt2_vae_path_matches": True,
            "server_qwen_tokenizer_path_matches": True,
            "server_websocket_implementation_matches_demian_source": True,
            "egl_vendor_json_path_and_sha256_match": True,
            "egl_mapping_receipt_reverified_against_live_gpu_identity": True,
            "robocasa_environment_and_ready_marker_match_egl_probe": True,
            "server_selected_serving_only_net_ema_bf16_tree": True,
            "checkpoint_atomic_publication_semantics_match": True,
            "all_required_pre_rollout_provenance_bound": True,
            "rank0_server_inherited_exact_nccl_watchdog_environment": True,
            "all_external_waits_have_finite_timeouts": True,
        },
    }
    validate_runtime_receipt(receipt)
    _reverify_runtime_egl_closure(
        receipt,
        live_gpu_identity=live_gpu_identity,
    )
    _atomic_write_exclusive(output, receipt)
    return receipt


def _validate_shutdown_receipt(
    value: Mapping[str, Any],
    *,
    runtime_descriptor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_exact_keys(
        value,
        frozenset(
            {
                "ack",
                "completed_at_utc",
                "kind",
                "launch_nonce",
                "request",
                "runtime_receipt",
                "schema_version",
                "server_pid",
                "status",
                "uri",
                "validation",
            }
        ),
        "controlled shutdown receipt",
    )
    if (
        value.get("schema_version") != 1
        or value.get("kind") != _SHUTDOWN_KIND
        or value.get("status") != "passed"
    ):
        raise ValueError("controlled shutdown receipt schema/kind/status mismatch")
    launch_nonce = _require_sha256(
        value.get("launch_nonce"), "controlled shutdown launch_nonce"
    )
    expected_request = {
        "__cwam_control__": "shutdown",
        "launch_nonce": launch_nonce,
    }
    expected_ack = {
        "actions": [],
        "cwam_control": "shutdown_ack",
        "launch_nonce": launch_nonce,
    }
    ack = value.get("ack")
    if not isinstance(ack, dict) or set(ack) != {*expected_ack, "server_timing"}:
        raise ValueError("controlled shutdown ACK keys mismatch")
    server_timing = ack.get("server_timing")
    if (
        not isinstance(server_timing, dict)
        or set(server_timing) != {"infer_ms"}
        or isinstance(server_timing.get("infer_ms"), bool)
        or not isinstance(server_timing.get("infer_ms"), (int, float))
        or not math.isfinite(float(server_timing["infer_ms"]))
        or float(server_timing["infer_ms"]) < 0.0
    ):
        raise ValueError("controlled shutdown ACK server_timing mismatch")
    ack_without_timing = {key: ack[key] for key in expected_ack}
    if value.get("request") != expected_request or ack_without_timing != expected_ack:
        raise ValueError("controlled shutdown request/ack mismatch")
    if value.get("uri") != "ws://127.0.0.1:16000":
        raise ValueError("controlled shutdown must use the fixed loopback endpoint")
    if type(value.get("server_pid")) is not int or value["server_pid"] <= 0:
        raise ValueError("controlled shutdown server_pid must be positive")
    descriptor = value.get("runtime_receipt")
    if not isinstance(descriptor, dict):
        raise TypeError("controlled shutdown runtime_receipt must be a descriptor")
    _descriptor_semantics(descriptor)
    if runtime_descriptor is not None and descriptor != runtime_descriptor:
        raise ValueError("controlled shutdown runtime receipt descriptor mismatch")
    validation = value.get("validation")
    if not isinstance(validation, dict) or not validation or any(flag is not True for flag in validation.values()):
        raise ValueError("controlled shutdown validation flags did not all pass")
    return dict(value)


def shutdown_server(
    *,
    uri: str,
    output: Path,
    timeout_seconds: float,
    expected_launch_nonce: str,
    runtime_receipt: Path,
) -> dict[str, Any]:
    """Request nonce-bound shutdown on the evaluated server and capture its ACK."""

    import msgpack
    import websockets.sync.client
    from capture_robocasa365_server_receipt import _validate_metadata

    if uri != "ws://127.0.0.1:16000":
        raise ValueError("controlled shutdown must use ws://127.0.0.1:16000")
    if timeout_seconds <= 0:
        raise ValueError("shutdown timeout_seconds must be positive")
    expected_launch_nonce = _require_sha256(
        expected_launch_nonce, "expected_launch_nonce"
    )
    runtime, runtime_descriptor = _load_platform_receipt(runtime_receipt)
    if (
        runtime["bindings"]["launch_nonce"] != expected_launch_nonce
        or runtime["topology"]["policy_server_uri"] != uri
    ):
        raise ValueError("controlled shutdown input differs from the runtime receipt")
    with websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        open_timeout=timeout_seconds,
        close_timeout=timeout_seconds,
        ping_interval=20,
        ping_timeout=timeout_seconds,
    ) as connection:
        metadata_wire = connection.recv(timeout=timeout_seconds)
        if not isinstance(metadata_wire, bytes):
            raise TypeError("controlled shutdown expected binary server metadata")
        metadata = msgpack.unpackb(metadata_wire, raw=False)
        if not isinstance(metadata, dict):
            raise TypeError("controlled shutdown server metadata must be a map")
        _validate_metadata(metadata)
        if metadata != runtime["policy_server"]:
            raise ValueError("controlled shutdown connected to metadata differing from runtime capture")
        request = {
            "__cwam_control__": "shutdown",
            "launch_nonce": expected_launch_nonce,
        }
        connection.send(msgpack.packb(request, use_bin_type=True))
        ack_wire = connection.recv(timeout=timeout_seconds)
        if not isinstance(ack_wire, bytes):
            raise TypeError("controlled shutdown expected a binary ACK")
        ack = msgpack.unpackb(ack_wire, raw=False)
    expected_ack_without_timing = {
        "actions": [],
        "cwam_control": "shutdown_ack",
        "launch_nonce": expected_launch_nonce,
    }
    if not isinstance(ack, dict) or set(ack) != {
        *expected_ack_without_timing,
        "server_timing",
    }:
        raise ValueError(f"controlled shutdown ACK mismatch: {ack!r}")
    server_timing = ack.get("server_timing")
    if (
        not isinstance(server_timing, dict)
        or set(server_timing) != {"infer_ms"}
        or isinstance(server_timing.get("infer_ms"), bool)
        or not isinstance(server_timing.get("infer_ms"), (int, float))
        or not math.isfinite(float(server_timing["infer_ms"]))
        or float(server_timing["infer_ms"]) < 0.0
    ):
        raise ValueError(
            f"controlled shutdown ACK has invalid server_timing: {server_timing!r}"
        )
    if {key: ack[key] for key in expected_ack_without_timing} != expected_ack_without_timing:
        raise ValueError(f"controlled shutdown ACK mismatch: {ack!r}")
    receipt = {
        "schema_version": 1,
        "kind": _SHUTDOWN_KIND,
        "status": "passed",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "uri": uri,
        "launch_nonce": expected_launch_nonce,
        "server_pid": runtime["bindings"]["rank0_server_pid"],
        "runtime_receipt": runtime_descriptor,
        "request": request,
        "ack": ack,
        "validation": {
            "loopback_endpoint_matches_runtime": True,
            "live_metadata_exactly_matches_runtime": True,
            "launch_nonce_matches": True,
            "shutdown_ack_and_server_timing_match_wire_contract": True,
        },
    }
    _validate_shutdown_receipt(receipt, runtime_descriptor=runtime_descriptor)
    _atomic_write_exclusive(output, receipt)
    return receipt


def _validate_terminal_receipt(
    value: Mapping[str, Any],
    *,
    runtime_descriptor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_exact_keys(
        value,
        frozenset(
            {
                "captured_at_utc",
                "kind",
                "launch_nonce",
                "runtime_receipt",
                "schema_version",
                "server_log",
                "shutdown_receipt",
                "status",
                "torchrun_exit_code",
                "torchrun_launcher_pid",
                "validation",
            }
        ),
        "server terminal receipt",
    )
    if (
        value.get("schema_version") != 1
        or value.get("kind") != _TERMINAL_KIND
        or value.get("status") != "passed"
        or value.get("torchrun_exit_code") != 0
    ):
        raise ValueError("server terminal receipt schema/kind/status/exit code mismatch")
    _require_sha256(value.get("launch_nonce"), "server terminal launch_nonce")
    for name in ("runtime_receipt", "shutdown_receipt", "server_log"):
        descriptor = value.get(name)
        if not isinstance(descriptor, dict):
            raise TypeError(f"server terminal {name} must be a descriptor")
        _descriptor_semantics(descriptor)
    if runtime_descriptor is not None and value["runtime_receipt"] != runtime_descriptor:
        raise ValueError("server terminal runtime receipt descriptor mismatch")
    if type(value.get("torchrun_launcher_pid")) is not int or value["torchrun_launcher_pid"] <= 0:
        raise ValueError("server terminal torchrun_launcher_pid must be positive")
    validation = value.get("validation")
    if not isinstance(validation, dict) or not validation or any(flag is not True for flag in validation.values()):
        raise ValueError("server terminal validation flags did not all pass")
    return dict(value)


def capture_terminal(
    *,
    output: Path,
    runtime_receipt: Path,
    shutdown_receipt: Path,
    server_log: Path,
    torchrun_launcher_pid: int,
    torchrun_exit_code: int,
) -> dict[str, Any]:
    """Bind a successful shutdown ACK to torchrun exit 0 and the final log."""

    runtime, runtime_descriptor = _load_platform_receipt(runtime_receipt)
    shutdown_value, shutdown_descriptor = _read_json(
        shutdown_receipt,
        expected_mode=0o444,
    )
    shutdown = _validate_shutdown_receipt(
        shutdown_value,
        runtime_descriptor=runtime_descriptor,
    )
    if torchrun_launcher_pid != runtime["bindings"]["torchrun_launcher_pid"]:
        raise ValueError("terminal torchrun launcher PID differs from runtime capture")
    if torchrun_exit_code != 0:
        raise ValueError(f"torchrun did not exit successfully: {torchrun_exit_code}")
    log_descriptor = _stable_sha256(server_log, expected_mode=0o444)
    if log_descriptor["size_bytes"] <= 0:
        raise ValueError("policy server torchrun log is empty")
    receipt = {
        "schema_version": 1,
        "kind": _TERMINAL_KIND,
        "status": "passed",
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "launch_nonce": runtime["bindings"]["launch_nonce"],
        "torchrun_launcher_pid": torchrun_launcher_pid,
        "torchrun_exit_code": torchrun_exit_code,
        "runtime_receipt": runtime_descriptor,
        "shutdown_receipt": shutdown_descriptor,
        "server_log": log_descriptor,
        "validation": {
            "shutdown_receipt_matches_runtime": True,
            "shutdown_ack_matches_launch_nonce": (
                shutdown["launch_nonce"] == runtime["bindings"]["launch_nonce"]
            ),
            "torchrun_launcher_pid_matches_runtime": True,
            "torchrun_exit_code_is_zero": True,
            "server_log_hashed_after_process_exit": True,
        },
    }
    _validate_terminal_receipt(receipt, runtime_descriptor=runtime_descriptor)
    _atomic_write_exclusive(output, receipt)
    return receipt


def _build_collection_contract(
    *,
    experiment_id: str,
    logical_run_id: str,
    run_name: str,
    results_dir: Path,
    checkpoint_path: Path,
    checkpoint_manifest_sha256: str,
    training_config_sha256: str,
    action_stats_sha256: str,
    wan2pt2_vae_sha256: str,
    demian_commit: str,
    base_seed: int,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "logical_run_id": logical_run_id,
        "checkpoint_path": str(checkpoint_path),
        "run_name": run_name,
        "results_dir": str(results_dir),
        "evaluation": {
            "role": "primary",
            "split": "atomic",
            "tasks": list(existing_aggregator.ATOMIC_TASKS),
            "episodes_per_task": 20,
            "base_seed": base_seed,
            "max_steps": 400,
            "open_loop_horizon": 12,
            "num_shards": 8,
        },
        "_verified_artifacts": {
            "checkpoint_manifest": {"sha256": checkpoint_manifest_sha256},
            "training_config": {"sha256": training_config_sha256},
            "action_stats": {"sha256": action_stats_sha256},
            "wan2pt2_vae": {"sha256": wan2pt2_vae_sha256},
        },
        "demian": {"commit": demian_commit},
    }


def _assert_exact_counts(
    episode_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> None:
    if len(episode_rows) != _EXPECTED_EPISODE_COUNT:
        raise ValueError(f"expected exactly 340 episode rows, got {len(episode_rows)}")
    if len(task_rows) != _EXPECTED_TASK_COUNT:
        raise ValueError(f"expected exactly 17 task rows, got {len(task_rows)}")
    tasks = [str(row["task"]) for row in task_rows]
    if tasks != list(existing_aggregator.ATOMIC_TASKS):
        raise ValueError("task_success_rates rows do not match the canonical 17 atomic tasks")
    for row in task_rows:
        if int(row["episodes"]) != 20 or row["execution_complete"] != "true":
            raise ValueError(f"task row is incomplete: {row['task']}")
        task_index = int(row["task_index"])
        if int(row["shard"]) != task_index % 8:
            raise ValueError(f"task row logical shard mismatch: {row['task']}")
    per_task: dict[str, int] = {}
    for row in episode_rows:
        task = str(row["task"])
        per_task[task] = per_task.get(task, 0) + 1
    if set(per_task) != set(existing_aggregator.ATOMIC_TASKS) or set(per_task.values()) != {20}:
        raise ValueError("summary.csv must contain exactly 20 episodes for every canonical atomic task")


def _csv_bytes(fieldnames: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _load_platform_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value, descriptor = _read_json(path, expected_mode=0o444)
    return validate_runtime_receipt(value), descriptor


def _load_terminal_receipt(
    path: Path,
    *,
    runtime_descriptor: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, descriptor = _read_json(path, expected_mode=0o444)
    return (
        _validate_terminal_receipt(value, runtime_descriptor=runtime_descriptor),
        descriptor,
    )


def _assert_pre_post_provenance_match(
    *,
    pre_values: Mapping[str, Mapping[str, Any]],
    post_values: Mapping[str, Mapping[str, Any]],
) -> None:
    pairs = (
        ("source_snapshot_pre", "source_snapshot_post"),
        ("static_inputs_pre", "static_inputs_post"),
        ("cosmos_environment_pre", "cosmos_environment_post"),
        ("robocasa_environment_pre", "robocasa_environment_post"),
        ("robocasa_assets_pre", "robocasa_assets_post"),
        ("qwen_tokenizer_pre", "qwen_tokenizer_post"),
    )
    for pre_name, post_name in pairs:
        before = dict(pre_values[pre_name])
        after = dict(post_values[post_name])
        before.pop("phase", None)
        after.pop("phase", None)
        if before != after:
            raise ValueError(
                f"pre/post provenance semantics differ: {pre_name} vs {post_name}"
            )
        if before.get("status") not in (None, "passed"):
            raise ValueError(f"pre-rollout provenance did not pass: {pre_name}")
        if after.get("status") not in (None, "passed"):
            raise ValueError(f"post-rollout provenance did not pass: {post_name}")
    gate = post_values.get("one_episode_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("schema_version") != 1
        or gate.get("kind") != _GATE_KIND
        or gate.get("status") != "passed"
        or gate.get("role") != "preflight_only_not_part_of_340_primary_metric"
        or gate.get("episode_count") != 1
    ):
        raise ValueError("one-episode preflight gate receipt is missing or invalid")


def summarize(
    *,
    experiment_id: str,
    logical_run_id: str,
    source_commit: str,
    run_name: str,
    results_dir: Path,
    checkpoint_path: Path,
    checkpoint_manifest_sha256: str,
    training_config_sha256: str,
    action_stats_sha256: str,
    wan2pt2_vae_sha256: str,
    qwen_tokenizer_manifest_sha256: str,
    source_manifest_sha256: str,
    demian_commit: str,
    base_seed: int,
    runtime_receipt: Path,
    terminal_receipt: Path,
    post_provenance_receipts: Mapping[str, Path],
    summary_csv: Path,
    task_success_rates_csv: Path,
    evidence_json: Path,
) -> dict[str, Any]:
    source_commit = _require_commit(source_commit, "source_commit")
    demian_commit = _require_commit(demian_commit, "demian_commit")
    digests = {
        "checkpoint_manifest_sha256": _require_sha256(checkpoint_manifest_sha256, "checkpoint_manifest_sha256"),
        "training_config_sha256": _require_sha256(training_config_sha256, "training_config_sha256"),
        "action_stats_sha256": _require_sha256(action_stats_sha256, "action_stats_sha256"),
        "wan2pt2_vae_sha256": _require_sha256(wan2pt2_vae_sha256, "wan2pt2_vae_sha256"),
        "qwen_tokenizer_manifest_sha256": _require_sha256(
            qwen_tokenizer_manifest_sha256,
            "qwen_tokenizer_manifest_sha256",
        ),
        "source_manifest_sha256": _require_sha256(source_manifest_sha256, "source_manifest_sha256"),
    }
    runtime, runtime_descriptor = _load_platform_receipt(runtime_receipt)
    bindings = runtime.get("bindings")
    if not isinstance(bindings, dict):
        raise TypeError("DSW runtime receipt bindings must be an object")
    runtime_digest_bindings = {
        "checkpoint_manifest_sha256": bindings["checkpoint"]["manifest"]["sha256"],
        "training_config_sha256": bindings["training_config"]["sha256"],
        "action_stats_sha256": bindings["action_stats"]["sha256"],
        "wan2pt2_vae_sha256": bindings["wan2pt2_vae"]["sha256"],
        "qwen_tokenizer_manifest_sha256": bindings["qwen_tokenizer_manifest"]["sha256"],
        "source_manifest_sha256": bindings["source_manifest"]["sha256"],
    }
    for key, value in runtime_digest_bindings.items():
        if value != digests[key]:
            raise ValueError(f"DSW runtime receipt {key} does not match aggregation inputs")
    if bindings.get("demian_commit") != demian_commit:
        raise ValueError("DSW runtime receipt DeMiAn commit does not match aggregation input")
    if bindings.get("source_commit") != source_commit:
        raise ValueError("DSW runtime receipt source commit does not match aggregation input")
    if bindings["checkpoint"]["path"] != str(checkpoint_path.expanduser().resolve()):
        raise ValueError("DSW runtime receipt checkpoint path does not match aggregation input")
    terminal, terminal_descriptor = _load_terminal_receipt(
        terminal_receipt,
        runtime_descriptor=runtime_descriptor,
    )
    if terminal["launch_nonce"] != bindings["launch_nonce"]:
        raise ValueError("terminal receipt launch nonce differs from runtime")
    post_values, post_descriptors = _load_receipt_descriptors(
        post_provenance_receipts,
        expected_labels=_REQUIRED_POST_PROVENANCE,
        label="post-rollout provenance",
    )
    pre_descriptors = runtime["provenance"]["receipts"]
    verified_pre_descriptors = _verify_bound_descriptors(
        pre_descriptors,
        expected_labels=_REQUIRED_RUNTIME_PROVENANCE,
        label="runtime provenance",
    )
    pre_values: dict[str, dict[str, Any]] = {}
    for name, descriptor in verified_pre_descriptors.items():
        value, observed = _read_json(Path(descriptor["path"]), expected_mode=0o444)
        if observed != descriptor:
            raise ValueError(f"runtime provenance descriptor changed for {name}")
        pre_values[name] = value
    _assert_pre_post_provenance_match(
        pre_values=pre_values,
        post_values=post_values,
    )

    results_dir = results_dir.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    contract = _build_collection_contract(
        experiment_id=experiment_id,
        logical_run_id=logical_run_id,
        run_name=run_name,
        results_dir=results_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        training_config_sha256=training_config_sha256,
        action_stats_sha256=action_stats_sha256,
        wan2pt2_vae_sha256=wan2pt2_vae_sha256,
        demian_commit=demian_commit,
        base_seed=base_seed,
    )
    # Reuse the audited H implementation only for raw DeMiAn result parsing,
    # seed derivation, and CSV rendering. DSW runtime/topology validation stays
    # in this platform-specific module.
    episode_rows, task_rows, raw_shards = existing_aggregator._collect_rows(contract, results_dir)
    _assert_exact_counts(episode_rows, task_rows)
    summary_csv = existing_aggregator._exclusive_artifact_path(summary_csv, "summary CSV")
    task_success_rates_csv = existing_aggregator._exclusive_artifact_path(
        task_success_rates_csv, "task success rates CSV"
    )
    evidence_json = existing_aggregator._exclusive_artifact_path(evidence_json, "DSW evaluation evidence")
    existing_aggregator._atomic_write_csv(
        summary_csv,
        existing_aggregator._CSV_EPISODE_FIELDS,
        episode_rows,
    )
    existing_aggregator._atomic_write_csv(
        task_success_rates_csv,
        existing_aggregator._CSV_TASK_FIELDS,
        task_rows,
    )
    summary_descriptor = _stable_sha256(summary_csv, expected_mode=0o444)
    task_descriptor = _stable_sha256(task_success_rates_csv, expected_mode=0o444)
    if Path(summary_descriptor["path"]).read_bytes() != _csv_bytes(
        existing_aggregator._CSV_EPISODE_FIELDS, episode_rows
    ):
        raise RuntimeError("summary.csv differs immediately after existing aggregator publication")
    if Path(task_descriptor["path"]).read_bytes() != _csv_bytes(existing_aggregator._CSV_TASK_FIELDS, task_rows):
        raise RuntimeError("task_success_rates.csv differs immediately after existing aggregator publication")

    success_rates = [float(row["success_rate"]) for row in task_rows]
    evidence = {
        "schema_version": 1,
        "kind": _EVIDENCE_KIND,
        "experiment_id": experiment_id,
        "logical_run_id": logical_run_id,
        "source_commit": source_commit,
        "run_name": run_name,
        "results_dir": str(results_dir),
        "checkpoint_path": str(checkpoint_path),
        "protocol": {
            "split": "atomic",
            "task_count": 17,
            "episodes_per_task": 20,
            "episode_count": 340,
            "logical_num_shards": 8,
            "base_seed": base_seed,
            "max_steps": 400,
            "open_loop_horizon": 12,
        },
        "execution": {
            "platform": dict(_PLATFORM),
            "physical_server_count": 1,
            "physical_policy_server_count": 1,
            "physical_gpu_count": 4,
            "distributed_world_size": 4,
            "logical_num_shards": 8,
            "evaluator_schedule": "sequential_shards_0_through_7",
        },
        "evaluation": {
            "execution_complete": True,
            "mean_success_rate": sum(success_rates) / len(success_rates),
            "successful_tasks": sum(rate > 0.0 for rate in success_rates),
        },
        "inputs": {
            **digests,
            "demian_commit": demian_commit,
            "runtime_bindings": bindings,
        },
        "artifacts": {
            "runtime_receipt": runtime_descriptor,
            "terminal_receipt": terminal_descriptor,
            "pre_provenance_receipts": verified_pre_descriptors,
            "post_provenance_receipts": post_descriptors,
            "summary_csv": summary_descriptor,
            "task_success_rates_csv": task_descriptor,
            "raw_shards": raw_shards,
            "aggregator_implementation": _stable_sha256(Path(existing_aggregator.__file__).resolve()),
        },
        "validation": {
            "existing_cwam_aggregator_used": True,
            "canonical_atomic_task_order_matches": True,
            "all_eight_logical_shards_present": True,
            "exactly_17_tasks_present": True,
            "exactly_20_episodes_per_task": True,
            "exactly_340_episodes_present": True,
            "no_task_error_or_incomplete_result": True,
            "summary_csv_published_mode_0444": True,
            "task_success_rates_csv_published_mode_0444": True,
            "controlled_server_shutdown_acknowledged": True,
            "torchrun_exited_zero_and_log_was_hashed": True,
            "all_pre_and_post_provenance_receipts_match": True,
        },
    }
    _atomic_write_exclusive(evidence_json, evidence)
    return evidence


def verify_sealed(
    *,
    evidence_json: Path,
    runtime_receipt: Path,
    terminal_receipt: Path,
    results_dir: Path,
    summary_csv: Path,
    task_success_rates_csv: Path,
    output: Path,
) -> dict[str, Any]:
    evidence, evidence_descriptor = _read_json(evidence_json, expected_mode=0o444)
    _require_exact_keys(
        evidence,
        frozenset(
            {
                "artifacts",
                "checkpoint_path",
                "evaluation",
                "execution",
                "experiment_id",
                "inputs",
                "kind",
                "logical_run_id",
                "protocol",
                "results_dir",
                "run_name",
                "schema_version",
                "source_commit",
                "validation",
            }
        ),
        "DSW evaluation evidence",
    )
    if evidence.get("schema_version") != 1 or evidence.get("kind") != _EVIDENCE_KIND:
        raise ValueError("unsupported DSW evaluation evidence schema/kind")
    runtime, runtime_descriptor = _load_platform_receipt(runtime_receipt)
    _reverify_runtime_egl_closure(runtime)
    terminal, terminal_descriptor = _load_terminal_receipt(
        terminal_receipt,
        runtime_descriptor=runtime_descriptor,
    )
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("DSW evaluation evidence artifacts must be an object")
    _require_exact_keys(
        artifacts,
        frozenset(
            {
                "aggregator_implementation",
                "post_provenance_receipts",
                "pre_provenance_receipts",
                "raw_shards",
                "runtime_receipt",
                "summary_csv",
                "task_success_rates_csv",
                "terminal_receipt",
            }
        ),
        "DSW evaluation evidence artifacts",
    )
    if artifacts.get("runtime_receipt") != runtime_descriptor:
        raise ValueError("DSW runtime receipt descriptor changed after aggregation")
    if artifacts.get("terminal_receipt") != terminal_descriptor:
        raise ValueError("DSW terminal receipt descriptor changed after aggregation")
    protocol = evidence.get("protocol")
    inputs = evidence.get("inputs")
    if not isinstance(protocol, dict) or not isinstance(inputs, dict):
        raise TypeError("DSW evaluation evidence protocol/inputs must be objects")
    expected_protocol = {
        "split": "atomic",
        "task_count": 17,
        "episodes_per_task": 20,
        "episode_count": 340,
        "logical_num_shards": 8,
        "base_seed": protocol.get("base_seed"),
        "max_steps": 400,
        "open_loop_horizon": 12,
    }
    if protocol != expected_protocol or type(protocol.get("base_seed")) is not int:
        raise ValueError("DSW evaluation evidence protocol differs from exact 17x20 contract")
    expected_execution = {
        "platform": dict(_PLATFORM),
        "physical_server_count": 1,
        "physical_policy_server_count": 1,
        "physical_gpu_count": 4,
        "distributed_world_size": 4,
        "logical_num_shards": 8,
        "evaluator_schedule": "sequential_shards_0_through_7",
    }
    if evidence.get("execution") != expected_execution:
        raise ValueError("DSW evaluation evidence execution topology mismatch")
    _require_exact_keys(
        inputs,
        frozenset(
            {
                "action_stats_sha256",
                "checkpoint_manifest_sha256",
                "demian_commit",
                "qwen_tokenizer_manifest_sha256",
                "runtime_bindings",
                "source_manifest_sha256",
                "training_config_sha256",
                "wan2pt2_vae_sha256",
            }
        ),
        "DSW evaluation evidence inputs",
    )
    bindings = runtime.get("bindings")
    if not isinstance(bindings, dict):
        raise TypeError("DSW runtime receipt bindings must be an object")
    if inputs.get("runtime_bindings") != bindings:
        raise ValueError("sealed DSW runtime bindings differ from evaluation evidence")
    runtime_digest_bindings = {
        "checkpoint_manifest_sha256": bindings["checkpoint"]["manifest"]["sha256"],
        "training_config_sha256": bindings["training_config"]["sha256"],
        "action_stats_sha256": bindings["action_stats"]["sha256"],
        "wan2pt2_vae_sha256": bindings["wan2pt2_vae"]["sha256"],
        "qwen_tokenizer_manifest_sha256": bindings["qwen_tokenizer_manifest"]["sha256"],
        "source_manifest_sha256": bindings["source_manifest"]["sha256"],
    }
    for key, runtime_digest in runtime_digest_bindings.items():
        digest = _require_sha256(str(inputs[key]), key)
        if runtime_digest != digest:
            raise ValueError(f"sealed DSW runtime receipt {key} does not match evaluation evidence")
    if bindings.get("demian_commit") != inputs.get("demian_commit"):
        raise ValueError("sealed DSW runtime receipt DeMiAn commit does not match evaluation evidence")
    if bindings.get("source_commit") != evidence.get("source_commit"):
        raise ValueError("sealed DSW source commit differs from runtime binding")
    if (
        bindings["checkpoint"]["path"] != evidence.get("checkpoint_path")
        or evidence.get("results_dir") != str(results_dir.expanduser().resolve())
    ):
        raise ValueError("sealed DSW checkpoint/results paths differ from runtime arguments")

    pre_descriptors = artifacts.get("pre_provenance_receipts")
    post_descriptors = artifacts.get("post_provenance_receipts")
    if not isinstance(pre_descriptors, dict) or not isinstance(post_descriptors, dict):
        raise TypeError("sealed DSW provenance descriptor sets must be objects")
    verified_pre = _verify_bound_descriptors(
        pre_descriptors,
        expected_labels=_REQUIRED_RUNTIME_PROVENANCE,
        label="sealed pre-rollout provenance",
    )
    verified_post = _verify_bound_descriptors(
        post_descriptors,
        expected_labels=_REQUIRED_POST_PROVENANCE,
        label="sealed post-rollout provenance",
    )
    if verified_pre != runtime["provenance"]["receipts"]:
        raise ValueError("sealed pre-rollout provenance differs from runtime")
    pre_values = {
        name: _read_json(Path(descriptor["path"]), expected_mode=0o444)[0]
        for name, descriptor in verified_pre.items()
    }
    post_values = {
        name: _read_json(Path(descriptor["path"]), expected_mode=0o444)[0]
        for name, descriptor in verified_post.items()
    }
    _assert_pre_post_provenance_match(
        pre_values=pre_values,
        post_values=post_values,
    )

    if _stable_sha256(
        Path(terminal["server_log"]["path"]),
        expected_mode=0o444,
    ) != terminal["server_log"]:
        raise ValueError("sealed policy server log descriptor changed")
    shutdown_value, shutdown_descriptor = _read_json(
        Path(terminal["shutdown_receipt"]["path"]),
        expected_mode=0o444,
    )
    _validate_shutdown_receipt(
        shutdown_value,
        runtime_descriptor=runtime_descriptor,
    )
    if shutdown_descriptor != terminal["shutdown_receipt"]:
        raise ValueError("sealed controlled shutdown receipt descriptor changed")

    contract = _build_collection_contract(
        experiment_id=str(evidence["experiment_id"]),
        logical_run_id=str(evidence["logical_run_id"]),
        run_name=str(evidence["run_name"]),
        results_dir=results_dir.expanduser().resolve(),
        checkpoint_path=Path(str(evidence["checkpoint_path"])).expanduser().resolve(),
        checkpoint_manifest_sha256=_require_sha256(
            str(inputs["checkpoint_manifest_sha256"]), "checkpoint_manifest_sha256"
        ),
        training_config_sha256=_require_sha256(str(inputs["training_config_sha256"]), "training_config_sha256"),
        action_stats_sha256=_require_sha256(str(inputs["action_stats_sha256"]), "action_stats_sha256"),
        wan2pt2_vae_sha256=_require_sha256(str(inputs["wan2pt2_vae_sha256"]), "wan2pt2_vae_sha256"),
        demian_commit=_require_commit(str(inputs["demian_commit"]), "demian_commit"),
        base_seed=int(protocol["base_seed"]),
    )
    episode_rows, task_rows, raw_shards = existing_aggregator._collect_rows(
        contract,
        results_dir.expanduser().resolve(),
        expected_raw_mode=0o444,
    )
    _assert_exact_counts(episode_rows, task_rows)
    summary_descriptor = _stable_sha256(summary_csv, expected_mode=0o444)
    task_descriptor = _stable_sha256(task_success_rates_csv, expected_mode=0o444)
    if artifacts.get("summary_csv") != summary_descriptor:
        raise ValueError("summary.csv descriptor changed after aggregation")
    if artifacts.get("task_success_rates_csv") != task_descriptor:
        raise ValueError("task_success_rates.csv descriptor changed after aggregation")
    if Path(summary_descriptor["path"]).read_bytes() != _csv_bytes(
        existing_aggregator._CSV_EPISODE_FIELDS, episode_rows
    ):
        raise ValueError("sealed summary.csv does not match independent reaggregation")
    if Path(task_descriptor["path"]).read_bytes() != _csv_bytes(existing_aggregator._CSV_TASK_FIELDS, task_rows):
        raise ValueError("sealed task_success_rates.csv does not match independent reaggregation")
    if artifacts.get("raw_shards") != raw_shards:
        raise ValueError("sealed raw shard descriptors changed after aggregation")
    success_rates = [float(row["success_rate"]) for row in task_rows]
    evaluation = evidence.get("evaluation")
    if not isinstance(evaluation, dict):
        raise TypeError("DSW evaluation field must be an object")
    _require_exact_keys(
        evaluation,
        frozenset({"execution_complete", "mean_success_rate", "successful_tasks"}),
        "DSW evaluation evidence evaluation",
    )
    if evaluation.get("execution_complete") is not True:
        raise ValueError("DSW evaluation evidence is not execution-complete")
    mean_success_rate = sum(success_rates) / len(success_rates)
    if not math.isclose(
        float(evaluation.get("mean_success_rate")),
        mean_success_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("DSW evaluation mean success rate changed during sealed verification")
    successful_tasks = sum(rate > 0.0 for rate in success_rates)
    if evaluation.get("successful_tasks") != successful_tasks:
        raise ValueError("DSW evaluation successful_tasks changed during sealed verification")
    validation = evidence.get("validation")
    if not isinstance(validation, dict) or not validation or any(flag is not True for flag in validation.values()):
        raise ValueError("DSW evaluation evidence validation flags did not all pass")
    receipt = {
        "schema_version": 1,
        "kind": _SEALED_KIND,
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "passed",
        "evaluation_evidence": evidence_descriptor,
        "runtime_receipt": runtime_descriptor,
        "terminal_receipt": terminal_descriptor,
        "pre_provenance_receipts": verified_pre,
        "post_provenance_receipts": verified_post,
        "summary_csv": summary_descriptor,
        "task_success_rates_csv": task_descriptor,
        "counts": {
            "physical_server_count": runtime["topology"]["physical_server_count"],
            "physical_gpu_count": runtime["topology"]["physical_gpu_count"],
            "logical_num_shards": runtime["topology"]["logical_num_shards"],
            "tasks": len(task_rows),
            "episodes": len(episode_rows),
        },
        "validation": {
            "raw_files_mode_0444_and_reaggregated": True,
            "summary_csv_mode_0444_content_and_hash_match": True,
            "task_success_rates_csv_mode_0444_content_and_hash_match": True,
            "runtime_receipt_mode_0444_content_and_hash_match": True,
            "terminal_receipt_and_server_log_mode_0444_content_and_hash_match": True,
            "evaluation_evidence_mode_0444_content_and_hash_match": True,
            "exact_17_by_20_protocol_complete": True,
            "logical_shards_remain_8_on_physical_fsdp4_server": True,
            "successful_tasks_recomputed": True,
            "pre_and_post_provenance_reverified": True,
            "egl_receipt_vendor_probe_and_36_evidence_files_reverified": True,
        },
    }
    _atomic_write_exclusive(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    qwen = subparsers.add_parser("verify-qwen-tokenizer")
    qwen.add_argument("--tokenizer-path", type=Path, required=True)
    qwen.add_argument("--manifest-path", type=Path, required=True)
    qwen.add_argument("--expected-manifest-sha256", required=True)
    qwen.add_argument("--output", type=Path, required=True)

    gate = subparsers.add_parser("verify-one-episode-gate")
    gate.add_argument("--gate-results-dir", type=Path, required=True)
    gate.add_argument("--checkpoint-path", type=Path, required=True)
    gate.add_argument("--run-name", required=True)
    gate.add_argument("--task", required=True)
    gate.add_argument("--runtime-receipt", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)

    capture = subparsers.add_parser("capture-runtime")
    capture.add_argument("--uri", required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--timeout-seconds", type=float, default=1800.0)
    capture.add_argument("--launcher-pid", type=int, required=True)
    capture.add_argument("--expected-launch-nonce", required=True)
    capture.add_argument("--checkpoint-path", type=Path, required=True)
    capture.add_argument("--checkpoint-manifest-path", type=Path, required=True)
    capture.add_argument("--checkpoint-manifest-sha256", required=True)
    capture.add_argument("--checkpoint-tree-sha256", required=True)
    capture.add_argument("--training-config-path", type=Path, required=True)
    capture.add_argument("--training-config-sha256", required=True)
    capture.add_argument("--action-stats-path", type=Path, required=True)
    capture.add_argument("--action-stats-sha256", required=True)
    capture.add_argument("--wan2pt2-vae-path", type=Path, required=True)
    capture.add_argument("--wan2pt2-vae-sha256", required=True)
    capture.add_argument("--qwen-tokenizer-path", type=Path, required=True)
    capture.add_argument("--qwen-tokenizer-manifest-path", type=Path, required=True)
    capture.add_argument("--qwen-tokenizer-manifest-sha256", required=True)
    capture.add_argument("--source-manifest-path", type=Path, required=True)
    capture.add_argument("--source-manifest-sha256", required=True)
    capture.add_argument("--source-commit", required=True)
    capture.add_argument("--demian-commit", required=True)
    capture.add_argument(
        "--demian-openpi-server-source-path",
        type=Path,
        required=True,
    )
    capture.add_argument("--robocasa-environment-path", type=Path, required=True)
    capture.add_argument("--robocasa-ready-marker-path", type=Path, required=True)
    capture.add_argument("--robocasa-ready-marker-sha256", required=True)
    capture.add_argument("--egl-vendor-json-path", type=Path, required=True)
    capture.add_argument("--egl-vendor-json-sha256", required=True)
    capture.add_argument("--egl-mapping-receipt-path", type=Path, required=True)
    capture.add_argument("--egl-mapping-receipt-sha256", required=True)
    capture.add_argument("--evaluator-timeout-seconds", type=int, required=True)
    capture.add_argument("--one-episode-gate-timeout-seconds", type=int, required=True)
    capture.add_argument("--server-exit-timeout-seconds", type=int, required=True)
    capture.add_argument(
        "--provenance-receipt",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )

    shutdown = subparsers.add_parser("shutdown-server")
    shutdown.add_argument("--uri", required=True)
    shutdown.add_argument("--output", type=Path, required=True)
    shutdown.add_argument("--timeout-seconds", type=float, default=60.0)
    shutdown.add_argument("--expected-launch-nonce", required=True)
    shutdown.add_argument("--runtime-receipt", type=Path, required=True)

    terminal = subparsers.add_parser("capture-terminal")
    terminal.add_argument("--output", type=Path, required=True)
    terminal.add_argument("--runtime-receipt", type=Path, required=True)
    terminal.add_argument("--shutdown-receipt", type=Path, required=True)
    terminal.add_argument("--server-log", type=Path, required=True)
    terminal.add_argument("--torchrun-launcher-pid", type=int, required=True)
    terminal.add_argument("--torchrun-exit-code", type=int, required=True)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--experiment-id", required=True)
    summary.add_argument("--logical-run-id", required=True)
    summary.add_argument("--source-commit", required=True)
    summary.add_argument("--run-name", required=True)
    summary.add_argument("--results-dir", type=Path, required=True)
    summary.add_argument("--checkpoint-path", type=Path, required=True)
    summary.add_argument("--checkpoint-manifest-sha256", required=True)
    summary.add_argument("--training-config-sha256", required=True)
    summary.add_argument("--action-stats-sha256", required=True)
    summary.add_argument("--wan2pt2-vae-sha256", required=True)
    summary.add_argument("--qwen-tokenizer-manifest-sha256", required=True)
    summary.add_argument("--source-manifest-sha256", required=True)
    summary.add_argument("--demian-commit", required=True)
    summary.add_argument("--base-seed", type=int, required=True)
    summary.add_argument("--runtime-receipt", type=Path, required=True)
    summary.add_argument("--terminal-receipt", type=Path, required=True)
    summary.add_argument(
        "--post-provenance-receipt",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    summary.add_argument("--summary-csv", type=Path, required=True)
    summary.add_argument("--task-success-rates-csv", type=Path, required=True)
    summary.add_argument("--evidence-json", type=Path, required=True)

    sealed = subparsers.add_parser("verify-sealed")
    sealed.add_argument("--evidence-json", type=Path, required=True)
    sealed.add_argument("--runtime-receipt", type=Path, required=True)
    sealed.add_argument("--terminal-receipt", type=Path, required=True)
    sealed.add_argument("--results-dir", type=Path, required=True)
    sealed.add_argument("--summary-csv", type=Path, required=True)
    sealed.add_argument("--task-success-rates-csv", type=Path, required=True)
    sealed.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify-qwen-tokenizer":
        result = verify_qwen_tokenizer_manifest(
            tokenizer_path=args.tokenizer_path,
            manifest_path=args.manifest_path,
            expected_manifest_sha256=args.expected_manifest_sha256,
            output=args.output,
        )
    elif args.command == "verify-one-episode-gate":
        result = verify_one_episode_gate(
            gate_results_dir=args.gate_results_dir,
            checkpoint_path=args.checkpoint_path,
            run_name=args.run_name,
            task=args.task,
            runtime_receipt=args.runtime_receipt,
            output=args.output,
        )
    elif args.command == "capture-runtime":
        result = capture_runtime(
            uri=args.uri,
            output=args.output,
            timeout_seconds=args.timeout_seconds,
            launcher_pid=args.launcher_pid,
            expected_launch_nonce=args.expected_launch_nonce,
            checkpoint_path=args.checkpoint_path,
            checkpoint_manifest_path=args.checkpoint_manifest_path,
            checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
            checkpoint_tree_sha256=args.checkpoint_tree_sha256,
            training_config_path=args.training_config_path,
            training_config_sha256=args.training_config_sha256,
            action_stats_path=args.action_stats_path,
            action_stats_sha256=args.action_stats_sha256,
            wan2pt2_vae_path=args.wan2pt2_vae_path,
            wan2pt2_vae_sha256=args.wan2pt2_vae_sha256,
            qwen_tokenizer_path=args.qwen_tokenizer_path,
            qwen_tokenizer_manifest_path=args.qwen_tokenizer_manifest_path,
            qwen_tokenizer_manifest_sha256=args.qwen_tokenizer_manifest_sha256,
            source_manifest_path=args.source_manifest_path,
            source_manifest_sha256=args.source_manifest_sha256,
            source_commit=args.source_commit,
            demian_commit=args.demian_commit,
            demian_openpi_server_source_path=args.demian_openpi_server_source_path,
            robocasa_environment_path=args.robocasa_environment_path,
            robocasa_ready_marker_path=args.robocasa_ready_marker_path,
            robocasa_ready_marker_sha256=args.robocasa_ready_marker_sha256,
            egl_vendor_json_path=args.egl_vendor_json_path,
            egl_vendor_json_sha256=args.egl_vendor_json_sha256,
            egl_mapping_receipt_path=args.egl_mapping_receipt_path,
            egl_mapping_receipt_sha256=args.egl_mapping_receipt_sha256,
            provenance_receipts=_parse_named_paths(
                args.provenance_receipt,
                "provenance receipt",
            ),
            evaluator_timeout_seconds=args.evaluator_timeout_seconds,
            one_episode_gate_timeout_seconds=args.one_episode_gate_timeout_seconds,
            server_exit_timeout_seconds=args.server_exit_timeout_seconds,
        )
    elif args.command == "shutdown-server":
        result = shutdown_server(
            uri=args.uri,
            output=args.output,
            timeout_seconds=args.timeout_seconds,
            expected_launch_nonce=args.expected_launch_nonce,
            runtime_receipt=args.runtime_receipt,
        )
    elif args.command == "capture-terminal":
        result = capture_terminal(
            output=args.output,
            runtime_receipt=args.runtime_receipt,
            shutdown_receipt=args.shutdown_receipt,
            server_log=args.server_log,
            torchrun_launcher_pid=args.torchrun_launcher_pid,
            torchrun_exit_code=args.torchrun_exit_code,
        )
    elif args.command == "summarize":
        result = summarize(
            experiment_id=args.experiment_id,
            logical_run_id=args.logical_run_id,
            source_commit=args.source_commit,
            run_name=args.run_name,
            results_dir=args.results_dir,
            checkpoint_path=args.checkpoint_path,
            checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
            training_config_sha256=args.training_config_sha256,
            action_stats_sha256=args.action_stats_sha256,
            wan2pt2_vae_sha256=args.wan2pt2_vae_sha256,
            qwen_tokenizer_manifest_sha256=args.qwen_tokenizer_manifest_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
            demian_commit=args.demian_commit,
            base_seed=args.base_seed,
            runtime_receipt=args.runtime_receipt,
            terminal_receipt=args.terminal_receipt,
            post_provenance_receipts=_parse_named_paths(
                args.post_provenance_receipt,
                "post provenance receipt",
            ),
            summary_csv=args.summary_csv,
            task_success_rates_csv=args.task_success_rates_csv,
            evidence_json=args.evidence_json,
        )
    else:
        result = verify_sealed(
            evidence_json=args.evidence_json,
            runtime_receipt=args.runtime_receipt,
            terminal_receipt=args.terminal_receipt,
            results_dir=args.results_dir,
            summary_csv=args.summary_csv,
            task_success_rates_csv=args.task_success_rates_csv,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
