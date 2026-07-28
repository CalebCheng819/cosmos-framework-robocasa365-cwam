#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Capture a fail-closed, normalized receipt from the H-cluster RJob CRDs.

The RJob custom object is authoritative for the submitted scheduler contract.
The related ``replicas`` custom objects expose the actual replica/Pod identity
and runtime status.  This helper deliberately imports Brainpp only in the CLI
adapter so that the validation logic remains deterministic and unit-testable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

_RECEIPT_KIND = "cwam_formal_rjob_scheduler_receipt"
_EXPECTED_SCHEMA_VERSION = 1
_RECEIPT_SCHEMA_VERSION = 1
_TASK_NAME = "t0"
_SHOW_NAME_ANNOTATION = "kubebrain.brainpp.cn/showname"
_RJOB_NAME_LABEL = "rjob.brainpp.cn/rjob-name"
_TASK_NAME_LABEL = "rjob.brainpp.cn/rjob-task-type"
_TASK_TYPE_KEY = "rjob.brainpp.cn/zs-task-type"
_QUOTA_GROUP_LABEL = "quotagroup.brainpp.cn/quotagroup"
_PRIVATE_MACHINE_LABEL = "privatemachine.brainpp.cn/privatemachine"
_PRIORITY_ANNOTATION = "volcano.brainpp.cn/priority"
_SCHEDULING_GROUP_KEY = "scheduling.k8s.io/group-name"
_GPFS_MOUNT_ANNOTATION = "mount.brainpp.cn/gpfs"
_REPLICA_INDEX_LABEL = "kubebrain.brainpp.cn/replica"
_WORKER_INIT_PREFIXES = (
    ("/kubebrain/worker-init", "run", "--enable-pid-monitor", "--"),
    (
        "/kubebrain/worker-init",
        "run",
        "--enable-pid-monitor",
        "--metrics-addr=",
        "--",
    ),
)
_API_REQUEST_TIMEOUT_SECONDS = (5.0, 15.0)
_PLATFORM_INJECTED_LIMIT_KEYS = frozenset(
    {
        "mellanox.com/mlnx_rdma",
        "rdma/mlnx_shared",
    }
)
_FORMAL_WRAPPER_RELATIVE_PATH = PurePosixPath("examples/h_cluster/run_cwam_cosmos3_nano_production_then_formal.sh")
_KUBERNETES_NAME_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_EXPECTED_KEYS = frozenset(
    {
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
    }
)


class ReceiptError(RuntimeError):
    """Base error for receipt capture or validation failures."""


class ReceiptValidationError(ReceiptError):
    """The scheduler object or caller-provided contract is invalid."""


class ReceiptNotReady(ReceiptError):
    """The scheduler has not materialized enough status to bind a worker yet."""


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReceiptValidationError(f"{label} must be a JSON object")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReceiptValidationError(f"{label} must be a JSON array")
    return value


def _require_string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ReceiptValidationError(f"{label} must be a non-empty string")
    return value


def _require_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ReceiptValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ReceiptValidationError(f"{label} must be at least {minimum}, got {value}")
    return value


def _require_exact(observed: object, expected: object, label: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise ReceiptValidationError(f"{label} mismatch: expected={expected!r}, actual={observed!r}")


def _require_metadata_identity(metadata: Mapping[str, Any], label: str) -> dict[str, str]:
    name = _require_string(metadata.get("name"), f"{label}.name")
    uid = _require_string(metadata.get("uid"), f"{label}.uid")
    try:
        uuid.UUID(uid)
    except ValueError as error:
        raise ReceiptValidationError(f"{label}.uid is not a UUID: {uid!r}") from error
    resource_version = _require_string(metadata.get("resourceVersion"), f"{label}.resourceVersion")
    if not resource_version.isdigit():
        raise ReceiptValidationError(f"{label}.resourceVersion must contain only decimal digits")
    creation_timestamp = _require_string(metadata.get("creationTimestamp"), f"{label}.creationTimestamp")
    if not creation_timestamp.endswith("Z"):
        raise ReceiptValidationError(f"{label}.creationTimestamp must be an RFC3339 UTC timestamp")
    try:
        datetime.fromisoformat(creation_timestamp.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ReceiptValidationError(f"{label}.creationTimestamp must be an RFC3339 UTC timestamp") from error
    return {
        "name": name,
        "uid": uid,
        "resource_version": resource_version,
        "creation_timestamp": creation_timestamp,
    }


def _require_absolute_posix_path(value: object, label: str) -> str:
    text = _require_string(value, label)
    path = PurePosixPath(text)
    if not path.is_absolute() or ".." in path.parts or str(path) != text:
        raise ReceiptValidationError(f"{label} must be a normalized absolute POSIX path")
    return text


def validate_expected_contract(value: object) -> dict[str, Any]:
    """Validate and return a defensive copy of the exact expected schema."""

    expected = _require_mapping(value, "expected contract")
    if set(expected) != _EXPECTED_KEYS:
        raise ReceiptValidationError(
            "expected contract key mismatch: "
            f"missing={sorted(_EXPECTED_KEYS - set(expected))}, "
            f"extra={sorted(set(expected) - _EXPECTED_KEYS)}"
        )

    _require_exact(expected["schema_version"], _EXPECTED_SCHEMA_VERSION, "expected.schema_version")
    scheduler_rjob_name = _require_string(expected["scheduler_rjob_name"], "expected.scheduler_rjob_name")
    showname = _require_string(expected["showname"], "expected.showname")
    if _KUBERNETES_NAME_RE.fullmatch(scheduler_rjob_name) is None:
        raise ReceiptValidationError("expected.scheduler_rjob_name is not a lowercase Kubernetes name")
    if _KUBERNETES_NAME_RE.fullmatch(showname) is None:
        raise ReceiptValidationError("expected.showname is not a lowercase Kubernetes name")

    quota_group = _require_string(expected["quota_group"], "expected.quota_group")
    if _SAFE_TOKEN_RE.fullmatch(quota_group) is None:
        raise ReceiptValidationError("expected.quota_group contains unsupported characters")
    _require_exact(expected["task_type"], "normal", "expected.task_type")
    _require_exact(expected["priority"], 9, "expected.priority")
    _require_exact(expected["backoff_limit"], 1, "expected.backoff_limit")
    _require_exact(expected["task_name"], _TASK_NAME, "expected.task_name")
    _require_exact(expected["replicas"], 1, "expected.replicas")
    _require_exact(expected["private_machine"], "group", "expected.private_machine")
    _require_exact(expected["restart_policy"], "Never", "expected.restart_policy")
    _require_exact(
        expected["termination_grace_period_seconds"],
        60,
        "expected.termination_grace_period_seconds",
    )

    image = _require_string(expected["image"], "expected.image")
    if _IMAGE_DIGEST_RE.fullmatch(image) is None:
        raise ReceiptValidationError("expected.image must be repository@sha256:<64 lowercase hex>")
    _require_int(expected["cpu"], "expected.cpu", minimum=1)
    _require_int(expected["memory_mib"], "expected.memory_mib", minimum=1)
    _require_int(expected["gpu"], "expected.gpu", minimum=1)

    negative_tags = _require_list(expected["negative_tags"], "expected.negative_tags")
    if len(negative_tags) != 1:
        raise ReceiptValidationError("expected.negative_tags must contain exactly one formal exclusion")
    _require_string(negative_tags[0], "expected.negative_tags[0]")

    mounts = _require_list(expected["mounts"], "expected.mounts")
    if not mounts:
        raise ReceiptValidationError("expected.mounts must not be empty")
    for index, mount in enumerate(mounts):
        mount_text = _require_string(mount, f"expected.mounts[{index}]")
        if ";" in mount_text:
            raise ReceiptValidationError(f"expected.mounts[{index}] must not contain ';'")
    if len(set(mounts)) != len(mounts):
        raise ReceiptValidationError("expected.mounts must not contain duplicates")

    snapshot_root = _require_absolute_posix_path(expected["snapshot_root"], "expected.snapshot_root")
    wrapper_path = _require_absolute_posix_path(expected["wrapper_path"], "expected.wrapper_path")
    required_wrapper = str(PurePosixPath(snapshot_root) / _FORMAL_WRAPPER_RELATIVE_PATH)
    if wrapper_path != required_wrapper:
        raise ReceiptValidationError(
            f"expected.wrapper_path must be the formal wrapper inside snapshot_root: {required_wrapper}"
        )
    experiment_id = _require_string(expected["experiment_id"], "expected.experiment_id")
    if _SAFE_TOKEN_RE.fullmatch(experiment_id) is None:
        raise ReceiptValidationError("expected.experiment_id contains unsupported characters")
    command_sha256 = _require_string(expected["command_sha256"], "expected.command_sha256")
    if _SHA256_RE.fullmatch(command_sha256) is None:
        raise ReceiptValidationError("expected.command_sha256 must be a lowercase SHA-256")

    return json.loads(json.dumps(expected))


def load_expected_contract(path: Path) -> dict[str, Any]:
    """Read an expected contract once from a non-symlink regular file."""

    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ReceiptValidationError(f"expected contract is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ReceiptValidationError(f"expected contract must be a non-symlink regular file: {path}")
    if metadata.st_size > 1024 * 1024:
        raise ReceiptValidationError("expected contract is unexpectedly larger than 1 MiB")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptValidationError(f"expected contract is not valid UTF-8 JSON: {path}") from error
    return validate_expected_contract(value)


def _normalize_conditions(value: object, label: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    conditions = _require_list(value, label)
    normalized: list[dict[str, Any]] = []
    allowed = (
        "type",
        "status",
        "reason",
        "message",
        "lastProbeTime",
        "lastTransitionTime",
        "lastUpdateTime",
    )
    for index, condition_value in enumerate(conditions):
        condition = _require_mapping(condition_value, f"{label}[{index}]")
        item: dict[str, Any] = {}
        for key in allowed:
            if key in condition:
                observed = condition[key]
                if observed is not None and not isinstance(observed, (str, bool, int, float)):
                    raise ReceiptValidationError(f"{label}[{index}].{key} has an unsupported type")
                item[key] = observed
        normalized.append(item)
    return normalized


def _normalize_json_value(value: object, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [_normalize_json_value(item, f"{label}[]") for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReceiptValidationError(f"{label} contains a non-string JSON object key")
            normalized[key] = _normalize_json_value(item, f"{label}.{key}")
        return normalized
    raise ReceiptValidationError(f"{label} contains a non-JSON value of type {type(value).__name__}")


def _parse_cpu_cores(value: object, label: str) -> int:
    text = _require_string(value, label)
    if text.isdigit():
        return int(text)
    if text.endswith("m") and text[:-1].isdigit() and int(text[:-1]) % 1000 == 0:
        return int(text[:-1]) // 1000
    raise ReceiptValidationError(f"{label} must be an integral Kubernetes CPU quantity")


def _parse_memory_mib(value: object, label: str) -> int:
    text = _require_string(value, label)
    units = {"Mi": 1, "Gi": 1024, "Ti": 1024 * 1024}
    for suffix, multiplier in units.items():
        if text.endswith(suffix) and text[: -len(suffix)].isdigit():
            return int(text[: -len(suffix)]) * multiplier
    raise ReceiptValidationError(f"{label} must be an integral Mi/Gi/Ti Kubernetes memory quantity")


def _parse_gpu_count(value: object, label: str) -> int:
    text = _require_string(value, label)
    if not text.isdigit():
        raise ReceiptValidationError(f"{label} must be an integral GPU quantity")
    return int(text)


def _parse_exports(shell_script: str) -> dict[str, list[str]]:
    exports: dict[str, list[str]] = {}
    for line_number, line in enumerate(shell_script.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            tokens = shlex.split(stripped, posix=True)
        except ValueError as error:
            raise ReceiptValidationError(f"scheduler command line {line_number} is not valid shell syntax") from error
        if not tokens or tokens[0] != "export":
            continue
        for assignment in tokens[1:]:
            if "=" not in assignment:
                raise ReceiptValidationError(f"scheduler command line {line_number} contains a non-assignment export")
            key, value = assignment.split("=", 1)
            if not key:
                raise ReceiptValidationError(f"scheduler command line {line_number} has an empty export name")
            exports.setdefault(key, []).append(value)
    return exports


def _require_one_export(exports: Mapping[str, list[str]], key: str, expected: str) -> None:
    values = exports.get(key, [])
    if values != [expected]:
        raise ReceiptValidationError(
            f"scheduler command must export {key} exactly once with {expected!r}, got {values!r}"
        )


def _validate_command(command_value: object, expected: Mapping[str, Any]) -> list[str]:
    command = _require_list(command_value, "rjob.spec.taskSpecs.t0.template.spec.containers[0].command")
    if len(command) != 3 or command[:2] != ["bash", "-lc"] or not isinstance(command[2], str):
        raise ReceiptValidationError("scheduler command must be exactly ['bash', '-lc', <formal shell script>]")
    shell_script = command[2]
    observed_command_sha256 = hashlib.sha256(shell_script.encode("utf-8")).hexdigest()
    if observed_command_sha256 != expected["command_sha256"]:
        raise ReceiptValidationError(
            "scheduler formal command SHA-256 mismatch: "
            f"expected={expected['command_sha256']}, actual={observed_command_sha256}"
        )
    exports = _parse_exports(shell_script)
    _require_one_export(exports, "COSMOS_SOURCE", expected["snapshot_root"])
    _require_one_export(exports, "FORMAL_EXPERIMENT_ID", expected["experiment_id"])
    _require_one_export(exports, "FORMAL_RUN_NAME", expected["experiment_id"])

    wrapper_invocations: list[list[str]] = []
    for line_number, line in enumerate(shell_script.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError as error:
            raise ReceiptValidationError(f"scheduler command line {line_number} is not valid shell syntax") from error
        if tokens in (
            ["bash", expected["wrapper_path"]],
            ["exec", "bash", expected["wrapper_path"]],
        ):
            wrapper_invocations.append(tokens)
    if len(wrapper_invocations) != 1:
        raise ReceiptValidationError("scheduler command must invoke the exact immutable snapshot wrapper exactly once")
    return [str(item) for item in command]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalize_task_statuses(value: object) -> dict[str, dict[str, int]]:
    if value is None:
        raise ReceiptNotReady("rjob has no taskStatuses yet")
    statuses = _require_mapping(value, "rjob.status.taskStatuses")
    if set(statuses) != {_TASK_NAME}:
        raise ReceiptValidationError(
            f"rjob.status.taskStatuses must contain exactly {_TASK_NAME!r}, got {sorted(statuses)}"
        )
    task = _require_mapping(statuses[_TASK_NAME], f"rjob.status.taskStatuses.{_TASK_NAME}")
    normalized: dict[str, int] = {}
    for key, count in task.items():
        _require_string(key, f"rjob.status.taskStatuses.{_TASK_NAME} key")
        normalized[key] = _require_int(count, f"rjob.status.taskStatuses.{_TASK_NAME}.{key}", minimum=0)
    if sum(normalized.values()) != 1:
        raise ReceiptValidationError("rjob.status.taskStatuses counts must sum to the one expected replica")
    return {_TASK_NAME: dict(sorted(normalized.items()))}


def _normalize_container_statuses(value: object, label: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    statuses = _require_list(value, label)
    normalized: list[dict[str, Any]] = []
    for index, status_value in enumerate(statuses):
        status = _require_mapping(status_value, f"{label}[{index}]")
        normalized.append(
            {
                "name": _require_string(status.get("name"), f"{label}[{index}].name"),
                "ready": status.get("ready"),
                "started": status.get("started"),
                "restart_count": status.get("restartCount"),
                "image": status.get("image"),
                "image_id": status.get("imageID"),
                "state": _normalize_json_value(status.get("state", {}), f"{label}[{index}].state"),
                "last_state": _normalize_json_value(status.get("lastState", {}), f"{label}[{index}].lastState"),
            }
        )
    return normalized


def _normalize_resources(
    resources_value: object,
    expected: Mapping[str, Any],
    *,
    label: str,
    allow_platform_injected_limits: bool,
) -> dict[str, int | bool | list[str]]:
    resources = _require_mapping(resources_value, label)
    requests = _require_mapping(resources.get("requests"), f"{label}.requests")
    limits = _require_mapping(resources.get("limits"), f"{label}.limits")
    resource_keys = {"cpu", "memory", "nvidia.com/gpu"}
    if set(requests) != resource_keys:
        raise ReceiptValidationError(f"{label}.requests must contain exactly cpu, memory, and nvidia.com/gpu")
    extra_limit_keys = set(limits) - resource_keys
    if not resource_keys.issubset(limits):
        raise ReceiptValidationError(f"{label}.limits must contain cpu, memory, and nvidia.com/gpu")
    if extra_limit_keys and (
        not allow_platform_injected_limits or not extra_limit_keys.issubset(_PLATFORM_INJECTED_LIMIT_KEYS)
    ):
        raise ReceiptValidationError(f"{label}.limits contains unsupported extra resources: {sorted(extra_limit_keys)}")
    for key in resource_keys:
        _require_exact(limits[key], requests[key], f"{label} requests/limits {key}")
    observed_resources = {
        "cpu": _parse_cpu_cores(requests["cpu"], f"{label}.requests.cpu"),
        "memory_mib": _parse_memory_mib(requests["memory"], f"{label}.requests.memory"),
        "gpu": _parse_gpu_count(requests["nvidia.com/gpu"], f"{label}.requests.nvidia.com/gpu"),
    }
    for field, value in observed_resources.items():
        _require_exact(value, expected[field], f"{label} normalized {field}")
    return {
        "requests_equal_core_limits": True,
        **observed_resources,
        "platform_injected_limit_keys": sorted(extra_limit_keys),
    }


def _validate_expanded_worker_command(
    value: object,
    expected: Mapping[str, Any],
    template_command: list[str],
) -> tuple[list[str], list[str]]:
    command = _require_list(value, "replica.spec.containers[0].command")
    if not all(isinstance(item, str) for item in command):
        raise ReceiptValidationError("replica.spec.containers[0].command must contain only strings")
    normalized_command = [str(item) for item in command]
    if normalized_command == template_command:
        worker_init_prefix: list[str] = []
        command_suffix = normalized_command
    else:
        matched_prefix = next(
            (
                list(prefix)
                for prefix in _WORKER_INIT_PREFIXES
                if normalized_command[: len(prefix)] == list(prefix)
            ),
            None,
        )
        if matched_prefix is None:
            raise ReceiptValidationError(
                "expanded worker command must be the formal command with only an audited worker-init prefix"
            )
        worker_init_prefix = matched_prefix
        command_suffix = normalized_command[len(matched_prefix) :]
    validated_suffix = _validate_command(command_suffix, expected)
    _require_exact(
        validated_suffix,
        template_command,
        "expanded worker formal command versus RJob template command",
    )
    return normalized_command, worker_init_prefix


def _normalize_replica(
    raw: object,
    expected: Mapping[str, Any],
    template_command: list[str],
) -> dict[str, Any]:
    replica = _require_mapping(raw, "replica")
    metadata = _require_mapping(replica.get("metadata"), "replica.metadata")
    identity = _require_metadata_identity(metadata, "replica.metadata")
    labels = _require_mapping(metadata.get("labels"), "replica.metadata.labels")
    annotations = _require_mapping(metadata.get("annotations"), "replica.metadata.annotations")
    _require_exact(labels.get(_RJOB_NAME_LABEL), expected["scheduler_rjob_name"], f"replica label {_RJOB_NAME_LABEL}")
    _require_exact(labels.get(_TASK_NAME_LABEL), _TASK_NAME, f"replica label {_TASK_NAME_LABEL}")
    _require_exact(labels.get(_QUOTA_GROUP_LABEL), expected["quota_group"], f"replica label {_QUOTA_GROUP_LABEL}")
    _require_exact(
        labels.get(_PRIVATE_MACHINE_LABEL),
        expected["private_machine"],
        f"replica label {_PRIVATE_MACHINE_LABEL}",
    )
    _require_exact(
        labels.get(_TASK_TYPE_KEY),
        f"rjob-{expected['task_type']}",
        f"replica label {_TASK_TYPE_KEY}",
    )
    replica_mount_annotation = _require_string(
        annotations.get(_GPFS_MOUNT_ANNOTATION),
        f"replica annotation {_GPFS_MOUNT_ANNOTATION}",
    )
    replica_mounts = replica_mount_annotation.split(";")
    _require_exact(replica_mounts, expected["mounts"], f"replica annotation {_GPFS_MOUNT_ANNOTATION}")
    replica_index = _require_string(labels.get(_REPLICA_INDEX_LABEL), f"replica label {_REPLICA_INDEX_LABEL}")
    if replica_index != "0":
        raise ReceiptValidationError(f"the only formal replica must have index '0', got {replica_index!r}")

    spec = _require_mapping(replica.get("spec"), "replica.spec")
    _require_exact(spec.get("type"), _TASK_NAME, "replica.spec.type")
    _require_exact(spec.get("restartPolicy"), expected["restart_policy"], "replica.spec.restartPolicy")
    _require_exact(
        spec.get("terminationGracePeriodSeconds"),
        expected["termination_grace_period_seconds"],
        "replica.spec.terminationGracePeriodSeconds",
    )
    replica_containers = _require_list(spec.get("containers"), "replica.spec.containers")
    if len(replica_containers) != 1:
        raise ReceiptValidationError("replica.spec.containers must contain exactly one expanded worker container")
    replica_container = _require_mapping(replica_containers[0], "replica.spec.containers[0]")
    _require_exact(replica_container.get("name"), _TASK_NAME, "replica.spec.containers[0].name")
    _require_exact(replica_container.get("image"), expected["image"], "replica.spec.containers[0].image")
    replica_resources = _normalize_resources(
        replica_container.get("resources"),
        expected,
        label="replica.spec.containers[0].resources",
        allow_platform_injected_limits=True,
    )
    replica_command, worker_init_prefix = _validate_expanded_worker_command(
        replica_container.get("command"),
        expected,
        template_command,
    )
    pod_hostname = _require_string(spec.get("hostname"), "replica.spec.hostname")
    status = _require_mapping(replica.get("status"), "replica.status")
    phase = status.get("phase")
    if not isinstance(phase, str) or not phase:
        raise ReceiptNotReady(f"replica {identity['name']} has no scheduler phase yet")

    return {
        "replica_name": identity["name"],
        "pod_name": pod_hostname,
        "pod_hostname": pod_hostname,
        "uid": identity["uid"],
        "resource_version": identity["resource_version"],
        "creation_timestamp": identity["creation_timestamp"],
        "task_name": _TASK_NAME,
        "replica_index": 0,
        "worker_spec": {
            "image": expected["image"],
            "resources": replica_resources,
            "restart_policy": expected["restart_policy"],
            "termination_grace_period_seconds": expected["termination_grace_period_seconds"],
            "mounts": replica_mounts,
            "command": replica_command,
            "worker_init_prefix": worker_init_prefix,
        },
        "status": {
            "phase": phase,
            "node_name": status.get("nodeName"),
            "pod_ip": status.get("podIP"),
            "pod_ips": _normalize_json_value(status.get("podIPs", []), "replica.status.podIPs"),
            "host_ip": status.get("hostIP"),
            "qos_class": status.get("qosClass"),
            "start_time": status.get("startTime"),
            "conditions": _normalize_conditions(status.get("conditions"), "replica.status.conditions"),
            "container_statuses": _normalize_container_statuses(
                status.get("containerStatuses"), "replica.status.containerStatuses"
            ),
        },
    }


def validate_and_normalize_receipt(
    raw_rjob: object,
    raw_replicas: object,
    expected_value: object,
    *,
    captured_at: str | None = None,
) -> dict[str, Any]:
    """Strictly validate raw scheduler objects and return a normalized receipt."""

    expected = validate_expected_contract(expected_value)
    rjob = _require_mapping(raw_rjob, "rjob")
    metadata = _require_mapping(rjob.get("metadata"), "rjob.metadata")
    identity = _require_metadata_identity(metadata, "rjob.metadata")
    _require_exact(identity["name"], expected["scheduler_rjob_name"], "rjob.metadata.name")

    labels = _require_mapping(metadata.get("labels"), "rjob.metadata.labels")
    annotations = _require_mapping(metadata.get("annotations"), "rjob.metadata.annotations")
    _require_exact(annotations.get(_SHOW_NAME_ANNOTATION), expected["showname"], _SHOW_NAME_ANNOTATION)
    _require_exact(labels.get(_RJOB_NAME_LABEL), expected["scheduler_rjob_name"], _RJOB_NAME_LABEL)
    _require_exact(labels.get(_SCHEDULING_GROUP_KEY), expected["scheduler_rjob_name"], _SCHEDULING_GROUP_KEY)
    _require_exact(
        annotations.get(_SCHEDULING_GROUP_KEY),
        expected["scheduler_rjob_name"],
        f"annotation {_SCHEDULING_GROUP_KEY}",
    )
    expected_task_type = f"rjob-{expected['task_type']}"
    _require_exact(labels.get(_TASK_TYPE_KEY), expected_task_type, f"label {_TASK_TYPE_KEY}")
    _require_exact(annotations.get(_TASK_TYPE_KEY), expected_task_type, f"annotation {_TASK_TYPE_KEY}")
    _require_exact(annotations.get(_PRIORITY_ANNOTATION), str(expected["priority"]), _PRIORITY_ANNOTATION)

    spec = _require_mapping(rjob.get("spec"), "rjob.spec")
    _require_exact(spec.get("backoffLimit"), expected["backoff_limit"], "rjob.spec.backoffLimit")
    task_specs = _require_mapping(spec.get("taskSpecs"), "rjob.spec.taskSpecs")
    if set(task_specs) != {_TASK_NAME}:
        raise ReceiptValidationError(f"rjob.spec.taskSpecs must contain exactly {_TASK_NAME!r}")
    task = _require_mapping(task_specs[_TASK_NAME], f"rjob.spec.taskSpecs.{_TASK_NAME}")
    _require_exact(task.get("replicas"), expected["replicas"], f"rjob.spec.taskSpecs.{_TASK_NAME}.replicas")
    negative_tags = _require_list(task.get("negativeTags"), f"rjob.spec.taskSpecs.{_TASK_NAME}.negativeTags")
    _require_exact(negative_tags, expected["negative_tags"], f"rjob.spec.taskSpecs.{_TASK_NAME}.negativeTags")

    template = _require_mapping(task.get("template"), f"rjob.spec.taskSpecs.{_TASK_NAME}.template")
    template_metadata = _require_mapping(
        template.get("metadata"), f"rjob.spec.taskSpecs.{_TASK_NAME}.template.metadata"
    )
    template_labels = _require_mapping(
        template_metadata.get("labels"), f"rjob.spec.taskSpecs.{_TASK_NAME}.template.metadata.labels"
    )
    template_annotations = _require_mapping(
        template_metadata.get("annotations"),
        f"rjob.spec.taskSpecs.{_TASK_NAME}.template.metadata.annotations",
    )
    _require_exact(template_labels.get(_RJOB_NAME_LABEL), expected["scheduler_rjob_name"], _RJOB_NAME_LABEL)
    _require_exact(template_labels.get(_TASK_NAME_LABEL), _TASK_NAME, _TASK_NAME_LABEL)
    _require_exact(template_labels.get(_TASK_TYPE_KEY), expected_task_type, _TASK_TYPE_KEY)
    _require_exact(template_labels.get(_QUOTA_GROUP_LABEL), expected["quota_group"], _QUOTA_GROUP_LABEL)
    _require_exact(
        template_labels.get(_PRIVATE_MACHINE_LABEL),
        expected["private_machine"],
        _PRIVATE_MACHINE_LABEL,
    )
    mount_annotation = _require_string(template_annotations.get(_GPFS_MOUNT_ANNOTATION), _GPFS_MOUNT_ANNOTATION)
    observed_mounts = mount_annotation.split(";")
    _require_exact(observed_mounts, expected["mounts"], _GPFS_MOUNT_ANNOTATION)

    pod_spec = _require_mapping(template.get("spec"), f"rjob.spec.taskSpecs.{_TASK_NAME}.template.spec")
    _require_exact(pod_spec.get("restartPolicy"), expected["restart_policy"], "template.spec.restartPolicy")
    _require_exact(
        pod_spec.get("terminationGracePeriodSeconds"),
        expected["termination_grace_period_seconds"],
        "template.spec.terminationGracePeriodSeconds",
    )
    containers = _require_list(pod_spec.get("containers"), "template.spec.containers")
    if len(containers) != 1:
        raise ReceiptValidationError("template.spec.containers must contain exactly one formal container")
    container = _require_mapping(containers[0], "template.spec.containers[0]")
    _require_exact(container.get("name"), _TASK_NAME, "template.spec.containers[0].name")
    _require_exact(container.get("image"), expected["image"], "template.spec.containers[0].image")
    command = _validate_command(container.get("command"), expected)

    observed_resources = _normalize_resources(
        container.get("resources"),
        expected,
        label="template.spec.containers[0].resources",
        allow_platform_injected_limits=False,
    )

    status = _require_mapping(rjob.get("status"), "rjob.status")
    phase = status.get("phase")
    if not isinstance(phase, str) or not phase:
        raise ReceiptNotReady("rjob has no scheduler phase yet")
    task_statuses = _normalize_task_statuses(status.get("taskStatuses"))

    replicas_value = _require_mapping(raw_replicas, "replicas list")
    replica_items = _require_list(replicas_value.get("items"), "replicas list.items")
    if not replica_items:
        raise ReceiptNotReady("the scheduler has not materialized the formal replica yet")
    if len(replica_items) != expected["replicas"]:
        raise ReceiptValidationError(
            f"expected exactly {expected['replicas']} scheduler replica, got {len(replica_items)}"
        )
    normalized_replicas = sorted(
        (_normalize_replica(item, expected, command) for item in replica_items),
        key=lambda item: (item["replica_index"], item["replica_name"]),
    )

    if captured_at is None:
        captured_at = datetime.now(timezone.utc).isoformat()
    else:
        _require_string(captured_at, "captured_at")
        try:
            parsed_capture = datetime.fromisoformat(captured_at)
        except ValueError as error:
            raise ReceiptValidationError("captured_at must be an ISO-8601 timestamp") from error
        if parsed_capture.tzinfo is None:
            raise ReceiptValidationError("captured_at must include a timezone")

    canonical_replicas = sorted(
        replica_items,
        key=lambda item: str(_require_mapping(item, "replica").get("metadata", {}).get("name", "")),
    )
    return {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "kind": _RECEIPT_KIND,
        "captured_at": captured_at,
        "validation": {
            "status": "passed",
            "expected_contract_schema_version": _EXPECTED_SCHEMA_VERSION,
        },
        "metadata": {
            **identity,
            "showname": expected["showname"],
        },
        "spec": {
            "quota_group": expected["quota_group"],
            "task_type": expected["task_type"],
            "priority": expected["priority"],
            "backoff_limit": expected["backoff_limit"],
            "task": {
                "name": _TASK_NAME,
                "replicas": expected["replicas"],
                "negative_tags": list(expected["negative_tags"]),
                "private_machine": expected["private_machine"],
                "restart_policy": expected["restart_policy"],
                "termination_grace_period_seconds": expected["termination_grace_period_seconds"],
                "mounts": list(expected["mounts"]),
                "container": {
                    "name": _TASK_NAME,
                    "image": expected["image"],
                    "resources": observed_resources,
                    "command": command,
                    "snapshot_root": expected["snapshot_root"],
                    "wrapper_path": expected["wrapper_path"],
                    "experiment_id": expected["experiment_id"],
                },
            },
        },
        "status": {
            "phase": phase,
            "reason": status.get("reason"),
            "message": status.get("message"),
            "task_statuses": task_statuses,
            "conditions": _normalize_conditions(status.get("conditions"), "rjob.status.conditions"),
        },
        "replica_names": [item["replica_name"] for item in normalized_replicas],
        "pod_names": [item["pod_name"] for item in normalized_replicas],
        "replicas": normalized_replicas,
        "source_objects": {
            "rjob_raw_sha256": _canonical_sha256(rjob),
            "replicas_raw_sha256": _canonical_sha256(canonical_replicas),
        },
    }


def fetch_scheduler_objects(client: Any, scheduler_rjob_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch the exact RJob and its Replica CRs through a Brainpp RJobClient."""

    raw_rjob = client.api.get_namespaced_custom_object(
        group=client.group,
        version=client.version,
        namespace=client.namespace,
        plural=client.plural,
        name=scheduler_rjob_name,
        _request_timeout=_API_REQUEST_TIMEOUT_SECONDS,
    )
    raw_replicas = client.api.list_namespaced_custom_object(
        group=client.group,
        version=client.version,
        namespace=client.namespace,
        resource_version="0",
        plural="replicas",
        label_selector=f"{_RJOB_NAME_LABEL}={scheduler_rjob_name}",
        _request_timeout=_API_REQUEST_TIMEOUT_SECONDS,
    )
    return (
        _require_mapping(raw_rjob, "Brainpp RJob response"),
        _require_mapping(raw_replicas, "Brainpp replicas response"),
    )


def capture_with_retries(
    client: Any,
    expected_value: object,
    *,
    attempts: int = 30,
    retry_seconds: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
    fetcher: Callable[[Any, str], tuple[dict[str, Any], dict[str, Any]]] = fetch_scheduler_objects,
    captured_at_factory: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat(),
) -> dict[str, Any]:
    """Fetch until status/replica materialization, never retrying a contract mismatch."""

    expected = validate_expected_contract(expected_value)
    if type(attempts) is not int or attempts < 1:
        raise ReceiptValidationError("attempts must be a positive integer")
    if not isinstance(retry_seconds, (int, float)) or retry_seconds < 0:
        raise ReceiptValidationError("retry_seconds must be non-negative")

    last_transient_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw_rjob, raw_replicas = fetcher(client, expected["scheduler_rjob_name"])
        except ReceiptValidationError:
            raise
        except Exception as error:  # Network/API exceptions are transient; no output is written.
            last_transient_error = error
        else:
            try:
                return validate_and_normalize_receipt(
                    raw_rjob,
                    raw_replicas,
                    expected,
                    captured_at=captured_at_factory(),
                )
            except ReceiptNotReady as error:
                last_transient_error = error
            except ReceiptValidationError:
                raise
        if attempt < attempts:
            sleeper(float(retry_seconds))

    detail = (
        f"{type(last_transient_error).__name__}: {last_transient_error}"
        if last_transient_error is not None
        else "unknown transient scheduler error"
    )
    raise ReceiptNotReady(f"scheduler receipt was not ready after {attempts} attempt(s): {detail}")


def write_json_atomic_exclusive(path: Path, payload: object) -> str:
    """Atomically publish a read-only JSON receipt without overwriting a target."""

    target = Path(path)
    literal = Path(os.path.abspath(target))
    if target != literal:
        raise ReceiptValidationError(f"output path must be absolute: {target}")
    parent = target.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise ReceiptValidationError(f"output parent is missing: {parent}") from error
    if resolved_parent != parent or not parent.is_dir():
        raise ReceiptValidationError(f"output parent must be a canonical non-symlink directory: {parent}")
    if target.exists() or target.is_symlink():
        raise ReceiptValidationError(f"output already exists; refusing to overwrite: {target}")

    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o444)
        try:
            os.link(temporary_path, target, follow_symlinks=False)
        except FileExistsError as error:
            raise ReceiptValidationError(f"output appeared concurrently; refusing to overwrite: {target}") from error
        directory_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return digest


def _create_brainpp_client() -> Any:
    try:
        from brainpp.rjob.client import RJobClient
    except ImportError as error:
        raise ReceiptError("Brainpp is unavailable; run this helper from an authenticated H workspace") from error
    return RJobClient(cluster_entry="", namespace="", access_key="", secret_key="")


def run_capture(
    *,
    scheduler_rjob_name: str,
    expected_json: Path,
    output: Path,
    attempts: int,
    retry_seconds: float,
    client_factory: Callable[[], Any] = _create_brainpp_client,
) -> tuple[dict[str, Any], str]:
    expected = load_expected_contract(expected_json)
    _require_exact(
        scheduler_rjob_name,
        expected["scheduler_rjob_name"],
        "CLI --scheduler-rjob-name versus expected.scheduler_rjob_name",
    )
    client = client_factory()
    receipt = capture_with_retries(
        client,
        expected,
        attempts=attempts,
        retry_seconds=retry_seconds,
    )
    digest = write_json_atomic_exclusive(output, receipt)
    return receipt, digest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-rjob-name",
        required=True,
        help="Actual scheduler-assigned RJob metadata.name, including its suffix.",
    )
    parser.add_argument(
        "--expected-json",
        type=Path,
        required=True,
        help="Strict expected-contract JSON; credentials must never be stored here.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New absolute receipt path. Existing targets are never overwritten.",
    )
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--retry-seconds", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        receipt, digest = run_capture(
            scheduler_rjob_name=arguments.scheduler_rjob_name,
            expected_json=arguments.expected_json,
            output=arguments.output,
            attempts=arguments.attempts,
            retry_seconds=arguments.retry_seconds,
        )
    except ReceiptError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "CWAM_FORMAL_RJOB_RECEIPT_CAPTURED "
        f"scheduler_rjob={receipt['metadata']['name']} "
        f"phase={receipt['status']['phase']} "
        f"replica={receipt['replica_names'][0]} "
        f"sha256={digest} "
        f"path={arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
