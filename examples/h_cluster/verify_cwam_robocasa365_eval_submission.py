#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Verify the submit-side identity and scheduler binding for RoboCasa365 eval.

This verifier deliberately makes a narrower claim than a general GPFS
time-of-check/time-of-use solution.  It performs stable reads of the two
immutable submission records, validates their complete schemas and hashes, and
binds the submitter-generated nonce and UTC timestamp to the scheduler
readback.  The caller must seal the emitted verification receipt into the
runtime and final evaluation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RESERVATION_KIND = "cwam_robocasa365_eval_submission_reservation"
READBACK_KIND = "cwam_robocasa365_eval_scheduler_readback_receipt"
VERIFICATION_KIND = "cwam_robocasa365_eval_submission_verification"
DEFAULT_MAX_AGE_SECONDS = 72 * 60 * 60
DEFAULT_MAX_FUTURE_SKEW_SECONDS = 5 * 60
DEFAULT_MAX_START_TO_VERIFICATION_SECONDS = 2 * 60 * 60
MAX_JSON_BYTES = 16 * 1024 * 1024
PINNED_IMAGE = (
    "registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
)
PINNED_NEGATIVE_TAGS = ["node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"]
PINNED_MOUNTS = [
    "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs",
    "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public",
]
ATOMIC_TASKS = frozenset(
    {
        "CloseBlenderLid",
        "CloseFridge",
        "CloseToasterOvenDoor",
        "CoffeeSetupMug",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
        "SlideDishwasherRack",
        "TurnOffStove",
        "TurnOnElectricKettle",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
_RJOB_NAME_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_RESERVATION_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "submitted_at_utc",
        "expires_at_utc",
        "experiment_id",
        "logical_run_id",
        "run_name",
        "submission_nonce",
        "rjob_showname",
        "source_commit",
        "source_manifest_sha256",
        "evaluation",
        "eval_intent_manifest",
        "preflight",
        "preflight_sha256",
        "scheduler_expected",
    }
)
_READBACK_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "accepted_at_utc",
        "experiment_id",
        "logical_run_id",
        "run_name",
        "submission_nonce",
        "submitted_at_utc",
        "expires_at_utc",
        "scheduler_rjob_name",
        "scheduler_rjob_uid",
        "scheduler_started_at_utc",
        "worker_command_sha256",
        "submission_reservation",
        "scheduler_expected",
        "artifacts",
    }
)
_EVALUATION_KEYS = frozenset(
    {
        "stage",
        "split",
        "tasks_csv",
        "episodes_per_task",
        "num_shards",
        "base_seed",
        "prior_gate_evidence",
    }
)
_SCHEDULER_KEYS = frozenset(
    {
        "showname",
        "quota_group",
        "private_machine",
        "task_type",
        "priority",
        "restart_policy",
        "termination_grace_period_seconds",
        "backoff_limit",
        "replicas",
        "image",
        "cpu",
        "memory_mib",
        "gpu",
        "negative_tags",
        "store_host_nvme",
        "mounts",
    }
)
_DESCRIPTOR_KEYS = frozenset({"path", "size_bytes", "sha256"})


class SubmissionVerificationError(RuntimeError):
    """The external submission identity or scheduler binding is invalid."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SubmissionVerificationError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SubmissionVerificationError(f"JSON contains forbidden non-finite value {value}")


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SubmissionVerificationError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SubmissionVerificationError(
            f"{label} keys mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(repr(key) for key in observed - expected)}"
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SubmissionVerificationError(f"{label} must be a non-empty trimmed string")
    return value


def _require_run_id(value: object, label: str) -> str:
    text = _require_string(value, label)
    if text in {".", ".."} or _RUN_ID_RE.fullmatch(text) is None:
        raise SubmissionVerificationError(f"{label} must be 1-128 path-safe characters with alphanumeric endpoints")
    return text


def _require_rjob_name(value: object, label: str) -> str:
    text = _require_string(value, label)
    if len(text) > 63 or _RJOB_NAME_RE.fullmatch(text) is None:
        raise SubmissionVerificationError(f"{label} is not a lowercase Kubernetes name")
    return text


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SubmissionVerificationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SubmissionVerificationError(f"{label} must be an integer >= {minimum}")
    return value


def _parse_utc(value: object, label: str) -> datetime:
    text = _require_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise SubmissionVerificationError(f"{label} is not valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise SubmissionVerificationError(f"{label} must identify UTC")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_unsafe_path(path: Path, label: str) -> Path:
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise SubmissionVerificationError(f"{label} must be a normalized absolute path")
    if any(".incomplete" in component.lower() for component in path.parts):
        raise SubmissionVerificationError(f"{label} contains a forbidden .incomplete component")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SubmissionVerificationError(f"{label} cannot be resolved: {path}") from error
    if resolved != path:
        raise SubmissionVerificationError(f"{label} must contain no symlink components: {path}")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise SubmissionVerificationError(f"{label} must be a regular non-symlink file")
    return path


def _read_stable(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    canonical = _reject_unsafe_path(path, label)
    before = canonical.stat()
    if before.st_size <= 0 or before.st_size > MAX_JSON_BYTES:
        raise SubmissionVerificationError(f"{label} has an invalid size: {before.st_size}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(canonical, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise SubmissionVerificationError(f"{label} changed before reading")
        chunks: list[bytes] = []
        remaining = MAX_JSON_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = canonical.stat()
    for observed in (after, after_path):
        if identity != (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        ):
            raise SubmissionVerificationError(f"{label} changed while reading")
    if len(payload) > MAX_JSON_BYTES:
        raise SubmissionVerificationError(f"{label} exceeds the JSON size limit")
    return payload, before


def _load_record(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, metadata = _read_stable(path, label)
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SubmissionVerificationError(f"{label} is not strict UTF-8 JSON") from error
    descriptor = {
        "path": str(path),
        "size_bytes": metadata.st_size,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return _require_mapping(value, label), descriptor


def _validate_artifact_descriptor(value: object, label: str) -> dict[str, Any]:
    descriptor = _require_mapping(value, label)
    _require_exact_keys(descriptor, _DESCRIPTOR_KEYS, label)
    path = Path(_require_string(descriptor.get("path"), f"{label}.path"))
    payload, metadata = _read_stable(path, label)
    size_bytes = _require_int(descriptor.get("size_bytes"), f"{label}.size_bytes", minimum=1)
    if metadata.st_size != size_bytes:
        raise SubmissionVerificationError(f"{label}.size_bytes does not match the live file")
    digest = _require_sha256(descriptor.get("sha256"), f"{label}.sha256")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise SubmissionVerificationError(f"{label}.sha256 does not match the live file")
    return {"path": str(path), "size_bytes": size_bytes, "sha256": digest}


def _validate_submission_terminal_state(
    *,
    scheduler_readback_receipt_path: Path,
    submission_unknown_path: Path,
) -> None:
    readback = Path(os.path.abspath(scheduler_readback_receipt_path))
    unknown = Path(os.path.abspath(submission_unknown_path))
    if readback.name != "scheduler-readback-receipt.json":
        raise SubmissionVerificationError("scheduler readback receipt must use the canonical terminal filename")
    if unknown.name != "submission-unknown.json":
        raise SubmissionVerificationError("submission unknown path must use the canonical terminal filename")
    if readback.parent != unknown.parent:
        raise SubmissionVerificationError("accepted and unknown terminal records must share one parent")
    try:
        parent = readback.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SubmissionVerificationError("submission terminal-state parent cannot be resolved") from error
    if parent != readback.parent or not parent.is_dir():
        raise SubmissionVerificationError("submission terminal-state parent must be a canonical directory")
    if unknown.exists() or unknown.is_symlink():
        raise SubmissionVerificationError("submission-unknown terminal record exists")


def _validate_scheduler(
    value: object,
    label: str,
    *,
    stage: str,
) -> dict[str, Any]:
    scheduler = _require_mapping(value, label)
    _require_exact_keys(scheduler, _SCHEDULER_KEYS, label)
    _require_rjob_name(scheduler.get("showname"), f"{label}.showname")
    if scheduler.get("quota_group") != "eailabagent_gpu":
        raise SubmissionVerificationError(f"{label}.quota_group is not pinned")
    if scheduler.get("private_machine") != "group" or scheduler.get("task_type") != "normal":
        raise SubmissionVerificationError(f"{label} scheduling class is not pinned")
    if scheduler.get("priority") != 9 or scheduler.get("restart_policy") != "Never":
        raise SubmissionVerificationError(f"{label} priority/restart policy is not pinned")
    if (
        scheduler.get("termination_grace_period_seconds") != 60
        or scheduler.get("backoff_limit") != 1
        or scheduler.get("replicas") != 1
    ):
        raise SubmissionVerificationError(f"{label} lifecycle policy is not pinned")
    expected_resources = (
        {"cpu": 16, "memory_mib": 187_500, "gpu": 1}
        if stage == "one_episode_gate"
        else {"cpu": 128, "memory_mib": 1_500_000, "gpu": 8}
    )
    if any(scheduler.get(name) != expected for name, expected in expected_resources.items()):
        raise SubmissionVerificationError(
            f"{label} resource contract does not match stage={stage!r}: expected={expected_resources}"
        )
    if scheduler.get("store_host_nvme") is not True:
        raise SubmissionVerificationError(f"{label}.store_host_nvme must be true")
    image = _require_string(scheduler.get("image"), f"{label}.image")
    if image != PINNED_IMAGE:
        raise SubmissionVerificationError(f"{label}.image is not the audited digest")
    negative_tags = scheduler.get("negative_tags")
    mounts = scheduler.get("mounts")
    if negative_tags != PINNED_NEGATIVE_TAGS:
        raise SubmissionVerificationError(f"{label}.negative_tags is not exact")
    if mounts != PINNED_MOUNTS:
        raise SubmissionVerificationError(f"{label}.mounts is not exact")
    return dict(scheduler)


def _validate_evaluation(value: object, label: str) -> dict[str, Any]:
    evaluation = _require_mapping(value, label)
    _require_exact_keys(evaluation, _EVALUATION_KEYS, label)
    stage = evaluation.get("stage")
    if stage not in {"one_episode_gate", "formal"}:
        raise SubmissionVerificationError(f"{label}.stage is unsupported")
    if evaluation.get("split") != "atomic":
        raise SubmissionVerificationError(f"{label}.split must be the canonical atomic split")
    tasks_csv = evaluation.get("tasks_csv")
    if not isinstance(tasks_csv, str):
        raise SubmissionVerificationError(f"{label}.tasks_csv must be a string")
    episodes = _require_int(evaluation.get("episodes_per_task"), f"{label}.episodes_per_task", minimum=1)
    shards = _require_int(evaluation.get("num_shards"), f"{label}.num_shards", minimum=1)
    base_seed = _require_int(evaluation.get("base_seed"), f"{label}.base_seed", minimum=0)
    if base_seed != 42:
        raise SubmissionVerificationError(f"{label}.base_seed must be exactly 42")
    prior_gate = evaluation.get("prior_gate_evidence")
    if stage == "one_episode_gate":
        if episodes != 1 or shards != 1 or tasks_csv not in ATOMIC_TASKS or prior_gate is not None:
            raise SubmissionVerificationError(f"{label} one_episode_gate contract is invalid")
        validated_prior_gate = None
    else:
        if episodes != 20 or shards != 8 or tasks_csv or prior_gate is None:
            raise SubmissionVerificationError(f"{label} formal contract is invalid")
        validated_prior_gate = _validate_artifact_descriptor(
            prior_gate,
            f"{label}.prior_gate_evidence",
        )
        prior_gate_path = Path(validated_prior_gate["path"])
        if prior_gate_path.lstat().st_mode & 0o222:
            raise SubmissionVerificationError(f"{label}.prior_gate_evidence must be read-only")
    return {
        **evaluation,
        "prior_gate_evidence": validated_prior_gate,
    }


def _validate_preflight(value: object, expected_sha256: object, label: str) -> dict[str, Any]:
    preflight = _require_mapping(value, label)
    _require_exact_keys(
        preflight,
        frozenset({"asset_tree", "repositories", "validation"}),
        label,
    )
    digest = _require_sha256(expected_sha256, f"{label}_sha256")
    try:
        encoded = json.dumps(
            preflight,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SubmissionVerificationError(f"{label} is not canonical finite JSON") from error
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise SubmissionVerificationError(f"{label} canonical SHA-256 mismatch")

    asset = _require_mapping(preflight.get("asset_tree"), f"{label}.asset_tree")
    if (
        asset.get("schema_version") != 1
        or asset.get("kind") != "cwam_robocasa_asset_tree_verification_receipt"
        or asset.get("status") != "passed"
    ):
        raise SubmissionVerificationError(f"{label}.asset_tree is not a passed full-tree receipt")
    _require_sha256(asset.get("manifest_sha256"), f"{label}.asset_tree.manifest_sha256")
    _require_sha256(
        asset.get("publisher_receipt_sha256"),
        f"{label}.asset_tree.publisher_receipt_sha256",
    )

    repositories = _require_mapping(preflight.get("repositories"), f"{label}.repositories")
    if set(repositories) != {"demian", "robocasa", "robosuite"}:
        raise SubmissionVerificationError(f"{label}.repositories coverage is not exact")
    for name, raw_report in repositories.items():
        report = _require_mapping(raw_report, f"{label}.repositories.{name}")
        _require_exact_keys(
            report,
            frozenset({"root", "expected_commit", "actual_commit", "clean", "tracked_files"}),
            f"{label}.repositories.{name}",
        )
        root = Path(_require_string(report.get("root"), f"{label}.repositories.{name}.root"))
        if not root.is_absolute() or root != Path(os.path.abspath(root)):
            raise SubmissionVerificationError(f"{label}.repositories.{name}.root is not normalized absolute")
        expected_commit = _require_string(
            report.get("expected_commit"),
            f"{label}.repositories.{name}.expected_commit",
        )
        actual_commit = _require_string(
            report.get("actual_commit"),
            f"{label}.repositories.{name}.actual_commit",
        )
        if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None or actual_commit != expected_commit:
            raise SubmissionVerificationError(f"{label}.repositories.{name} commit binding is invalid")
        if report.get("clean") is not True:
            raise SubmissionVerificationError(f"{label}.repositories.{name} is not clean")
        tracked = _require_mapping(
            report.get("tracked_files"),
            f"{label}.repositories.{name}.tracked_files",
        )
        if tracked.get("all_tracked_regular_files_match_head_blobs") is not True:
            raise SubmissionVerificationError(f"{label}.repositories.{name} tracked blobs did not pass")

    validation = _require_mapping(preflight.get("validation"), f"{label}.validation")
    expected_validation = {
        "asset_tree_full_hash_match": True,
        "repository_heads_clean": True,
        "tracked_regular_files_match_head_blobs": True,
    }
    if validation != expected_validation:
        raise SubmissionVerificationError(f"{label}.validation is not the exact all-true verdict")
    return dict(preflight)


def _validate_eval_intent_manifest(
    value: object,
    *,
    experiment_id: str,
    logical_run_id: str,
    run_name: str,
    source_commit: str,
    source_manifest_sha256: str,
    evaluation: Mapping[str, Any],
    scheduler: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = _require_mapping(value, "reservation.eval_intent_manifest")
    _require_exact_keys(descriptor, _DESCRIPTOR_KEYS, "reservation.eval_intent_manifest")
    path = Path(_require_string(descriptor.get("path"), "reservation.eval_intent_manifest.path"))
    payload, metadata = _read_stable(path, "reservation.eval_intent_manifest")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SubmissionVerificationError("reservation.eval_intent_manifest must be immutable mode 0444")
    size_bytes = _require_int(
        descriptor.get("size_bytes"),
        "reservation.eval_intent_manifest.size_bytes",
        minimum=1,
    )
    digest = _require_sha256(
        descriptor.get("sha256"),
        "reservation.eval_intent_manifest.sha256",
    )
    if size_bytes != len(payload) or hashlib.sha256(payload).hexdigest() != digest:
        raise SubmissionVerificationError("reservation.eval_intent_manifest descriptor differs from live bytes")
    try:
        intent = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SubmissionVerificationError("reservation.eval_intent_manifest is not strict UTF-8 JSON") from error
    intent = _require_mapping(intent, "eval intent")
    _require_exact_keys(
        intent,
        frozenset(
            {
                "schema_version",
                "kind",
                "identity",
                "protocol",
                "thresholds",
                "source",
                "serving_checkpoint",
                "artifacts",
                "repositories",
                "runtime",
                "resources",
                "prior_gate",
            }
        ),
        "eval intent",
    )
    if intent.get("schema_version") != 1 or intent.get("kind") != "cwam_robocasa365_eval_intent":
        raise SubmissionVerificationError("eval intent schema/kind is invalid")
    identity = _require_mapping(intent.get("identity"), "eval intent.identity")
    if identity.get("experiment_id") != experiment_id or identity.get("logical_run_id") != logical_run_id:
        raise SubmissionVerificationError("eval intent experiment/logical identity mismatch")
    if identity.get("run_name") != run_name or identity.get("stage") != evaluation["stage"]:
        raise SubmissionVerificationError("eval intent run/stage identity mismatch")
    evaluation_class = identity.get("evaluation_class")
    if evaluation_class not in {"exploratory", "confirmatory"}:
        raise SubmissionVerificationError("eval intent evaluation_class is invalid")

    source = _require_mapping(intent.get("source"), "eval intent.source")
    if source.get("commit") != source_commit:
        raise SubmissionVerificationError("eval intent source commit mismatch")
    source_manifest = _require_mapping(source.get("manifest"), "eval intent.source.manifest")
    if source_manifest.get("sha256") != source_manifest_sha256:
        raise SubmissionVerificationError("eval intent source manifest mismatch")
    if type(source.get("git_pushed")) is not bool:
        raise SubmissionVerificationError("eval intent source.git_pushed must be boolean")
    if evaluation_class == "confirmatory" and source.get("git_pushed") is not True:
        raise SubmissionVerificationError("confirmatory eval intent is not pushed")

    protocol = _require_mapping(intent.get("protocol"), "eval intent.protocol")
    expected_protocol = {
        "split": evaluation["split"],
        "episodes_per_task": evaluation["episodes_per_task"],
        "num_shards": evaluation["num_shards"],
        "base_seed": evaluation["base_seed"],
    }
    if any(protocol.get(key) != expected for key, expected in expected_protocol.items()):
        raise SubmissionVerificationError("eval intent protocol differs from reservation")
    tasks = protocol.get("tasks")
    expected_tasks = [evaluation["tasks_csv"]] if evaluation["stage"] == "one_episode_gate" else sorted(ATOMIC_TASKS)
    if not isinstance(tasks, list) or sorted(tasks) != expected_tasks:
        raise SubmissionVerificationError("eval intent task coverage differs from reservation")
    if intent.get("prior_gate") != evaluation["prior_gate_evidence"]:
        raise SubmissionVerificationError("eval intent prior gate differs from reservation")

    resources = _require_mapping(intent.get("resources"), "eval intent.resources")
    resource_pairs = {
        "cpu": scheduler["cpu"],
        "memory_mib": scheduler["memory_mib"],
        "gpu": scheduler["gpu"],
        "quota_group": scheduler["quota_group"],
        "negative_tag": scheduler["negative_tags"][0],
        "task_type": scheduler["task_type"],
        "private_machine": scheduler["private_machine"],
        "priority": scheduler["priority"],
        "backoff_limit": scheduler["backoff_limit"],
        "replicas": scheduler["replicas"],
    }
    if any(resources.get(key) != expected for key, expected in resource_pairs.items()):
        raise SubmissionVerificationError("eval intent resources differ from reservation scheduler")
    return {"path": str(path), "size_bytes": size_bytes, "sha256": digest}


def verify_submission(
    *,
    reservation_path: Path,
    expected_reservation_sha256: str,
    scheduler_readback_receipt_path: Path,
    submission_unknown_path: Path,
    experiment_id: str,
    logical_run_id: str,
    run_name: str,
    submission_nonce: str,
    submitted_at_utc: str,
    observed_scheduler_rjob_name: str,
    observed_scheduler_rjob_uid: str,
    max_age_seconds: int,
    max_future_skew_seconds: int,
    max_start_to_verification_seconds: int,
    output: Path,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> tuple[dict[str, Any], str]:
    """Verify both immutable records and publish a read-only verification receipt."""

    experiment_id = _require_run_id(experiment_id, "experiment_id")
    logical_run_id = _require_run_id(logical_run_id, "logical_run_id")
    run_name = _require_run_id(run_name, "run_name")
    if not logical_run_id.startswith(f"{experiment_id}-"):
        raise SubmissionVerificationError("logical_run_id must be namespaced below experiment_id")
    if run_name != logical_run_id:
        raise SubmissionVerificationError("run_name must equal logical_run_id")
    submission_nonce = _require_sha256(submission_nonce, "submission_nonce")
    observed_scheduler_rjob_name = _require_rjob_name(
        observed_scheduler_rjob_name,
        "observed_scheduler_rjob_name",
    )
    observed_scheduler_rjob_uid = _require_string(
        observed_scheduler_rjob_uid,
        "observed_scheduler_rjob_uid",
    )
    try:
        observed_scheduler_rjob_uid = str(uuid.UUID(observed_scheduler_rjob_uid))
    except ValueError as error:
        raise SubmissionVerificationError("observed_scheduler_rjob_uid is not a UUID") from error
    expected_reservation_sha256 = _require_sha256(
        expected_reservation_sha256,
        "expected_reservation_sha256",
    )
    if type(max_age_seconds) is not int or max_age_seconds != DEFAULT_MAX_AGE_SECONDS:
        raise SubmissionVerificationError(f"max_age_seconds must be exactly {DEFAULT_MAX_AGE_SECONDS}")
    if type(max_future_skew_seconds) is not int or max_future_skew_seconds != DEFAULT_MAX_FUTURE_SKEW_SECONDS:
        raise SubmissionVerificationError(f"max_future_skew_seconds must be exactly {DEFAULT_MAX_FUTURE_SKEW_SECONDS}")
    if (
        type(max_start_to_verification_seconds) is not int
        or max_start_to_verification_seconds != DEFAULT_MAX_START_TO_VERIFICATION_SECONDS
    ):
        raise SubmissionVerificationError(
            f"max_start_to_verification_seconds must be exactly {DEFAULT_MAX_START_TO_VERIFICATION_SECONDS}"
        )
    supplied_submitted = _parse_utc(submitted_at_utc, "submitted_at_utc")
    now = clock()
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise SubmissionVerificationError("clock must return a timezone-aware datetime")
    now = now.astimezone(timezone.utc)

    _validate_submission_terminal_state(
        scheduler_readback_receipt_path=scheduler_readback_receipt_path,
        submission_unknown_path=submission_unknown_path,
    )
    reservation, reservation_descriptor = _load_record(
        reservation_path,
        "submission reservation",
    )
    if reservation_descriptor["sha256"] != expected_reservation_sha256:
        raise SubmissionVerificationError("submission reservation SHA-256 mismatch")
    _require_exact_keys(reservation, _RESERVATION_KEYS, "submission reservation")
    if (
        reservation.get("schema_version") != SCHEMA_VERSION
        or reservation.get("kind") != RESERVATION_KIND
        or reservation.get("status") != "submitting"
    ):
        raise SubmissionVerificationError("submission reservation schema/kind/status is invalid")
    expected_identity = {
        "experiment_id": experiment_id,
        "logical_run_id": logical_run_id,
        "run_name": run_name,
        "submission_nonce": submission_nonce,
        "submitted_at_utc": submitted_at_utc,
    }
    for key, expected in expected_identity.items():
        if reservation.get(key) != expected:
            raise SubmissionVerificationError(
                f"submission reservation {key} mismatch: expected={expected!r}, actual={reservation.get(key)!r}"
            )
    submitted = _parse_utc(reservation.get("submitted_at_utc"), "reservation.submitted_at_utc")
    expires = _parse_utc(reservation.get("expires_at_utc"), "reservation.expires_at_utc")
    if submitted != supplied_submitted:
        raise SubmissionVerificationError("submitted_at_utc changed after parsing")
    if expires - submitted != timedelta(seconds=max_age_seconds):
        raise SubmissionVerificationError("reservation expiry does not match the pinned freshness window")
    age_seconds = (now - submitted).total_seconds()
    if age_seconds < -max_future_skew_seconds:
        raise SubmissionVerificationError("submission timestamp is too far in the future")
    if age_seconds > max_age_seconds or now > expires:
        raise SubmissionVerificationError("submission reservation is stale")
    _require_rjob_name(reservation.get("rjob_showname"), "reservation.rjob_showname")
    source_commit = _require_string(reservation.get("source_commit"), "reservation.source_commit")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise SubmissionVerificationError("reservation.source_commit must be a full Git SHA")
    _require_sha256(
        reservation.get("source_manifest_sha256"),
        "reservation.source_manifest_sha256",
    )
    evaluation = _validate_evaluation(reservation.get("evaluation"), "reservation.evaluation")
    preflight = _validate_preflight(
        reservation.get("preflight"),
        reservation.get("preflight_sha256"),
        "reservation.preflight",
    )
    scheduler_expected = _validate_scheduler(
        reservation.get("scheduler_expected"),
        "reservation.scheduler_expected",
        stage=evaluation["stage"],
    )
    eval_intent_manifest = _validate_eval_intent_manifest(
        reservation.get("eval_intent_manifest"),
        experiment_id=experiment_id,
        logical_run_id=logical_run_id,
        run_name=run_name,
        source_commit=source_commit,
        source_manifest_sha256=reservation["source_manifest_sha256"],
        evaluation=evaluation,
        scheduler=scheduler_expected,
    )

    readback, readback_descriptor = _load_record(
        scheduler_readback_receipt_path,
        "scheduler readback receipt",
    )
    _require_exact_keys(readback, _READBACK_KEYS, "scheduler readback receipt")
    if (
        readback.get("schema_version") != SCHEMA_VERSION
        or readback.get("kind") != READBACK_KIND
        or readback.get("status") != "accepted"
    ):
        raise SubmissionVerificationError("scheduler readback receipt schema/kind/status is invalid")
    for key in (
        "experiment_id",
        "logical_run_id",
        "run_name",
        "submission_nonce",
        "submitted_at_utc",
        "expires_at_utc",
    ):
        if readback.get(key) != reservation.get(key):
            raise SubmissionVerificationError(f"scheduler readback {key} does not match reservation")
    accepted = _parse_utc(readback.get("accepted_at_utc"), "readback.accepted_at_utc")
    if accepted < submitted - timedelta(seconds=max_future_skew_seconds):
        raise SubmissionVerificationError("scheduler accepted timestamp predates submission")
    if accepted > now + timedelta(seconds=max_future_skew_seconds):
        raise SubmissionVerificationError("scheduler accepted timestamp is too far in the future")
    scheduler_name = _require_rjob_name(
        readback.get("scheduler_rjob_name"),
        "readback.scheduler_rjob_name",
    )
    scheduler_uid = _require_string(
        readback.get("scheduler_rjob_uid"),
        "readback.scheduler_rjob_uid",
    )
    try:
        scheduler_uid = str(uuid.UUID(scheduler_uid))
    except ValueError as error:
        raise SubmissionVerificationError("readback.scheduler_rjob_uid is not a UUID") from error
    scheduler_started = _parse_utc(
        readback.get("scheduler_started_at_utc"),
        "readback.scheduler_started_at_utc",
    )
    if scheduler_started < submitted - timedelta(seconds=max_future_skew_seconds):
        raise SubmissionVerificationError("scheduler start timestamp predates submission")
    start_age_seconds = (now - scheduler_started).total_seconds()
    if start_age_seconds < -max_future_skew_seconds:
        raise SubmissionVerificationError("scheduler start timestamp is too far in the future")
    if start_age_seconds > max_start_to_verification_seconds:
        raise SubmissionVerificationError("scheduler-start-to-verification window is stale")
    if accepted < scheduler_started - timedelta(seconds=max_future_skew_seconds):
        raise SubmissionVerificationError("scheduler accepted timestamp predates scheduler start")
    showname = _require_rjob_name(reservation.get("rjob_showname"), "reservation.rjob_showname")
    if not scheduler_name.startswith(f"{showname}-"):
        raise SubmissionVerificationError("scheduler RJob name does not derive from the showname")
    if scheduler_name != observed_scheduler_rjob_name:
        raise SubmissionVerificationError("scheduler RJob name does not match the worker-injected identity")
    if scheduler_uid != observed_scheduler_rjob_uid:
        raise SubmissionVerificationError("scheduler RJob UID does not match the worker-injected identity")
    _require_sha256(
        readback.get("worker_command_sha256"),
        "readback.worker_command_sha256",
    )
    reservation_binding = _require_mapping(
        readback.get("submission_reservation"),
        "readback.submission_reservation",
    )
    _require_exact_keys(
        reservation_binding,
        frozenset({"path", "sha256"}),
        "readback.submission_reservation",
    )
    if reservation_binding != {
        "path": str(reservation_path),
        "sha256": expected_reservation_sha256,
    }:
        raise SubmissionVerificationError("scheduler readback does not bind the exact reservation")
    if readback.get("scheduler_expected") != scheduler_expected:
        raise SubmissionVerificationError("scheduler readback contract differs from reservation")
    artifacts = _require_mapping(readback.get("artifacts"), "readback.artifacts")
    expected_artifact_names = frozenset(
        {
            "submission_log",
            "scheduler_expected_contract",
            "scheduler_capture_log",
            "scheduler_structured_receipt",
            "scheduler_text_readback",
        }
    )
    _require_exact_keys(artifacts, expected_artifact_names, "readback.artifacts")
    validated_artifacts = {
        name: _validate_artifact_descriptor(value, f"readback.artifacts.{name}") for name, value in artifacts.items()
    }
    _validate_submission_terminal_state(
        scheduler_readback_receipt_path=scheduler_readback_receipt_path,
        submission_unknown_path=submission_unknown_path,
    )

    verification = {
        "schema_version": SCHEMA_VERSION,
        "kind": VERIFICATION_KIND,
        "status": "passed",
        "verified_at_utc": _format_utc(now),
        "identity": expected_identity,
        "freshness": {
            "age_seconds": age_seconds,
            "expires_at_utc": _format_utc(expires),
            "max_age_seconds": max_age_seconds,
            "max_future_skew_seconds": max_future_skew_seconds,
            "scheduler_started_at_utc": _format_utc(scheduler_started),
            "start_age_seconds": start_age_seconds,
            "max_start_to_verification_seconds": max_start_to_verification_seconds,
        },
        "evaluation": evaluation,
        "eval_intent_manifest": eval_intent_manifest,
        "preflight": preflight,
        "preflight_sha256": reservation["preflight_sha256"],
        "scheduler": {
            "rjob_name": scheduler_name,
            "rjob_uid": scheduler_uid,
            "showname": showname,
            "worker_command_sha256": readback["worker_command_sha256"],
            "expected": scheduler_expected,
        },
        "reservation": reservation_descriptor,
        "scheduler_readback_receipt": readback_descriptor,
        "scheduler_artifacts": validated_artifacts,
        "validation": {
            "submit_identity_matches": True,
            "freshness_window_passed": True,
            "scheduler_readback_matches": True,
            "scheduler_artifact_hashes_match": True,
            "eval_intent_manifest_matches": True,
            "pre_submission_full_inputs_passed": True,
            "submission_unknown_absent": True,
        },
    }
    output = Path(os.path.abspath(output))
    if output.exists() or output.is_symlink():
        raise SubmissionVerificationError(f"refusing to overwrite output: {output}")
    parent = output.parent.resolve(strict=True)
    if output.parent != parent or not parent.is_dir():
        raise SubmissionVerificationError("output parent must be an existing canonical directory")
    encoded = (json.dumps(verification, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o444)
    published = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_submission_terminal_state(
            scheduler_readback_receipt_path=scheduler_readback_receipt_path,
            submission_unknown_path=submission_unknown_path,
        )
        os.link(temporary, output, follow_symlinks=False)
        published = True
        _validate_submission_terminal_state(
            scheduler_readback_receipt_path=scheduler_readback_receipt_path,
            submission_unknown_path=submission_unknown_path,
        )
        os.chmod(output, 0o444, follow_symlinks=False)
        directory_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if published:
            output.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(encoded).hexdigest()
    return verification, digest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--expected-reservation-sha256", required=True)
    parser.add_argument("--scheduler-readback-receipt", type=Path, required=True)
    parser.add_argument("--submission-unknown", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--logical-run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--submission-nonce", required=True)
    parser.add_argument("--submitted-at-utc", required=True)
    parser.add_argument("--observed-scheduler-rjob-name", required=True)
    parser.add_argument("--observed-scheduler-rjob-uid", required=True)
    parser.add_argument("--max-age-seconds", type=int, required=True)
    parser.add_argument("--max-future-skew-seconds", type=int, required=True)
    parser.add_argument("--max-start-to-verification-seconds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        receipt, digest = verify_submission(
            reservation_path=arguments.reservation,
            expected_reservation_sha256=arguments.expected_reservation_sha256,
            scheduler_readback_receipt_path=arguments.scheduler_readback_receipt,
            submission_unknown_path=arguments.submission_unknown,
            experiment_id=arguments.experiment_id,
            logical_run_id=arguments.logical_run_id,
            run_name=arguments.run_name,
            submission_nonce=arguments.submission_nonce,
            submitted_at_utc=arguments.submitted_at_utc,
            observed_scheduler_rjob_name=arguments.observed_scheduler_rjob_name,
            observed_scheduler_rjob_uid=arguments.observed_scheduler_rjob_uid,
            max_age_seconds=arguments.max_age_seconds,
            max_future_skew_seconds=arguments.max_future_skew_seconds,
            max_start_to_verification_seconds=arguments.max_start_to_verification_seconds,
            output=arguments.output,
        )
    except SubmissionVerificationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        "ROBOCASA365_EVAL_SUBMISSION_VERIFIED "
        f"rjob={receipt['scheduler']['rjob_name']} "
        f"sha256={digest} path={arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
