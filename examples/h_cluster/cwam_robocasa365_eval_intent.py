#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Create or verify an immutable RoboCasa365 evaluation intent manifest.

The intent is deliberately created before scheduler submission.  It binds the
evaluation identity and scientific decision rule to the exact live code,
checkpoint, environments, assets, repositories, graphics runtime, and RJob
resources.  ``stage`` and ``evaluation_class`` are independent: a full formal
rollout may still be exploratory, while confirmatory intent additionally
requires an explicitly pushed source commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from . import cwam_serving_checkpoint_manifest
except ImportError:  # pragma: no cover - direct script execution
    import cwam_serving_checkpoint_manifest

SCHEMA_VERSION = 1
MANIFEST_KIND = "cwam_robocasa365_eval_intent"
HASH_CHUNK_BYTES = 8 * 1024 * 1024

ATOMIC_TASKS = (
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
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
_IMAGE_DIGEST_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_GIT = Path("/usr/bin/git")
_GIT_TIMEOUT_SECONDS = 600
_SOURCE_DCP_SELECTION_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "source_checkpoint_path",
        "source_checkpoint_iteration",
        "serving_checkpoint_path",
        "checkpoint_manifest_sha256",
        "checkpoint_tree_sha256",
        "validation",
    }
)
_SOURCE_DCP_SELECTION_VALIDATION = {
    "source_iteration_path_matches": True,
    "serving_checkpoint_is_exact_model_child": True,
    "serving_checkpoint_manifest_matches": True,
}

_TOP_LEVEL_KEYS = frozenset(
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
)


class EvalIntentError(RuntimeError):
    """Raised when an evaluation intent contract is violated."""


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EvalIntentError(f"{label} must be a lowercase SHA-256, got {value!r}")
    return value


def _require_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise EvalIntentError(f"{label} must be a full lowercase Git SHA-1, got {value!r}")
    return value


def _require_safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or value in {".", ".."} or _SAFE_ID_RE.fullmatch(value) is None:
        raise EvalIntentError(f"{label} must be a 1-128 character path-safe identifier")
    return value


def _reject_incomplete(path: Path, label: str) -> None:
    if any(".incomplete" in component for component in path.parts):
        raise EvalIntentError(f"{label} must not contain .incomplete: {path}")


def _canonical_existing(path: Path, label: str, *, directory: bool) -> Path:
    supplied = Path(os.fspath(path))
    if not supplied.is_absolute() or os.path.normpath(os.fspath(supplied)) != os.fspath(supplied):
        raise EvalIntentError(f"{label} must be an exact canonical absolute path: {path}")
    try:
        resolved = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise EvalIntentError(f"{label} does not resolve safely: {path}: {error}") from error
    if supplied != resolved:
        raise EvalIntentError(f"{label} must contain no symlink components: supplied={supplied}, resolved={resolved}")
    _reject_incomplete(resolved, label)
    try:
        metadata = resolved.lstat()
    except OSError as error:
        raise EvalIntentError(f"cannot inspect {label}: {resolved}: {error}") from error
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected:
        kind = "directory" if directory else "regular file"
        raise EvalIntentError(f"{label} must be a {kind}: {resolved}")
    return resolved


def _read_stable(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    canonical = _canonical_existing(path, label, directory=False)
    before = canonical.lstat()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(canonical, flags)
    except OSError as error:
        raise EvalIntentError(f"cannot safely open {label}: {canonical}: {error}") from error

    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise EvalIntentError(f"{label} changed to a non-regular file: {canonical}")
        if _stable_identity(before) != _stable_identity(opened):
            raise EvalIntentError(f"{label} changed before reading: {canonical}")
        while chunk := os.read(descriptor, HASH_CHUNK_BYTES):
            payload.extend(chunk)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    after_path = canonical.lstat()
    identity = _stable_identity(before)
    if identity != _stable_identity(after_read) or identity != _stable_identity(after_path):
        raise EvalIntentError(f"{label} changed while reading: {canonical}")
    return bytes(payload), before


def _descriptor(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    require_read_only: bool = False,
) -> dict[str, str | int]:
    expected = _require_sha256(expected_sha256, f"expected {label} SHA-256")
    payload, metadata = _read_stable(path, label)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise EvalIntentError(f"{label} SHA-256 mismatch: expected={expected}, actual={actual}, path={path}")
    mode = stat.S_IMODE(metadata.st_mode)
    if require_read_only and mode != 0o444:
        raise EvalIntentError(f"{label} must have mode 0444, got {mode:04o}: {path}")
    return {
        "path": str(_canonical_existing(path, label, directory=False)),
        "size_bytes": len(payload),
        "sha256": actual,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EvalIntentError(f"JSON contains a duplicate object key: {key!r}")
        value[key] = item
    return value


def _load_json_descriptor(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    require_read_only: bool = False,
) -> tuple[dict[str, str | int], dict[str, Any]]:
    expected = _require_sha256(expected_sha256, f"expected {label} SHA-256")
    canonical = _canonical_existing(path, label, directory=False)
    payload, metadata = _read_stable(canonical, label)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise EvalIntentError(f"{label} SHA-256 mismatch: expected={expected}, actual={actual}, path={canonical}")
    mode = stat.S_IMODE(metadata.st_mode)
    if require_read_only and mode != 0o444:
        raise EvalIntentError(f"{label} must have mode 0444, got {mode:04o}: {canonical}")
    descriptor: dict[str, str | int] = {
        "path": str(canonical),
        "size_bytes": len(payload),
        "sha256": actual,
    }
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvalIntentError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise EvalIntentError(f"{label} JSON root must be an object")
    return descriptor, value


def _verify_source_dcp_selection_receipt(
    receipt_path: Path,
    expected_receipt_sha256: str,
    *,
    checkpoint_root: Path,
    checkpoint_manifest_sha256: str,
    checkpoint_tree_sha256: str,
) -> dict[str, str | int]:
    label = "source DCP export receipt"
    descriptor, receipt = _load_json_descriptor(
        receipt_path,
        expected_receipt_sha256,
        label,
        require_read_only=True,
    )
    observed = set(receipt)
    if observed != _SOURCE_DCP_SELECTION_RECEIPT_KEYS:
        raise EvalIntentError(
            f"{label} keys mismatch: "
            f"missing={sorted(_SOURCE_DCP_SELECTION_RECEIPT_KEYS - observed)}, "
            f"extra={sorted(observed - _SOURCE_DCP_SELECTION_RECEIPT_KEYS)}"
        )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "cwam_serving_dcp_selection_receipt"
        or receipt.get("status") != "passed"
    ):
        raise EvalIntentError(f"{label} schema/kind/status mismatch")

    iteration = receipt.get("source_checkpoint_iteration")
    if type(iteration) is not int or iteration < 1:
        raise EvalIntentError(f"{label} source_checkpoint_iteration must be a positive integer")
    source_checkpoint = _canonical_existing(
        Path(str(receipt.get("source_checkpoint_path", ""))),
        f"{label} source checkpoint",
        directory=True,
    )
    serving_checkpoint = _canonical_existing(
        Path(str(receipt.get("serving_checkpoint_path", ""))),
        f"{label} serving checkpoint",
        directory=True,
    )
    expected_source_checkpoint = checkpoint_root.parent
    if (
        serving_checkpoint != checkpoint_root
        or source_checkpoint != expected_source_checkpoint
        or checkpoint_root.name != "model"
        or source_checkpoint.name != f"iter_{iteration:09d}"
    ):
        raise EvalIntentError(f"{label} source/serving checkpoint selection mismatch")
    if (
        _require_sha256(
            receipt.get("checkpoint_manifest_sha256"),
            f"{label} checkpoint_manifest_sha256",
        )
        != checkpoint_manifest_sha256
        or _require_sha256(
            receipt.get("checkpoint_tree_sha256"),
            f"{label} checkpoint_tree_sha256",
        )
        != checkpoint_tree_sha256
    ):
        raise EvalIntentError(f"{label} checkpoint manifest/tree binding mismatch")
    validation = receipt.get("validation")
    if (
        not isinstance(validation, dict)
        or set(validation) != set(_SOURCE_DCP_SELECTION_VALIDATION)
        or any(validation.get(key) is not True for key in _SOURCE_DCP_SELECTION_VALIDATION)
    ):
        raise EvalIntentError(f"{label} validation must be the exact all-true verdict")
    return descriptor


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _verify_environment(
    root_path: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    label: str,
) -> tuple[str, dict[str, str | int]]:
    root = _canonical_existing(root_path, f"{label} root", directory=True)
    descriptor, manifest = _load_json_descriptor(
        manifest_path,
        expected_manifest_sha256,
        f"{label} manifest",
    )
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "cwam_python_environment_manifest":
        raise EvalIntentError(f"{label} manifest schema/kind mismatch")
    interpreter = manifest.get("interpreter")
    if not isinstance(interpreter, dict):
        raise EvalIntentError(f"{label} manifest interpreter must be an object")
    resolved_path = interpreter.get("resolved_path")
    if not isinstance(resolved_path, str):
        raise EvalIntentError(f"{label} manifest interpreter.resolved_path must be a string")
    interpreter_path = _canonical_existing(
        Path(resolved_path),
        f"{label} interpreter",
        directory=False,
    )
    if not _path_is_within(interpreter_path, root):
        raise EvalIntentError(f"{label} interpreter is outside its environment root")
    interpreter_sha256 = _require_sha256(
        interpreter.get("sha256"),
        f"{label} manifest interpreter.sha256",
    )
    interpreter_size = interpreter.get("size_bytes")
    live_descriptor = _descriptor(
        interpreter_path,
        interpreter_sha256,
        f"{label} interpreter",
    )
    if type(interpreter_size) is not int or interpreter_size != live_descriptor["size_bytes"]:
        raise EvalIntentError(f"{label} interpreter size does not match its manifest")
    return str(root), descriptor


def _verify_snapshot_source(
    root_path: Path,
    commit: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[str, dict[str, str | int]]:
    root = _canonical_existing(root_path, "source root", directory=True)
    source_commit = _require_commit(commit, "source commit")
    manifest_file = _canonical_existing(manifest_path, "source manifest", directory=False)
    if manifest_file != root / "SNAPSHOT_MANIFEST.json":
        raise EvalIntentError("source manifest must be the exact SNAPSHOT_MANIFEST.json under source root")
    descriptor, manifest = _load_json_descriptor(
        manifest_file,
        expected_manifest_sha256,
        "source manifest",
    )
    if manifest.get("schema_version") != 1:
        raise EvalIntentError("source manifest schema_version must be 1")
    if manifest.get("source_head") != source_commit:
        raise EvalIntentError("source manifest source_head does not match source commit")
    if manifest.get("source_dirty") is not False:
        raise EvalIntentError("source manifest must assert source_dirty=false")
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_count") != len(files):
        raise EvalIntentError("source manifest file inventory is malformed")
    return str(root), descriptor


def _verify_git_repository(root_path: Path, commit: str, label: str) -> dict[str, str]:
    root = _canonical_existing(root_path, f"{label} root", directory=True)
    expected_commit = _require_commit(commit, f"{label} commit")
    if not _GIT.is_file():
        raise EvalIntentError(f"required Git executable is unavailable: {_GIT}")

    def git(*arguments: str) -> str:
        command = [
            str(_GIT),
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
            *arguments,
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise EvalIntentError(f"cannot verify {label} repository: {error}") from error
        return completed.stdout

    top_level = git("rev-parse", "--show-toplevel").strip()
    if top_level != str(root):
        raise EvalIntentError(f"{label} root is not the Git top-level: {root}")
    actual_commit = git("rev-parse", "--verify", "HEAD").strip()
    if actual_commit != expected_commit:
        raise EvalIntentError(f"{label} commit mismatch: expected={expected_commit}, actual={actual_commit}")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise EvalIntentError(f"{label} repository must be clean: {root}")
    return {"root": str(root), "commit": expected_commit}


def _require_exact_protocol(args: argparse.Namespace) -> dict[str, Any]:
    if args.split != "atomic":
        raise EvalIntentError("primary evaluation split must be exactly atomic")
    exact = {
        "base_seed": (args.base_seed, 42),
        "max_steps": (args.max_steps, 400),
        "open_loop_horizon": (args.open_loop_horizon, 12),
        "policy_sampler": (args.policy_sampler, "unipc"),
        "policy_seed": (args.policy_seed, 42),
        "policy_guidance": (args.policy_guidance, 1.0),
        "policy_num_steps": (args.policy_num_steps, 30),
        "base_port": (args.base_port, 16000),
        "server_ready_timeout_seconds": (args.server_ready_timeout_seconds, 1800),
    }
    for label, (actual, expected) in exact.items():
        if type(actual) is not type(expected) or actual != expected:
            raise EvalIntentError(f"protocol {label} must be exactly {expected!r}, got {actual!r}")

    if args.stage == "one_episode_gate":
        if args.episodes_per_task != 1 or args.num_shards != 1:
            raise EvalIntentError("one_episode_gate requires one episode and one shard")
        tasks = tuple(item for item in args.tasks_csv.split(",") if item)
        if len(tasks) != 1 or tasks[0] not in ATOMIC_TASKS or args.tasks_csv != tasks[0]:
            raise EvalIntentError("one_episode_gate requires exactly one canonical atomic task")
    elif args.stage == "formal":
        if args.episodes_per_task != 20 or args.num_shards != 8:
            raise EvalIntentError("formal requires 20 episodes per task and eight shards")
        if args.tasks_csv:
            raise EvalIntentError("formal must use the complete canonical atomic split")
        tasks = ATOMIC_TASKS
    else:
        raise EvalIntentError(f"unsupported evaluation stage: {args.stage!r}")

    return {
        "split": args.split,
        "tasks": list(tasks),
        "task_count": len(tasks),
        "episodes_per_task": args.episodes_per_task,
        "expected_episode_count": len(tasks) * args.episodes_per_task,
        "num_shards": args.num_shards,
        "base_seed": args.base_seed,
        "max_steps": args.max_steps,
        "open_loop_horizon": args.open_loop_horizon,
        "policy_sampler": args.policy_sampler,
        "policy_seed": args.policy_seed,
        "policy_guidance": args.policy_guidance,
        "policy_num_steps": args.policy_num_steps,
        "base_port": args.base_port,
        "server_ready_timeout_seconds": args.server_ready_timeout_seconds,
    }


def _thresholds(args: argparse.Namespace) -> dict[str, float | int]:
    mean_rate = args.minimum_mean_success_rate
    successful_tasks = args.minimum_successful_tasks
    if not math.isfinite(mean_rate) or not 0.0 <= mean_rate <= 1.0:
        raise EvalIntentError("minimum mean success rate must be finite and in [0,1]")
    if type(successful_tasks) is not int or not 0 <= successful_tasks <= 17:
        raise EvalIntentError("minimum successful tasks must be an integer in [0,17]")
    if args.stage == "one_episode_gate" and (mean_rate != 0.0 or successful_tasks != 0):
        raise EvalIntentError("one_episode_gate thresholds must be exactly 0.0 and 0")
    return {
        "minimum_mean_success_rate": mean_rate,
        "minimum_successful_tasks": successful_tasks,
    }


def _resources(args: argparse.Namespace) -> dict[str, str | int]:
    expected_stage_resources = {
        "one_episode_gate": (16, 187_500, 1),
        "formal": (128, 1_500_000, 8),
    }
    expected_cpu, expected_memory, expected_gpu = expected_stage_resources[args.stage]
    if (args.cpu, args.memory_mib, args.gpu) != (
        expected_cpu,
        expected_memory,
        expected_gpu,
    ):
        raise EvalIntentError(
            f"{args.stage} resources must be cpu={expected_cpu}, memory_mib={expected_memory}, gpu={expected_gpu}"
        )
    expected = {
        "quota_group": "eailabagent_gpu",
        "negative_tag": "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn",
        "task_type": "normal",
        "private_machine": "group",
        "priority": 9,
        "restart_policy": "never",
        "backoff_limit": 1,
        "replicas": 1,
    }
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise EvalIntentError(f"resource {key} must be exactly {value!r}")
    return {
        "cpu": args.cpu,
        "memory_mib": args.memory_mib,
        "gpu": args.gpu,
        **expected,
    }


def build_intent(args: argparse.Namespace) -> dict[str, Any]:
    """Build a deterministic intent after verifying every live input."""

    experiment_id = _require_safe_id(args.experiment_id, "experiment ID")
    logical_run_id = _require_safe_id(args.logical_run_id, "logical run ID")
    run_name = _require_safe_id(args.run_name, "run name")
    if not logical_run_id.startswith(f"{experiment_id}-"):
        raise EvalIntentError("logical run ID must be namespaced below experiment ID")
    if run_name != logical_run_id:
        raise EvalIntentError("run name must equal logical run ID")
    if args.evaluation_class not in {"exploratory", "confirmatory"}:
        raise EvalIntentError("evaluation class must be exploratory or confirmatory")
    if args.evaluation_class == "confirmatory" and not args.git_pushed:
        raise EvalIntentError("confirmatory evaluation requires git_pushed=true")

    source_root, source_manifest = _verify_snapshot_source(
        args.source_root,
        args.source_commit,
        args.source_manifest,
        args.expected_source_manifest_sha256,
    )
    checkpoint_root = _canonical_existing(
        args.serving_checkpoint_path,
        "serving checkpoint",
        directory=True,
    )
    checkpoint_descriptor, checkpoint_manifest = _load_json_descriptor(
        args.checkpoint_manifest,
        args.expected_checkpoint_manifest_sha256,
        "checkpoint manifest",
    )
    try:
        checkpoint_verification = cwam_serving_checkpoint_manifest.verify_serving_checkpoint_manifest(
            checkpoint_manifest,
            checkpoint_root,
        )
    except cwam_serving_checkpoint_manifest.ServingCheckpointManifestError as error:
        raise EvalIntentError(f"serving checkpoint verification failed: {error}") from error

    cosmos_env_root, cosmos_env_manifest = _verify_environment(
        args.cosmos_env_root,
        args.cosmos_environment_manifest,
        args.expected_cosmos_environment_manifest_sha256,
        "Cosmos environment",
    )
    robocasa_env_root, robocasa_env_manifest = _verify_environment(
        args.robocasa_env_root,
        args.robocasa_environment_manifest,
        args.expected_robocasa_environment_manifest_sha256,
        "RoboCasa environment",
    )
    asset_root = _canonical_existing(args.robocasa_asset_root, "RoboCasa asset root", directory=True)

    prior_gate: dict[str, str | int] | None
    if args.stage == "one_episode_gate":
        if args.prior_gate_evidence is not None or args.expected_prior_gate_evidence_sha256 is not None:
            raise EvalIntentError("one_episode_gate must not bind prior gate evidence")
        prior_gate = None
    else:
        if args.prior_gate_evidence is None or args.expected_prior_gate_evidence_sha256 is None:
            raise EvalIntentError("formal evaluation requires prior gate evidence and its SHA-256")
        prior_gate = _descriptor(
            args.prior_gate_evidence,
            args.expected_prior_gate_evidence_sha256,
            "prior gate evidence",
            require_read_only=True,
        )

    if _IMAGE_DIGEST_RE.fullmatch(args.container_image) is None:
        raise EvalIntentError("container image must be digest pinned with @sha256")

    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "identity": {
            "experiment_id": experiment_id,
            "logical_run_id": logical_run_id,
            "run_name": run_name,
            "stage": args.stage,
            "evaluation_class": args.evaluation_class,
        },
        "protocol": _require_exact_protocol(args),
        "thresholds": _thresholds(args),
        "source": {
            "root": source_root,
            "commit": _require_commit(args.source_commit, "source commit"),
            "git_pushed": args.git_pushed,
            "manifest": source_manifest,
        },
        "serving_checkpoint": {
            "path": str(checkpoint_root),
            "manifest": checkpoint_descriptor,
            "tree_sha256": checkpoint_verification["tree_sha256"],
            "source_dcp_export_receipt": _verify_source_dcp_selection_receipt(
                args.source_dcp_export_receipt,
                args.expected_source_dcp_export_receipt_sha256,
                checkpoint_root=checkpoint_root,
                checkpoint_manifest_sha256=str(checkpoint_descriptor["sha256"]),
                checkpoint_tree_sha256=str(checkpoint_verification["tree_sha256"]),
            ),
        },
        "artifacts": {
            "training_config": _descriptor(
                args.training_config,
                args.expected_training_config_sha256,
                "training config",
            ),
            "action_stats": _descriptor(
                args.action_stats,
                args.expected_action_stats_sha256,
                "action stats",
            ),
            "cosmos_environment": {
                "root": cosmos_env_root,
                "manifest": cosmos_env_manifest,
            },
            "robocasa_environment": {
                "root": robocasa_env_root,
                "manifest": robocasa_env_manifest,
            },
            "robocasa_assets": {
                "root": str(asset_root),
                "manifest": _descriptor(
                    args.robocasa_asset_manifest,
                    args.expected_robocasa_asset_manifest_sha256,
                    "RoboCasa asset manifest",
                ),
                "receipt": _descriptor(
                    args.robocasa_asset_receipt,
                    args.expected_robocasa_asset_receipt_sha256,
                    "RoboCasa asset receipt",
                ),
            },
        },
        "repositories": {
            "demian": _verify_git_repository(
                args.demian_root,
                args.demian_commit,
                "DeMiAn",
            ),
            "robocasa": _verify_git_repository(
                args.robocasa_root,
                args.robocasa_commit,
                "RoboCasa",
            ),
            "robosuite": _verify_git_repository(
                args.robosuite_root,
                args.robosuite_commit,
                "robosuite",
            ),
        },
        "runtime": {
            "container_image": args.container_image,
            "egl_vendor_json": _descriptor(
                args.egl_vendor_json,
                args.expected_egl_vendor_json_sha256,
                "EGL vendor JSON",
            ),
        },
        "resources": _resources(args),
        "prior_gate": prior_gate,
    }
    if set(intent) != _TOP_LEVEL_KEYS:
        raise AssertionError("internal intent top-level schema drifted")
    return intent


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def create_intent_file(output: Path, intent: Mapping[str, Any]) -> dict[str, str | int]:
    """Create ``output`` exactly once, seal it 0444, and return its descriptor."""

    supplied = Path(os.fspath(output))
    if not supplied.is_absolute() or os.path.normpath(os.fspath(supplied)) != os.fspath(supplied):
        raise EvalIntentError("output must be an exact canonical absolute path")
    _reject_incomplete(supplied, "output")
    parent = _canonical_existing(supplied.parent, "output parent", directory=True)
    if supplied.parent != parent:
        raise EvalIntentError("output parent must contain no symlink components")

    payload = _canonical_json(intent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(supplied, flags, 0o444)
    except OSError as error:
        raise EvalIntentError(f"cannot exclusively create intent manifest {supplied}: {error}") from error

    created = True
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        sealed = os.fstat(descriptor)
        if stat.S_IMODE(sealed.st_mode) != 0o444 or sealed.st_size != len(payload):
            raise EvalIntentError("intent manifest was not sealed exactly as mode 0444")
    except BaseException:
        try:
            os.close(descriptor)
        finally:
            if created:
                supplied.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)

    directory_descriptor = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)

    result = _descriptor(
        supplied,
        hashlib.sha256(payload).hexdigest(),
        "created intent manifest",
        require_read_only=True,
    )
    return {"status": "created", **result}


def verify_intent_file(
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_intent: Mapping[str, Any],
) -> dict[str, str | int | bool]:
    """Verify the sealed intent and exact equivalence to current live inputs."""

    descriptor, observed = _load_json_descriptor(
        manifest_path,
        expected_manifest_sha256,
        "evaluation intent manifest",
        require_read_only=True,
    )
    if set(observed) != _TOP_LEVEL_KEYS:
        raise EvalIntentError(
            "evaluation intent top-level keys mismatch: "
            f"missing={sorted(_TOP_LEVEL_KEYS - set(observed))}, "
            f"extra={sorted(set(observed) - _TOP_LEVEL_KEYS)}"
        )
    if observed != expected_intent:
        raise EvalIntentError("evaluation intent does not exactly match current expected inputs")
    return {
        "status": "passed",
        **descriptor,
        "live_inputs_reverified": True,
        "exact_intent_match": True,
    }


def _parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected exactly 'true' or 'false'")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--logical-run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=("one_episode_gate", "formal"), required=True)
    parser.add_argument("--evaluation-class", choices=("exploratory", "confirmatory"), required=True)
    parser.add_argument("--git-pushed", type=_parse_bool, required=True)

    parser.add_argument("--split", required=True)
    parser.add_argument("--tasks-csv", required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--open-loop-horizon", type=int, required=True)
    parser.add_argument("--policy-sampler", required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--policy-guidance", type=float, required=True)
    parser.add_argument("--policy-num-steps", type=int, required=True)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--server-ready-timeout-seconds", type=int, required=True)
    parser.add_argument("--minimum-mean-success-rate", type=float, required=True)
    parser.add_argument("--minimum-successful-tasks", type=int, required=True)

    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--serving-checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--source-dcp-export-receipt", type=Path, required=True)
    parser.add_argument("--expected-source-dcp-export-receipt-sha256", required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--expected-training-config-sha256", required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--expected-action-stats-sha256", required=True)
    parser.add_argument("--cosmos-env-root", type=Path, required=True)
    parser.add_argument("--cosmos-environment-manifest", type=Path, required=True)
    parser.add_argument("--expected-cosmos-environment-manifest-sha256", required=True)
    parser.add_argument("--robocasa-env-root", type=Path, required=True)
    parser.add_argument("--robocasa-environment-manifest", type=Path, required=True)
    parser.add_argument("--expected-robocasa-environment-manifest-sha256", required=True)
    parser.add_argument("--robocasa-asset-root", type=Path, required=True)
    parser.add_argument("--robocasa-asset-manifest", type=Path, required=True)
    parser.add_argument("--expected-robocasa-asset-manifest-sha256", required=True)
    parser.add_argument("--robocasa-asset-receipt", type=Path, required=True)
    parser.add_argument("--expected-robocasa-asset-receipt-sha256", required=True)
    parser.add_argument("--demian-root", type=Path, required=True)
    parser.add_argument("--demian-commit", required=True)
    parser.add_argument("--robocasa-root", type=Path, required=True)
    parser.add_argument("--robocasa-commit", required=True)
    parser.add_argument("--robosuite-root", type=Path, required=True)
    parser.add_argument("--robosuite-commit", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--egl-vendor-json", type=Path, required=True)
    parser.add_argument("--expected-egl-vendor-json-sha256", required=True)
    parser.add_argument("--prior-gate-evidence", type=Path)
    parser.add_argument("--expected-prior-gate-evidence-sha256")

    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--memory-mib", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--quota-group", required=True)
    parser.add_argument("--negative-tag", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--private-machine", required=True)
    parser.add_argument("--priority", type=int, required=True)
    parser.add_argument("--restart-policy", required=True)
    parser.add_argument("--backoff-limit", type=int, required=True)
    parser.add_argument("--replicas", type=int, required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create and seal a new intent manifest")
    create.add_argument("--output", type=Path, required=True)
    _add_common_arguments(create)
    verify = subparsers.add_parser("verify", help="verify a sealed intent against live inputs")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-manifest-sha256", required=True)
    _add_common_arguments(verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        intent = build_intent(args)
        if args.command == "create":
            result = create_intent_file(args.output, intent)
        else:
            result = verify_intent_file(
                args.manifest,
                args.expected_manifest_sha256,
                intent,
            )
    except (EvalIntentError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
