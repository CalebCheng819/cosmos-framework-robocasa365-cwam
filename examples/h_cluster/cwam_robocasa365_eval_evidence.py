#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build and verify formal RoboCasa365 closed-loop evaluation evidence.

DeMiAn writes one ``results_shardN.json`` per task shard.  Its stock merger
does not reject incomplete tasks and does not emit the two CSV artifacts used
by the CWAM research workflow.  This verifier therefore treats the immutable
eval contract plus every shard args/result pair as one fail-closed input and
emits:

* ``summary.csv`` — one row per task/episode with the deterministic seed;
* ``task_success_rates.csv`` — one row per task; and
* ``evaluation_evidence.json`` — hashes, aggregate metrics, and execution/effect
  verdicts.

The episode seed is derived from the verified DeMiAn invocation contract.  The
upstream evaluator does not currently serialize it into each episode result.
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
import stat
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

try:
    from . import cwam_formal_provenance, cwam_robocasa_asset_manifest, cwam_serving_checkpoint_manifest
except ImportError:  # pragma: no cover - direct script execution
    import cwam_formal_provenance
    import cwam_robocasa_asset_manifest
    import cwam_serving_checkpoint_manifest

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

COMPOSITE_TASKS = (
    "ArrangeBreadBasket",
    "ArrangeTea",
    "BreadSelection",
    "CategorizeCondiments",
    "CuttingToolSelection",
    "DeliverStraw",
    "GarnishPancake",
    "GatherTableware",
    "GetToastedBread",
    "HeatKebabSandwich",
    "KettleBoiling",
    "LoadDishwasher",
    "MakeIceLemonade",
    "PackIdenticalLunches",
    "PanTransfer",
    "PortionHotDogs",
    "PreSoakPan",
    "PrepareCoffee",
    "RecycleBottlesByType",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SeparateFreezerRack",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WaffleReheat",
    "WashFruitColander",
    "WashLettuce",
    "WeighIngredients",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
_NORMALIZED_GPU_UUID_RE = re.compile(r"^[0-9a-f]{32}$")
_NORMALIZE_DISTRIBUTION_NAME_RE = re.compile(r"[-_.]+")
_CONTRACT_KIND = "cwam_robocasa365_formal_eval_contract"
_EVIDENCE_KIND = "cwam_robocasa365_formal_eval_evidence"
_EXPECTED_H_DRIVER_LIBRARY_PATH = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
_EXPECTED_EGL_VENDOR_JSON = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
_EXPECTED_CONTAINER_IMAGE = (
    "registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
)
_ONLINE_WAN2PT2_TARGET = "cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface"
_EXPECTED_SUBMISSION_MOUNTS = [
    "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs",
    "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public",
]
_EXPECTED_FORBIDDEN_GRAPHICS_ENVIRONMENT = (
    "LD_PRELOAD",
    "LD_AUDIT",
    "__EGL_VENDOR_LIBRARY_DIRS",
    "LIBGL_ALWAYS_SOFTWARE",
    "MESA_LOADER_DRIVER_OVERRIDE",
    "LIBGL_DRIVERS_PATH",
)
_SUBMISSION_VERIFICATION_KIND = "cwam_robocasa365_eval_submission_verification"
_ASSET_VERIFICATION_KIND = "cwam_robocasa_asset_tree_verification_receipt"
_EVAL_INTENT_KIND = "cwam_robocasa365_eval_intent"
_PRIOR_GATE_VERIFICATION_KIND = "cwam_robocasa365_prior_gate_verification"
_POST_ROLLOUT_VERIFICATION_KIND = "cwam_robocasa365_post_rollout_live_input_verification_receipt"
_SEALED_EVIDENCE_VERIFICATION_KIND = "cwam_robocasa365_sealed_evidence_verification_receipt"
_POST_ROLLOUT_SCOPE = "point_in_time_post_rollout_live_input_reverification"
_POST_ROLLOUT_SNAPSHOT_ALLOWED_ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots")
_GIT_EXECUTABLE = Path("/usr/bin/git")
_GIT_COMMAND_TIMEOUT_SECONDS = 600
_MAX_START_TO_CAPTURE_SECONDS = 2 * 60 * 60
_MAX_FUTURE_SKEW_SECONDS = 5 * 60
_SEED_PROVENANCE = (
    "The pinned evaluator ran with eval_dataset=null, so target-environment "
    "resets used the seeds derived from verified DeMiAn shard args: "
    "base_seed + within_shard_task_index * 10000 + episode_index; "
    "the pinned evaluator does not serialize seed per episode."
)
_EXPECTED_DEMIAN_COMMIT = "2600ece5bf0eeacf010a95f084dcda2de6d30d59"
_EXPECTED_DEMIAN_EVALUATOR_SHA256 = "c38ed1f4049d079b84c185cb572667a47eaf984c2c22b492efc8985401119836"
_EXPECTED_DEMIAN_TASK_REGISTRY_SHA256 = "e902311cf435986859ae01920548aba169e085ad6d26187cad9fb0d1f9c315de"
_CHECKPOINT_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "checkpoint_path",
        "files",
        "tree_sha256",
    }
)
_CHECKPOINT_VERIFICATION_KEYS = frozenset(
    {
        "status",
        "checkpoint_path",
        "file_count",
        "tree_sha256",
    }
)
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
_PYTHON_ENVIRONMENT_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "interpreter",
        "distribution_count",
        "distributions",
        "distributions_sha256",
    }
)
_PYTHON_ENVIRONMENT_INTERPRETER_KEYS = frozenset(
    {
        "resolved_path",
        "size_bytes",
        "sha256",
        "python_version",
        "version_info",
        "implementation",
        "cache_tag",
        "prefix",
        "base_prefix",
    }
)
_RUNTIME_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "capture_started_at_utc",
        "captured_at_utc",
        "driver_environment",
        "inputs",
        "environment_verification",
        "cosmos_runtime",
        "robocasa_runtime",
        "robocasa_egl",
        "graphics_userspace",
        "nvidia_smi",
        "validation",
    }
)
_RUNTIME_INPUT_KEYS = frozenset(
    {
        "experiment_id",
        "logical_run_id",
        "capture_nonce",
        "submitted_at_utc",
        "num_gpus",
        "submission_verification_receipt",
        "asset_verification_receipt",
        "interpreters",
        "environment_manifests",
        "sources",
    }
)
_RUNTIME_ENVIRONMENT_DESCRIPTOR_KEYS = frozenset(
    {
        "path",
        "size_bytes",
        "sha256",
        "expected_sha256",
        "kind",
        "schema_version",
    }
)
_RUNTIME_ENVIRONMENT_REPORT_KEYS = frozenset(
    {
        "status",
        "interpreter_resolved_path",
        "interpreter_sha256",
        "python_version",
        "distribution_count",
        "distributions_sha256",
    }
)
_RUNTIME_SOURCE_KEYS = frozenset(
    {
        "root",
        "expected_commit",
        "actual_commit",
        "clean",
    }
)
_RUNTIME_VALIDATION_KEYS = frozenset(
    {
        "environment_manifests_match_live_interpreters",
        "source_commits_match_and_are_clean",
        "source_module_paths_match_exact_roots",
        "cuda_available",
        "gpu_count_matches",
        "all_gpus_are_h200_compute_capability_9_0",
        "torch_and_nvidia_smi_uuid_sets_match",
        "h_driver_library_path_is_exact",
        "robocasa_egl_contexts_created_per_shard",
        "submission_identity_and_freshness_match",
        "asset_tree_verification_is_bound",
        "graphics_userspace_files_match_live_hashes",
        "nvidia_gl_vendor_and_renderer_match",
    }
)
_RUNTIME_BOUND_RECEIPT_DESCRIPTOR_KEYS = frozenset(
    {
        "path",
        "size_bytes",
        "sha256",
        "expected_sha256",
        "kind",
        "schema_version",
    }
)
_SUBMISSION_SCHEDULER_EXPECTED_KEYS = frozenset(
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
        "host_network",
        "negative_tags",
        "store_host_nvme",
        "mounts",
    }
)
_EVAL_INTENT_TOP_LEVEL_KEYS = frozenset(
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
_EVAL_INTENT_PROTOCOL_KEYS = frozenset(
    {
        "split",
        "tasks",
        "task_count",
        "episodes_per_task",
        "expected_episode_count",
        "num_shards",
        "base_seed",
        "max_steps",
        "open_loop_horizon",
        "policy_sampler",
        "policy_seed",
        "policy_guidance",
        "policy_num_steps",
        "base_port",
        "server_ready_timeout_seconds",
    }
)
_EVAL_INTENT_RESOURCE_KEYS = frozenset(
    {
        "cpu",
        "memory_mib",
        "gpu",
        "host_network",
        "quota_group",
        "negative_tag",
        "task_type",
        "private_machine",
        "priority",
        "restart_policy",
        "backoff_limit",
        "replicas",
    }
)
_GRAPHICS_FILE_DESCRIPTOR_KEYS = frozenset(
    {
        "supplied_path",
        "canonical_path",
        "size_bytes",
        "sha256",
    }
)
_GRAPHICS_LIBRARY_DESCRIPTOR_KEYS = _GRAPHICS_FILE_DESCRIPTOR_KEYS | frozenset(
    {
        "name",
        "requested_soname",
        "symbol",
        "evidence_class",
        "was_mapped_after_egl_context",
    }
)
_POST_FILE_DESCRIPTOR_KEYS = frozenset({"path", "size_bytes", "sha256"})
_PRIOR_GATE_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "experiment_id",
        "logical_run_id",
        "source_commit",
        "demian",
        "simulator",
        "contract_path",
        "contract_sha256",
        "results_dir",
        "prior_gate_evidence",
        "external_execution_binding",
        "evaluation",
        "execution_complete",
        "effect_pass",
        "artifacts",
        "seed_provenance",
    }
)
_PRIOR_GATE_EVALUATION_KEYS = frozenset(
    {
        "stage",
        "evaluation_class",
        "source_git_pushed_assertion",
        "role",
        "split",
        "tasks",
        "episodes",
        "episodes_per_task",
        "mean_success_rate",
        "successful_tasks",
        "effect_thresholds",
    }
)
_PRIOR_GATE_REQUIRED_ARTIFACT_KEYS = frozenset(
    {
        "source_manifest",
        "checkpoint_manifest",
        "source_dcp_export_receipt",
        "training_config",
        "action_stats",
        "wan2pt2_vae",
        "cosmos_environment_manifest",
        "robocasa_environment_manifest",
        "eval_intent_manifest",
        "eval_runtime_receipt",
    }
)
_PRIOR_GATE_ARTIFACT_KEYS = frozenset(
    {
        "summary_csv",
        "task_success_rates_csv",
        "raw_shards",
        "source_manifest",
        "cosmos_server_module",
        "checkpoint_manifest",
        "checkpoint_verification_receipt",
        "source_dcp_export_receipt",
        "training_config",
        "action_stats",
        "wan2pt2_vae",
        "cosmos_environment_manifest",
        "robocasa_environment_manifest",
        "eval_intent_manifest",
        "submission_verification_receipt",
        "asset_verification_receipt",
        "eval_runtime_receipt",
        "demian_evaluator",
        "demian_task_registry",
        "server_runtime_receipts",
    }
)
_PRIOR_GATE_RUNTIME_EGL_VALIDATION_KEYS = frozenset(
    {
        "h_driver_library_path_is_exact",
        "robocasa_egl_contexts_created_per_shard",
        "graphics_userspace_files_match_live_hashes",
        "nvidia_gl_vendor_and_renderer_match",
    }
)
_PRIOR_GATE_VERIFICATION_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "prior_gate_evidence",
        "gate_identity",
        "gate_evaluation",
        "current_inputs",
        "gate_receipts",
        "validation",
    }
)
_PRIOR_GATE_CURRENT_INPUT_KEYS = frozenset(
    {
        "source_root",
        "source_commit",
        "source_manifest_sha256",
        "checkpoint_path",
        "checkpoint_manifest_sha256",
        "source_dcp_export_receipt_sha256",
        "asset_manifest_sha256",
        "asset_publisher_receipt_sha256",
        "training_config_sha256",
        "action_stats_sha256",
        "wan2pt2_vae_path",
        "wan2pt2_vae_sha256",
        "cosmos_environment_manifest_sha256",
        "robocasa_environment_manifest_sha256",
        "demian_commit",
        "robocasa_commit",
        "robosuite_commit",
    }
)
_PRIOR_GATE_VALIDATION_KEYS = frozenset(
    {
        "evidence_hash_matches_expected",
        "gate_terminal_verdict_passed",
        "gate_contract_matches_evidence",
        "gate_runtime_egl_validation_all_true",
        "gate_inputs_match_current_formal_inputs",
    }
)
_POST_CONTRACT_DESCRIPTOR_KEYS = _POST_FILE_DESCRIPTOR_KEYS | frozenset(
    {
        "kind",
        "schema_version",
    }
)
_POST_ROLLOUT_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "scope",
        "verified_at_utc",
        "contract",
        "inputs",
        "live_verification",
        "semantic_matches",
        "elapsed_seconds",
        "validation",
    }
)
_POST_ROLLOUT_INPUT_KEYS = frozenset(
    {
        "cosmos_snapshot_manifest",
        "serving_checkpoint_manifest",
        "pre_rollout_checkpoint_verification_receipt",
        "pre_rollout_asset_verification_receipt",
        "wan2pt2_vae",
    }
)
_POST_ROLLOUT_LIVE_KEYS = frozenset(
    {
        "cosmos_snapshot",
        "serving_checkpoint",
        "robocasa_assets",
        "wan2pt2_vae",
        "source_repositories",
    }
)
_POST_ROLLOUT_SEMANTIC_MATCH_KEYS = frozenset(
    {
        "checkpoint_pre_and_post",
        "robocasa_assets_pre_and_post",
        "wan2pt2_vae_pre_and_post",
    }
)
_POST_ROLLOUT_ELAPSED_KEYS = frozenset(
    {
        "cosmos_snapshot",
        "serving_checkpoint",
        "robocasa_assets",
        "wan2pt2_vae",
        "source_repositories",
        "total",
    }
)
_POST_ROLLOUT_VALIDATION_KEYS = frozenset(
    {
        "cosmos_snapshot_full_file_verification_passed",
        "serving_checkpoint_full_file_verification_passed",
        "robocasa_asset_full_tree_verification_passed",
        "wan2pt2_vae_content_rehash_passed",
        "source_heads_match_and_are_clean",
        "tracked_source_files_match_head_blobs",
        "pre_and_post_checkpoint_receipts_semantically_match",
        "pre_and_post_asset_receipts_semantically_match",
        "pre_and_post_wan2pt2_vae_semantically_match",
    }
)
_POST_ROLLOUT_BINDING_KEYS = frozenset(
    {
        "scope",
        "receipt",
        "receipt_verified_at_utc",
        "receipt_elapsed_seconds",
        "summary_reverification_elapsed_seconds",
        "semantic_matches",
        "validation",
        "live_verification",
    }
)
_CSV_EPISODE_FIELDS = (
    "experiment_id",
    "logical_run_id",
    "evaluation_role",
    "split",
    "task",
    "task_index",
    "shard",
    "episode_index",
    "seed",
    "success",
    "place_anytime",
    "steps",
    "instruction",
    "checkpoint_manifest_sha256",
    "training_config_sha256",
    "action_stats_sha256",
    "wan2pt2_vae_sha256",
    "demian_commit",
)
_CSV_TASK_FIELDS = (
    "experiment_id",
    "logical_run_id",
    "evaluation_role",
    "split",
    "task",
    "task_index",
    "shard",
    "episodes",
    "successes",
    "success_rate",
    "place_anytime_count",
    "place_anytime_rate",
    "execution_complete",
    "checkpoint_manifest_sha256",
    "training_config_sha256",
    "action_stats_sha256",
    "wan2pt2_vae_sha256",
    "demian_commit",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"JSON contains forbidden non-finite value {value}")


def _parse_json_bytes(payload: bytes, path: Path) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 in {path}: {exc}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: dict[str, Any], expected: frozenset[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(repr(key) for key in observed - expected)}"
        )


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_nonempty_str(value, label)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return digest


def _require_commit(value: Any, label: str) -> str:
    commit = _require_nonempty_str(value, label)
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError(f"{label} must be a full lowercase 40-character Git commit")
    return commit


def _require_run_id(value: Any, label: str) -> str:
    run_id = _require_nonempty_str(value, label)
    if run_id in {".", ".."} or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError(f"{label} must be 1-128 path-safe characters with alphanumeric endpoints")
    return run_id


def _normalized_gpu_uuid(value: Any, label: str) -> str:
    raw = _require_nonempty_str(value, label).lower()
    if raw.startswith("gpu-"):
        raw = raw[4:]
    normalized = raw.replace("-", "")
    if _NORMALIZED_GPU_UUID_RE.fullmatch(normalized) is None:
        raise ValueError(f"{label} is not a valid NVIDIA GPU UUID")
    return normalized


def _close_rate(actual: Any, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise TypeError(f"{label} must be numeric")
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: expected={expected}, actual={actual}")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_exclusive(
    path: Path,
    text: str,
    *,
    create_parent: bool = True,
) -> None:
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"Artifact parent must be an existing non-symlink directory: {path.parent}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    encoded = text.encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    published = False
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path, follow_symlinks=False)
        published = True
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if published:
            path.unlink(missing_ok=True)
        raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _read_stable_regular_file(
    path: Path,
    label: str,
    *,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Read one immutable byte stream from a canonical regular file descriptor."""

    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        canonical = supplied.resolve(strict=True)
        path_before = supplied.lstat()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise ValueError(f"{label} does not resolve to an existing file: {supplied}") from error
    if supplied != canonical or stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"{label} must be a canonical regular non-symlink file: {supplied}")
    if expected_mode is not None and stat.S_IMODE(path_before.st_mode) != expected_mode:
        raise ValueError(
            f"{label} mode mismatch: expected={oct(expected_mode)}, actual={oct(stat.S_IMODE(path_before.st_mode))}"
        )
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for sealed evaluation artifact verification")

    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(supplied, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened without following symlinks: {supplied}") from error
    try:
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode) or _stable_file_identity(path_before) != _stable_file_identity(
            descriptor_before
        ):
            raise RuntimeError(f"{label} was replaced while it was being opened: {supplied}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        descriptor_after = os.fstat(descriptor)
        try:
            path_after = supplied.lstat()
        except OSError as error:
            raise RuntimeError(f"{label} disappeared while it was being read: {supplied}") from error
        stable_identity = _stable_file_identity(descriptor_before)
        if stable_identity != _stable_file_identity(descriptor_after) or stable_identity != _stable_file_identity(
            path_after
        ):
            raise RuntimeError(f"{label} changed or was replaced while it was being read: {supplied}")
    finally:
        os.close(descriptor)

    payload = b"".join(chunks)
    if len(payload) != int(descriptor_before.st_size):
        raise RuntimeError(f"{label} byte count changed while it was being read: {supplied}")
    return payload, {
        "path": str(supplied),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _hash_stable_regular_file(path: Path, label: str) -> dict[str, Any]:
    """Stream-hash a canonical regular file while rejecting path/inode races."""

    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        canonical = supplied.resolve(strict=True)
        path_before = supplied.lstat()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise ValueError(f"{label} does not resolve to an existing file: {supplied}") from error
    if supplied != canonical or stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"{label} must be a canonical regular non-symlink file: {supplied}")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for sealed evaluation artifact verification")

    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(supplied, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened without following symlinks: {supplied}") from error
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode) or _stable_file_identity(path_before) != _stable_file_identity(
            descriptor_before
        ):
            raise RuntimeError(f"{label} was replaced while it was being opened: {supplied}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
        descriptor_after = os.fstat(descriptor)
        try:
            path_after = supplied.lstat()
        except OSError as error:
            raise RuntimeError(f"{label} disappeared while it was being hashed: {supplied}") from error
        stable_identity = _stable_file_identity(descriptor_before)
        if stable_identity != _stable_file_identity(descriptor_after) or stable_identity != _stable_file_identity(
            path_after
        ):
            raise RuntimeError(f"{label} changed or was replaced while it was being hashed: {supplied}")
    finally:
        os.close(descriptor)
    if size_bytes != int(descriptor_before.st_size):
        raise RuntimeError(f"{label} byte count changed while it was being hashed: {supplied}")
    return {
        "path": str(supplied),
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _load_stable_json(
    path: Path,
    label: str,
    *,
    expected_mode: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    payload, descriptor = _read_stable_regular_file(
        path,
        label,
        expected_mode=expected_mode,
    )
    return _parse_json_bytes(payload, Path(descriptor["path"])), descriptor


def _validate_post_file_descriptor(
    value: Any,
    *,
    expected: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    descriptor = _require_dict(value, label)
    _require_exact_keys(descriptor, _POST_FILE_DESCRIPTOR_KEYS, label)
    path = _absolute_resolved_path(descriptor.get("path"), f"{label}.path")
    if path != Path(expected["path"]):
        raise ValueError(f"{label}.path does not match the exact contract-derived file")
    size_bytes = _require_int(descriptor.get("size_bytes"), f"{label}.size_bytes", minimum=1)
    sha256 = _require_sha256(descriptor.get("sha256"), f"{label}.sha256")
    if size_bytes != expected["size_bytes"] or sha256 != expected["sha256"]:
        raise ValueError(f"{label} does not match the exact live file descriptor")
    return dict(descriptor)


def _canonical_git_root(root: Path, label: str) -> Path:
    supplied = Path(os.path.abspath(root.expanduser()))
    try:
        canonical = supplied.resolve(strict=True)
        metadata = canonical.lstat()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise ValueError(f"{label} does not resolve to an existing repository: {supplied}") from error
    if supplied != canonical or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a canonical non-symlink directory: {supplied}")
    return canonical


def _run_git(
    root: Path,
    arguments: list[str],
    *,
    label: str,
    input_bytes: bytes | None = None,
) -> bytes:
    git = _GIT_EXECUTABLE.resolve(strict=True)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    command = [
        str(git),
        "--no-optional-locks",
        "-c",
        f"safe.directory={root}",
        "-C",
        str(root),
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=_GIT_COMMAND_TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"{label} could not execute safely: {error}") from error
    if result.returncode != 0:
        diagnostics = (result.stderr or result.stdout)[-4000:].decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{label} failed with exit code {result.returncode}: {diagnostics or 'no diagnostics'}")
    if len(result.stdout) > 256 * 1024 * 1024 or len(result.stderr) > 1024 * 1024:
        raise RuntimeError(f"{label} produced an unexpectedly large response")
    return result.stdout


def _single_git_line(root: Path, arguments: list[str], *, label: str) -> str:
    try:
        text = _run_git(root, arguments, label=label).decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} did not emit valid UTF-8") from error
    lines = text.splitlines()
    if len(lines) != 1 or lines[0] != lines[0].strip():
        raise ValueError(f"{label} must emit exactly one trimmed line")
    return lines[0]


def _git_status_clean(root: Path, label: str) -> None:
    status = _run_git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        label=f"{label} git status",
    )
    if status:
        detail = status[:2000].decode("utf-8", errors="replace")
        raise ValueError(f"{label} source checkout is dirty: {detail}")


def _verify_tracked_regular_files(root: Path, label: str) -> dict[str, Any]:
    object_format = _single_git_line(
        root,
        ["rev-parse", "--show-object-format"],
        label=f"{label} git object format",
    )
    if object_format not in {"sha1", "sha256"}:
        raise ValueError(f"{label} uses an unsupported Git object format: {object_format!r}")
    oid_length = 40 if object_format == "sha1" else 64
    tree_oid = _single_git_line(
        root,
        ["rev-parse", "--verify", "HEAD^{tree}"],
        label=f"{label} HEAD tree",
    )
    if re.fullmatch(rf"[0-9a-f]{{{oid_length}}}", tree_oid) is None:
        raise ValueError(f"{label} HEAD tree did not produce a canonical object ID")

    tree_payload = _run_git(
        root,
        ["ls-tree", "-r", "-z", "--full-tree", "HEAD"],
        label=f"{label} tracked tree",
    )
    raw_entries = tree_payload.split(b"\0")
    if raw_entries[-1:] != [b""]:
        raise ValueError(f"{label} tracked tree output is not NUL terminated")
    tracked_entry_count = 0
    non_regular_entry_count = 0
    regular_entries: list[tuple[str, str, str, os.stat_result]] = []
    seen_paths: set[str] = set()
    for raw_entry in raw_entries[:-1]:
        tracked_entry_count += 1
        try:
            header, raw_path = raw_entry.split(b"\t", maxsplit=1)
            mode, object_type, object_id = header.decode("ascii", errors="strict").split(" ")
            relative_path = raw_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"{label} tracked tree entry is malformed") from error
        if (
            not relative_path
            or "\n" in relative_path
            or "\r" in relative_path
            or "\x00" in relative_path
            or PurePosixPath(relative_path).is_absolute()
            or any(component in {"", ".", ".."} for component in PurePosixPath(relative_path).parts)
            or PurePosixPath(relative_path).as_posix() != relative_path
            or relative_path in seen_paths
        ):
            raise ValueError(f"{label} contains a non-canonical tracked path: {relative_path!r}")
        seen_paths.add(relative_path)
        if re.fullmatch(rf"[0-9a-f]{{{oid_length}}}", object_id) is None:
            raise ValueError(f"{label} tracked entry has an invalid object ID: {relative_path!r}")
        if object_type != "blob" or mode not in {"100644", "100755"}:
            non_regular_entry_count += 1
            continue
        live_path = root.joinpath(*PurePosixPath(relative_path).parts)
        try:
            metadata = live_path.lstat()
        except OSError as error:
            raise ValueError(f"{label} tracked regular file is missing: {relative_path!r}") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} tracked regular file is not a live regular file: {relative_path!r}")
        live_mode = stat.S_IMODE(metadata.st_mode)
        executable = bool(live_mode & 0o111)
        expected_executable = mode == "100755"
        if executable is not expected_executable:
            raise ValueError(
                f"{label} tracked regular file executable-bit semantics differ from HEAD: "
                f"path={relative_path!r}, git_mode={mode}, live_mode={oct(live_mode)}"
            )
        regular_entries.append((relative_path, mode, object_id, metadata))
    if not regular_entries:
        raise ValueError(f"{label} must contain at least one tracked regular file")

    path_input = "".join(f"{relative_path}\n" for relative_path, *_ in regular_entries).encode("utf-8")
    live_hashes_payload = _run_git(
        root,
        ["hash-object", "--no-filters", "--stdin-paths"],
        label=f"{label} live tracked file hashing",
        input_bytes=path_input,
    )
    try:
        live_hashes = live_hashes_payload.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} live tracked file hashes are not ASCII") from error
    if len(live_hashes) != len(regular_entries):
        raise ValueError(f"{label} live tracked file hash count differs from HEAD")

    records: list[dict[str, Any]] = []
    total_bytes = 0
    for (relative_path, mode, expected_oid, before), actual_oid in zip(
        regular_entries,
        live_hashes,
        strict=True,
    ):
        if actual_oid != expected_oid:
            raise ValueError(
                f"{label} tracked file content differs from HEAD: "
                f"path={relative_path!r}, expected={expected_oid}, actual={actual_oid}"
            )
        live_path = root.joinpath(*PurePosixPath(relative_path).parts)
        after = live_path.lstat()
        if _stable_file_identity(before) != _stable_file_identity(after):
            raise RuntimeError(f"{label} tracked file changed while hashing: {relative_path!r}")
        total_bytes += int(after.st_size)
        records.append(
            {
                "path": relative_path,
                "git_mode": mode,
                "head_blob_oid": expected_oid,
                "live_mode": stat.S_IMODE(after.st_mode),
                "size_bytes": int(after.st_size),
            }
        )
    manifest_payload = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "object_format": object_format,
        "head_tree_oid": tree_oid,
        "tracked_entry_count": tracked_entry_count,
        "tracked_regular_file_count": len(records),
        "tracked_non_regular_entry_count": non_regular_entry_count,
        "tracked_regular_total_bytes": total_bytes,
        "tracked_regular_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "all_tracked_regular_files_match_head_blobs": True,
    }


def _verify_git_repository(
    *,
    name: str,
    root: Path,
    expected_commit: str,
    verify_tracked_files: bool,
) -> dict[str, Any]:
    canonical = _canonical_git_root(root, f"{name} source root")
    expected = _require_commit(expected_commit, f"{name} expected commit")
    top_level = _single_git_line(
        canonical,
        ["rev-parse", "--show-toplevel"],
        label=f"{name} Git top-level",
    )
    if Path(top_level).resolve(strict=True) != canonical:
        raise ValueError(f"{name} Git top-level differs from the contract root")
    actual = _single_git_line(
        canonical,
        ["rev-parse", "--verify", "HEAD"],
        label=f"{name} Git HEAD",
    )
    if _require_commit(actual, f"{name} actual commit") != expected:
        raise ValueError(f"{name} Git HEAD differs from the eval contract")
    _git_status_clean(canonical, name)
    tracked_files = (
        _verify_tracked_regular_files(canonical, name)
        if verify_tracked_files
        else {
            "verification_mode": "covered_by_full_cosmos_snapshot_verification",
        }
    )
    if (
        _single_git_line(
            canonical,
            ["rev-parse", "--verify", "HEAD"],
            label=f"{name} final Git HEAD",
        )
        != expected
    ):
        raise RuntimeError(f"{name} Git HEAD changed during post-rollout verification")
    _git_status_clean(canonical, f"{name} final")
    return {
        "root": str(canonical),
        "expected_commit": expected,
        "actual_commit": actual,
        "clean": True,
        "tracked_files": tracked_files,
    }


def _render_csv(fieldnames: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _atomic_write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_exclusive(
        path,
        _render_csv(fieldnames, rows),
        create_parent=False,
    )


def _read_bound_file_artifact(
    artifacts: dict[str, Any],
    name: str,
    *,
    expected_mode: int | None = None,
) -> tuple[dict[str, str], bytes, dict[str, Any]]:
    descriptor = _require_dict(artifacts.get(name), f"contract.artifacts.{name}")
    _require_exact_keys(
        descriptor,
        frozenset({"path", "sha256"}),
        f"contract.artifacts.{name}",
    )
    path = Path(_require_nonempty_str(descriptor.get("path"), f"contract.artifacts.{name}.path"))
    if not path.is_absolute():
        raise ValueError(f"contract.artifacts.{name}.path must be absolute: {path}")
    expected = _require_sha256(
        descriptor.get("sha256"),
        f"contract.artifacts.{name}.sha256",
    )
    payload, stable_descriptor = _read_stable_regular_file(
        path,
        f"contract.artifacts.{name}",
        expected_mode=expected_mode,
    )
    if stable_descriptor["sha256"] != expected:
        raise RuntimeError(
            f"{name} SHA256 mismatch: expected={expected}, actual={stable_descriptor['sha256']}, path={path}"
        )
    public_descriptor = {
        "path": stable_descriptor["path"],
        "sha256": stable_descriptor["sha256"],
    }
    return public_descriptor, payload, stable_descriptor


def _absolute_resolved_path(value: Any, label: str) -> Path:
    path = Path(_require_nonempty_str(value, label))
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    try:
        return path.expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise ValueError(f"{label} must resolve to an existing path: {path}") from error


def _absolute_lexical_path(value: Any, label: str) -> Path:
    text = _require_nonempty_str(value, label)
    path = Path(text)
    if not path.is_absolute() or os.path.normpath(text) != text or os.fspath(path) != text:
        raise ValueError(f"{label} must be an exact canonical-form absolute path: {path}")
    return path


def _absolute_canonical_existing_path(
    value: Any,
    label: str,
    *,
    directory: bool,
) -> Path:
    supplied = _absolute_lexical_path(value, label)
    try:
        resolved = supplied.resolve(strict=True)
        metadata = supplied.lstat()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise ValueError(f"{label} must resolve to an existing path: {supplied}") from error
    expected_kind = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if supplied != resolved or stat.S_ISLNK(metadata.st_mode) or not expected_kind:
        kind = "directory" if directory else "regular file"
        raise ValueError(f"{label} must be a canonical non-symlink {kind}: {supplied}")
    return resolved


def _validated_environment_launcher(
    *,
    root: Path,
    manifest_interpreter: dict[str, Any],
    label: str,
) -> tuple[Path, Path]:
    bin_directory = _absolute_canonical_existing_path(
        str(root / "bin"),
        f"{label} bin directory",
        directory=True,
    )
    launcher = bin_directory / "python"
    try:
        before = launcher.lstat()
        resolved = launcher.resolve(strict=True)
        after = launcher.lstat()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise ValueError(f"{label} launcher must resolve safely: {launcher}") from error
    if not (stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)):
        raise ValueError(f"{label} launcher must be a regular file or leaf symlink: {launcher}")
    if _stable_file_identity(before) != _stable_file_identity(after):
        raise ValueError(f"{label} launcher changed while resolving: {launcher}")
    resolved = _absolute_canonical_existing_path(
        str(resolved),
        f"{label} launcher target",
        directory=False,
    )
    manifest_resolved = _absolute_canonical_existing_path(
        manifest_interpreter.get("resolved_path"),
        f"{label} manifest interpreter.resolved_path",
        directory=False,
    )
    if resolved != manifest_resolved:
        raise ValueError(f"{label} launcher target does not match the environment manifest")
    if not os.access(launcher, os.X_OK) or not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} launcher and target must be executable: {launcher}")
    launcher_identity = _stable_file_identity(after)
    target_identity = _stable_file_identity(resolved.lstat())
    _payload, target_descriptor = _read_stable_regular_file(
        resolved,
        f"{label} launcher target",
    )
    expected_size = _require_int(
        manifest_interpreter.get("size_bytes"),
        f"{label} manifest interpreter.size_bytes",
        minimum=1,
    )
    expected_sha256 = _require_sha256(
        manifest_interpreter.get("sha256"),
        f"{label} manifest interpreter.sha256",
    )
    if target_descriptor["size_bytes"] != expected_size or target_descriptor["sha256"] != expected_sha256:
        raise ValueError(f"{label} launcher target bytes do not match the environment manifest")
    try:
        launcher_before_resolve = launcher.lstat()
        launcher_target_after = launcher.resolve(strict=True)
        launcher_after_resolve = launcher.lstat()
        target_after = resolved.lstat()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise ValueError(f"{label} launcher changed while hashing: {launcher}") from error
    if (
        _stable_file_identity(launcher_before_resolve) != launcher_identity
        or _stable_file_identity(launcher_after_resolve) != launcher_identity
        or launcher_target_after != manifest_resolved
        or _stable_file_identity(target_after) != target_identity
        or not os.access(launcher, os.X_OK)
        or not os.access(resolved, os.X_OK)
    ):
        raise ValueError(f"{label} launcher or target changed while hashing: {launcher}")
    return launcher, resolved


def _parse_utc(value: Any, label: str) -> dt.datetime:
    text = _require_nonempty_str(value, label)
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} is not a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError(f"{label} must identify UTC")
    return parsed.astimezone(dt.timezone.utc)


def _validate_bound_receipt_descriptor(
    value: Any,
    *,
    artifact: dict[str, str],
    stable_artifact: dict[str, Any],
    expected_kind: str,
    label: str,
) -> dict[str, Any]:
    descriptor = _require_dict(value, label)
    _require_exact_keys(descriptor, _RUNTIME_BOUND_RECEIPT_DESCRIPTOR_KEYS, label)
    artifact_path = Path(artifact["path"]).resolve(strict=True)
    if _absolute_resolved_path(descriptor.get("path"), f"{label}.path") != artifact_path:
        raise ValueError(f"{label}.path does not match the eval contract artifact")
    size_bytes = _require_int(descriptor.get("size_bytes"), f"{label}.size_bytes", minimum=1)
    if size_bytes != stable_artifact["size_bytes"]:
        raise ValueError(f"{label}.size_bytes does not match the live receipt")
    if (
        _require_sha256(descriptor.get("sha256"), f"{label}.sha256") != artifact["sha256"]
        or _require_sha256(
            descriptor.get("expected_sha256"),
            f"{label}.expected_sha256",
        )
        != artifact["sha256"]
        or stable_artifact["sha256"] != artifact["sha256"]
        or stable_artifact["path"] != artifact["path"]
    ):
        raise ValueError(f"{label} does not bind the exact live receipt SHA256")
    if descriptor.get("kind") != expected_kind or descriptor.get("schema_version") != 1:
        raise ValueError(f"{label} has an unsupported schema/kind")
    return dict(descriptor)


def _validate_graphics_file_descriptor(
    value: Any,
    *,
    label: str,
    expected_keys: frozenset[str],
) -> dict[str, Any]:
    descriptor = _require_dict(value, label)
    _require_exact_keys(descriptor, expected_keys, label)
    supplied_path = Path(_require_nonempty_str(descriptor.get("supplied_path"), f"{label}.supplied_path"))
    if not supplied_path.is_absolute():
        raise ValueError(f"{label}.supplied_path must be absolute")
    try:
        canonical_path = supplied_path.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise ValueError(f"{label}.supplied_path does not resolve to a live file") from error
    if not canonical_path.is_file():
        raise ValueError(f"{label}.supplied_path must resolve to a regular file")
    recorded_canonical = Path(
        _require_nonempty_str(
            descriptor.get("canonical_path"),
            f"{label}.canonical_path",
        )
    )
    if not recorded_canonical.is_absolute() or recorded_canonical != canonical_path:
        raise ValueError(f"{label}.canonical_path does not match the live resolution")
    payload, stable_descriptor = _read_stable_regular_file(
        canonical_path,
        f"{label}.live_file",
    )
    size_bytes = _require_int(descriptor.get("size_bytes"), f"{label}.size_bytes", minimum=1)
    if stable_descriptor["size_bytes"] != size_bytes:
        raise ValueError(f"{label}.size_bytes does not match the live graphics file")
    sha256 = _require_sha256(descriptor.get("sha256"), f"{label}.sha256")
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"{label}.sha256 does not match the live graphics file")
    return dict(descriptor)


def _validate_graphics_userspace(value: Any, label: str) -> dict[str, Any]:
    graphics = _require_dict(value, label)
    _require_exact_keys(
        graphics,
        frozenset({"egl_vendor_json", "libraries", "gl_identity"}),
        label,
    )
    vendor_json = _validate_graphics_file_descriptor(
        graphics.get("egl_vendor_json"),
        label=f"{label}.egl_vendor_json",
        expected_keys=_GRAPHICS_FILE_DESCRIPTOR_KEYS | frozenset({"file_format_version", "icd_library_path"}),
    )
    if (
        vendor_json["supplied_path"] != _EXPECTED_EGL_VENDOR_JSON
        or vendor_json["file_format_version"] != "1.0.0"
        or vendor_json["icd_library_path"] != "libEGL_nvidia.so.0"
    ):
        raise ValueError(f"{label}.egl_vendor_json does not match the pinned NVIDIA ICD")

    expected_libraries = {
        "libEGL": ("libEGL.so.1", "eglGetDisplay", "egl_context_mapped", True),
        "libEGL_nvidia": (
            "libEGL_nvidia.so.0",
            "__egl_Main",
            "egl_context_mapped",
            True,
        ),
        "libGL": (
            "libGL.so.1",
            "glGetString",
            "resolved_and_dlopen_verified",
            None,
        ),
        "libGLX": (
            "libGLX.so.0",
            "glXGetProcAddress",
            "resolved_and_dlopen_verified",
            None,
        ),
        "libGLX_nvidia": (
            "libGLX_nvidia.so.0",
            "__glx_Main",
            "resolved_and_dlopen_verified",
            None,
        ),
    }
    libraries = _require_dict(graphics.get("libraries"), f"{label}.libraries")
    _require_exact_keys(libraries, frozenset(expected_libraries), f"{label}.libraries")
    normalized_libraries: dict[str, Any] = {}
    for name, (soname, symbol, evidence_class, required_mapped) in expected_libraries.items():
        library_label = f"{label}.libraries.{name}"
        record = _validate_graphics_file_descriptor(
            libraries.get(name),
            label=library_label,
            expected_keys=_GRAPHICS_LIBRARY_DESCRIPTOR_KEYS,
        )
        expected = {
            "name": name,
            "requested_soname": soname,
            "symbol": symbol,
            "evidence_class": evidence_class,
        }
        for key, expected_value in expected.items():
            if record.get(key) != expected_value:
                raise ValueError(
                    f"{library_label}.{key} mismatch: expected={expected_value!r}, actual={record.get(key)!r}"
                )
        mapped = record.get("was_mapped_after_egl_context")
        if type(mapped) is not bool:
            raise TypeError(f"{library_label}.was_mapped_after_egl_context must be boolean")
        if required_mapped is True and mapped is not True:
            raise ValueError(f"{library_label} was not mapped by the live EGL context")
        normalized_libraries[name] = record

    gl_identity = _require_dict(graphics.get("gl_identity"), f"{label}.gl_identity")
    _require_exact_keys(gl_identity, frozenset({"vendor", "renderer"}), f"{label}.gl_identity")
    vendor = _require_nonempty_str(gl_identity.get("vendor"), f"{label}.gl_identity.vendor")
    renderer = _require_nonempty_str(
        gl_identity.get("renderer"),
        f"{label}.gl_identity.renderer",
    )
    if "NVIDIA" not in vendor.upper():
        raise ValueError(f"{label}.gl_identity.vendor is not NVIDIA")
    if "MESA" in renderer.upper() or "LLVMPIPE" in renderer.upper():
        raise ValueError(f"{label}.gl_identity.renderer is software/Mesa")
    return {
        "egl_vendor_json": vendor_json,
        "libraries": normalized_libraries,
        "gl_identity": {"vendor": vendor, "renderer": renderer},
    }


def _validated_environment_manifest(
    manifest_value: Any,
    label: str,
) -> dict[str, Any]:
    manifest = _require_dict(manifest_value, label)
    _require_exact_keys(manifest, _PYTHON_ENVIRONMENT_MANIFEST_KEYS, label)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "cwam_python_environment_manifest":
        raise ValueError(f"{label} has an unsupported schema/kind")

    interpreter = _require_dict(manifest.get("interpreter"), f"{label}.interpreter")
    _require_exact_keys(
        interpreter,
        _PYTHON_ENVIRONMENT_INTERPRETER_KEYS,
        f"{label}.interpreter",
    )
    resolved_interpreter = _absolute_canonical_existing_path(
        interpreter.get("resolved_path"),
        f"{label}.interpreter.resolved_path",
        directory=False,
    )
    _absolute_canonical_existing_path(
        interpreter.get("prefix"),
        f"{label}.interpreter.prefix",
        directory=True,
    )
    _absolute_canonical_existing_path(
        interpreter.get("base_prefix"),
        f"{label}.interpreter.base_prefix",
        directory=True,
    )
    if not os.access(resolved_interpreter, os.X_OK):
        raise ValueError(f"{label}.interpreter.resolved_path must be executable")
    interpreter_size = _require_int(
        interpreter.get("size_bytes"),
        f"{label}.interpreter.size_bytes",
        minimum=1,
    )
    interpreter_payload, interpreter_descriptor = _read_stable_regular_file(
        resolved_interpreter,
        f"{label}.interpreter.live_file",
    )
    if interpreter_descriptor["size_bytes"] != interpreter_size:
        raise ValueError(f"{label}.interpreter.size_bytes does not match the interpreter file")
    interpreter_sha256 = _require_sha256(
        interpreter.get("sha256"),
        f"{label}.interpreter.sha256",
    )
    if hashlib.sha256(interpreter_payload).hexdigest() != interpreter_sha256:
        raise ValueError(f"{label}.interpreter.sha256 does not match the interpreter file")
    python_version = _require_nonempty_str(
        interpreter.get("python_version"),
        f"{label}.interpreter.python_version",
    )
    version_info = _require_dict(
        interpreter.get("version_info"),
        f"{label}.interpreter.version_info",
    )
    _require_exact_keys(
        version_info,
        frozenset({"major", "minor", "micro", "releaselevel", "serial"}),
        f"{label}.interpreter.version_info",
    )
    major = _require_int(version_info.get("major"), f"{label}.interpreter.version_info.major", minimum=0)
    minor = _require_int(version_info.get("minor"), f"{label}.interpreter.version_info.minor", minimum=0)
    micro = _require_int(version_info.get("micro"), f"{label}.interpreter.version_info.micro", minimum=0)
    _require_int(
        version_info.get("serial"),
        f"{label}.interpreter.version_info.serial",
        minimum=0,
    )
    _require_nonempty_str(
        version_info.get("releaselevel"),
        f"{label}.interpreter.version_info.releaselevel",
    )
    if python_version != f"{major}.{minor}.{micro}":
        raise ValueError(f"{label}.interpreter.python_version does not match version_info")
    _require_nonempty_str(
        interpreter.get("implementation"),
        f"{label}.interpreter.implementation",
    )
    _require_nonempty_str(
        interpreter.get("cache_tag"),
        f"{label}.interpreter.cache_tag",
    )

    distributions = _require_dict(manifest.get("distributions"), f"{label}.distributions")
    distribution_count = _require_int(
        manifest.get("distribution_count"),
        f"{label}.distribution_count",
        minimum=1,
    )
    if distribution_count != len(distributions):
        raise ValueError(
            f"{label}.distribution_count mismatch: expected={len(distributions)}, actual={distribution_count}"
        )
    for name, raw_record in distributions.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}.distributions contains an invalid name: {name!r}")
        record = _require_dict(raw_record, f"{label}.distributions[{name!r}]")
        _require_exact_keys(
            record,
            frozenset({"name", "version"}),
            f"{label}.distributions[{name!r}]",
        )
        recorded_name = _require_nonempty_str(
            record.get("name"),
            f"{label}.distributions[{name!r}].name",
        )
        canonical_name = _NORMALIZE_DISTRIBUTION_NAME_RE.sub("-", recorded_name).lower()
        if name != canonical_name:
            raise ValueError(
                f"{label}.distributions key is not the canonical form of its recorded name: "
                f"key={name!r}, recorded_name={recorded_name!r}"
            )
        _require_nonempty_str(
            record.get("version"),
            f"{label}.distributions[{name!r}].version",
        )
    distributions_sha256 = hashlib.sha256(
        json.dumps(
            distributions,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    if (
        _require_sha256(
            manifest.get("distributions_sha256"),
            f"{label}.distributions_sha256",
        )
        != distributions_sha256
    ):
        raise ValueError(f"{label}.distributions_sha256 does not match its inventory")
    return manifest


def _validate_environment_runtime_binding(
    *,
    name: str,
    artifact: dict[str, str],
    stable_artifact: dict[str, Any],
    manifest: dict[str, Any],
    receipt_descriptor: Any,
    receipt_interpreter: Any,
    verification_report: Any,
) -> None:
    descriptor_label = f"eval_runtime_receipt.inputs.environment_manifests.{name}"
    descriptor = _require_dict(receipt_descriptor, descriptor_label)
    _require_exact_keys(
        descriptor,
        _RUNTIME_ENVIRONMENT_DESCRIPTOR_KEYS,
        descriptor_label,
    )
    if _absolute_resolved_path(descriptor.get("path"), f"{descriptor_label}.path") != Path(artifact["path"]).resolve():
        raise ValueError(f"{descriptor_label}.path does not match the eval contract")
    if descriptor.get("kind") != "cwam_python_environment_manifest" or descriptor.get("schema_version") != 1:
        raise ValueError(f"{descriptor_label} has an unsupported schema/kind")
    if (
        _require_sha256(descriptor.get("sha256"), f"{descriptor_label}.sha256") != artifact["sha256"]
        or _require_sha256(
            descriptor.get("expected_sha256"),
            f"{descriptor_label}.expected_sha256",
        )
        != artifact["sha256"]
    ):
        raise ValueError(f"{descriptor_label} SHA256 does not match the eval contract")
    if (
        _require_int(
            descriptor.get("size_bytes"),
            f"{descriptor_label}.size_bytes",
            minimum=1,
        )
        != stable_artifact["size_bytes"]
    ):
        raise ValueError(f"{descriptor_label}.size_bytes does not match the manifest file")

    interpreter_label = f"eval_runtime_receipt.inputs.interpreters.{name}"
    interpreter = _require_dict(receipt_interpreter, interpreter_label)
    _require_exact_keys(
        interpreter,
        frozenset({"launch_path", "resolved_path"}),
        interpreter_label,
    )
    launch_path = _absolute_lexical_path(
        interpreter.get("launch_path"),
        f"{interpreter_label}.launch_path",
    )
    resolved_path = _absolute_canonical_existing_path(
        interpreter.get("resolved_path"),
        f"{interpreter_label}.resolved_path",
        directory=False,
    )
    manifest_interpreter = _require_dict(
        manifest.get("interpreter"),
        f"{name}_environment_manifest.interpreter",
    )
    environment_root = _absolute_canonical_existing_path(
        manifest_interpreter.get("prefix"),
        f"{name}_environment_manifest.interpreter.prefix",
        directory=True,
    )
    expected_launch_path, manifest_resolved_path = _validated_environment_launcher(
        root=environment_root,
        manifest_interpreter=manifest_interpreter,
        label=f"{name} environment",
    )
    if launch_path != expected_launch_path:
        raise ValueError(f"{interpreter_label}.launch_path is not the exact environment launcher")
    if resolved_path != manifest_resolved_path:
        raise ValueError(f"{interpreter_label} does not resolve to the manifest interpreter")

    report_label = f"eval_runtime_receipt.environment_verification.{name}"
    report = _require_dict(verification_report, report_label)
    _require_exact_keys(report, _RUNTIME_ENVIRONMENT_REPORT_KEYS, report_label)
    expected_report = {
        "status": "passed",
        "interpreter_resolved_path": str(manifest_resolved_path),
        "interpreter_sha256": manifest_interpreter["sha256"],
        "python_version": manifest_interpreter["python_version"],
        "distribution_count": manifest["distribution_count"],
        "distributions_sha256": manifest["distributions_sha256"],
    }
    if report != expected_report:
        raise ValueError(f"{report_label} does not match the environment manifest")


def _stable_public_descriptor(stable: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": stable["path"],
        "size_bytes": stable["size_bytes"],
        "sha256": stable["sha256"],
    }


def _validate_submission_preflight(
    *,
    preflight_value: Any,
    expected_sha256: Any,
    contract: dict[str, Any],
    asset_verification: dict[str, Any],
) -> dict[str, Any]:
    preflight = _require_dict(preflight_value, "submission_verification_receipt.preflight")
    _require_exact_keys(
        preflight,
        frozenset({"asset_tree", "repositories", "validation"}),
        "submission_verification_receipt.preflight",
    )
    encoded = json.dumps(
        preflight,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = _require_sha256(
        expected_sha256,
        "submission_verification_receipt.preflight_sha256",
    )
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError("submission preflight canonical SHA256 does not match")

    asset_tree = _require_dict(
        preflight.get("asset_tree"),
        "submission_verification_receipt.preflight.asset_tree",
    )
    if asset_tree != asset_verification:
        raise ValueError("submission preflight asset tree differs from the bound asset verification")

    repositories = _require_dict(
        preflight.get("repositories"),
        "submission_verification_receipt.preflight.repositories",
    )
    _require_exact_keys(
        repositories,
        frozenset({"demian", "robocasa", "robosuite"}),
        "submission_verification_receipt.preflight.repositories",
    )
    expected_repositories = {
        "demian": (
            Path(contract["demian"]["repo"]).resolve(),
            contract["demian"]["commit"],
        ),
        "robocasa": (
            Path(contract["simulator"]["robocasa_repo"]).resolve(),
            contract["simulator"]["robocasa_commit"],
        ),
        "robosuite": (
            Path(contract["simulator"]["robosuite_repo"]).resolve(),
            contract["simulator"]["robosuite_commit"],
        ),
    }
    tracked_keys = frozenset(
        {
            "object_format",
            "head_tree_oid",
            "tracked_entry_count",
            "tracked_regular_file_count",
            "tracked_non_regular_entry_count",
            "tracked_regular_total_bytes",
            "tracked_regular_manifest_sha256",
            "all_tracked_regular_files_match_head_blobs",
        }
    )
    for name, (expected_root, expected_commit) in expected_repositories.items():
        label = f"submission_verification_receipt.preflight.repositories.{name}"
        report = _require_dict(repositories.get(name), label)
        _require_exact_keys(
            report,
            frozenset({"root", "expected_commit", "actual_commit", "clean", "tracked_files"}),
            label,
        )
        if (
            _absolute_resolved_path(report.get("root"), f"{label}.root") != expected_root
            or _require_commit(report.get("expected_commit"), f"{label}.expected_commit") != expected_commit
            or _require_commit(report.get("actual_commit"), f"{label}.actual_commit") != expected_commit
            or report.get("clean") is not True
        ):
            raise ValueError(f"{label} differs from the clean contract repository")
        tracked = _require_dict(report.get("tracked_files"), f"{label}.tracked_files")
        _require_exact_keys(tracked, tracked_keys, f"{label}.tracked_files")
        object_format = tracked.get("object_format")
        if object_format not in {"sha1", "sha256"}:
            raise ValueError(f"{label}.tracked_files.object_format is unsupported")
        oid_length = 40 if object_format == "sha1" else 64
        if (
            not isinstance(tracked.get("head_tree_oid"), str)
            or re.fullmatch(rf"[0-9a-f]{{{oid_length}}}", tracked["head_tree_oid"]) is None
        ):
            raise ValueError(f"{label}.tracked_files.head_tree_oid is invalid")
        for key in (
            "tracked_entry_count",
            "tracked_regular_file_count",
            "tracked_non_regular_entry_count",
            "tracked_regular_total_bytes",
        ):
            minimum = 1 if key in {"tracked_entry_count", "tracked_regular_file_count"} else 0
            _require_int(tracked.get(key), f"{label}.tracked_files.{key}", minimum=minimum)
        _require_sha256(
            tracked.get("tracked_regular_manifest_sha256"),
            f"{label}.tracked_files.tracked_regular_manifest_sha256",
        )
        if tracked.get("all_tracked_regular_files_match_head_blobs") is not True:
            raise ValueError(f"{label}.tracked_files is not an all-true tracked blob verdict")

    validation = _require_dict(
        preflight.get("validation"),
        "submission_verification_receipt.preflight.validation",
    )
    expected_validation = {
        "asset_tree_full_hash_match": True,
        "repository_heads_clean": True,
        "tracked_regular_files_match_head_blobs": True,
    }
    if validation != expected_validation:
        raise ValueError("submission preflight validation is not the exact all-true verdict")
    return {
        "sha256": digest,
        "asset_tree": {
            "manifest_sha256": asset_tree["manifest_sha256"],
            "publisher_receipt_sha256": asset_tree["publisher_receipt_sha256"],
            "robocasa_git_commit": asset_tree["robocasa_git_commit"],
        },
        "repositories": repositories,
        "validation": validation,
    }


def _validate_external_execution_bindings(
    *,
    contract: dict[str, Any],
    verified: dict[str, Any],
    stable_artifacts: dict[str, dict[str, Any]],
    bound_payloads: dict[str, Any],
    bound_json: dict[str, Any],
    inputs: dict[str, Any],
    capture_started: dt.datetime,
    captured_at: dt.datetime,
) -> dict[str, Any]:
    submission_descriptor = _validate_bound_receipt_descriptor(
        inputs.get("submission_verification_receipt"),
        artifact=verified["submission_verification_receipt"],
        stable_artifact=stable_artifacts["submission_verification_receipt"],
        expected_kind=_SUBMISSION_VERIFICATION_KIND,
        label="eval_runtime_receipt.inputs.submission_verification_receipt",
    )
    asset_descriptor = _validate_bound_receipt_descriptor(
        inputs.get("asset_verification_receipt"),
        artifact=verified["asset_verification_receipt"],
        stable_artifact=stable_artifacts["asset_verification_receipt"],
        expected_kind=_ASSET_VERIFICATION_KIND,
        label="eval_runtime_receipt.inputs.asset_verification_receipt",
    )

    submission = _require_dict(bound_json["submission_verification_receipt"], "submission_verification_receipt")
    _require_exact_keys(
        submission,
        frozenset(
            {
                "schema_version",
                "kind",
                "status",
                "verified_at_utc",
                "identity",
                "freshness",
                "evaluation",
                "eval_intent_manifest",
                "preflight",
                "preflight_sha256",
                "scheduler",
                "reservation",
                "scheduler_readback_receipt",
                "scheduler_artifacts",
                "validation",
            }
        ),
        "submission_verification_receipt",
    )
    if (
        submission.get("schema_version") != 1
        or submission.get("kind") != _SUBMISSION_VERIFICATION_KIND
        or submission.get("status") != "passed"
    ):
        raise ValueError("submission_verification_receipt has an unsupported schema/kind/status")
    intent_descriptor = _validate_post_file_descriptor(
        submission.get("eval_intent_manifest"),
        expected=stable_artifacts["eval_intent_manifest"],
        label="submission_verification_receipt.eval_intent_manifest",
    )
    identity = _require_dict(
        submission.get("identity"),
        "submission_verification_receipt.identity",
    )
    _require_exact_keys(
        identity,
        frozenset(
            {
                "experiment_id",
                "logical_run_id",
                "run_name",
                "submission_nonce",
                "submitted_at_utc",
            }
        ),
        "submission_verification_receipt.identity",
    )
    expected_identity = {
        "experiment_id": contract["experiment_id"],
        "logical_run_id": contract["logical_run_id"],
        "run_name": contract["run_name"],
        "submission_nonce": inputs["capture_nonce"],
        "submitted_at_utc": inputs["submitted_at_utc"],
    }
    if identity != expected_identity:
        raise ValueError("submission_verification_receipt.identity does not match the eval contract")

    freshness = _require_dict(
        submission.get("freshness"),
        "submission_verification_receipt.freshness",
    )
    _require_exact_keys(
        freshness,
        frozenset(
            {
                "age_seconds",
                "expires_at_utc",
                "max_age_seconds",
                "max_future_skew_seconds",
                "scheduler_started_at_utc",
                "start_age_seconds",
                "max_start_to_verification_seconds",
            }
        ),
        "submission_verification_receipt.freshness",
    )
    if (
        freshness.get("max_future_skew_seconds") != _MAX_FUTURE_SKEW_SECONDS
        or freshness.get("max_start_to_verification_seconds") != _MAX_START_TO_CAPTURE_SECONDS
    ):
        raise ValueError("submission_verification_receipt freshness policy is not pinned")
    submitted_at = _parse_utc(
        inputs.get("submitted_at_utc"),
        "eval_runtime_receipt.inputs.submitted_at_utc",
    )
    scheduler_started = _parse_utc(
        freshness.get("scheduler_started_at_utc"),
        "submission_verification_receipt.freshness.scheduler_started_at_utc",
    )
    verified_at = _parse_utc(
        submission.get("verified_at_utc"),
        "submission_verification_receipt.verified_at_utc",
    )
    skew = dt.timedelta(seconds=_MAX_FUTURE_SKEW_SECONDS)
    if submitted_at > scheduler_started + skew:
        raise ValueError("submission timestamp is after scheduler start")
    if capture_started < scheduler_started - skew:
        raise ValueError("runtime capture predates scheduler start")
    if capture_started > scheduler_started + dt.timedelta(
        seconds=_MAX_START_TO_CAPTURE_SECONDS
    ) or captured_at > scheduler_started + dt.timedelta(seconds=_MAX_START_TO_CAPTURE_SECONDS):
        raise ValueError("runtime capture is outside the scheduler-start freshness window")
    if verified_at > capture_started + skew:
        raise ValueError("submission verification timestamp is after runtime capture")

    evaluation = _require_dict(
        submission.get("evaluation"),
        "submission_verification_receipt.evaluation",
    )
    _require_exact_keys(
        evaluation,
        frozenset(
            {
                "stage",
                "split",
                "tasks_csv",
                "episodes_per_task",
                "num_shards",
                "base_seed",
                "prior_gate_evidence",
            }
        ),
        "submission_verification_receipt.evaluation",
    )
    contract_evaluation = contract["evaluation"]
    expected_evaluation = {
        "stage": contract_evaluation["stage"],
        "split": contract_evaluation["split"],
        "tasks_csv": (
            ",".join(contract_evaluation["tasks"]) if contract_evaluation["stage"] == "one_episode_gate" else ""
        ),
        "episodes_per_task": contract_evaluation["episodes_per_task"],
        "num_shards": contract_evaluation["num_shards"],
        "base_seed": contract_evaluation["base_seed"],
        "prior_gate_evidence": contract_evaluation["prior_gate_evidence"],
    }
    if evaluation != expected_evaluation:
        raise ValueError("submission_verification_receipt.evaluation differs from the eval contract")

    scheduler = _require_dict(
        submission.get("scheduler"),
        "submission_verification_receipt.scheduler",
    )
    _require_exact_keys(
        scheduler,
        frozenset({"rjob_name", "rjob_uid", "showname", "worker_command_sha256", "expected"}),
        "submission_verification_receipt.scheduler",
    )
    _require_nonempty_str(
        scheduler.get("rjob_name"),
        "submission_verification_receipt.scheduler.rjob_name",
    )
    _require_nonempty_str(
        scheduler.get("rjob_uid"),
        "submission_verification_receipt.scheduler.rjob_uid",
    )
    _require_sha256(
        scheduler.get("worker_command_sha256"),
        "submission_verification_receipt.scheduler.worker_command_sha256",
    )
    scheduler_expected = _require_dict(
        scheduler.get("expected"),
        "submission_verification_receipt.scheduler.expected",
    )
    _require_exact_keys(
        scheduler_expected,
        _SUBMISSION_SCHEDULER_EXPECTED_KEYS,
        "submission_verification_receipt.scheduler.expected",
    )
    expected_resources = (
        {"cpu": 16, "memory_mib": 187_500, "gpu": 1}
        if contract_evaluation["stage"] == "one_episode_gate"
        else {"cpu": 128, "memory_mib": 1_500_000, "gpu": 8}
    )
    if any(scheduler_expected.get(key) != value for key, value in expected_resources.items()):
        raise ValueError("submission scheduler resources do not match the evaluation phase")
    expected_host_network = contract_evaluation["stage"] == "formal"
    if (
        scheduler_expected.get("showname") != scheduler.get("showname")
        or scheduler_expected.get("image") != _EXPECTED_CONTAINER_IMAGE
        or scheduler_expected.get("quota_group") != "eailabagent_gpu"
        or scheduler_expected.get("private_machine") != "group"
        or scheduler_expected.get("task_type") != "normal"
        or scheduler_expected.get("priority") != 9
        or scheduler_expected.get("restart_policy") != "Never"
        or scheduler_expected.get("termination_grace_period_seconds") != 60
        or scheduler_expected.get("backoff_limit") != 1
        or scheduler_expected.get("replicas") != 1
        or scheduler_expected.get("host_network") is not expected_host_network
        or scheduler_expected.get("store_host_nvme") is not True
        or scheduler_expected.get("negative_tags") != ["node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"]
        or scheduler_expected.get("mounts") != _EXPECTED_SUBMISSION_MOUNTS
    ):
        raise ValueError("submission scheduler expected contract is not fully pinned")
    submission_validation = _require_dict(
        submission.get("validation"),
        "submission_verification_receipt.validation",
    )
    if set(submission_validation) != {
        "submit_identity_matches",
        "freshness_window_passed",
        "scheduler_readback_matches",
        "scheduler_artifact_hashes_match",
        "eval_intent_manifest_matches",
        "pre_submission_full_inputs_passed",
        "submission_unknown_absent",
    } or any(value is not True for value in submission_validation.values()):
        raise ValueError("submission_verification_receipt.validation is not an exact all-true verdict")

    asset = _require_dict(bound_json["asset_verification_receipt"], "asset_verification_receipt")
    _require_exact_keys(
        asset,
        frozenset(
            {
                "schema_version",
                "kind",
                "status",
                "asset_root",
                "asset_root_mode",
                "excluded_top_level_paths",
                "manifest_path",
                "manifest_size_bytes",
                "manifest_sha256",
                "publisher_receipt_path",
                "publisher_receipt_size_bytes",
                "publisher_receipt_sha256",
                "publisher_generator_id",
                "publisher_generated_at_utc",
                "robocasa_git_commit",
                "payload_directory_count",
                "payload_file_count",
                "payload_symlink_count",
                "payload_total_bytes",
                "source_archive_directory",
                "source_sha256sums_sha256",
                "source_sizes_sha256",
            }
        ),
        "asset_verification_receipt",
    )
    if (
        asset.get("schema_version") != 1
        or asset.get("kind") != _ASSET_VERIFICATION_KIND
        or asset.get("status") != "passed"
    ):
        raise ValueError("asset_verification_receipt has an unsupported schema/kind/status")
    asset_root = _absolute_resolved_path(
        asset.get("asset_root"),
        "asset_verification_receipt.asset_root",
    )
    expected_asset_root = Path(contract["simulator"]["robocasa_repo"]).resolve(strict=True)
    if asset_root != expected_asset_root or not asset_root.is_dir():
        raise ValueError("asset_verification_receipt.asset_root differs from the RoboCasa checkout")
    if asset.get("robocasa_git_commit") != contract["simulator"]["robocasa_commit"]:
        raise ValueError("asset verification RoboCasa commit differs from the eval contract")
    if asset.get("payload_symlink_count") != 0:
        raise ValueError("asset verification must report zero payload symlinks")
    for key in ("payload_directory_count", "payload_file_count", "payload_total_bytes"):
        _require_int(asset.get(key), f"asset_verification_receipt.{key}", minimum=1)
    manifest_path = _absolute_resolved_path(
        asset.get("manifest_path"),
        "asset_verification_receipt.manifest_path",
    )
    publisher_receipt_path = _absolute_resolved_path(
        asset.get("publisher_receipt_path"),
        "asset_verification_receipt.publisher_receipt_path",
    )
    if (
        manifest_path.parent != asset_root / ".cwam_provenance"
        or publisher_receipt_path.parent != asset_root / ".cwam_provenance"
    ):
        raise ValueError("asset provenance files are outside the exact RoboCasa checkout")
    manifest_payload, manifest_descriptor = _read_stable_regular_file(
        manifest_path,
        "asset_verification_receipt.manifest",
        expected_mode=cwam_robocasa_asset_manifest.PROVENANCE_FILE_MODE,
    )
    publisher_receipt_payload, publisher_receipt_descriptor = _read_stable_regular_file(
        publisher_receipt_path,
        "asset_verification_receipt.publisher_receipt",
        expected_mode=cwam_robocasa_asset_manifest.PROVENANCE_FILE_MODE,
    )
    if (
        manifest_descriptor["size_bytes"]
        != _require_int(
            asset.get("manifest_size_bytes"),
            "asset_verification_receipt.manifest_size_bytes",
            minimum=1,
        )
        or publisher_receipt_descriptor["size_bytes"]
        != _require_int(
            asset.get("publisher_receipt_size_bytes"),
            "asset_verification_receipt.publisher_receipt_size_bytes",
            minimum=1,
        )
        or manifest_descriptor["sha256"]
        != _require_sha256(
            asset.get("manifest_sha256"),
            "asset_verification_receipt.manifest_sha256",
        )
        or publisher_receipt_descriptor["sha256"]
        != _require_sha256(
            asset.get("publisher_receipt_sha256"),
            "asset_verification_receipt.publisher_receipt_sha256",
        )
    ):
        raise ValueError("asset provenance SHA256 does not match the live files")
    bound_payloads["asset_manifest"] = manifest_payload
    bound_payloads["asset_publisher_receipt"] = publisher_receipt_payload
    stable_artifacts["asset_manifest"] = manifest_descriptor
    stable_artifacts["asset_publisher_receipt"] = publisher_receipt_descriptor
    preflight_binding = _validate_submission_preflight(
        preflight_value=submission.get("preflight"),
        expected_sha256=submission.get("preflight_sha256"),
        contract=contract,
        asset_verification=asset,
    )

    return {
        "evaluation_intent_manifest": intent_descriptor,
        "pre_submission": preflight_binding,
        "submission": {
            "verification_receipt": submission_descriptor,
            "submitted_at_utc": inputs["submitted_at_utc"],
            "scheduler_started_at_utc": freshness["scheduler_started_at_utc"],
            "scheduler_rjob_name": scheduler["rjob_name"],
            "scheduler_rjob_uid": scheduler["rjob_uid"],
        },
        "assets": {
            "verification_receipt": asset_descriptor,
            "asset_root": str(asset_root),
            "manifest_sha256": asset["manifest_sha256"],
            "publisher_receipt_sha256": asset["publisher_receipt_sha256"],
            "robocasa_git_commit": asset["robocasa_git_commit"],
            "payload_directory_count": asset["payload_directory_count"],
            "payload_file_count": asset["payload_file_count"],
            "payload_symlink_count": asset["payload_symlink_count"],
            "payload_total_bytes": asset["payload_total_bytes"],
        },
    }


def _validate_source_dcp_selection_receipt(
    *,
    value: Any,
    stable_artifact: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_manifest_sha256: str,
    checkpoint_tree_sha256: str,
) -> dict[str, Any]:
    label = "source_dcp_export_receipt"
    receipt = _require_dict(value, label)
    _require_exact_keys(receipt, _SOURCE_DCP_SELECTION_RECEIPT_KEYS, label)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "cwam_serving_dcp_selection_receipt"
        or receipt.get("status") != "passed"
    ):
        raise ValueError(f"{label} has an unsupported schema/kind/status")

    iteration = _require_int(
        receipt.get("source_checkpoint_iteration"),
        f"{label}.source_checkpoint_iteration",
        minimum=1,
    )
    source_path_text = _require_nonempty_str(
        receipt.get("source_checkpoint_path"),
        f"{label}.source_checkpoint_path",
    )
    serving_path_text = _require_nonempty_str(
        receipt.get("serving_checkpoint_path"),
        f"{label}.serving_checkpoint_path",
    )
    source_path = _absolute_resolved_path(source_path_text, f"{label}.source_checkpoint_path")
    serving_path = _absolute_resolved_path(serving_path_text, f"{label}.serving_checkpoint_path")
    expected_serving_path = checkpoint_path.resolve(strict=True)
    expected_source_path = expected_serving_path.parent
    if source_path_text != str(source_path) or serving_path_text != str(serving_path):
        raise ValueError(f"{label} checkpoint paths must be exact canonical paths")
    if (
        serving_path != expected_serving_path
        or source_path != expected_source_path
        or expected_serving_path.name != "model"
        or source_path.name != f"iter_{iteration:09d}"
    ):
        raise ValueError(f"{label} source/serving checkpoint selection mismatch")
    if (
        _require_sha256(
            receipt.get("checkpoint_manifest_sha256"),
            f"{label}.checkpoint_manifest_sha256",
        )
        != checkpoint_manifest_sha256
        or _require_sha256(
            receipt.get("checkpoint_tree_sha256"),
            f"{label}.checkpoint_tree_sha256",
        )
        != checkpoint_tree_sha256
    ):
        raise ValueError(f"{label} checkpoint manifest/tree binding mismatch")
    validation = _require_dict(receipt.get("validation"), f"{label}.validation")
    if set(validation) != set(_SOURCE_DCP_SELECTION_VALIDATION) or any(
        validation.get(key) is not True for key in _SOURCE_DCP_SELECTION_VALIDATION
    ):
        raise ValueError(f"{label}.validation must be the exact all-true verdict")
    return {
        "receipt": _stable_public_descriptor(stable_artifact),
        "source_checkpoint_path": str(source_path),
        "source_checkpoint_iteration": iteration,
        "serving_checkpoint_path": str(serving_path),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "validation": validation,
    }


def _validate_eval_intent_binding(
    *,
    contract: dict[str, Any],
    verified: dict[str, Any],
    stable_artifacts: dict[str, dict[str, Any]],
    bound_json: dict[str, Any],
    graphics_userspace: dict[str, Any],
    source_dcp_selection: dict[str, Any],
) -> dict[str, Any]:
    intent = _require_dict(bound_json["eval_intent_manifest"], "eval_intent_manifest")
    _require_exact_keys(intent, _EVAL_INTENT_TOP_LEVEL_KEYS, "eval_intent_manifest")
    if intent.get("schema_version") != 1 or intent.get("kind") != _EVAL_INTENT_KIND:
        raise ValueError("eval_intent_manifest has an unsupported schema/kind")

    evaluation = contract["evaluation"]
    identity = _require_dict(intent.get("identity"), "eval_intent_manifest.identity")
    _require_exact_keys(
        identity,
        frozenset({"experiment_id", "logical_run_id", "run_name", "stage", "evaluation_class"}),
        "eval_intent_manifest.identity",
    )
    if {
        "experiment_id": identity.get("experiment_id"),
        "logical_run_id": identity.get("logical_run_id"),
        "run_name": identity.get("run_name"),
        "stage": identity.get("stage"),
    } != {
        "experiment_id": contract["experiment_id"],
        "logical_run_id": contract["logical_run_id"],
        "run_name": contract["run_name"],
        "stage": evaluation["stage"],
    }:
        raise ValueError("eval intent identity differs from the eval contract")
    evaluation_class = identity.get("evaluation_class")
    if evaluation_class not in {"exploratory", "confirmatory"}:
        raise ValueError("eval intent evaluation_class is invalid")

    protocol = _require_dict(intent.get("protocol"), "eval_intent_manifest.protocol")
    _require_exact_keys(protocol, _EVAL_INTENT_PROTOCOL_KEYS, "eval_intent_manifest.protocol")
    policy_sampling = evaluation["policy_sampling"]
    expected_protocol = {
        "split": evaluation["split"],
        "tasks": evaluation["tasks"],
        "task_count": len(evaluation["tasks"]),
        "episodes_per_task": evaluation["episodes_per_task"],
        "expected_episode_count": len(evaluation["tasks"]) * evaluation["episodes_per_task"],
        "num_shards": evaluation["num_shards"],
        "base_seed": evaluation["base_seed"],
        "max_steps": evaluation["max_steps"],
        "open_loop_horizon": evaluation["open_loop_horizon"],
        "policy_sampler": policy_sampling["sampler"],
        "policy_seed": policy_sampling["seed"],
        "policy_guidance": policy_sampling["guidance"],
        "policy_num_steps": policy_sampling["num_steps"],
        "base_port": 16000,
        "server_ready_timeout_seconds": 1800,
    }
    if protocol != expected_protocol:
        raise ValueError("eval intent protocol differs from the complete eval contract")

    thresholds = _require_dict(intent.get("thresholds"), "eval_intent_manifest.thresholds")
    _require_exact_keys(
        thresholds,
        frozenset({"minimum_mean_success_rate", "minimum_successful_tasks"}),
        "eval_intent_manifest.thresholds",
    )
    if thresholds != evaluation["effect_thresholds"]:
        raise ValueError("eval intent thresholds differ from the preregistered contract thresholds")

    source = _require_dict(intent.get("source"), "eval_intent_manifest.source")
    _require_exact_keys(
        source,
        frozenset({"root", "commit", "git_pushed", "manifest"}),
        "eval_intent_manifest.source",
    )
    source_git_pushed = source.get("git_pushed")
    if type(source_git_pushed) is not bool:
        raise TypeError("eval_intent_manifest.source.git_pushed must be boolean")
    if evaluation_class == "confirmatory" and source_git_pushed is not True:
        raise ValueError("confirmatory eval intent requires a pushed source commit")
    if (
        _absolute_resolved_path(source.get("root"), "eval_intent_manifest.source.root")
        != Path(contract["source_root"]).resolve()
        or _require_commit(source.get("commit"), "eval_intent_manifest.source.commit") != contract["source_commit"]
    ):
        raise ValueError("eval intent source root/commit differs from the eval contract")
    _validate_post_file_descriptor(
        source.get("manifest"),
        expected=stable_artifacts["source_manifest"],
        label="eval_intent_manifest.source.manifest",
    )

    serving_checkpoint = _require_dict(
        intent.get("serving_checkpoint"),
        "eval_intent_manifest.serving_checkpoint",
    )
    _require_exact_keys(
        serving_checkpoint,
        frozenset({"path", "manifest", "tree_sha256", "source_dcp_export_receipt"}),
        "eval_intent_manifest.serving_checkpoint",
    )
    if (
        _absolute_resolved_path(
            serving_checkpoint.get("path"),
            "eval_intent_manifest.serving_checkpoint.path",
        )
        != Path(contract["checkpoint_path"]).resolve()
    ):
        raise ValueError("eval intent serving checkpoint path differs from the eval contract")
    _validate_post_file_descriptor(
        serving_checkpoint.get("manifest"),
        expected=stable_artifacts["checkpoint_manifest"],
        label="eval_intent_manifest.serving_checkpoint.manifest",
    )
    checkpoint_manifest = _require_dict(bound_json["checkpoint_manifest"], "checkpoint_manifest")
    if (
        _require_sha256(
            serving_checkpoint.get("tree_sha256"),
            "eval_intent_manifest.serving_checkpoint.tree_sha256",
        )
        != checkpoint_manifest["tree_sha256"]
    ):
        raise ValueError("eval intent checkpoint tree differs from the verified serving checkpoint")
    source_dcp_descriptor = _validate_post_file_descriptor(
        serving_checkpoint.get("source_dcp_export_receipt"),
        expected=stable_artifacts["source_dcp_export_receipt"],
        label="eval_intent_manifest.serving_checkpoint.source_dcp_export_receipt",
    )
    if source_dcp_descriptor != source_dcp_selection["receipt"]:
        raise ValueError("eval intent source DCP receipt differs from the validated selection receipt")

    artifacts = _require_dict(intent.get("artifacts"), "eval_intent_manifest.artifacts")
    _require_exact_keys(
        artifacts,
        frozenset(
            {
                "training_config",
                "action_stats",
                "wan2pt2_vae",
                "cosmos_environment",
                "robocasa_environment",
                "robocasa_assets",
            }
        ),
        "eval_intent_manifest.artifacts",
    )
    for name in ("training_config", "action_stats"):
        _validate_post_file_descriptor(
            artifacts.get(name),
            expected=stable_artifacts[name],
            label=f"eval_intent_manifest.artifacts.{name}",
        )
    _validate_post_file_descriptor(
        artifacts.get("wan2pt2_vae"),
        expected=stable_artifacts["wan2pt2_vae"],
        label="eval_intent_manifest.artifacts.wan2pt2_vae",
    )
    for name in ("cosmos", "robocasa"):
        label = f"eval_intent_manifest.artifacts.{name}_environment"
        environment = _require_dict(artifacts.get(f"{name}_environment"), label)
        _require_exact_keys(environment, frozenset({"root", "manifest"}), label)
        manifest = _require_dict(
            bound_json[f"{name}_environment_manifest"],
            f"{name}_environment_manifest",
        )
        interpreter = _require_dict(
            manifest.get("interpreter"),
            f"{name}_environment_manifest.interpreter",
        )
        environment_root = _absolute_canonical_existing_path(
            environment.get("root"),
            f"{label}.root",
            directory=True,
        )
        manifest_prefix = _absolute_canonical_existing_path(
            interpreter.get("prefix"),
            f"{name}_environment_manifest.interpreter.prefix",
            directory=True,
        )
        if environment_root != manifest_prefix:
            raise ValueError(f"{label}.root does not match the environment manifest prefix")
        _validated_environment_launcher(
            root=environment_root,
            manifest_interpreter=interpreter,
            label=label,
        )
        _validate_post_file_descriptor(
            environment.get("manifest"),
            expected=stable_artifacts[f"{name}_environment_manifest"],
            label=f"{label}.manifest",
        )

    robocasa_assets = _require_dict(
        artifacts.get("robocasa_assets"),
        "eval_intent_manifest.artifacts.robocasa_assets",
    )
    _require_exact_keys(
        robocasa_assets,
        frozenset({"root", "manifest", "receipt"}),
        "eval_intent_manifest.artifacts.robocasa_assets",
    )
    if (
        _absolute_resolved_path(
            robocasa_assets.get("root"),
            "eval_intent_manifest.artifacts.robocasa_assets.root",
        )
        != Path(contract["simulator"]["robocasa_repo"]).resolve()
    ):
        raise ValueError("eval intent RoboCasa asset root differs from the simulator checkout")
    _validate_post_file_descriptor(
        robocasa_assets.get("manifest"),
        expected=stable_artifacts["asset_manifest"],
        label="eval_intent_manifest.artifacts.robocasa_assets.manifest",
    )
    _validate_post_file_descriptor(
        robocasa_assets.get("receipt"),
        expected=stable_artifacts["asset_publisher_receipt"],
        label="eval_intent_manifest.artifacts.robocasa_assets.receipt",
    )

    repositories = _require_dict(intent.get("repositories"), "eval_intent_manifest.repositories")
    _require_exact_keys(
        repositories,
        frozenset({"demian", "robocasa", "robosuite"}),
        "eval_intent_manifest.repositories",
    )
    expected_repositories = {
        "demian": (contract["demian"]["repo"], contract["demian"]["commit"]),
        "robocasa": (
            contract["simulator"]["robocasa_repo"],
            contract["simulator"]["robocasa_commit"],
        ),
        "robosuite": (
            contract["simulator"]["robosuite_repo"],
            contract["simulator"]["robosuite_commit"],
        ),
    }
    for name, (expected_root, expected_commit) in expected_repositories.items():
        label = f"eval_intent_manifest.repositories.{name}"
        repository = _require_dict(repositories.get(name), label)
        _require_exact_keys(repository, frozenset({"root", "commit"}), label)
        if (
            _absolute_resolved_path(repository.get("root"), f"{label}.root") != Path(expected_root).resolve()
            or _require_commit(repository.get("commit"), f"{label}.commit") != expected_commit
        ):
            raise ValueError(f"{label} differs from the eval contract")

    submission = _require_dict(
        bound_json["submission_verification_receipt"],
        "submission_verification_receipt",
    )
    scheduler = _require_dict(submission["scheduler"], "submission_verification_receipt.scheduler")
    scheduler_expected = _require_dict(
        scheduler["expected"],
        "submission_verification_receipt.scheduler.expected",
    )
    runtime = _require_dict(intent.get("runtime"), "eval_intent_manifest.runtime")
    _require_exact_keys(
        runtime,
        frozenset({"container_image", "egl_vendor_json"}),
        "eval_intent_manifest.runtime",
    )
    if (
        runtime.get("container_image") != scheduler_expected["image"]
        or runtime.get("container_image") != _EXPECTED_CONTAINER_IMAGE
    ):
        raise ValueError("eval intent container image differs from the submission contract")
    egl_descriptor = _require_dict(
        runtime.get("egl_vendor_json"),
        "eval_intent_manifest.runtime.egl_vendor_json",
    )
    _require_exact_keys(
        egl_descriptor,
        _POST_FILE_DESCRIPTOR_KEYS,
        "eval_intent_manifest.runtime.egl_vendor_json",
    )
    graphics_egl = _require_dict(
        graphics_userspace.get("egl_vendor_json"),
        "eval_runtime_receipt.graphics_userspace.egl_vendor_json",
    )
    expected_egl_descriptor = {
        "path": graphics_egl["canonical_path"],
        "size_bytes": graphics_egl["size_bytes"],
        "sha256": graphics_egl["sha256"],
    }
    if egl_descriptor != expected_egl_descriptor:
        raise ValueError("eval intent EGL vendor JSON differs from the live runtime binding")

    resources = _require_dict(intent.get("resources"), "eval_intent_manifest.resources")
    _require_exact_keys(resources, _EVAL_INTENT_RESOURCE_KEYS, "eval_intent_manifest.resources")
    expected_resources = {
        "cpu": scheduler_expected["cpu"],
        "memory_mib": scheduler_expected["memory_mib"],
        "gpu": scheduler_expected["gpu"],
        "host_network": scheduler_expected["host_network"],
        "quota_group": scheduler_expected["quota_group"],
        "negative_tag": scheduler_expected["negative_tags"][0],
        "task_type": scheduler_expected["task_type"],
        "private_machine": scheduler_expected["private_machine"],
        "priority": scheduler_expected["priority"],
        "restart_policy": scheduler_expected["restart_policy"].lower(),
        "backoff_limit": scheduler_expected["backoff_limit"],
        "replicas": scheduler_expected["replicas"],
    }
    if resources != expected_resources:
        raise ValueError("eval intent resources differ from the accepted submission contract")
    if intent.get("prior_gate") != evaluation["prior_gate_evidence"]:
        raise ValueError("eval intent prior gate differs from the eval contract")

    return {
        "manifest": _stable_public_descriptor(stable_artifacts["eval_intent_manifest"]),
        "evaluation_class": evaluation_class,
        "source_git_pushed": source_git_pushed,
        "protocol": protocol,
        "thresholds": thresholds,
        "source_dcp_export_receipt": source_dcp_descriptor,
        "source_dcp_selection": source_dcp_selection,
        "container_image": runtime["container_image"],
        "resources": resources,
        "validation": {
            "identity_stage_and_class_match": True,
            "protocol_and_thresholds_match": True,
            "source_checkpoint_and_dcp_receipt_match": True,
            "environments_assets_and_repositories_match": True,
            "container_egl_and_resources_match": True,
            "prior_gate_binding_matches": True,
        },
    }


def _validate_eval_runtime_receipt(
    *,
    contract: dict[str, Any],
    verified: dict[str, Any],
    stable_artifacts: dict[str, dict[str, Any]],
    bound_payloads: dict[str, Any],
    bound_json: dict[str, Any],
    cosmos_environment_manifest: dict[str, Any],
    robocasa_environment_manifest: dict[str, Any],
    source_dcp_selection: dict[str, Any],
) -> dict[str, Any]:
    receipt = _require_dict(bound_json["eval_runtime_receipt"], "eval_runtime_receipt")
    _require_exact_keys(receipt, _RUNTIME_RECEIPT_KEYS, "eval_runtime_receipt")
    if (
        receipt.get("schema_version") != 2
        or receipt.get("kind") != "cwam_robocasa365_eval_runtime_receipt"
        or receipt.get("status") != "passed"
    ):
        raise ValueError("eval_runtime_receipt has an unsupported schema/kind/status")
    capture_started = _parse_utc(
        receipt.get("capture_started_at_utc"),
        "eval_runtime_receipt.capture_started_at_utc",
    )
    captured_at = _parse_utc(
        receipt.get("captured_at_utc"),
        "eval_runtime_receipt.captured_at_utc",
    )
    if captured_at < capture_started:
        raise ValueError("eval_runtime_receipt completion predates capture start")

    driver_environment = _require_dict(
        receipt.get("driver_environment"),
        "eval_runtime_receipt.driver_environment",
    )
    _require_exact_keys(
        driver_environment,
        frozenset(
            {
                "ld_library_path",
                "egl_vendor_library_filenames",
                "glx_vendor_library_name",
                "forbidden_environment_cleared",
            }
        ),
        "eval_runtime_receipt.driver_environment",
    )
    if driver_environment.get("ld_library_path") != _EXPECTED_H_DRIVER_LIBRARY_PATH:
        raise ValueError("eval_runtime_receipt did not use the exact audited H driver library path")
    if (
        driver_environment.get("egl_vendor_library_filenames") != _EXPECTED_EGL_VENDOR_JSON
        or driver_environment.get("glx_vendor_library_name") != "nvidia"
        or driver_environment.get("forbidden_environment_cleared") != list(_EXPECTED_FORBIDDEN_GRAPHICS_ENVIRONMENT)
    ):
        raise ValueError("eval_runtime_receipt did not pin and scrub the audited NVIDIA GL environment")

    inputs = _require_dict(receipt.get("inputs"), "eval_runtime_receipt.inputs")
    _require_exact_keys(inputs, _RUNTIME_INPUT_KEYS, "eval_runtime_receipt.inputs")
    if (
        _require_nonempty_str(
            inputs.get("experiment_id"),
            "eval_runtime_receipt.inputs.experiment_id",
        )
        != contract["experiment_id"]
        or _require_nonempty_str(
            inputs.get("logical_run_id"),
            "eval_runtime_receipt.inputs.logical_run_id",
        )
        != contract["logical_run_id"]
    ):
        raise ValueError("eval runtime run identity does not match the eval contract")
    _require_sha256(
        inputs.get("capture_nonce"),
        "eval_runtime_receipt.inputs.capture_nonce",
    )
    _parse_utc(
        inputs.get("submitted_at_utc"),
        "eval_runtime_receipt.inputs.submitted_at_utc",
    )
    num_gpus = _require_int(
        inputs.get("num_gpus"),
        "eval_runtime_receipt.inputs.num_gpus",
        minimum=1,
    )
    if num_gpus != contract["evaluation"]["num_shards"]:
        raise ValueError("eval runtime GPU count does not match the eval shard count")
    execution_bindings = _validate_external_execution_bindings(
        contract=contract,
        verified=verified,
        stable_artifacts=stable_artifacts,
        bound_payloads=bound_payloads,
        bound_json=bound_json,
        inputs=inputs,
        capture_started=capture_started,
        captured_at=captured_at,
    )

    environment_descriptors = _require_dict(
        inputs.get("environment_manifests"),
        "eval_runtime_receipt.inputs.environment_manifests",
    )
    interpreters = _require_dict(
        inputs.get("interpreters"),
        "eval_runtime_receipt.inputs.interpreters",
    )
    environment_verification = _require_dict(
        receipt.get("environment_verification"),
        "eval_runtime_receipt.environment_verification",
    )
    for value, label in (
        (environment_descriptors, "eval_runtime_receipt.inputs.environment_manifests"),
        (interpreters, "eval_runtime_receipt.inputs.interpreters"),
        (environment_verification, "eval_runtime_receipt.environment_verification"),
    ):
        _require_exact_keys(value, frozenset({"cosmos", "robocasa"}), label)
    for name, manifest in (
        ("cosmos", cosmos_environment_manifest),
        ("robocasa", robocasa_environment_manifest),
    ):
        _validate_environment_runtime_binding(
            name=name,
            artifact=verified[f"{name}_environment_manifest"],
            stable_artifact=stable_artifacts[f"{name}_environment_manifest"],
            manifest=manifest,
            receipt_descriptor=environment_descriptors[name],
            receipt_interpreter=interpreters[name],
            verification_report=environment_verification[name],
        )

    sources = _require_dict(inputs.get("sources"), "eval_runtime_receipt.inputs.sources")
    _require_exact_keys(
        sources,
        frozenset({"cosmos", "demian", "robocasa", "robosuite"}),
        "eval_runtime_receipt.inputs.sources",
    )
    source_expectations = {
        "cosmos": (Path(contract["source_root"]).resolve(), contract["source_commit"]),
        "demian": (Path(contract["demian"]["repo"]).resolve(), contract["demian"]["commit"]),
        "robocasa": (
            Path(contract["simulator"]["robocasa_repo"]).resolve(),
            contract["simulator"]["robocasa_commit"],
        ),
        "robosuite": (
            Path(contract["simulator"]["robosuite_repo"]).resolve(),
            contract["simulator"]["robosuite_commit"],
        ),
    }
    for name, (expected_root, expected_commit) in source_expectations.items():
        label = f"eval_runtime_receipt.inputs.sources.{name}"
        source = _require_dict(sources.get(name), label)
        _require_exact_keys(source, _RUNTIME_SOURCE_KEYS, label)
        if _absolute_resolved_path(source.get("root"), f"{label}.root") != expected_root:
            raise ValueError(f"{label}.root does not match the eval contract")
        if (
            _require_commit(source.get("expected_commit"), f"{label}.expected_commit") != expected_commit
            or _require_commit(source.get("actual_commit"), f"{label}.actual_commit") != expected_commit
            or source.get("clean") is not True
        ):
            raise ValueError(f"{label} does not bind the clean expected source commit")

    cosmos_runtime = _require_dict(
        receipt.get("cosmos_runtime"),
        "eval_runtime_receipt.cosmos_runtime",
    )
    _require_exact_keys(
        cosmos_runtime,
        frozenset({"python", "torch", "cosmos_framework", "gpus"}),
        "eval_runtime_receipt.cosmos_runtime",
    )
    cosmos_python = _require_dict(
        cosmos_runtime.get("python"),
        "eval_runtime_receipt.cosmos_runtime.python",
    )
    _require_exact_keys(
        cosmos_python,
        frozenset({"version", "executable", "prefix", "base_prefix"}),
        "eval_runtime_receipt.cosmos_runtime.python",
    )
    cosmos_manifest_interpreter = _require_dict(
        cosmos_environment_manifest["interpreter"],
        "cosmos_environment_manifest.interpreter",
    )
    if _absolute_resolved_path(
        cosmos_python.get("executable"),
        "eval_runtime_receipt.cosmos_runtime.python.executable",
    ) != _absolute_resolved_path(
        cosmos_manifest_interpreter["resolved_path"],
        "cosmos_environment_manifest.interpreter.resolved_path",
    ):
        raise ValueError("Cosmos runtime interpreter does not match its environment manifest")
    cosmos_runtime_version = _require_nonempty_str(
        cosmos_python.get("version"),
        "eval_runtime_receipt.cosmos_runtime.python.version",
    ).split(maxsplit=1)[0]
    if cosmos_runtime_version != cosmos_manifest_interpreter["python_version"]:
        raise ValueError("Cosmos runtime Python version does not match its environment manifest")
    cosmos_prefix = _absolute_resolved_path(
        cosmos_python.get("prefix"),
        "eval_runtime_receipt.cosmos_runtime.python.prefix",
    )
    cosmos_base_prefix = _absolute_resolved_path(
        cosmos_python.get("base_prefix"),
        "eval_runtime_receipt.cosmos_runtime.python.base_prefix",
    )
    if cosmos_prefix != _absolute_resolved_path(
        cosmos_manifest_interpreter["prefix"],
        "cosmos_environment_manifest.interpreter.prefix",
    ) or cosmos_base_prefix != _absolute_resolved_path(
        cosmos_manifest_interpreter["base_prefix"],
        "cosmos_environment_manifest.interpreter.base_prefix",
    ):
        raise ValueError("Cosmos runtime Python prefixes do not match its environment manifest")
    torch_record = _require_dict(
        cosmos_runtime.get("torch"),
        "eval_runtime_receipt.cosmos_runtime.torch",
    )
    _require_exact_keys(
        torch_record,
        frozenset(
            {
                "version",
                "cuda_version",
                "cuda_available",
                "cuda_device_count",
                "module_path",
            }
        ),
        "eval_runtime_receipt.cosmos_runtime.torch",
    )
    torch_version = _require_nonempty_str(
        torch_record.get("version"),
        "eval_runtime_receipt.cosmos_runtime.torch.version",
    )
    _require_nonempty_str(
        torch_record.get("cuda_version"),
        "eval_runtime_receipt.cosmos_runtime.torch.cuda_version",
    )
    torch_module_path = _absolute_resolved_path(
        torch_record.get("module_path"),
        "eval_runtime_receipt.cosmos_runtime.torch.module_path",
    )
    if cosmos_prefix not in torch_module_path.parents:
        raise ValueError("Cosmos runtime torch module resolved outside its environment prefix")
    cosmos_distributions = _require_dict(
        cosmos_environment_manifest["distributions"],
        "cosmos_environment_manifest.distributions",
    )
    torch_distribution = _require_dict(
        cosmos_distributions.get("torch"),
        "cosmos_environment_manifest.distributions.torch",
    )
    if torch_distribution.get("version") != torch_version:
        raise ValueError("Cosmos runtime torch version does not match its environment manifest")
    if (
        torch_record.get("cuda_available") is not True
        or _require_int(
            torch_record.get("cuda_device_count"),
            "eval_runtime_receipt.cosmos_runtime.torch.cuda_device_count",
            minimum=1,
        )
        != num_gpus
    ):
        raise ValueError("Cosmos runtime CUDA state does not match the eval contract")
    cosmos_framework = _require_dict(
        cosmos_runtime.get("cosmos_framework"),
        "eval_runtime_receipt.cosmos_runtime.cosmos_framework",
    )
    _require_exact_keys(
        cosmos_framework,
        frozenset({"module_path"}),
        "eval_runtime_receipt.cosmos_runtime.cosmos_framework",
    )
    expected_cosmos_module = Path(contract["source_root"]) / "cosmos_framework" / "__init__.py"
    if (
        _absolute_resolved_path(
            cosmos_framework.get("module_path"),
            "eval_runtime_receipt.cosmos_runtime.cosmos_framework.module_path",
        )
        != expected_cosmos_module.resolve()
    ):
        raise ValueError("Cosmos runtime module resolved outside the exact source root")
    cosmos_gpus = cosmos_runtime.get("gpus")
    if not isinstance(cosmos_gpus, list) or len(cosmos_gpus) != num_gpus:
        raise ValueError("Cosmos runtime GPU inventory does not match the eval contract")
    cosmos_gpu_uuids: set[str] = set()
    for expected_index, gpu_value in enumerate(cosmos_gpus):
        label = f"eval_runtime_receipt.cosmos_runtime.gpus[{expected_index}]"
        gpu = _require_dict(gpu_value, label)
        _require_exact_keys(
            gpu,
            frozenset(
                {
                    "index",
                    "name",
                    "uuid",
                    "normalized_uuid",
                    "capability",
                    "total_memory_bytes",
                }
            ),
            label,
        )
        if (
            gpu.get("index") != expected_index
            or "H200" not in _require_nonempty_str(gpu.get("name"), f"{label}.name").upper()
            or gpu.get("capability") != [9, 0]
        ):
            raise ValueError(f"{label} is not the expected H200 device")
        _require_int(
            gpu.get("total_memory_bytes"),
            f"{label}.total_memory_bytes",
            minimum=1,
        )
        normalized_uuid = _require_nonempty_str(
            gpu.get("normalized_uuid"),
            f"{label}.normalized_uuid",
        )
        if (
            _NORMALIZED_GPU_UUID_RE.fullmatch(normalized_uuid) is None
            or _normalized_gpu_uuid(gpu.get("uuid"), f"{label}.uuid") != normalized_uuid
        ):
            raise ValueError(f"{label} raw and normalized GPU UUIDs differ")
        cosmos_gpu_uuids.add(normalized_uuid)
    if len(cosmos_gpu_uuids) != num_gpus:
        raise ValueError("Cosmos runtime GPU UUIDs must be unique")

    robocasa_runtime = _require_dict(
        receipt.get("robocasa_runtime"),
        "eval_runtime_receipt.robocasa_runtime",
    )
    _require_exact_keys(
        robocasa_runtime,
        frozenset({"python", "packages", "demian_evaluator"}),
        "eval_runtime_receipt.robocasa_runtime",
    )
    robocasa_python = _require_dict(
        robocasa_runtime.get("python"),
        "eval_runtime_receipt.robocasa_runtime.python",
    )
    _require_exact_keys(
        robocasa_python,
        frozenset({"version", "executable", "prefix", "base_prefix"}),
        "eval_runtime_receipt.robocasa_runtime.python",
    )
    robocasa_manifest_interpreter = _require_dict(
        robocasa_environment_manifest["interpreter"],
        "robocasa_environment_manifest.interpreter",
    )
    if _absolute_resolved_path(
        robocasa_python.get("executable"),
        "eval_runtime_receipt.robocasa_runtime.python.executable",
    ) != _absolute_resolved_path(
        robocasa_manifest_interpreter["resolved_path"],
        "robocasa_environment_manifest.interpreter.resolved_path",
    ):
        raise ValueError("RoboCasa runtime interpreter does not match its environment manifest")
    robocasa_runtime_version = _require_nonempty_str(
        robocasa_python.get("version"),
        "eval_runtime_receipt.robocasa_runtime.python.version",
    ).split(maxsplit=1)[0]
    if robocasa_runtime_version != robocasa_manifest_interpreter["python_version"]:
        raise ValueError("RoboCasa runtime Python version does not match its environment manifest")
    robocasa_prefix = _absolute_resolved_path(
        robocasa_python.get("prefix"),
        "eval_runtime_receipt.robocasa_runtime.python.prefix",
    )
    robocasa_base_prefix = _absolute_resolved_path(
        robocasa_python.get("base_prefix"),
        "eval_runtime_receipt.robocasa_runtime.python.base_prefix",
    )
    if robocasa_prefix != _absolute_resolved_path(
        robocasa_manifest_interpreter["prefix"],
        "robocasa_environment_manifest.interpreter.prefix",
    ) or robocasa_base_prefix != _absolute_resolved_path(
        robocasa_manifest_interpreter["base_prefix"],
        "robocasa_environment_manifest.interpreter.base_prefix",
    ):
        raise ValueError("RoboCasa runtime Python prefixes do not match its environment manifest")
    runtime_evaluator = _require_dict(
        robocasa_runtime.get("demian_evaluator"),
        "eval_runtime_receipt.robocasa_runtime.demian_evaluator",
    )
    _require_exact_keys(
        runtime_evaluator,
        frozenset({"module_path"}),
        "eval_runtime_receipt.robocasa_runtime.demian_evaluator",
    )
    if (
        _absolute_resolved_path(
            runtime_evaluator.get("module_path"),
            "eval_runtime_receipt.robocasa_runtime.demian_evaluator.module_path",
        )
        != Path(verified["demian_evaluator"]["path"]).resolve()
    ):
        raise ValueError("RoboCasa runtime did not import the exact DeMiAn evaluator")
    runtime_packages = _require_dict(
        robocasa_runtime.get("packages"),
        "eval_runtime_receipt.robocasa_runtime.packages",
    )
    _require_exact_keys(
        runtime_packages,
        frozenset({"numpy", "mujoco", "gymnasium", "robocasa", "robosuite"}),
        "eval_runtime_receipt.robocasa_runtime.packages",
    )
    robocasa_distributions = _require_dict(
        robocasa_environment_manifest["distributions"],
        "robocasa_environment_manifest.distributions",
    )
    expected_source_modules = {
        "robocasa": (Path(contract["simulator"]["robocasa_repo"]).resolve() / "robocasa" / "__init__.py").resolve(),
        "robosuite": (Path(contract["simulator"]["robosuite_repo"]).resolve() / "robosuite" / "__init__.py").resolve(),
    }
    for name in ("numpy", "mujoco", "gymnasium", "robocasa", "robosuite"):
        label = f"eval_runtime_receipt.robocasa_runtime.packages.{name}"
        package = _require_dict(runtime_packages.get(name), label)
        _require_exact_keys(
            package,
            frozenset(
                {
                    "distribution_version",
                    "module_reported_version",
                    "version_source",
                    "module_path",
                }
            ),
            label,
        )
        distribution_version = _require_nonempty_str(
            package.get("distribution_version"),
            f"{label}.distribution_version",
        )
        module_reported_version = package.get("module_reported_version")
        if module_reported_version is not None:
            _require_nonempty_str(
                module_reported_version,
                f"{label}.module_reported_version",
            )
        if package.get("version_source") != "importlib.metadata":
            raise ValueError(f"{label}.version_source must be importlib.metadata")
        manifest_distribution = _require_dict(
            robocasa_distributions.get(name),
            f"robocasa_environment_manifest.distributions.{name}",
        )
        if manifest_distribution.get("version") != distribution_version:
            raise ValueError(f"{label}.distribution_version does not match the environment manifest")
        module_path = _absolute_resolved_path(package.get("module_path"), f"{label}.module_path")
        if name in expected_source_modules:
            if module_path != expected_source_modules[name]:
                raise ValueError(f"{label}.module_path resolved outside the exact source root")
        elif robocasa_prefix not in module_path.parents:
            raise ValueError(f"{label}.module_path resolved outside the RoboCasa environment prefix")

    robocasa_egl = receipt.get("robocasa_egl")
    if not isinstance(robocasa_egl, list) or len(robocasa_egl) != num_gpus:
        raise ValueError("eval_runtime_receipt.robocasa_egl must contain one record per shard")
    graphics_userspace = _validate_graphics_userspace(
        receipt.get("graphics_userspace"),
        "eval_runtime_receipt.graphics_userspace",
    )
    expected_robosuite_module = expected_source_modules["robosuite"]
    for expected_shard, raw_egl in enumerate(robocasa_egl):
        label = f"eval_runtime_receipt.robocasa_egl[{expected_shard}]"
        egl = _require_dict(raw_egl, label)
        _require_exact_keys(
            egl,
            frozenset(
                {
                    "status",
                    "shard_index",
                    "cuda_visible_devices",
                    "mujoco_egl_device_id",
                    "mujoco_gl",
                    "pyopengl_platform",
                    "ld_library_path",
                    "egl_vendor_library_filenames",
                    "glx_vendor_library_name",
                    "robosuite_module_path",
                    "graphics_userspace",
                }
            ),
            label,
        )
        expected_device = str(expected_shard)
        if (
            egl.get("status") != "passed"
            or egl.get("shard_index") != expected_shard
            or egl.get("cuda_visible_devices") != expected_device
            or egl.get("mujoco_egl_device_id") != expected_device
            or egl.get("mujoco_gl") != "egl"
            or egl.get("pyopengl_platform") != "egl"
            or egl.get("ld_library_path") != _EXPECTED_H_DRIVER_LIBRARY_PATH
            or egl.get("egl_vendor_library_filenames") != _EXPECTED_EGL_VENDOR_JSON
            or egl.get("glx_vendor_library_name") != "nvidia"
        ):
            raise ValueError(f"{label} does not bind the exact per-shard EGL environment")
        if egl.get("graphics_userspace") != graphics_userspace:
            raise ValueError(f"{label}.graphics_userspace differs from the cross-shard live binding")
        if (
            _absolute_resolved_path(
                egl.get("robosuite_module_path"),
                f"{label}.robosuite_module_path",
            )
            != expected_robosuite_module
        ):
            raise ValueError(f"{label} imported robosuite outside the exact source root")

    nvidia_smi = _require_dict(
        receipt.get("nvidia_smi"),
        "eval_runtime_receipt.nvidia_smi",
    )
    _require_exact_keys(
        nvidia_smi,
        frozenset({"executable", "driver_version", "gpus"}),
        "eval_runtime_receipt.nvidia_smi",
    )
    _absolute_resolved_path(
        nvidia_smi.get("executable"),
        "eval_runtime_receipt.nvidia_smi.executable",
    )
    _require_nonempty_str(
        nvidia_smi.get("driver_version"),
        "eval_runtime_receipt.nvidia_smi.driver_version",
    )
    nvidia_gpus = nvidia_smi.get("gpus")
    if not isinstance(nvidia_gpus, list) or len(nvidia_gpus) != num_gpus:
        raise ValueError("nvidia-smi GPU inventory does not match the eval contract")
    nvidia_gpu_uuids: set[str] = set()
    for index, raw_gpu in enumerate(nvidia_gpus):
        label = f"eval_runtime_receipt.nvidia_smi.gpus[{index}]"
        gpu = _require_dict(raw_gpu, label)
        _require_exact_keys(
            gpu,
            frozenset({"index", "uuid", "normalized_uuid", "name", "driver_version"}),
            label,
        )
        if (
            gpu.get("index") != index
            or "H200" not in _require_nonempty_str(gpu.get("name"), f"{label}.name").upper()
            or gpu.get("driver_version") != nvidia_smi["driver_version"]
        ):
            raise ValueError(f"{label} does not match the expected H200 inventory")
        normalized_uuid = _require_nonempty_str(
            gpu.get("normalized_uuid"),
            f"{label}.normalized_uuid",
        )
        if (
            _NORMALIZED_GPU_UUID_RE.fullmatch(normalized_uuid) is None
            or _normalized_gpu_uuid(gpu.get("uuid"), f"{label}.uuid") != normalized_uuid
        ):
            raise ValueError(f"{label} raw and normalized GPU UUIDs differ")
        if normalized_uuid in nvidia_gpu_uuids:
            raise ValueError("nvidia-smi GPU UUIDs must be unique")
        nvidia_gpu_uuids.add(normalized_uuid)
    if nvidia_gpu_uuids != cosmos_gpu_uuids:
        raise ValueError("runtime receipt PyTorch and nvidia-smi GPU UUID sets differ")

    validation = _require_dict(
        receipt.get("validation"),
        "eval_runtime_receipt.validation",
    )
    _require_exact_keys(
        validation,
        _RUNTIME_VALIDATION_KEYS,
        "eval_runtime_receipt.validation",
    )
    if any(value is not True for value in validation.values()):
        raise ValueError("eval_runtime_receipt.validation must contain only true verdicts")
    intent_binding = _validate_eval_intent_binding(
        contract=contract,
        verified=verified,
        stable_artifacts=stable_artifacts,
        bound_json=bound_json,
        graphics_userspace=graphics_userspace,
        source_dcp_selection=source_dcp_selection,
    )
    return {
        **execution_bindings,
        "evaluation_intent": intent_binding,
        "graphics_userspace": graphics_userspace,
        "capture_started_at_utc": receipt["capture_started_at_utc"],
        "captured_at_utc": receipt["captured_at_utc"],
    }


def _canonical_existing_directory(path: Path, label: str) -> Path:
    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        canonical = supplied.resolve(strict=True)
        metadata = canonical.lstat()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise ValueError(f"{label} does not resolve to an existing directory: {supplied}") from error
    if supplied != canonical or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a canonical non-symlink directory: {supplied}")
    return canonical


def _exclusive_artifact_path(path: Path, label: str) -> Path:
    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        canonical_parent = supplied.parent.resolve(strict=True)
        parent_metadata = canonical_parent.lstat()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise ValueError(f"{label} parent must be an existing canonical directory") from error
    if (
        supplied.parent != canonical_parent
        or stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
    ):
        raise ValueError(f"{label} parent must be an existing canonical non-symlink directory")
    if supplied.exists() or supplied.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing {label}: {supplied}")
    return supplied


def verify_prior_gate(
    *,
    gate_evidence_path: Path,
    expected_gate_evidence_sha256: str,
    current_source_root: Path,
    current_source_commit: str,
    current_source_manifest_sha256: str,
    current_checkpoint_path: Path,
    current_checkpoint_manifest_sha256: str,
    current_source_dcp_export_receipt_sha256: str,
    current_asset_manifest_sha256: str,
    current_asset_publisher_receipt_sha256: str,
    current_training_config_sha256: str,
    current_action_stats_sha256: str,
    current_wan2pt2_vae_path: Path,
    current_wan2pt2_vae_sha256: str,
    current_cosmos_environment_manifest_sha256: str,
    current_robocasa_environment_manifest_sha256: str,
    current_demian_commit: str,
    current_robocasa_commit: str,
    current_robosuite_commit: str,
) -> dict[str, Any]:
    """Verify one terminal gate without rehashing checkpoint or asset payload trees."""

    expected_gate_sha256 = _require_sha256(
        expected_gate_evidence_sha256,
        "expected prior gate evidence SHA256",
    )
    gate_evidence_value, evidence_descriptor = _load_stable_json(
        gate_evidence_path,
        "prior gate evidence",
        expected_mode=0o444,
    )
    if evidence_descriptor["sha256"] != expected_gate_sha256:
        raise ValueError("prior gate evidence SHA256 does not match the expected digest")
    source_root = _canonical_existing_directory(current_source_root, "current formal source root")
    checkpoint_path = _canonical_existing_directory(
        current_checkpoint_path,
        "current formal checkpoint path",
    )
    wan2pt2_vae_path = _absolute_canonical_existing_path(
        str(Path(os.path.abspath(current_wan2pt2_vae_path.expanduser()))),
        "current formal Wan2.2 VAE path",
        directory=False,
    )
    current_inputs = {
        "source_root": str(source_root),
        "source_commit": _require_commit(current_source_commit, "current formal source commit"),
        "source_manifest_sha256": _require_sha256(
            current_source_manifest_sha256,
            "current formal source manifest SHA256",
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_manifest_sha256": _require_sha256(
            current_checkpoint_manifest_sha256,
            "current formal checkpoint manifest SHA256",
        ),
        "source_dcp_export_receipt_sha256": _require_sha256(
            current_source_dcp_export_receipt_sha256,
            "current formal source DCP export receipt SHA256",
        ),
        "asset_manifest_sha256": _require_sha256(
            current_asset_manifest_sha256,
            "current formal asset manifest SHA256",
        ),
        "asset_publisher_receipt_sha256": _require_sha256(
            current_asset_publisher_receipt_sha256,
            "current formal asset publisher receipt SHA256",
        ),
        "training_config_sha256": _require_sha256(
            current_training_config_sha256,
            "current formal training config SHA256",
        ),
        "action_stats_sha256": _require_sha256(
            current_action_stats_sha256,
            "current formal action stats SHA256",
        ),
        "wan2pt2_vae_path": str(wan2pt2_vae_path),
        "wan2pt2_vae_sha256": _require_sha256(
            current_wan2pt2_vae_sha256,
            "current formal Wan2.2 VAE SHA256",
        ),
        "cosmos_environment_manifest_sha256": _require_sha256(
            current_cosmos_environment_manifest_sha256,
            "current formal Cosmos environment manifest SHA256",
        ),
        "robocasa_environment_manifest_sha256": _require_sha256(
            current_robocasa_environment_manifest_sha256,
            "current formal RoboCasa environment manifest SHA256",
        ),
        "demian_commit": _require_commit(current_demian_commit, "current formal DeMiAn commit"),
        "robocasa_commit": _require_commit(current_robocasa_commit, "current formal RoboCasa commit"),
        "robosuite_commit": _require_commit(current_robosuite_commit, "current formal robosuite commit"),
    }
    _require_exact_keys(
        current_inputs,
        _PRIOR_GATE_CURRENT_INPUT_KEYS,
        "current formal prior-gate inputs",
    )

    gate_evidence = _require_dict(gate_evidence_value, "prior_gate_evidence")
    _require_exact_keys(gate_evidence, _PRIOR_GATE_EVIDENCE_KEYS, "prior_gate_evidence")
    if gate_evidence.get("schema_version") != 1 or gate_evidence.get("kind") != _EVIDENCE_KIND:
        raise ValueError("prior_gate_evidence has an unsupported schema/kind")
    if gate_evidence.get("execution_complete") is not True or gate_evidence.get("effect_pass") is not True:
        raise ValueError("prior gate must have terminal execution_complete=true and effect_pass=true")
    if gate_evidence.get("prior_gate_evidence") is not None:
        raise ValueError("one_episode_gate evidence must not recursively bind another prior gate")

    gate_evaluation = _require_dict(gate_evidence.get("evaluation"), "prior_gate_evidence.evaluation")
    _require_exact_keys(
        gate_evaluation,
        _PRIOR_GATE_EVALUATION_KEYS,
        "prior_gate_evidence.evaluation",
    )
    gate_tasks = _require_int(gate_evaluation.get("tasks"), "prior_gate_evidence.evaluation.tasks", minimum=1)
    gate_evaluation_class = gate_evaluation.get("evaluation_class")
    if gate_evaluation_class not in {"exploratory", "confirmatory"}:
        raise ValueError("prior_gate_evidence evaluation_class is invalid")
    gate_source_git_pushed = gate_evaluation.get("source_git_pushed_assertion")
    if type(gate_source_git_pushed) is not bool:
        raise TypeError("prior_gate_evidence source_git_pushed_assertion must be boolean")
    gate_episodes = _require_int(
        gate_evaluation.get("episodes"),
        "prior_gate_evidence.evaluation.episodes",
        minimum=1,
    )
    gate_episodes_per_task = _require_int(
        gate_evaluation.get("episodes_per_task"),
        "prior_gate_evidence.evaluation.episodes_per_task",
        minimum=1,
    )
    if (
        gate_evaluation.get("stage") != "one_episode_gate"
        or gate_evaluation.get("role") != "primary"
        or gate_evaluation.get("split") != "atomic"
        or gate_tasks != 1
        or gate_episodes != 1
        or gate_episodes_per_task != 1
    ):
        raise ValueError("prior gate evidence is not exactly one atomic task, one episode")
    successful_tasks = _require_int(
        gate_evaluation.get("successful_tasks"),
        "prior_gate_evidence.evaluation.successful_tasks",
        minimum=0,
    )
    mean_success_rate = gate_evaluation.get("mean_success_rate")
    if (
        isinstance(mean_success_rate, bool)
        or not isinstance(mean_success_rate, (int, float))
        or not math.isfinite(float(mean_success_rate))
        or float(mean_success_rate) not in {0.0, 1.0}
        or successful_tasks != int(float(mean_success_rate) > 0.0)
    ):
        raise ValueError("prior_gate_evidence evaluation success aggregates are inconsistent")

    gate_contract_path = Path(
        _require_nonempty_str(
            gate_evidence.get("contract_path"),
            "prior_gate_evidence.contract_path",
        )
    )
    if not gate_contract_path.is_absolute():
        raise ValueError("prior_gate_evidence.contract_path must be absolute")
    gate_contract_sha256 = _require_sha256(
        gate_evidence.get("contract_sha256"),
        "prior_gate_evidence.contract_sha256",
    )
    gate_contract = _validate_contract(gate_contract_path)
    gate_contract_descriptor = dict(gate_contract["_contract_descriptor"])
    if gate_contract_descriptor["sha256"] != gate_contract_sha256:
        raise ValueError("prior gate eval contract SHA256 does not match the evidence")
    gate_contract_descriptor.update({"kind": _CONTRACT_KIND, "schema_version": 1})
    gate_contract_evaluation = gate_contract["evaluation"]
    if (
        gate_contract_evaluation["stage"] != "one_episode_gate"
        or gate_contract_evaluation["episodes_per_task"] != 1
        or gate_contract_evaluation["num_shards"] != 1
        or len(gate_contract_evaluation["tasks"]) != 1
        or gate_contract_evaluation.get("prior_gate_evidence") is not None
    ):
        raise ValueError("prior gate contract is not the exact one-task/one-episode/one-shard gate")
    if gate_evaluation["effect_thresholds"] != gate_contract_evaluation["effect_thresholds"]:
        raise ValueError("prior gate evidence thresholds differ from its eval contract")
    gate_intent_binding = gate_contract["_runtime_bindings"]["evaluation_intent"]
    if (
        gate_evaluation_class != gate_intent_binding["evaluation_class"]
        or gate_source_git_pushed != gate_intent_binding["source_git_pushed"]
    ):
        raise ValueError("prior gate evidence lifecycle class differs from its eval intent")
    if (
        gate_evidence.get("experiment_id") != gate_contract["experiment_id"]
        or gate_evidence.get("logical_run_id") != gate_contract["logical_run_id"]
        or gate_evidence.get("source_commit") != gate_contract["source_commit"]
        or gate_evidence.get("demian") != gate_contract["demian"]
        or gate_evidence.get("simulator") != gate_contract["simulator"]
        or Path(_require_nonempty_str(gate_evidence.get("results_dir"), "prior_gate_evidence.results_dir"))
        != Path(gate_contract["results_dir"])
    ):
        raise ValueError("prior gate evidence identity differs from its eval contract")

    gate_artifacts = _require_dict(gate_evidence.get("artifacts"), "prior_gate_evidence.artifacts")
    _require_exact_keys(gate_artifacts, _PRIOR_GATE_ARTIFACT_KEYS, "prior_gate_evidence.artifacts")
    for name in _PRIOR_GATE_REQUIRED_ARTIFACT_KEYS:
        descriptor = _require_dict(gate_artifacts.get(name), f"prior_gate_evidence.artifacts.{name}")
        _require_exact_keys(
            descriptor,
            frozenset({"path", "sha256"}),
            f"prior_gate_evidence.artifacts.{name}",
        )
        if descriptor != gate_contract["_verified_artifacts"][name]:
            raise ValueError(f"prior gate evidence artifact {name} differs from its eval contract")
    raw_shards = gate_artifacts.get("raw_shards")
    if (
        not isinstance(raw_shards, list)
        or len(raw_shards) != 1
        or not isinstance(raw_shards[0], dict)
        or raw_shards[0].get("shard") != 0
    ):
        raise ValueError("prior gate evidence must bind exactly one raw result shard")
    server_receipts = gate_artifacts.get("server_runtime_receipts")
    if not isinstance(server_receipts, list) or len(server_receipts) != 1:
        raise ValueError("prior gate evidence must bind exactly one server runtime receipt")
    _validate_server_receipt(gate_contract)
    gate_results_dir = _canonical_existing_directory(
        Path(gate_contract["results_dir"]),
        "prior gate results directory",
    )
    episode_rows, task_rows, live_raw_shards = _collect_rows(
        gate_contract,
        gate_results_dir,
        expected_raw_mode=0o444,
    )
    if len(episode_rows) != 1 or len(task_rows) != 1 or raw_shards != live_raw_shards:
        raise ValueError("prior gate raw result artifacts do not prove exactly one complete episode")
    for raw_descriptor_name in ("args", "results"):
        raw_descriptor = _require_dict(
            raw_shards[0].get(raw_descriptor_name),
            f"prior_gate_evidence.artifacts.raw_shards[0].{raw_descriptor_name}",
        )
        _require_exact_keys(
            raw_descriptor,
            frozenset({"path", "sha256"}),
            f"prior_gate_evidence.artifacts.raw_shards[0].{raw_descriptor_name}",
        )
        if raw_descriptor != live_raw_shards[0][raw_descriptor_name]:
            raise ValueError(f"prior gate raw {raw_descriptor_name} differs from the stable raw result")
    for summary_name in ("summary_csv", "task_success_rates_csv"):
        summary_descriptor = _require_dict(
            gate_artifacts.get(summary_name),
            f"prior_gate_evidence.artifacts.{summary_name}",
        )
        _require_exact_keys(
            summary_descriptor,
            frozenset({"path", "sha256"}),
            f"prior_gate_evidence.artifacts.{summary_name}",
        )
        _, live_summary_descriptor = _read_stable_regular_file(
            Path(summary_descriptor["path"]),
            f"prior gate {summary_name}",
            expected_mode=0o444,
        )
        if {
            "path": live_summary_descriptor["path"],
            "sha256": live_summary_descriptor["sha256"],
        } != summary_descriptor:
            raise ValueError(f"prior gate {summary_name} differs from the stable summary artifact")
    live_success_rate = float(task_rows[0]["success_rate"])
    if float(mean_success_rate) != live_success_rate or successful_tasks != int(live_success_rate > 0.0):
        raise ValueError("prior gate evidence metrics differ from the verified raw result")

    runtime_descriptor = gate_contract["_verified_artifacts"]["eval_runtime_receipt"]
    runtime_file_descriptor = dict(gate_contract["_stable_artifacts"]["eval_runtime_receipt"])
    if runtime_file_descriptor["sha256"] != runtime_descriptor["sha256"]:
        raise ValueError("prior gate runtime receipt differs from the stable contract artifact")
    runtime_receipt = _require_dict(
        gate_contract["_bound_json"]["eval_runtime_receipt"],
        "prior gate runtime receipt",
    )
    runtime_validation = _require_dict(
        runtime_receipt.get("validation"),
        "prior gate runtime receipt.validation",
    )
    _require_exact_keys(
        runtime_validation,
        _RUNTIME_VALIDATION_KEYS,
        "prior gate runtime receipt.validation",
    )
    if any(runtime_validation.get(key) is not True for key in _PRIOR_GATE_RUNTIME_EGL_VALIDATION_KEYS):
        raise ValueError("prior gate runtime EGL validation is not an exact all-true verdict")
    runtime_file_descriptor.update(
        {
            "kind": "cwam_robocasa365_eval_runtime_receipt",
            "schema_version": 2,
        }
    )

    gate_inputs = {
        "source_root": gate_contract["source_root"],
        "source_commit": gate_contract["source_commit"],
        "source_manifest_sha256": gate_contract["source_manifest_sha256"],
        "checkpoint_path": gate_contract["checkpoint_path"],
        "checkpoint_manifest_sha256": gate_contract["_verified_artifacts"]["checkpoint_manifest"]["sha256"],
        "source_dcp_export_receipt_sha256": gate_contract["_verified_artifacts"]["source_dcp_export_receipt"]["sha256"],
        "asset_manifest_sha256": gate_contract["_runtime_bindings"]["assets"]["manifest_sha256"],
        "asset_publisher_receipt_sha256": gate_contract["_runtime_bindings"]["assets"]["publisher_receipt_sha256"],
        "training_config_sha256": gate_contract["_verified_artifacts"]["training_config"]["sha256"],
        "action_stats_sha256": gate_contract["_verified_artifacts"]["action_stats"]["sha256"],
        "wan2pt2_vae_path": gate_contract["_verified_artifacts"]["wan2pt2_vae"]["path"],
        "wan2pt2_vae_sha256": gate_contract["_verified_artifacts"]["wan2pt2_vae"]["sha256"],
        "cosmos_environment_manifest_sha256": gate_contract["_verified_artifacts"]["cosmos_environment_manifest"][
            "sha256"
        ],
        "robocasa_environment_manifest_sha256": gate_contract["_verified_artifacts"]["robocasa_environment_manifest"][
            "sha256"
        ],
        "demian_commit": gate_contract["demian"]["commit"],
        "robocasa_commit": gate_contract["simulator"]["robocasa_commit"],
        "robosuite_commit": gate_contract["simulator"]["robosuite_commit"],
    }
    if gate_inputs != current_inputs:
        differences = sorted(key for key in current_inputs if gate_inputs.get(key) != current_inputs[key])
        raise ValueError(f"prior gate inputs differ from current formal inputs: {differences}")

    validation = {
        "evidence_hash_matches_expected": True,
        "gate_terminal_verdict_passed": True,
        "gate_contract_matches_evidence": True,
        "gate_runtime_egl_validation_all_true": True,
        "gate_inputs_match_current_formal_inputs": True,
    }
    _require_exact_keys(validation, _PRIOR_GATE_VALIDATION_KEYS, "prior gate validation")
    result = {
        "schema_version": 1,
        "kind": _PRIOR_GATE_VERIFICATION_KIND,
        "status": "passed",
        "prior_gate_evidence": evidence_descriptor,
        "gate_identity": {
            "experiment_id": gate_contract["experiment_id"],
            "logical_run_id": gate_contract["logical_run_id"],
        },
        "gate_evaluation": {
            "stage": "one_episode_gate",
            "evaluation_class": gate_evaluation_class,
            "source_git_pushed_assertion": gate_source_git_pushed,
            "task": gate_contract_evaluation["tasks"][0],
            "tasks": 1,
            "episodes": 1,
            "episodes_per_task": 1,
            "num_shards": 1,
            "execution_complete": True,
            "effect_pass": True,
        },
        "current_inputs": current_inputs,
        "gate_receipts": {
            "contract": gate_contract_descriptor,
            "runtime": runtime_file_descriptor,
        },
        "validation": validation,
    }
    _require_exact_keys(result, _PRIOR_GATE_VERIFICATION_KEYS, "prior gate verification")
    return result


def _expected_formal_tasks(split: str) -> tuple[str, ...]:
    if split == "atomic":
        return ATOMIC_TASKS
    if split == "composite":
        return COMPOSITE_TASKS
    raise ValueError(f"Unsupported formal split: {split!r}")


def create_contract(
    *,
    output: Path,
    experiment_id: str,
    logical_run_id: str,
    source_commit: str,
    source_root: Path,
    source_manifest: Path,
    cosmos_server_module: Path,
    checkpoint_path: Path,
    checkpoint_manifest: Path,
    checkpoint_verification_receipt: Path,
    source_dcp_export_receipt: Path,
    training_config: Path,
    action_stats: Path,
    wan2pt2_vae: Path,
    expected_wan2pt2_vae_sha256: str,
    cosmos_environment_manifest: Path,
    robocasa_environment_manifest: Path,
    eval_intent_manifest: Path,
    submission_verification_receipt: Path,
    asset_verification_receipt: Path,
    eval_runtime_receipt: Path,
    server_runtime_receipts: list[Path],
    demian_repo: Path,
    demian_commit: str,
    demian_evaluator: Path,
    demian_task_registry: Path,
    robocasa_repo: Path,
    robocasa_commit: str,
    robosuite_repo: Path,
    robosuite_commit: str,
    stage: str,
    split: str,
    tasks: list[str] | None,
    episodes_per_task: int,
    base_seed: int,
    max_steps: int,
    open_loop_horizon: int,
    num_shards: int,
    policy_sampler: str,
    policy_seed: int,
    policy_guidance: float,
    policy_num_steps: int,
    results_dir: Path,
    run_name: str,
    minimum_mean_success_rate: float,
    minimum_successful_tasks: int,
    prior_gate_evidence: Path | None = None,
    expected_prior_gate_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    """Create the post-server-start immutable eval contract."""
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing eval contract: {output}")
    source_root = source_root.expanduser().resolve()
    source_manifest = source_manifest.expanduser().resolve()
    cosmos_server_module = cosmos_server_module.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint_manifest = checkpoint_manifest.expanduser().resolve()
    checkpoint_verification_receipt = checkpoint_verification_receipt.expanduser().resolve()
    source_dcp_export_receipt = source_dcp_export_receipt.expanduser().resolve()
    training_config = training_config.expanduser().resolve()
    action_stats = action_stats.expanduser().resolve()
    wan2pt2_vae = Path(os.path.abspath(wan2pt2_vae.expanduser()))
    cosmos_environment_manifest = cosmos_environment_manifest.expanduser().resolve()
    robocasa_environment_manifest = robocasa_environment_manifest.expanduser().resolve()
    eval_intent_manifest = eval_intent_manifest.expanduser().resolve()
    submission_verification_receipt = submission_verification_receipt.expanduser().resolve()
    asset_verification_receipt = asset_verification_receipt.expanduser().resolve()
    eval_runtime_receipt = eval_runtime_receipt.expanduser().resolve()
    server_runtime_receipts = [path.expanduser().resolve() for path in server_runtime_receipts]
    demian_repo = demian_repo.expanduser().resolve()
    demian_evaluator = demian_evaluator.expanduser().resolve()
    demian_task_registry = demian_task_registry.expanduser().resolve()
    robocasa_repo = robocasa_repo.expanduser().resolve()
    robosuite_repo = robosuite_repo.expanduser().resolve()
    results_dir = results_dir.expanduser().resolve()
    if prior_gate_evidence is not None:
        prior_gate_evidence = Path(os.path.abspath(prior_gate_evidence.expanduser()))
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Cosmos source directory not found: {source_root}")
    if not demian_repo.is_dir():
        raise FileNotFoundError(f"DeMiAn repository not found: {demian_repo}")
    if not robocasa_repo.is_dir() or not robosuite_repo.is_dir():
        raise FileNotFoundError("Pinned RoboCasa and robosuite repositories must both exist")
    task_pool = list(tasks) if tasks else list(_expected_formal_tasks(split))
    if len(server_runtime_receipts) != num_shards:
        raise ValueError(
            f"Expected one server runtime receipt per shard: receipts={len(server_runtime_receipts)}, "
            f"num_shards={num_shards}"
        )

    def artifact(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": _sha256(path)}

    source_artifact = artifact(source_manifest)
    server_module_artifact = artifact(cosmos_server_module)
    evaluator_artifact = artifact(demian_evaluator)
    task_registry_artifact = artifact(demian_task_registry)
    checkpoint_manifest_artifact = artifact(checkpoint_manifest)
    source_dcp_export_receipt_artifact = artifact(source_dcp_export_receipt)
    training_config_artifact = artifact(training_config)
    action_stats_artifact = artifact(action_stats)
    wan2pt2_vae_stable = _hash_stable_regular_file(
        wan2pt2_vae,
        "Wan2.2 VAE",
    )
    expected_wan2pt2_vae_sha256 = _require_sha256(
        expected_wan2pt2_vae_sha256,
        "expected Wan2.2 VAE SHA256",
    )
    if wan2pt2_vae_stable["sha256"] != expected_wan2pt2_vae_sha256:
        raise ValueError(
            f"Wan2.2 VAE SHA256 mismatch: expected={expected_wan2pt2_vae_sha256}, actual={wan2pt2_vae_stable['sha256']}"
        )
    wan2pt2_vae_artifact = {
        "path": wan2pt2_vae_stable["path"],
        "sha256": wan2pt2_vae_stable["sha256"],
    }
    cosmos_environment_manifest_artifact = artifact(cosmos_environment_manifest)
    robocasa_environment_manifest_artifact = artifact(robocasa_environment_manifest)
    eval_intent_manifest_artifact = artifact(eval_intent_manifest)
    asset_verification, asset_verification_stable = _load_stable_json(
        asset_verification_receipt,
        "asset verification receipt",
    )
    if (
        asset_verification.get("schema_version") != 1
        or asset_verification.get("kind") != _ASSET_VERIFICATION_KIND
        or asset_verification.get("status") != "passed"
    ):
        raise ValueError("asset verification receipt has an unsupported schema/kind/status")
    current_asset_manifest_sha256 = _require_sha256(
        asset_verification.get("manifest_sha256"),
        "asset verification receipt manifest_sha256",
    )
    current_asset_publisher_receipt_sha256 = _require_sha256(
        asset_verification.get("publisher_receipt_sha256"),
        "asset verification receipt publisher_receipt_sha256",
    )
    asset_verification_receipt_artifact = {
        "path": asset_verification_stable["path"],
        "sha256": asset_verification_stable["sha256"],
    }
    if stage == "one_episode_gate":
        if prior_gate_evidence is not None or expected_prior_gate_evidence_sha256 is not None:
            raise ValueError("one_episode_gate must not accept prior_gate_evidence")
        prior_gate_descriptor = None
    elif stage == "formal":
        if prior_gate_evidence is None or expected_prior_gate_evidence_sha256 is None:
            raise ValueError("formal evaluation requires prior_gate_evidence and its expected SHA256")
        prior_gate_verification = verify_prior_gate(
            gate_evidence_path=prior_gate_evidence,
            expected_gate_evidence_sha256=expected_prior_gate_evidence_sha256,
            current_source_root=source_root,
            current_source_commit=source_commit,
            current_source_manifest_sha256=source_artifact["sha256"],
            current_checkpoint_path=checkpoint_path,
            current_checkpoint_manifest_sha256=checkpoint_manifest_artifact["sha256"],
            current_source_dcp_export_receipt_sha256=source_dcp_export_receipt_artifact["sha256"],
            current_asset_manifest_sha256=current_asset_manifest_sha256,
            current_asset_publisher_receipt_sha256=current_asset_publisher_receipt_sha256,
            current_training_config_sha256=training_config_artifact["sha256"],
            current_action_stats_sha256=action_stats_artifact["sha256"],
            current_wan2pt2_vae_path=Path(wan2pt2_vae_artifact["path"]),
            current_wan2pt2_vae_sha256=wan2pt2_vae_artifact["sha256"],
            current_cosmos_environment_manifest_sha256=cosmos_environment_manifest_artifact["sha256"],
            current_robocasa_environment_manifest_sha256=robocasa_environment_manifest_artifact["sha256"],
            current_demian_commit=demian_commit,
            current_robocasa_commit=robocasa_commit,
            current_robosuite_commit=robosuite_commit,
        )
        prior_gate_descriptor = prior_gate_verification["prior_gate_evidence"]
    else:
        raise ValueError(f"Unsupported evaluation stage: {stage!r}")
    contract = {
        "schema_version": 1,
        "kind": _CONTRACT_KIND,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "logical_run_id": logical_run_id,
        "source_commit": source_commit,
        "source_root": str(source_root),
        "source_manifest_sha256": source_artifact["sha256"],
        "checkpoint_path": str(checkpoint_path),
        "run_name": run_name,
        "results_dir": str(results_dir),
        "demian": {
            "repo": str(demian_repo),
            "commit": demian_commit,
            "evaluator_sha256": evaluator_artifact["sha256"],
            "task_registry_sha256": task_registry_artifact["sha256"],
        },
        "simulator": {
            "split": "target",
            "robocasa_repo": str(robocasa_repo),
            "robocasa_commit": robocasa_commit,
            "robosuite_repo": str(robosuite_repo),
            "robosuite_commit": robosuite_commit,
        },
        "evaluation": {
            "stage": stage,
            "role": "primary" if split == "atomic" else "zero_shot_secondary",
            "split": split,
            "tasks": task_pool,
            "episodes_per_task": episodes_per_task,
            "base_seed": base_seed,
            "max_steps": max_steps,
            "open_loop_horizon": open_loop_horizon,
            "num_shards": num_shards,
            "prior_gate_evidence": prior_gate_descriptor,
            "initialization_mode": "seeded_target_env",
            "seed_rule": "base_seed + within_shard_task_index * 10000 + episode_index",
            "causal_prompt_only": True,
            "policy_sampling": {
                "sampler": policy_sampler,
                "seed": policy_seed,
                "guidance": policy_guidance,
                "num_steps": policy_num_steps,
                "reuse_seed_per_request": True,
                "use_ema_weights": True,
            },
            "effect_thresholds": {
                "minimum_mean_success_rate": minimum_mean_success_rate,
                "minimum_successful_tasks": minimum_successful_tasks,
            },
        },
        "artifacts": {
            "source_manifest": source_artifact,
            "cosmos_server_module": server_module_artifact,
            "checkpoint_manifest": checkpoint_manifest_artifact,
            "checkpoint_verification_receipt": artifact(checkpoint_verification_receipt),
            "source_dcp_export_receipt": source_dcp_export_receipt_artifact,
            "training_config": training_config_artifact,
            "action_stats": action_stats_artifact,
            "wan2pt2_vae": wan2pt2_vae_artifact,
            "cosmos_environment_manifest": cosmos_environment_manifest_artifact,
            "robocasa_environment_manifest": robocasa_environment_manifest_artifact,
            "eval_intent_manifest": eval_intent_manifest_artifact,
            "submission_verification_receipt": artifact(submission_verification_receipt),
            "asset_verification_receipt": asset_verification_receipt_artifact,
            "eval_runtime_receipt": artifact(eval_runtime_receipt),
            "demian_evaluator": evaluator_artifact,
            "demian_task_registry": task_registry_artifact,
            "server_runtime_receipts": [artifact(path) for path in server_runtime_receipts],
        },
    }
    _atomic_write_exclusive(output, json.dumps(contract, indent=2, sort_keys=True) + "\n")
    try:
        validated = _validate_contract(output)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {key: value for key, value in validated.items() if not key.startswith("_")}


def _validate_contract(contract_path: Path) -> dict[str, Any]:
    contract_value, contract_descriptor = _load_stable_json(
        contract_path,
        "contract",
        expected_mode=0o444,
    )
    contract = _require_dict(contract_value, "contract")
    _require_exact_keys(
        contract,
        frozenset(
            {
                "schema_version",
                "kind",
                "created_at",
                "experiment_id",
                "logical_run_id",
                "source_commit",
                "source_root",
                "source_manifest_sha256",
                "checkpoint_path",
                "run_name",
                "results_dir",
                "demian",
                "simulator",
                "evaluation",
                "artifacts",
            }
        ),
        "contract",
    )
    if contract.get("schema_version") != 1 or contract.get("kind") != _CONTRACT_KIND:
        raise ValueError(
            f"Unsupported eval contract schema/kind: {contract.get('schema_version')!r}/{contract.get('kind')!r}"
        )
    contract_created_at = _parse_utc(contract.get("created_at"), "contract.created_at")
    experiment_id = _require_run_id(contract.get("experiment_id"), "contract.experiment_id")
    logical_run_id = _require_run_id(contract.get("logical_run_id"), "contract.logical_run_id")
    if not logical_run_id.startswith(f"{experiment_id}-"):
        raise ValueError("contract.logical_run_id must be namespaced below contract.experiment_id")
    _require_commit(contract.get("source_commit"), "contract.source_commit")
    source_root = Path(_require_nonempty_str(contract.get("source_root"), "contract.source_root"))
    if not source_root.is_absolute() or not source_root.is_dir():
        raise ValueError(f"contract.source_root must be an existing absolute directory: {source_root}")
    _require_sha256(contract.get("source_manifest_sha256"), "contract.source_manifest_sha256")
    checkpoint_path = Path(_require_nonempty_str(contract.get("checkpoint_path"), "contract.checkpoint_path"))
    results_dir = Path(_require_nonempty_str(contract.get("results_dir"), "contract.results_dir"))
    run_name = _require_run_id(contract.get("run_name"), "contract.run_name")
    if not checkpoint_path.is_absolute() or not results_dir.is_absolute():
        raise ValueError("contract checkpoint_path and results_dir must be absolute")
    if results_dir.name != run_name:
        raise ValueError(f"contract results_dir must end in run_name: results_dir={results_dir}, run_name={run_name!r}")
    if run_name != logical_run_id:
        raise ValueError("contract.run_name must equal contract.logical_run_id")

    demian = _require_dict(contract.get("demian"), "contract.demian")
    _require_exact_keys(
        demian,
        frozenset(
            {
                "repo",
                "commit",
                "evaluator_sha256",
                "task_registry_sha256",
            }
        ),
        "contract.demian",
    )
    demian_commit = _require_commit(demian.get("commit"), "contract.demian.commit")
    evaluator_sha256 = _require_sha256(
        demian.get("evaluator_sha256"),
        "contract.demian.evaluator_sha256",
    )
    task_registry_sha256 = _require_sha256(
        demian.get("task_registry_sha256"),
        "contract.demian.task_registry_sha256",
    )
    if (
        demian_commit != _EXPECTED_DEMIAN_COMMIT
        or evaluator_sha256 != _EXPECTED_DEMIAN_EVALUATOR_SHA256
        or task_registry_sha256 != _EXPECTED_DEMIAN_TASK_REGISTRY_SHA256
    ):
        raise ValueError("contract.demian does not match the audited DeMiAn revision and evaluator files")
    demian_repo = Path(_require_nonempty_str(demian.get("repo"), "contract.demian.repo"))
    if not demian_repo.is_absolute() or not demian_repo.is_dir():
        raise ValueError(f"contract.demian.repo must be an existing absolute directory: {demian_repo}")

    simulator = _require_dict(contract.get("simulator"), "contract.simulator")
    _require_exact_keys(
        simulator,
        frozenset(
            {
                "split",
                "robocasa_repo",
                "robocasa_commit",
                "robosuite_repo",
                "robosuite_commit",
            }
        ),
        "contract.simulator",
    )
    if simulator.get("split") != "target":
        raise ValueError("contract.simulator.split must be target")
    _require_commit(simulator.get("robocasa_commit"), "contract.simulator.robocasa_commit")
    _require_commit(simulator.get("robosuite_commit"), "contract.simulator.robosuite_commit")
    for key in ("robocasa_repo", "robosuite_repo"):
        repo = Path(_require_nonempty_str(simulator.get(key), f"contract.simulator.{key}"))
        if not repo.is_absolute() or not repo.is_dir():
            raise ValueError(f"contract.simulator.{key} must be an existing absolute directory: {repo}")

    evaluation = _require_dict(contract.get("evaluation"), "contract.evaluation")
    _require_exact_keys(
        evaluation,
        frozenset(
            {
                "stage",
                "role",
                "split",
                "tasks",
                "episodes_per_task",
                "base_seed",
                "max_steps",
                "open_loop_horizon",
                "num_shards",
                "prior_gate_evidence",
                "initialization_mode",
                "seed_rule",
                "causal_prompt_only",
                "policy_sampling",
                "effect_thresholds",
            }
        ),
        "contract.evaluation",
    )
    stage = evaluation.get("stage")
    if stage not in {"one_episode_gate", "formal"}:
        raise ValueError("contract.evaluation.stage must be one_episode_gate or formal")
    prior_gate_descriptor = evaluation.get("prior_gate_evidence")
    if stage == "one_episode_gate":
        if prior_gate_descriptor is not None:
            raise ValueError("one_episode_gate contract.evaluation.prior_gate_evidence must be null")
    else:
        if prior_gate_descriptor is None:
            raise ValueError("formal contract.evaluation.prior_gate_evidence is required")
        prior_gate_descriptor = _require_dict(
            prior_gate_descriptor,
            "contract.evaluation.prior_gate_evidence",
        )
        _require_exact_keys(
            prior_gate_descriptor,
            _POST_FILE_DESCRIPTOR_KEYS,
            "contract.evaluation.prior_gate_evidence",
        )
        _absolute_resolved_path(
            prior_gate_descriptor.get("path"),
            "contract.evaluation.prior_gate_evidence.path",
        )
        _require_int(
            prior_gate_descriptor.get("size_bytes"),
            "contract.evaluation.prior_gate_evidence.size_bytes",
            minimum=1,
        )
        _require_sha256(
            prior_gate_descriptor.get("sha256"),
            "contract.evaluation.prior_gate_evidence.sha256",
        )
    split = evaluation.get("split")
    if split not in {"atomic", "composite"}:
        raise ValueError("contract.evaluation.split must be atomic or composite")
    if split != "atomic":
        raise ValueError("this primary CWAM evaluation contract requires the atomic split")
    expected_role = "primary" if split == "atomic" else "zero_shot_secondary"
    if evaluation.get("role") != expected_role:
        raise ValueError(f"contract.evaluation.role must be {expected_role!r} for split={split!r}")
    task_pool = evaluation.get("tasks")
    if not isinstance(task_pool, list) or not task_pool:
        raise ValueError("contract.evaluation.tasks must be a non-empty list")
    if len(task_pool) != len(set(task_pool)) or not all(isinstance(task, str) and task for task in task_pool):
        raise ValueError("contract.evaluation.tasks must contain unique non-empty strings")
    canonical = _expected_formal_tasks(split)
    unknown = sorted(set(task_pool) - set(canonical))
    if unknown:
        raise ValueError(f"contract.evaluation.tasks contains non-canonical {split} tasks: {unknown}")
    episodes = _require_int(evaluation.get("episodes_per_task"), "episodes_per_task", minimum=1)
    if stage == "formal":
        if tuple(task_pool) != canonical:
            raise ValueError(f"formal {split} evaluation must use the canonical ordered task list")
        if episodes != 20:
            raise ValueError(f"formal {split} evaluation requires exactly 20 episodes per task")
    elif episodes != 1 or len(task_pool) != 1:
        raise ValueError("one_episode_gate requires exactly one canonical task and one episode")

    if _require_int(evaluation.get("base_seed"), "base_seed", minimum=0) != 42:
        raise ValueError("base_seed must be exactly 42")
    if _require_int(evaluation.get("max_steps"), "max_steps", minimum=1) != 400:
        raise ValueError("max_steps must be exactly 400")
    if _require_int(evaluation.get("open_loop_horizon"), "open_loop_horizon", minimum=1) != 12:
        raise ValueError("open_loop_horizon must be exactly 12")
    num_shards = _require_int(evaluation.get("num_shards"), "num_shards", minimum=1)
    if num_shards > len(task_pool):
        raise ValueError("num_shards cannot exceed the number of tasks")
    if stage == "formal" and num_shards != 8:
        raise ValueError("formal evaluation requires exactly eight shards")
    if stage == "one_episode_gate" and num_shards != 1:
        raise ValueError("one_episode_gate requires exactly one shard")
    if evaluation.get("seed_rule") != "base_seed + within_shard_task_index * 10000 + episode_index":
        raise ValueError("contract.evaluation.seed_rule does not match the pinned DeMiAn evaluator")
    if evaluation.get("causal_prompt_only") is not True:
        raise ValueError("contract.evaluation.causal_prompt_only must be true")
    if evaluation.get("initialization_mode") != "seeded_target_env":
        raise ValueError("contract.evaluation.initialization_mode must be seeded_target_env")
    sampling = _require_dict(evaluation.get("policy_sampling"), "policy_sampling")
    _require_exact_keys(
        sampling,
        frozenset(
            {
                "sampler",
                "seed",
                "guidance",
                "num_steps",
                "reuse_seed_per_request",
                "use_ema_weights",
            }
        ),
        "policy_sampling",
    )
    if sampling.get("sampler") != "unipc":
        raise ValueError("policy_sampling.sampler must be exactly unipc")
    if _require_int(sampling.get("seed"), "policy_sampling.seed", minimum=0) != 42:
        raise ValueError("policy_sampling.seed must be exactly 42")
    guidance = sampling.get("guidance")
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(float(guidance))
        or float(guidance) != 1.0
    ):
        raise TypeError("policy_sampling.guidance must be exactly 1.0")
    if _require_int(sampling.get("num_steps"), "policy_sampling.num_steps", minimum=1) != 30:
        raise ValueError("policy_sampling.num_steps must be exactly 30")
    if sampling.get("reuse_seed_per_request") is not True or sampling.get("use_ema_weights") is not True:
        raise ValueError("formal policy sampling must reuse its seed and load EMA weights")

    thresholds = _require_dict(evaluation.get("effect_thresholds"), "effect_thresholds")
    _require_exact_keys(
        thresholds,
        frozenset(
            {
                "minimum_mean_success_rate",
                "minimum_successful_tasks",
            }
        ),
        "effect_thresholds",
    )
    minimum_mean_sr = thresholds.get("minimum_mean_success_rate")
    minimum_successful_tasks = thresholds.get("minimum_successful_tasks")
    if isinstance(minimum_mean_sr, bool) or not isinstance(minimum_mean_sr, (int, float)):
        raise TypeError("minimum_mean_success_rate must be numeric")
    if not 0.0 <= float(minimum_mean_sr) <= 1.0:
        raise ValueError("minimum_mean_success_rate must be in [0,1]")
    _require_int(minimum_successful_tasks, "minimum_successful_tasks", minimum=0)
    if minimum_successful_tasks > len(task_pool):
        raise ValueError("minimum_successful_tasks exceeds task count")
    if stage == "one_episode_gate" and (float(minimum_mean_sr) != 0.0 or minimum_successful_tasks != 0):
        raise ValueError("one_episode_gate effect thresholds must be exactly 0.0 and 0")

    artifacts = _require_dict(contract.get("artifacts"), "contract.artifacts")
    _require_exact_keys(
        artifacts,
        frozenset(
            {
                "source_manifest",
                "cosmos_server_module",
                "checkpoint_manifest",
                "checkpoint_verification_receipt",
                "source_dcp_export_receipt",
                "training_config",
                "action_stats",
                "wan2pt2_vae",
                "cosmos_environment_manifest",
                "robocasa_environment_manifest",
                "eval_intent_manifest",
                "submission_verification_receipt",
                "asset_verification_receipt",
                "eval_runtime_receipt",
                "demian_evaluator",
                "demian_task_registry",
                "server_runtime_receipts",
            }
        ),
        "contract.artifacts",
    )
    verified: dict[str, Any] = {}
    bound_payloads: dict[str, Any] = {}
    stable_artifacts: dict[str, Any] = {}
    for name in (
        "source_manifest",
        "cosmos_server_module",
        "checkpoint_manifest",
        "checkpoint_verification_receipt",
        "source_dcp_export_receipt",
        "training_config",
        "action_stats",
        "cosmos_environment_manifest",
        "robocasa_environment_manifest",
        "eval_intent_manifest",
        "submission_verification_receipt",
        "asset_verification_receipt",
        "eval_runtime_receipt",
        "demian_evaluator",
        "demian_task_registry",
    ):
        expected_mode = 0o444 if name in {"eval_intent_manifest", "source_dcp_export_receipt"} else None
        verified[name], bound_payloads[name], stable_artifacts[name] = _read_bound_file_artifact(
            artifacts,
            name,
            expected_mode=expected_mode,
        )
    wan2pt2_vae_descriptor = _require_dict(
        artifacts.get("wan2pt2_vae"),
        "contract.artifacts.wan2pt2_vae",
    )
    _require_exact_keys(
        wan2pt2_vae_descriptor,
        frozenset({"path", "sha256"}),
        "contract.artifacts.wan2pt2_vae",
    )
    wan2pt2_vae_path = Path(
        _require_nonempty_str(
            wan2pt2_vae_descriptor.get("path"),
            "contract.artifacts.wan2pt2_vae.path",
        )
    )
    if not wan2pt2_vae_path.is_absolute():
        raise ValueError("contract.artifacts.wan2pt2_vae.path must be absolute")
    expected_wan2pt2_vae_sha256 = _require_sha256(
        wan2pt2_vae_descriptor.get("sha256"),
        "contract.artifacts.wan2pt2_vae.sha256",
    )
    wan2pt2_vae_stable = _hash_stable_regular_file(
        wan2pt2_vae_path,
        "contract.artifacts.wan2pt2_vae",
    )
    if wan2pt2_vae_stable["sha256"] != expected_wan2pt2_vae_sha256:
        raise RuntimeError(
            "wan2pt2_vae SHA256 mismatch: "
            f"expected={expected_wan2pt2_vae_sha256}, "
            f"actual={wan2pt2_vae_stable['sha256']}, "
            f"path={wan2pt2_vae_path}"
        )
    verified["wan2pt2_vae"] = {
        "path": wan2pt2_vae_stable["path"],
        "sha256": wan2pt2_vae_stable["sha256"],
    }
    stable_artifacts["wan2pt2_vae"] = wan2pt2_vae_stable
    receipt_descriptors = artifacts.get("server_runtime_receipts")
    if not isinstance(receipt_descriptors, list) or len(receipt_descriptors) != num_shards:
        raise ValueError("contract.artifacts.server_runtime_receipts must contain one entry per shard")
    verified_receipts = []
    receipt_payloads = []
    stable_receipts = []
    for index, descriptor in enumerate(receipt_descriptors):
        synthetic = {"receipt": descriptor}
        public_descriptor, payload, stable_descriptor = _read_bound_file_artifact(synthetic, "receipt")
        verified_receipts.append(public_descriptor)
        receipt_payloads.append(payload)
        stable_receipts.append(stable_descriptor)
    verified["server_runtime_receipts"] = verified_receipts
    bound_payloads["server_runtime_receipts"] = receipt_payloads
    stable_artifacts["server_runtime_receipts"] = stable_receipts
    json_artifact_names = (
        "source_manifest",
        "checkpoint_manifest",
        "checkpoint_verification_receipt",
        "source_dcp_export_receipt",
        "cosmos_environment_manifest",
        "robocasa_environment_manifest",
        "eval_intent_manifest",
        "submission_verification_receipt",
        "asset_verification_receipt",
        "eval_runtime_receipt",
    )
    bound_json = {
        name: _parse_json_bytes(bound_payloads[name], Path(stable_artifacts[name]["path"]))
        for name in json_artifact_names
    }
    bound_json["server_runtime_receipts"] = [
        _parse_json_bytes(payload, Path(stable_descriptor["path"]))
        for payload, stable_descriptor in zip(receipt_payloads, stable_receipts, strict=True)
    ]
    expected_server_module = source_root / "cosmos_framework" / "scripts" / "action_policy_server_robocasa365.py"
    if Path(verified["cosmos_server_module"]["path"]).resolve() != expected_server_module.resolve():
        raise ValueError("cosmos_server_module artifact resolved outside the exact source root")
    if verified["source_manifest"]["sha256"] != contract["source_manifest_sha256"]:
        raise ValueError("source_manifest artifact hash does not match contract.source_manifest_sha256")
    checkpoint_manifest = _require_dict(bound_json["checkpoint_manifest"], "checkpoint_manifest")
    _require_exact_keys(
        checkpoint_manifest,
        _CHECKPOINT_MANIFEST_KEYS,
        "checkpoint_manifest",
    )
    if (
        checkpoint_manifest.get("schema_version") != 1
        or checkpoint_manifest.get("kind") != "cwam_serving_checkpoint_manifest"
    ):
        raise ValueError("checkpoint_manifest has an unsupported schema/kind")
    if (
        Path(
            _require_nonempty_str(
                checkpoint_manifest.get("checkpoint_path"),
                "checkpoint_manifest.checkpoint_path",
            )
        ).resolve()
        != checkpoint_path.resolve()
    ):
        raise ValueError("checkpoint_manifest checkpoint_path does not match the eval contract")
    checkpoint_tree_sha256 = _require_sha256(
        checkpoint_manifest.get("tree_sha256"),
        "checkpoint_manifest.tree_sha256",
    )
    checkpoint_files = checkpoint_manifest.get("files")
    if not isinstance(checkpoint_files, list) or not checkpoint_files:
        raise ValueError("checkpoint_manifest.files must be a non-empty list")
    checkpoint_verification = _require_dict(
        bound_json["checkpoint_verification_receipt"],
        "checkpoint_verification_receipt",
    )
    _require_exact_keys(
        checkpoint_verification,
        _CHECKPOINT_VERIFICATION_KEYS,
        "checkpoint_verification_receipt",
    )
    if checkpoint_verification.get("status") != "passed":
        raise ValueError("checkpoint_verification_receipt status must be passed")
    if (
        Path(
            _require_nonempty_str(
                checkpoint_verification.get("checkpoint_path"),
                "checkpoint_verification_receipt.checkpoint_path",
            )
        ).resolve()
        != checkpoint_path.resolve()
    ):
        raise ValueError("checkpoint_verification_receipt path does not match the eval contract")
    checkpoint_file_count = _require_int(
        checkpoint_verification.get("file_count"),
        "checkpoint_verification_receipt.file_count",
        minimum=1,
    )
    if checkpoint_file_count != len(checkpoint_files):
        raise ValueError("checkpoint verification file_count does not match checkpoint_manifest.files")
    if (
        _require_sha256(
            checkpoint_verification.get("tree_sha256"),
            "checkpoint_verification_receipt.tree_sha256",
        )
        != checkpoint_tree_sha256
    ):
        raise ValueError("checkpoint verification tree SHA does not match the checkpoint manifest")
    source_dcp_selection = _validate_source_dcp_selection_receipt(
        value=bound_json["source_dcp_export_receipt"],
        stable_artifact=stable_artifacts["source_dcp_export_receipt"],
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_sha256=verified["checkpoint_manifest"]["sha256"],
        checkpoint_tree_sha256=checkpoint_tree_sha256,
    )
    if verified["demian_evaluator"]["sha256"] != demian["evaluator_sha256"]:
        raise ValueError("demian_evaluator artifact hash does not match contract.demian.evaluator_sha256")
    if verified["demian_task_registry"]["sha256"] != demian["task_registry_sha256"]:
        raise ValueError("demian_task_registry artifact hash does not match contract.demian.task_registry_sha256")
    cosmos_environment_manifest = _validated_environment_manifest(
        bound_json["cosmos_environment_manifest"],
        "cosmos_environment_manifest",
    )
    robocasa_environment_manifest = _validated_environment_manifest(
        bound_json["robocasa_environment_manifest"],
        "robocasa_environment_manifest",
    )
    runtime_bindings = _validate_eval_runtime_receipt(
        contract=contract,
        verified=verified,
        stable_artifacts=stable_artifacts,
        bound_payloads=bound_payloads,
        bound_json=bound_json,
        cosmos_environment_manifest=cosmos_environment_manifest,
        robocasa_environment_manifest=robocasa_environment_manifest,
        source_dcp_selection=source_dcp_selection,
    )
    prior_gate_binding = None
    if stage == "formal":
        assert isinstance(prior_gate_descriptor, dict)
        prior_gate_binding = verify_prior_gate(
            gate_evidence_path=Path(prior_gate_descriptor["path"]),
            expected_gate_evidence_sha256=prior_gate_descriptor["sha256"],
            current_source_root=source_root,
            current_source_commit=contract["source_commit"],
            current_source_manifest_sha256=contract["source_manifest_sha256"],
            current_checkpoint_path=checkpoint_path,
            current_checkpoint_manifest_sha256=verified["checkpoint_manifest"]["sha256"],
            current_source_dcp_export_receipt_sha256=verified["source_dcp_export_receipt"]["sha256"],
            current_asset_manifest_sha256=runtime_bindings["assets"]["manifest_sha256"],
            current_asset_publisher_receipt_sha256=runtime_bindings["assets"]["publisher_receipt_sha256"],
            current_training_config_sha256=verified["training_config"]["sha256"],
            current_action_stats_sha256=verified["action_stats"]["sha256"],
            current_wan2pt2_vae_path=Path(verified["wan2pt2_vae"]["path"]),
            current_wan2pt2_vae_sha256=verified["wan2pt2_vae"]["sha256"],
            current_cosmos_environment_manifest_sha256=verified["cosmos_environment_manifest"]["sha256"],
            current_robocasa_environment_manifest_sha256=verified["robocasa_environment_manifest"]["sha256"],
            current_demian_commit=demian["commit"],
            current_robocasa_commit=simulator["robocasa_commit"],
            current_robosuite_commit=simulator["robosuite_commit"],
        )
        if prior_gate_binding["prior_gate_evidence"] != prior_gate_descriptor:
            raise ValueError("formal contract prior gate descriptor changed during verification")
    if contract_created_at < _parse_utc(
        runtime_bindings["captured_at_utc"],
        "eval_runtime_receipt.captured_at_utc",
    ):
        raise ValueError("contract.created_at predates runtime capture completion")
    contract["_verified_artifacts"] = verified
    contract["_bound_payloads"] = bound_payloads
    contract["_bound_json"] = bound_json
    contract["_stable_artifacts"] = stable_artifacts
    contract["_runtime_bindings"] = runtime_bindings
    contract["_source_dcp_selection"] = source_dcp_selection
    contract["_prior_gate_binding"] = prior_gate_binding
    contract["_contract_descriptor"] = contract_descriptor
    contract["_contract_sha256"] = contract_descriptor["sha256"]
    return contract


def _post_rollout_input_descriptors(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    verified = contract["_verified_artifacts"]
    stable_artifacts = contract["_stable_artifacts"]
    source_manifest_path = Path(verified["source_manifest"]["path"])
    expected_source_manifest_path = Path(contract["source_root"]) / "SNAPSHOT_MANIFEST.json"
    if source_manifest_path != expected_source_manifest_path.resolve(strict=True):
        raise ValueError("contract source_manifest must be the exact Cosmos SNAPSHOT_MANIFEST.json")
    return {
        "cosmos_snapshot_manifest": dict(stable_artifacts["source_manifest"]),
        "serving_checkpoint_manifest": dict(stable_artifacts["checkpoint_manifest"]),
        "pre_rollout_checkpoint_verification_receipt": dict(stable_artifacts["checkpoint_verification_receipt"]),
        "pre_rollout_asset_verification_receipt": dict(stable_artifacts["asset_verification_receipt"]),
        "wan2pt2_vae": dict(stable_artifacts["wan2pt2_vae"]),
    }


def _verify_snapshot_with_bound_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    root = _canonical_existing_directory(Path(contract["source_root"]), "Cosmos snapshot root")
    allowed_root = _POST_ROLLOUT_SNAPSHOT_ALLOWED_ROOT.resolve(strict=True)
    try:
        root.relative_to(allowed_root)
    except ValueError as error:
        raise ValueError(f"Cosmos snapshot root is outside the allowed root: {root}") from error

    manifest_descriptor = contract["_stable_artifacts"]["source_manifest"]
    manifest_path = root / "SNAPSHOT_MANIFEST.json"
    if (
        manifest_descriptor["path"] != str(manifest_path)
        or manifest_descriptor["sha256"] != contract["source_manifest_sha256"]
    ):
        raise ValueError("bound Cosmos snapshot manifest path or SHA256 differs from the eval contract")
    manifest = _require_dict(contract["_bound_json"]["source_manifest"], "Cosmos snapshot manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("Cosmos snapshot manifest schema_version must be 1")
    if (
        _require_commit(manifest.get("source_head"), "Cosmos snapshot manifest source_head")
        != contract["source_commit"]
    ):
        raise ValueError("Cosmos snapshot manifest source_head differs from the eval contract")
    if manifest.get("source_dirty") is not False:
        raise ValueError("Cosmos snapshot manifest must assert source_dirty=false")
    entries = manifest.get("files")
    if not isinstance(entries, list) or manifest.get("file_count") != len(entries):
        raise ValueError("Cosmos snapshot manifest file inventory is malformed")

    expected: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(entries):
        entry = _require_dict(raw_entry, f"Cosmos snapshot manifest files[{index}]")
        _require_exact_keys(
            entry,
            frozenset({"path", "sha256", "bytes"}),
            f"Cosmos snapshot manifest files[{index}]",
        )
        raw_relative = _require_nonempty_str(entry.get("path"), f"Cosmos snapshot manifest files[{index}].path")
        relative = PurePosixPath(raw_relative)
        if (
            relative.is_absolute()
            or relative.as_posix() != raw_relative
            or "." in relative.parts
            or ".." in relative.parts
            or raw_relative == "SNAPSHOT_MANIFEST.json"
            or raw_relative in expected
        ):
            raise ValueError(f"Cosmos snapshot manifest path is unsafe or duplicated: {raw_relative!r}")
        size_bytes = _require_int(
            entry.get("bytes"),
            f"Cosmos snapshot manifest files[{index}].bytes",
            minimum=0,
        )
        expected[raw_relative] = {
            "path": raw_relative,
            "bytes": size_bytes,
            "sha256": _require_sha256(
                entry.get("sha256"),
                f"Cosmos snapshot manifest files[{index}].sha256",
            ),
        }

    actual = cwam_formal_provenance._scan_regular_tree(root, reject_writable=True)
    actual.pop("SNAPSHOT_MANIFEST.json", None)
    if set(actual) != set(expected):
        raise ValueError(
            "Cosmos snapshot file set mismatch: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    total_bytes = 0
    for relative in sorted(expected):
        _, descriptor = _read_stable_regular_file(actual[relative], f"Cosmos snapshot file {relative}")
        if (
            descriptor["size_bytes"] != expected[relative]["bytes"]
            or descriptor["sha256"] != expected[relative]["sha256"]
        ):
            raise ValueError(f"Cosmos snapshot file differs from its bound manifest: {relative}")
        total_bytes += descriptor["size_bytes"]
    return {
        "schema_version": 1,
        "status": "passed",
        "snapshot_root": str(root),
        "snapshot_manifest": manifest_descriptor["path"],
        "snapshot_manifest_sha256": manifest_descriptor["sha256"],
        "file_count": len(expected),
        "file_bytes": total_bytes,
    }


def _verify_asset_tree_with_bound_provenance(
    contract: dict[str, Any],
    *,
    asset_workers: int = cwam_robocasa_asset_manifest.DEFAULT_HASH_WORKERS,
) -> dict[str, Any]:
    root, _ = cwam_robocasa_asset_manifest._require_path(
        contract["simulator"]["robocasa_repo"],
        "asset root",
        expected_kind="directory",
    )
    receipt = cwam_robocasa_asset_manifest._validate_publisher_receipt(
        cwam_robocasa_asset_manifest._decode_json(
            contract["_bound_payloads"]["asset_publisher_receipt"],
            "asset publisher receipt",
        ),
        root,
    )
    manifest_descriptor = contract["_stable_artifacts"]["asset_manifest"]
    publisher_descriptor = contract["_stable_artifacts"]["asset_publisher_receipt"]
    manifest_bytes = contract["_bound_payloads"]["asset_manifest"]
    if manifest_descriptor["sha256"] != receipt["manifest_sha256"]:
        raise ValueError("bound asset manifest SHA256 differs from the bound publisher receipt")
    header, expected_records = cwam_robocasa_asset_manifest._parse_asset_manifest(manifest_bytes, root)
    if header["excluded_top_level_paths"] != receipt["excluded_top_level_paths"]:
        raise ValueError("bound asset manifest and publisher receipt exclusion policies differ")
    expected_summary = cwam_robocasa_asset_manifest._payload_summary(expected_records)
    for summary_key, receipt_key in (
        ("directory_count", "payload_directory_count"),
        ("file_count", "payload_file_count"),
        ("symlink_count", "payload_symlink_count"),
        ("total_bytes", "payload_total_bytes"),
    ):
        if expected_summary[summary_key] != receipt[receipt_key]:
            raise ValueError(f"bound asset manifest {summary_key} differs from the publisher receipt")

    actual_records, asset_root_mode = cwam_robocasa_asset_manifest._scan_payload_tree(
        root,
        workers=asset_workers,
    )
    cwam_robocasa_asset_manifest._compare_manifest_and_live_tree(expected_records, actual_records)
    actual_summary = cwam_robocasa_asset_manifest._payload_summary(actual_records)
    if actual_summary != expected_summary:
        raise ValueError("live asset payload summary differs from the bound asset manifest")
    return {
        "schema_version": cwam_robocasa_asset_manifest.SCHEMA_VERSION,
        "kind": cwam_robocasa_asset_manifest.VERIFICATION_RECEIPT_KIND,
        "status": "passed",
        "asset_root": str(root),
        "asset_root_mode": asset_root_mode,
        "excluded_top_level_paths": list(cwam_robocasa_asset_manifest.EXCLUDED_TOP_LEVEL_PATHS),
        "manifest_path": manifest_descriptor["path"],
        "manifest_size_bytes": manifest_descriptor["size_bytes"],
        "manifest_sha256": manifest_descriptor["sha256"],
        "publisher_receipt_path": publisher_descriptor["path"],
        "publisher_receipt_size_bytes": publisher_descriptor["size_bytes"],
        "publisher_receipt_sha256": publisher_descriptor["sha256"],
        "publisher_generator_id": receipt["generator_id"],
        "publisher_generated_at_utc": receipt["generated_at_utc"],
        "robocasa_git_commit": receipt["git_commit"],
        "payload_directory_count": actual_summary["directory_count"],
        "payload_file_count": actual_summary["file_count"],
        "payload_symlink_count": actual_summary["symlink_count"],
        "payload_total_bytes": actual_summary["total_bytes"],
        "source_archive_directory": receipt["source_archive_directory"],
        "source_sha256sums_sha256": receipt["source_sha256sums_sha256"],
        "source_sizes_sha256": receipt["source_sizes_sha256"],
    }


def _collect_post_rollout_live_verification(
    *,
    contract_path: Path,
    contract: dict[str, Any],
    asset_workers: int = cwam_robocasa_asset_manifest.DEFAULT_HASH_WORKERS,
) -> dict[str, Any]:
    """Re-verify live payload trees after rollout.

    Bound control artifacts are parsed only from the stable O_NOFOLLOW reads
    captured by ``_validate_contract``. Large payload trees are independently
    reverified against those immutable control bytes.
    """

    asset_workers = cwam_robocasa_asset_manifest._validated_hash_workers(asset_workers)
    started = time.perf_counter()
    contract_descriptor = dict(contract["_contract_descriptor"])
    contract_descriptor.update(
        {
            "kind": _CONTRACT_KIND,
            "schema_version": 1,
        }
    )
    input_descriptors = _post_rollout_input_descriptors(contract)

    phase_started = time.perf_counter()
    snapshot = _verify_snapshot_with_bound_manifest(contract)
    snapshot_elapsed = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    checkpoint_manifest = _require_dict(
        contract["_bound_json"]["checkpoint_manifest"],
        "serving checkpoint manifest",
    )
    checkpoint = cwam_serving_checkpoint_manifest.verify_serving_checkpoint_manifest(
        checkpoint_manifest,
        contract["checkpoint_path"],
    )
    checkpoint_elapsed = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    assets = _verify_asset_tree_with_bound_provenance(
        contract,
        asset_workers=asset_workers,
    )
    assets_elapsed = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    wan2pt2_vae = _hash_stable_regular_file(
        Path(contract["_verified_artifacts"]["wan2pt2_vae"]["path"]),
        "post-rollout Wan2.2 VAE",
    )
    wan2pt2_vae_elapsed = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    source_repositories = {
        "cosmos": {
            "root": snapshot["snapshot_root"],
            "expected_commit": contract["source_commit"],
            "actual_commit": contract["source_commit"],
            "clean": True,
            "tracked_files": {
                "verification_mode": "covered_by_full_cosmos_snapshot_verification",
            },
        },
        "demian": _verify_git_repository(
            name="DeMiAn",
            root=Path(contract["demian"]["repo"]),
            expected_commit=contract["demian"]["commit"],
            verify_tracked_files=True,
        ),
        "robocasa": _verify_git_repository(
            name="RoboCasa",
            root=Path(contract["simulator"]["robocasa_repo"]),
            expected_commit=contract["simulator"]["robocasa_commit"],
            verify_tracked_files=True,
        ),
        "robosuite": _verify_git_repository(
            name="robosuite",
            root=Path(contract["simulator"]["robosuite_repo"]),
            expected_commit=contract["simulator"]["robosuite_commit"],
            verify_tracked_files=True,
        ),
    }
    source_elapsed = time.perf_counter() - phase_started

    pre_checkpoint = _require_dict(
        contract["_bound_json"]["checkpoint_verification_receipt"],
        "pre_rollout_checkpoint_verification_receipt",
    )
    pre_assets = _require_dict(
        contract["_bound_json"]["asset_verification_receipt"],
        "pre_rollout_asset_verification_receipt",
    )
    if checkpoint != pre_checkpoint:
        raise ValueError("post-rollout checkpoint verification does not exactly match the pre-rollout receipt")
    if assets != pre_assets:
        raise ValueError("post-rollout asset verification does not exactly match the pre-rollout receipt")
    if wan2pt2_vae != contract["_stable_artifacts"]["wan2pt2_vae"]:
        raise ValueError("post-rollout Wan2.2 VAE descriptor does not exactly match the pre-rollout descriptor")

    semantic_matches = {
        "checkpoint_pre_and_post": True,
        "robocasa_assets_pre_and_post": True,
        "wan2pt2_vae_pre_and_post": True,
    }
    validation = {
        "cosmos_snapshot_full_file_verification_passed": True,
        "serving_checkpoint_full_file_verification_passed": True,
        "robocasa_asset_full_tree_verification_passed": True,
        "wan2pt2_vae_content_rehash_passed": True,
        "source_heads_match_and_are_clean": True,
        "tracked_source_files_match_head_blobs": True,
        "pre_and_post_checkpoint_receipts_semantically_match": True,
        "pre_and_post_asset_receipts_semantically_match": True,
        "pre_and_post_wan2pt2_vae_semantically_match": True,
    }
    return {
        "schema_version": 1,
        "kind": _POST_ROLLOUT_VERIFICATION_KIND,
        "status": "passed",
        "scope": _POST_ROLLOUT_SCOPE,
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract": contract_descriptor,
        "inputs": input_descriptors,
        "live_verification": {
            "cosmos_snapshot": snapshot,
            "serving_checkpoint": checkpoint,
            "robocasa_assets": assets,
            "wan2pt2_vae": wan2pt2_vae,
            "source_repositories": source_repositories,
        },
        "semantic_matches": semantic_matches,
        "elapsed_seconds": {
            "cosmos_snapshot": snapshot_elapsed,
            "serving_checkpoint": checkpoint_elapsed,
            "robocasa_assets": assets_elapsed,
            "wan2pt2_vae": wan2pt2_vae_elapsed,
            "source_repositories": source_elapsed,
            "total": time.perf_counter() - started,
        },
        "validation": validation,
    }


def _validate_post_rollout_elapsed(value: Any, label: str) -> dict[str, float]:
    elapsed = _require_dict(value, label)
    _require_exact_keys(elapsed, _POST_ROLLOUT_ELAPSED_KEYS, label)
    normalized: dict[str, float] = {}
    for key in sorted(_POST_ROLLOUT_ELAPSED_KEYS):
        duration = elapsed.get(key)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise TypeError(f"{label}.{key} must be numeric")
        normalized[key] = float(duration)
        if not math.isfinite(normalized[key]) or normalized[key] < 0.0:
            raise ValueError(f"{label}.{key} must be finite and non-negative")
    component_total = sum(
        normalized[key]
        for key in (
            "cosmos_snapshot",
            "serving_checkpoint",
            "robocasa_assets",
            "wan2pt2_vae",
            "source_repositories",
        )
    )
    if normalized["total"] + 1e-9 < component_total:
        raise ValueError(f"{label}.total is smaller than the measured component durations")
    return normalized


def verify_post_rollout(
    *,
    contract_path: Path,
    output: Path,
    asset_workers: int = cwam_robocasa_asset_manifest.DEFAULT_HASH_WORKERS,
) -> dict[str, Any]:
    contract_path = Path(os.path.abspath(contract_path.expanduser()))
    output = Path(os.path.abspath(output.expanduser()))
    try:
        canonical_contract_parent = contract_path.parent.resolve(strict=True)
        contract_parent_metadata = canonical_contract_parent.lstat()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise ValueError("eval contract parent is not an existing canonical directory") from error
    if (
        contract_path.parent != canonical_contract_parent
        or stat.S_ISLNK(contract_parent_metadata.st_mode)
        or not stat.S_ISDIR(contract_parent_metadata.st_mode)
    ):
        raise ValueError("eval contract parent must be an existing canonical non-symlink directory")
    if output.parent != canonical_contract_parent:
        raise ValueError("post-rollout verification receipt must share the eval contract canonical parent")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite post-rollout verification receipt: {output}")
    contract = _validate_contract(contract_path)
    payload = _collect_post_rollout_live_verification(
        contract_path=contract_path,
        contract=contract,
        asset_workers=asset_workers,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_exclusive(output, rendered, create_parent=False)
    published, _ = _read_stable_regular_file(
        output,
        "post-rollout verification receipt",
        expected_mode=0o444,
    )
    if published != rendered.encode("utf-8"):
        raise RuntimeError("post-rollout verification receipt changed during publication")
    return payload


def _validate_post_rollout_verification_receipt(
    *,
    receipt_path: Path,
    contract_path: Path,
    contract: dict[str, Any],
    asset_workers: int = cwam_robocasa_asset_manifest.DEFAULT_HASH_WORKERS,
) -> dict[str, Any]:
    contract_path = Path(os.path.abspath(contract_path.expanduser()))
    receipt_path = Path(os.path.abspath(receipt_path.expanduser()))
    if receipt_path.parent != contract_path.parent:
        raise ValueError("post-rollout verification receipt must share the eval contract canonical parent")
    receipt_value, receipt_descriptor = _load_stable_json(
        receipt_path,
        "post-rollout verification receipt",
        expected_mode=0o444,
    )
    receipt = _require_dict(receipt_value, "post_rollout_verification_receipt")
    _require_exact_keys(
        receipt,
        _POST_ROLLOUT_RECEIPT_KEYS,
        "post_rollout_verification_receipt",
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != _POST_ROLLOUT_VERIFICATION_KIND
        or receipt.get("status") != "passed"
        or receipt.get("scope") != _POST_ROLLOUT_SCOPE
    ):
        raise ValueError("post_rollout_verification_receipt has an unsupported schema/kind/status/scope")
    verified_at = _parse_utc(
        receipt.get("verified_at_utc"),
        "post_rollout_verification_receipt.verified_at_utc",
    )
    if verified_at < _parse_utc(contract["created_at"], "contract.created_at"):
        raise ValueError("post-rollout verification receipt predates the eval contract")
    if verified_at > dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=_MAX_FUTURE_SKEW_SECONDS):
        raise ValueError("post-rollout verification receipt is implausibly far in the future")

    expected = _collect_post_rollout_live_verification(
        contract_path=contract_path,
        contract=contract,
        asset_workers=asset_workers,
    )
    contract_descriptor = _require_dict(
        receipt.get("contract"),
        "post_rollout_verification_receipt.contract",
    )
    _require_exact_keys(
        contract_descriptor,
        _POST_CONTRACT_DESCRIPTOR_KEYS,
        "post_rollout_verification_receipt.contract",
    )
    if contract_descriptor != expected["contract"]:
        raise ValueError("post-rollout verification receipt does not bind the exact live eval contract")

    inputs = _require_dict(receipt.get("inputs"), "post_rollout_verification_receipt.inputs")
    _require_exact_keys(inputs, _POST_ROLLOUT_INPUT_KEYS, "post_rollout_verification_receipt.inputs")
    for name in sorted(_POST_ROLLOUT_INPUT_KEYS):
        _validate_post_file_descriptor(
            inputs.get(name),
            expected=expected["inputs"][name],
            label=f"post_rollout_verification_receipt.inputs.{name}",
        )
    if inputs != expected["inputs"]:
        raise ValueError("post-rollout verification receipt input descriptors differ from live inputs")

    live_verification = _require_dict(
        receipt.get("live_verification"),
        "post_rollout_verification_receipt.live_verification",
    )
    _require_exact_keys(
        live_verification,
        _POST_ROLLOUT_LIVE_KEYS,
        "post_rollout_verification_receipt.live_verification",
    )
    if live_verification != expected["live_verification"]:
        raise ValueError("post-rollout receipt conclusions differ from strict live re-verification")

    semantic_matches = _require_dict(
        receipt.get("semantic_matches"),
        "post_rollout_verification_receipt.semantic_matches",
    )
    _require_exact_keys(
        semantic_matches,
        _POST_ROLLOUT_SEMANTIC_MATCH_KEYS,
        "post_rollout_verification_receipt.semantic_matches",
    )
    if semantic_matches != expected["semantic_matches"] or any(
        value is not True for value in semantic_matches.values()
    ):
        raise ValueError("post-rollout checkpoint/asset semantic match conclusions are not all true")

    validation = _require_dict(
        receipt.get("validation"),
        "post_rollout_verification_receipt.validation",
    )
    _require_exact_keys(
        validation,
        _POST_ROLLOUT_VALIDATION_KEYS,
        "post_rollout_verification_receipt.validation",
    )
    if validation != expected["validation"] or any(value is not True for value in validation.values()):
        raise ValueError("post-rollout verification receipt validation is not an exact all-true verdict")
    receipt_elapsed = _validate_post_rollout_elapsed(
        receipt.get("elapsed_seconds"),
        "post_rollout_verification_receipt.elapsed_seconds",
    )
    summary_elapsed = _validate_post_rollout_elapsed(
        expected["elapsed_seconds"],
        "post_rollout_live_reverification.elapsed_seconds",
    )
    return {
        "scope": _POST_ROLLOUT_SCOPE,
        "receipt": {
            **receipt_descriptor,
            "kind": _POST_ROLLOUT_VERIFICATION_KIND,
            "schema_version": 1,
        },
        "receipt_verified_at_utc": receipt["verified_at_utc"],
        "receipt_elapsed_seconds": receipt_elapsed,
        "summary_reverification_elapsed_seconds": summary_elapsed,
        "semantic_matches": semantic_matches,
        "validation": validation,
        "live_verification": live_verification,
    }


def _validate_server_receipt(contract: dict[str, Any]) -> None:
    artifacts = contract["_verified_artifacts"]
    receipt_values = contract["_bound_json"]["server_runtime_receipts"]
    expected = {
        "checkpoint_manifest_sha256": artifacts["checkpoint_manifest"]["sha256"],
        "training_config_sha256": artifacts["training_config"]["sha256"],
        "action_stats_sha256": artifacts["action_stats"]["sha256"],
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "demian_commit": contract["demian"]["commit"],
        "sampler": contract["evaluation"]["policy_sampling"]["sampler"],
        "policy_seed": contract["evaluation"]["policy_sampling"]["seed"],
        "guidance": contract["evaluation"]["policy_sampling"]["guidance"],
        "num_steps": contract["evaluation"]["policy_sampling"]["num_steps"],
        "reuse_policy_seed_per_request": True,
        "use_ema_weights": True,
        "guardrails": False,
        "vision_tokenizer": _ONLINE_WAN2PT2_TARGET,
        "online_pixel_encoding": True,
        "model_action": "[dpos3,rot6d_columns6,gripper1]",
        "output_action": "[dpos3,axisangle3,gripper1]",
        "format_prompt_as_json": True,
        "source_module_sha256": artifacts["cosmos_server_module"]["sha256"],
    }
    seen_shards: set[int] = set()
    seen_nonces: set[str] = set()
    for expected_shard, receipt_value in enumerate(receipt_values):
        receipt = _require_dict(receipt_value, f"server_runtime_receipt[{expected_shard}]")
        if receipt.get("schema_version") != 1 or receipt.get("service") != "cwam_cosmos3_robocasa365":
            raise ValueError("server runtime receipt has an unsupported service/schema")
        if receipt.get("shard_index") != expected_shard:
            raise ValueError(
                f"server runtime receipt shard mismatch: expected={expected_shard}, actual={receipt.get('shard_index')}"
            )
        seen_shards.add(expected_shard)
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(
                    f"server runtime receipt {key} mismatch: expected={value!r}, actual={receipt.get(key)!r}"
                )
        source_module_path = (
            Path(
                _require_nonempty_str(
                    receipt.get("source_module_path"),
                    f"server_runtime_receipt[{expected_shard}].source_module_path",
                )
            )
            .expanduser()
            .resolve()
        )
        if source_module_path != Path(artifacts["cosmos_server_module"]["path"]).resolve():
            raise ValueError("server runtime receipt source_module_path does not match the eval contract")
        _require_int(
            receipt.get("server_pid"),
            f"server_runtime_receipt[{expected_shard}].server_pid",
            minimum=1,
        )
        launch_nonce = _require_sha256(
            receipt.get("launch_nonce"),
            f"server_runtime_receipt[{expected_shard}].launch_nonce",
        )
        if launch_nonce in seen_nonces:
            raise ValueError("server runtime receipt launch nonces must be unique across shards")
        seen_nonces.add(launch_nonce)
        if (
            Path(str(receipt.get("checkpoint", ""))).expanduser().resolve()
            != Path(contract["checkpoint_path"]).resolve()
        ):
            raise ValueError("server runtime receipt checkpoint path does not match the eval contract")
        if (
            Path(str(receipt.get("action_stats_path", ""))).expanduser().resolve()
            != Path(artifacts["action_stats"]["path"]).resolve()
        ):
            raise ValueError("server runtime receipt action_stats_path does not match the eval contract")
        if receipt.get("wire_inputs") != [
            "observation/agentview_left",
            "observation/eye_in_hand",
            "observation/agentview_right",
            "observation/state",
            "prompt",
        ]:
            raise ValueError("server runtime receipt wire_inputs drifted")
        if receipt.get("conditioning_views") != [
            "observation/agentview_left",
            "observation/eye_in_hand",
        ]:
            raise ValueError("server runtime receipt conditioning_views drifted")
        if receipt.get("state_conditioning") is not False:
            raise ValueError("server runtime receipt must state state_conditioning=false")
        if receipt.get("gripper_transform") != "identity":
            raise ValueError("server runtime receipt gripper_transform must be identity")
        if receipt.get("fps") != 20 or receipt.get("action_chunk_size") != 16:
            raise ValueError("server runtime receipt fps/action_chunk_size drifted")
    if seen_shards != set(range(contract["evaluation"]["num_shards"])):
        raise ValueError("server runtime receipt shard coverage is incomplete")


def _args_path(results_dir: Path, shard: int, num_shards: int) -> Path:
    return results_dir / ("args.json" if num_shards == 1 else f"args_shard{shard}.json")


def _validate_shard_args(
    args_json: dict[str, Any],
    *,
    contract: dict[str, Any],
    shard: int,
) -> list[str]:
    evaluation = contract["evaluation"]
    expected = {
        "task_shard": shard,
        "num_shards": evaluation["num_shards"],
        "episodes": evaluation["episodes_per_task"],
        "episodes_start": 0,
        "max_steps": evaluation["max_steps"],
        "open_loop_horizon": evaluation["open_loop_horizon"],
        "seed": evaluation["base_seed"],
        "checkpoint": contract["checkpoint_path"],
        "run_name": contract["run_name"],
        "results_dir": str(Path(contract["results_dir"]).parent),
        "eval_dataset": None,
        "headless": True,
        "save_video": False,
    }
    for key, value in expected.items():
        if key not in args_json or args_json[key] != value:
            raise ValueError(f"args shard {shard} {key} mismatch: expected={value!r}, actual={args_json.get(key)!r}")
    for forbidden in (
        "dynamic_instructions_file",
        "gt_reasoning_file",
        "vlm_captions_file",
        "fg_captions_path",
    ):
        if args_json.get(forbidden) not in (None, ""):
            raise ValueError(f"args shard {shard} enables forbidden non-causal input {forbidden}")

    requested_tasks = args_json.get("tasks")
    if requested_tasks not in ([], None):
        if not isinstance(requested_tasks, list) or not all(isinstance(task, str) for task in requested_tasks):
            raise TypeError(f"args shard {shard} tasks must be a string list")
        flattened_tasks = [
            task for value in requested_tasks for task in (part.strip() for part in value.split(",")) if task
        ]
        if flattened_tasks != evaluation["tasks"]:
            raise ValueError(f"args shard {shard} explicit task pool does not match contract")
    elif args_json.get("split") != evaluation["split"]:
        raise ValueError(f"args shard {shard} split does not match contract")
    task_pool = evaluation["tasks"]
    return [task for index, task in enumerate(task_pool) if index % evaluation["num_shards"] == shard]


def _validate_episode(
    episode: Any,
    *,
    task: str,
    episode_index: int,
    max_steps: int,
) -> dict[str, Any]:
    value = _require_dict(episode, f"{task}.episode_results[{episode_index}]")
    success = value.get("success")
    place_anytime = value.get("place_anytime")
    if type(success) is not bool or type(place_anytime) is not bool:
        raise TypeError(f"{task} episode {episode_index} success/place_anytime must be booleans")
    steps = _require_int(value.get("steps"), f"{task} episode {episode_index} steps", minimum=1)
    if steps > max_steps:
        raise ValueError(f"{task} episode {episode_index} steps={steps} exceeds max_steps={max_steps}")
    instruction = _require_nonempty_str(
        value.get("instruction"),
        f"{task} episode {episode_index} instruction",
    )
    action_log = value.get("action_log")
    if not isinstance(action_log, list) or len(action_log) != steps:
        raise ValueError(f"{task} episode {episode_index} action_log must contain exactly {steps} actions")
    for step_index, action in enumerate(action_log):
        array = list(action) if isinstance(action, (list, tuple)) else None
        if array is None or len(array) != 7:
            raise ValueError(f"{task} episode {episode_index} step {step_index} action must have width 7")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item))
            for item in array
        ):
            raise ValueError(f"{task} episode {episode_index} step {step_index} action is non-finite/non-numeric")
    return {
        "success": success,
        "place_anytime": place_anytime,
        "steps": steps,
        "instruction": instruction,
    }


def _collect_rows(
    contract: dict[str, Any],
    results_dir: Path,
    *,
    expected_raw_mode: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = contract["evaluation"]
    task_pool = evaluation["tasks"]
    artifacts = contract["_verified_artifacts"]
    episode_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    raw_shard_artifacts: list[dict[str, Any]] = []
    observed_tasks: set[str] = set()

    for shard in range(evaluation["num_shards"]):
        args_path = _args_path(results_dir, shard, evaluation["num_shards"])
        args_value, args_descriptor = _load_stable_json(
            args_path,
            f"args shard {shard}",
            expected_mode=expected_raw_mode,
        )
        args_json = _require_dict(args_value, f"args shard {shard}")
        expected_tasks = _validate_shard_args(args_json, contract=contract, shard=shard)
        results_path = results_dir / f"results_shard{shard}.json"
        results_value, results_descriptor = _load_stable_json(
            results_path,
            f"results shard {shard}",
            expected_mode=expected_raw_mode,
        )
        results = _require_dict(results_value, f"results shard {shard}")
        if set(results) != set(expected_tasks):
            raise ValueError(
                f"results shard {shard} task coverage mismatch: "
                f"missing={sorted(set(expected_tasks) - set(results))}, "
                f"unexpected={sorted(set(results) - set(expected_tasks))}"
            )
        raw_shard_artifacts.append(
            {
                "shard": shard,
                "args": {
                    "path": args_descriptor["path"],
                    "sha256": args_descriptor["sha256"],
                },
                "results": {
                    "path": results_descriptor["path"],
                    "sha256": results_descriptor["sha256"],
                },
            }
        )

        for within_shard_index, task in enumerate(expected_tasks):
            if task in observed_tasks:
                raise ValueError(f"Duplicate task across result shards: {task}")
            observed_tasks.add(task)
            result = _require_dict(results[task], f"results[{task}]")
            if "error" in result:
                raise RuntimeError(f"Task {task} contains an evaluation error: {result['error']}")
            if result.get("complete") is not True:
                raise RuntimeError(f"Task {task} is not complete")
            if result.get("split") != evaluation["split"]:
                raise ValueError(f"Task {task} split mismatch: {result.get('split')!r}")
            if result.get("episodes") != evaluation["episodes_per_task"]:
                raise ValueError(f"Task {task} episode count field mismatch")
            episode_results = result.get("episode_results")
            if not isinstance(episode_results, list) or len(episode_results) != evaluation["episodes_per_task"]:
                raise ValueError(f"Task {task} must contain exactly {evaluation['episodes_per_task']} episode results")

            validated_episodes = [
                _validate_episode(
                    episode,
                    task=task,
                    episode_index=episode_index,
                    max_steps=evaluation["max_steps"],
                )
                for episode_index, episode in enumerate(episode_results)
            ]
            successes = sum(int(episode["success"]) for episode in validated_episodes)
            place_anytime_count = sum(int(episode["place_anytime"]) for episode in validated_episodes)
            episodes = len(validated_episodes)
            success_rate = successes / episodes
            place_anytime_rate = place_anytime_count / episodes
            if result.get("successes") != successes:
                raise ValueError(f"Task {task} successes mismatch")
            if result.get("place_anytime_count") != place_anytime_count:
                raise ValueError(f"Task {task} place_anytime_count mismatch")
            _close_rate(result.get("success_rate"), success_rate, f"Task {task} success_rate")
            _close_rate(result.get("place_anytime_rate"), place_anytime_rate, f"Task {task} place_anytime_rate")

            task_index = task_pool.index(task)
            common = {
                "experiment_id": contract["experiment_id"],
                "logical_run_id": contract["logical_run_id"],
                "evaluation_role": evaluation["role"],
                "split": evaluation["split"],
                "task": task,
                "task_index": task_index,
                "shard": shard,
                "checkpoint_manifest_sha256": artifacts["checkpoint_manifest"]["sha256"],
                "training_config_sha256": artifacts["training_config"]["sha256"],
                "action_stats_sha256": artifacts["action_stats"]["sha256"],
                "wan2pt2_vae_sha256": artifacts["wan2pt2_vae"]["sha256"],
                "demian_commit": contract["demian"]["commit"],
            }
            for episode_index, episode in enumerate(validated_episodes):
                seed = evaluation["base_seed"] + within_shard_index * 10000 + episode_index
                episode_rows.append(
                    {
                        **common,
                        "episode_index": episode_index,
                        "seed": seed,
                        "success": str(episode["success"]).lower(),
                        "place_anytime": str(episode["place_anytime"]).lower(),
                        "steps": episode["steps"],
                        "instruction": episode["instruction"],
                    }
                )
            task_rows.append(
                {
                    **common,
                    "episodes": episodes,
                    "successes": successes,
                    "success_rate": f"{success_rate:.12f}",
                    "place_anytime_count": place_anytime_count,
                    "place_anytime_rate": f"{place_anytime_rate:.12f}",
                    "execution_complete": "true",
                }
            )

    if observed_tasks != set(task_pool):
        raise ValueError(
            f"Global task coverage mismatch: missing={sorted(set(task_pool) - observed_tasks)}, "
            f"unexpected={sorted(observed_tasks - set(task_pool))}"
        )
    episode_rows.sort(key=lambda row: (int(row["task_index"]), int(row["episode_index"])))
    task_rows.sort(key=lambda row: int(row["task_index"]))
    return episode_rows, task_rows, raw_shard_artifacts


def _build_evidence_payload(
    *,
    contract_path: Path,
    contract: dict[str, Any],
    post_rollout_binding: dict[str, Any],
    results_dir: Path,
    summary_csv_descriptor: dict[str, Any],
    task_success_rates_csv_descriptor: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    raw_shard_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    task_success_rates = [float(row["success_rate"]) for row in task_rows]
    successful_tasks = sum(rate > 0.0 for rate in task_success_rates)
    mean_success_rate = sum(task_success_rates) / len(task_success_rates)
    thresholds = contract["evaluation"]["effect_thresholds"]
    effect_pass = mean_success_rate >= float(thresholds["minimum_mean_success_rate"]) and successful_tasks >= int(
        thresholds["minimum_successful_tasks"]
    )
    return {
        "schema_version": 1,
        "kind": _EVIDENCE_KIND,
        "experiment_id": contract["experiment_id"],
        "logical_run_id": contract["logical_run_id"],
        "source_commit": contract["source_commit"],
        "demian": contract["demian"],
        "simulator": contract["simulator"],
        "contract_path": str(contract_path),
        "contract_sha256": contract["_contract_sha256"],
        "results_dir": str(results_dir),
        "prior_gate_evidence": contract["_prior_gate_binding"],
        "external_execution_binding": {
            **contract["_runtime_bindings"],
            "post_rollout_live_inputs": post_rollout_binding,
        },
        "evaluation": {
            "stage": contract["evaluation"]["stage"],
            "evaluation_class": contract["_runtime_bindings"]["evaluation_intent"]["evaluation_class"],
            "source_git_pushed_assertion": contract["_runtime_bindings"]["evaluation_intent"]["source_git_pushed"],
            "role": contract["evaluation"]["role"],
            "split": contract["evaluation"]["split"],
            "tasks": len(task_rows),
            "episodes": len(episode_rows),
            "episodes_per_task": contract["evaluation"]["episodes_per_task"],
            "mean_success_rate": mean_success_rate,
            "successful_tasks": successful_tasks,
            "effect_thresholds": thresholds,
        },
        "execution_complete": True,
        "effect_pass": effect_pass,
        "artifacts": {
            "summary_csv": {
                "path": summary_csv_descriptor["path"],
                "sha256": summary_csv_descriptor["sha256"],
            },
            "task_success_rates_csv": {
                "path": task_success_rates_csv_descriptor["path"],
                "sha256": task_success_rates_csv_descriptor["sha256"],
            },
            "raw_shards": raw_shard_artifacts,
            **contract["_verified_artifacts"],
        },
        "seed_provenance": _SEED_PROVENANCE,
    }


def summarize(
    *,
    contract_path: Path,
    post_rollout_verification_receipt: Path,
    results_dir: Path,
    summary_csv: Path,
    task_success_rates_csv: Path,
    evidence_json: Path,
    require_effect_pass: bool,
    asset_workers: int = cwam_robocasa_asset_manifest.DEFAULT_HASH_WORKERS,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve()
    results_dir = results_dir.expanduser().resolve()
    contract = _validate_contract(contract_path)
    _validate_server_receipt(contract)
    post_rollout_binding = _validate_post_rollout_verification_receipt(
        receipt_path=post_rollout_verification_receipt,
        contract_path=contract_path,
        contract=contract,
        asset_workers=asset_workers,
    )
    expected_results_dir = Path(contract["results_dir"]).expanduser().resolve()
    if results_dir != expected_results_dir:
        raise ValueError(f"results_dir mismatch: contract={expected_results_dir}, requested={results_dir}")

    summary_csv = _exclusive_artifact_path(summary_csv, "summary CSV")
    task_success_rates_csv = _exclusive_artifact_path(task_success_rates_csv, "task success rates CSV")
    evidence_json = _exclusive_artifact_path(evidence_json, "evaluation evidence")
    episode_rows, task_rows, raw_shard_artifacts = _collect_rows(contract, results_dir)
    _atomic_write_csv(summary_csv, _CSV_EPISODE_FIELDS, episode_rows)
    _atomic_write_csv(task_success_rates_csv, _CSV_TASK_FIELDS, task_rows)
    summary_bytes, summary_descriptor = _read_stable_regular_file(
        summary_csv,
        "summary CSV",
        expected_mode=0o444,
    )
    task_bytes, task_descriptor = _read_stable_regular_file(
        task_success_rates_csv,
        "task success rates CSV",
        expected_mode=0o444,
    )
    if summary_bytes != _render_csv(_CSV_EPISODE_FIELDS, episode_rows).encode("utf-8"):
        raise RuntimeError("summary CSV differs immediately after exclusive publication")
    if task_bytes != _render_csv(_CSV_TASK_FIELDS, task_rows).encode("utf-8"):
        raise RuntimeError("task success rates CSV differs immediately after exclusive publication")
    evidence = _build_evidence_payload(
        contract_path=contract_path,
        contract=contract,
        post_rollout_binding=post_rollout_binding,
        results_dir=results_dir,
        summary_csv_descriptor=summary_descriptor,
        task_success_rates_csv_descriptor=task_descriptor,
        episode_rows=episode_rows,
        task_rows=task_rows,
        raw_shard_artifacts=raw_shard_artifacts,
    )
    evidence_text = json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_exclusive(
        evidence_json,
        evidence_text,
        create_parent=False,
    )
    evidence_bytes, evidence_descriptor = _read_stable_regular_file(
        evidence_json,
        "evaluation evidence",
        expected_mode=0o444,
    )
    if evidence_bytes != evidence_text.encode("utf-8"):
        raise RuntimeError("evaluation evidence differs immediately after exclusive publication")
    evidence["artifacts"]["evaluation_evidence"] = {
        "path": evidence_descriptor["path"],
        "sha256": evidence_descriptor["sha256"],
    }
    if require_effect_pass and not evidence["effect_pass"]:
        raise RuntimeError(
            "Formal evaluation execution is complete, but the preregistered effect threshold failed: "
            f"mean_success_rate={evidence['evaluation']['mean_success_rate']:.6f}, "
            f"successful_tasks={evidence['evaluation']['successful_tasks']}"
        )
    return evidence


def _validate_sealed_post_rollout_binding(
    *,
    binding_value: Any,
    receipt_path: Path,
    contract_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    binding = _require_dict(binding_value, "evaluation_evidence.post_rollout_live_inputs")
    _require_exact_keys(binding, _POST_ROLLOUT_BINDING_KEYS, "evaluation_evidence.post_rollout_live_inputs")
    if binding.get("scope") != _POST_ROLLOUT_SCOPE:
        raise ValueError("sealed evidence post-rollout scope mismatch")

    receipt, live_receipt_descriptor = _load_stable_json(
        receipt_path,
        "sealed post-rollout verification receipt",
        expected_mode=0o444,
    )
    receipt = _require_dict(receipt, "sealed post-rollout verification receipt")
    _require_exact_keys(receipt, _POST_ROLLOUT_RECEIPT_KEYS, "sealed post-rollout verification receipt")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != _POST_ROLLOUT_VERIFICATION_KIND
        or receipt.get("status") != "passed"
        or receipt.get("scope") != _POST_ROLLOUT_SCOPE
    ):
        raise ValueError("sealed post-rollout verification receipt has an unsupported terminal verdict")

    live_contract_descriptor = dict(contract["_contract_descriptor"])
    live_contract_descriptor.update({"kind": _CONTRACT_KIND, "schema_version": 1})
    if receipt.get("contract") != live_contract_descriptor:
        raise ValueError("sealed post-rollout verification receipt contract binding mismatch")
    if receipt.get("inputs") != _post_rollout_input_descriptors(contract):
        raise ValueError("sealed post-rollout verification receipt input descriptors changed")

    semantic_matches = _require_dict(receipt.get("semantic_matches"), "sealed post-rollout semantic matches")
    _require_exact_keys(semantic_matches, _POST_ROLLOUT_SEMANTIC_MATCH_KEYS, "sealed post-rollout semantic matches")
    validation = _require_dict(receipt.get("validation"), "sealed post-rollout validation")
    _require_exact_keys(validation, _POST_ROLLOUT_VALIDATION_KEYS, "sealed post-rollout validation")
    if any(value is not True for value in semantic_matches.values()) or any(
        value is not True for value in validation.values()
    ):
        raise ValueError("sealed post-rollout verification receipt is not an all-true verdict")
    live_verification = _require_dict(receipt.get("live_verification"), "sealed post-rollout live verification")
    _require_exact_keys(live_verification, _POST_ROLLOUT_LIVE_KEYS, "sealed post-rollout live verification")
    receipt_elapsed = _validate_post_rollout_elapsed(receipt.get("elapsed_seconds"), "sealed receipt elapsed_seconds")
    summary_elapsed = _validate_post_rollout_elapsed(
        binding.get("summary_reverification_elapsed_seconds"),
        "sealed evidence summary_reverification_elapsed_seconds",
    )
    receipt_verified_at = _require_nonempty_str(receipt.get("verified_at_utc"), "sealed receipt verified_at_utc")
    _parse_utc(receipt_verified_at, "sealed receipt verified_at_utc")

    bound_receipt = {
        **live_receipt_descriptor,
        "kind": _POST_ROLLOUT_VERIFICATION_KIND,
        "schema_version": 1,
    }
    expected_binding = {
        "scope": _POST_ROLLOUT_SCOPE,
        "receipt": bound_receipt,
        "receipt_verified_at_utc": receipt_verified_at,
        "receipt_elapsed_seconds": receipt_elapsed,
        "summary_reverification_elapsed_seconds": summary_elapsed,
        "semantic_matches": semantic_matches,
        "validation": validation,
        "live_verification": live_verification,
    }
    if binding != expected_binding:
        raise ValueError("sealed evidence post-rollout binding differs from its immutable receipt")
    return binding


def verify_sealed_evidence(
    *,
    contract_path: Path,
    post_rollout_verification_receipt: Path,
    results_dir: Path,
    summary_csv: Path,
    task_success_rates_csv: Path,
    evidence_json: Path,
    output: Path,
) -> dict[str, Any]:
    contract_path = Path(os.path.abspath(contract_path.expanduser()))
    post_rollout_verification_receipt = Path(os.path.abspath(post_rollout_verification_receipt.expanduser()))
    results_dir = _canonical_existing_directory(results_dir, "sealed results directory")
    results_metadata = results_dir.lstat()
    if stat.S_IMODE(results_metadata.st_mode) != 0o555:
        raise ValueError(
            "sealed results directory mode mismatch: "
            f"expected={oct(0o555)}, actual={oct(stat.S_IMODE(results_metadata.st_mode))}"
        )

    expected_paths = {
        "summary CSV": results_dir / "summary.csv",
        "task success rates CSV": results_dir / "task_success_rates.csv",
        "evaluation evidence": results_dir / "evaluation_evidence.json",
    }
    supplied_paths = {
        "summary CSV": Path(os.path.abspath(summary_csv.expanduser())),
        "task success rates CSV": Path(os.path.abspath(task_success_rates_csv.expanduser())),
        "evaluation evidence": Path(os.path.abspath(evidence_json.expanduser())),
    }
    for label, expected_path in expected_paths.items():
        if supplied_paths[label] != expected_path:
            raise ValueError(f"{label} must be the canonical sealed results artifact {expected_path}")
    output = _exclusive_artifact_path(output, "sealed evidence verification receipt")

    contract = _validate_contract(contract_path)
    _validate_server_receipt(contract)
    if results_dir != Path(contract["results_dir"]):
        raise ValueError("sealed results directory differs from the eval contract")
    contract_descriptor = dict(contract["_contract_descriptor"])

    evidence_value, evidence_descriptor = _load_stable_json(
        supplied_paths["evaluation evidence"],
        "sealed evaluation evidence",
        expected_mode=0o444,
    )
    evidence_value = _require_dict(evidence_value, "sealed evaluation evidence")
    _require_exact_keys(evidence_value, _PRIOR_GATE_EVIDENCE_KEYS, "sealed evaluation evidence")
    external_binding = _require_dict(
        evidence_value.get("external_execution_binding"),
        "sealed evaluation evidence external_execution_binding",
    )
    post_rollout_binding = _validate_sealed_post_rollout_binding(
        binding_value=external_binding.get("post_rollout_live_inputs"),
        receipt_path=post_rollout_verification_receipt,
        contract_path=contract_path,
        contract=contract,
    )

    episode_rows, task_rows, raw_shard_artifacts = _collect_rows(
        contract,
        results_dir,
        expected_raw_mode=0o444,
    )
    summary_bytes, summary_descriptor = _read_stable_regular_file(
        supplied_paths["summary CSV"],
        "sealed summary CSV",
        expected_mode=0o444,
    )
    task_bytes, task_descriptor = _read_stable_regular_file(
        supplied_paths["task success rates CSV"],
        "sealed task success rates CSV",
        expected_mode=0o444,
    )
    if summary_bytes != _render_csv(_CSV_EPISODE_FIELDS, episode_rows).encode("utf-8"):
        raise ValueError("sealed summary CSV content differs from raw shard reaggregation")
    if task_bytes != _render_csv(_CSV_TASK_FIELDS, task_rows).encode("utf-8"):
        raise ValueError("sealed task success rates CSV content differs from raw shard reaggregation")

    expected_evidence = _build_evidence_payload(
        contract_path=contract_path,
        contract=contract,
        post_rollout_binding=post_rollout_binding,
        results_dir=results_dir,
        summary_csv_descriptor=summary_descriptor,
        task_success_rates_csv_descriptor=task_descriptor,
        episode_rows=episode_rows,
        task_rows=task_rows,
        raw_shard_artifacts=raw_shard_artifacts,
    )
    if evidence_value != expected_evidence:
        raise ValueError("sealed evaluation evidence content or artifact hashes differ from live reaggregation")
    if _stable_file_identity(results_metadata) != _stable_file_identity(results_dir.lstat()):
        raise RuntimeError("sealed results directory changed during final verification")

    receipt = {
        "schema_version": 1,
        "kind": _SEALED_EVIDENCE_VERIFICATION_KIND,
        "status": "passed",
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "receipt_path": str(output),
        "contract": contract_descriptor,
        "post_rollout_verification_receipt": post_rollout_binding["receipt"],
        "results_dir": {
            "path": str(results_dir),
            "mode": "0555",
        },
        "artifacts": {
            "summary_csv": summary_descriptor,
            "task_success_rates_csv": task_descriptor,
            "evaluation_evidence": evidence_descriptor,
            "raw_shards": raw_shard_artifacts,
        },
        "evaluation": {
            **expected_evidence["evaluation"],
            "execution_complete": expected_evidence["execution_complete"],
            "effect_pass": expected_evidence["effect_pass"],
        },
        "validation": {
            "results_directory_mode_0555": True,
            "raw_files_mode_0444_and_stably_reaggregated": True,
            "summary_csv_mode_0444_content_and_hash_match": True,
            "task_success_rates_csv_mode_0444_content_and_hash_match": True,
            "evaluation_evidence_mode_0444_content_and_hash_match": True,
            "post_rollout_receipt_binding_match": True,
        },
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_exclusive(output, rendered, create_parent=False)
    published, _ = _read_stable_regular_file(
        output,
        "sealed evidence verification receipt",
        expected_mode=0o444,
    )
    if published != rendered.encode("utf-8"):
        raise RuntimeError("sealed evidence verification receipt changed during publication")
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract_parser = subparsers.add_parser("create-contract")
    contract_parser.add_argument("--output", type=Path, required=True)
    contract_parser.add_argument("--experiment-id", required=True)
    contract_parser.add_argument("--logical-run-id", required=True)
    contract_parser.add_argument("--source-commit", required=True)
    contract_parser.add_argument("--source-root", type=Path, required=True)
    contract_parser.add_argument("--source-manifest", type=Path, required=True)
    contract_parser.add_argument("--cosmos-server-module", type=Path, required=True)
    contract_parser.add_argument("--checkpoint-path", type=Path, required=True)
    contract_parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    contract_parser.add_argument("--checkpoint-verification-receipt", type=Path, required=True)
    contract_parser.add_argument("--source-dcp-export-receipt", type=Path, required=True)
    contract_parser.add_argument("--training-config", type=Path, required=True)
    contract_parser.add_argument("--action-stats", type=Path, required=True)
    contract_parser.add_argument("--wan2pt2-vae", type=Path, required=True)
    contract_parser.add_argument("--expected-wan2pt2-vae-sha256", required=True)
    contract_parser.add_argument("--cosmos-environment-manifest", type=Path, required=True)
    contract_parser.add_argument("--robocasa-environment-manifest", type=Path, required=True)
    contract_parser.add_argument("--eval-intent-manifest", type=Path, required=True)
    contract_parser.add_argument("--submission-verification-receipt", type=Path, required=True)
    contract_parser.add_argument("--asset-verification-receipt", type=Path, required=True)
    contract_parser.add_argument("--eval-runtime-receipt", type=Path, required=True)
    contract_parser.add_argument("--server-runtime-receipt", type=Path, action="append", required=True)
    contract_parser.add_argument("--demian-repo", type=Path, required=True)
    contract_parser.add_argument("--demian-commit", required=True)
    contract_parser.add_argument("--demian-evaluator", type=Path, required=True)
    contract_parser.add_argument("--demian-task-registry", type=Path, required=True)
    contract_parser.add_argument("--robocasa-repo", type=Path, required=True)
    contract_parser.add_argument("--robocasa-commit", required=True)
    contract_parser.add_argument("--robosuite-repo", type=Path, required=True)
    contract_parser.add_argument("--robosuite-commit", required=True)
    contract_parser.add_argument("--stage", choices=("one_episode_gate", "formal"), required=True)
    contract_parser.add_argument("--split", choices=("atomic", "composite"), required=True)
    contract_parser.add_argument("--tasks", nargs="*")
    contract_parser.add_argument("--episodes-per-task", type=int, required=True)
    contract_parser.add_argument("--base-seed", type=int, required=True)
    contract_parser.add_argument("--max-steps", type=int, default=400)
    contract_parser.add_argument("--open-loop-horizon", type=int, default=12)
    contract_parser.add_argument("--num-shards", type=int, required=True)
    contract_parser.add_argument("--policy-sampler", choices=("unipc", "edm"), default="unipc")
    contract_parser.add_argument("--policy-seed", type=int, required=True)
    contract_parser.add_argument("--policy-guidance", type=float, required=True)
    contract_parser.add_argument("--policy-num-steps", type=int, required=True)
    contract_parser.add_argument("--results-dir", type=Path, required=True)
    contract_parser.add_argument("--run-name", required=True)
    contract_parser.add_argument("--minimum-mean-success-rate", type=float, required=True)
    contract_parser.add_argument("--minimum-successful-tasks", type=int, required=True)
    contract_parser.add_argument("--prior-gate-evidence", type=Path)
    contract_parser.add_argument("--expected-prior-gate-evidence-sha256")
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--contract", type=Path, required=True)
    summarize_parser.add_argument(
        "--post-rollout-verification-receipt",
        type=Path,
        required=True,
    )
    summarize_parser.add_argument("--results-dir", type=Path, required=True)
    summarize_parser.add_argument("--summary-csv", type=Path, required=True)
    summarize_parser.add_argument("--task-success-rates-csv", type=Path, required=True)
    summarize_parser.add_argument("--evidence-json", type=Path, required=True)
    summarize_parser.add_argument("--require-effect-pass", action="store_true")
    summarize_parser.add_argument(
        "--asset-workers",
        type=int,
        default=cwam_robocasa_asset_manifest.DEFAULT_HASH_WORKERS,
    )
    sealed_parser = subparsers.add_parser("verify-sealed-evidence")
    sealed_parser.add_argument("--contract", type=Path, required=True)
    sealed_parser.add_argument(
        "--post-rollout-verification-receipt",
        type=Path,
        required=True,
    )
    sealed_parser.add_argument("--results-dir", type=Path, required=True)
    sealed_parser.add_argument("--summary-csv", type=Path, required=True)
    sealed_parser.add_argument("--task-success-rates-csv", type=Path, required=True)
    sealed_parser.add_argument("--evidence-json", type=Path, required=True)
    sealed_parser.add_argument("--output", type=Path, required=True)
    post_rollout_parser = subparsers.add_parser("verify-post-rollout")
    post_rollout_parser.add_argument("--contract", type=Path, required=True)
    post_rollout_parser.add_argument("--output", type=Path, required=True)
    post_rollout_parser.add_argument(
        "--asset-workers",
        type=int,
        default=cwam_robocasa_asset_manifest.DEFAULT_HASH_WORKERS,
    )
    prior_gate_parser = subparsers.add_parser("verify-prior-gate")
    prior_gate_parser.add_argument("--gate-evidence", type=Path, required=True)
    prior_gate_parser.add_argument("--expected-gate-evidence-sha256", required=True)
    prior_gate_parser.add_argument("--current-source-root", type=Path, required=True)
    prior_gate_parser.add_argument("--current-source-commit", required=True)
    prior_gate_parser.add_argument("--current-source-manifest-sha256", required=True)
    prior_gate_parser.add_argument("--current-checkpoint-path", type=Path, required=True)
    prior_gate_parser.add_argument("--current-checkpoint-manifest-sha256", required=True)
    prior_gate_parser.add_argument("--current-source-dcp-export-receipt-sha256", required=True)
    prior_gate_parser.add_argument("--current-asset-manifest-sha256", required=True)
    prior_gate_parser.add_argument("--current-asset-publisher-receipt-sha256", required=True)
    prior_gate_parser.add_argument("--current-training-config-sha256", required=True)
    prior_gate_parser.add_argument("--current-action-stats-sha256", required=True)
    prior_gate_parser.add_argument("--current-wan2pt2-vae-path", type=Path, required=True)
    prior_gate_parser.add_argument("--current-wan2pt2-vae-sha256", required=True)
    prior_gate_parser.add_argument("--current-cosmos-environment-manifest-sha256", required=True)
    prior_gate_parser.add_argument("--current-robocasa-environment-manifest-sha256", required=True)
    prior_gate_parser.add_argument("--current-demian-commit", required=True)
    prior_gate_parser.add_argument("--current-robocasa-commit", required=True)
    prior_gate_parser.add_argument("--current-robosuite-commit", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "create-contract":
        contract = create_contract(
            output=args.output,
            experiment_id=args.experiment_id,
            logical_run_id=args.logical_run_id,
            source_commit=args.source_commit,
            source_root=args.source_root,
            source_manifest=args.source_manifest,
            cosmos_server_module=args.cosmos_server_module,
            checkpoint_path=args.checkpoint_path,
            checkpoint_manifest=args.checkpoint_manifest,
            checkpoint_verification_receipt=args.checkpoint_verification_receipt,
            source_dcp_export_receipt=args.source_dcp_export_receipt,
            training_config=args.training_config,
            action_stats=args.action_stats,
            wan2pt2_vae=args.wan2pt2_vae,
            expected_wan2pt2_vae_sha256=args.expected_wan2pt2_vae_sha256,
            cosmos_environment_manifest=args.cosmos_environment_manifest,
            robocasa_environment_manifest=args.robocasa_environment_manifest,
            eval_intent_manifest=args.eval_intent_manifest,
            submission_verification_receipt=args.submission_verification_receipt,
            asset_verification_receipt=args.asset_verification_receipt,
            eval_runtime_receipt=args.eval_runtime_receipt,
            server_runtime_receipts=args.server_runtime_receipt,
            demian_repo=args.demian_repo,
            demian_commit=args.demian_commit,
            demian_evaluator=args.demian_evaluator,
            demian_task_registry=args.demian_task_registry,
            robocasa_repo=args.robocasa_repo,
            robocasa_commit=args.robocasa_commit,
            robosuite_repo=args.robosuite_repo,
            robosuite_commit=args.robosuite_commit,
            stage=args.stage,
            split=args.split,
            tasks=args.tasks,
            episodes_per_task=args.episodes_per_task,
            base_seed=args.base_seed,
            max_steps=args.max_steps,
            open_loop_horizon=args.open_loop_horizon,
            num_shards=args.num_shards,
            policy_sampler=args.policy_sampler,
            policy_seed=args.policy_seed,
            policy_guidance=args.policy_guidance,
            policy_num_steps=args.policy_num_steps,
            results_dir=args.results_dir,
            run_name=args.run_name,
            minimum_mean_success_rate=args.minimum_mean_success_rate,
            minimum_successful_tasks=args.minimum_successful_tasks,
            prior_gate_evidence=args.prior_gate_evidence,
            expected_prior_gate_evidence_sha256=args.expected_prior_gate_evidence_sha256,
        )
        print(json.dumps(contract, indent=2, sort_keys=True))
        return
    if args.command == "summarize":
        evidence = summarize(
            contract_path=args.contract,
            post_rollout_verification_receipt=args.post_rollout_verification_receipt,
            results_dir=args.results_dir,
            summary_csv=args.summary_csv,
            task_success_rates_csv=args.task_success_rates_csv,
            evidence_json=args.evidence_json,
            require_effect_pass=args.require_effect_pass,
            asset_workers=args.asset_workers,
        )
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return
    if args.command == "verify-sealed-evidence":
        receipt = verify_sealed_evidence(
            contract_path=args.contract,
            post_rollout_verification_receipt=args.post_rollout_verification_receipt,
            results_dir=args.results_dir,
            summary_csv=args.summary_csv,
            task_success_rates_csv=args.task_success_rates_csv,
            evidence_json=args.evidence_json,
            output=args.output,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    if args.command == "verify-post-rollout":
        receipt = verify_post_rollout(
            contract_path=args.contract,
            output=args.output,
            asset_workers=args.asset_workers,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    if args.command == "verify-prior-gate":
        verification = verify_prior_gate(
            gate_evidence_path=args.gate_evidence,
            expected_gate_evidence_sha256=args.expected_gate_evidence_sha256,
            current_source_root=args.current_source_root,
            current_source_commit=args.current_source_commit,
            current_source_manifest_sha256=args.current_source_manifest_sha256,
            current_checkpoint_path=args.current_checkpoint_path,
            current_checkpoint_manifest_sha256=args.current_checkpoint_manifest_sha256,
            current_source_dcp_export_receipt_sha256=args.current_source_dcp_export_receipt_sha256,
            current_asset_manifest_sha256=args.current_asset_manifest_sha256,
            current_asset_publisher_receipt_sha256=args.current_asset_publisher_receipt_sha256,
            current_training_config_sha256=args.current_training_config_sha256,
            current_action_stats_sha256=args.current_action_stats_sha256,
            current_wan2pt2_vae_path=args.current_wan2pt2_vae_path,
            current_wan2pt2_vae_sha256=args.current_wan2pt2_vae_sha256,
            current_cosmos_environment_manifest_sha256=args.current_cosmos_environment_manifest_sha256,
            current_robocasa_environment_manifest_sha256=args.current_robocasa_environment_manifest_sha256,
            current_demian_commit=args.current_demian_commit,
            current_robocasa_commit=args.current_robocasa_commit,
            current_robosuite_commit=args.current_robosuite_commit,
        )
        print(json.dumps(verification, indent=2, sort_keys=True))
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
