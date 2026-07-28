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
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

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
_CONTRACT_KIND = "cwam_robocasa365_formal_eval_contract"
_EVIDENCE_KIND = "cwam_robocasa365_formal_eval_evidence"
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
    "demian_commit",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required evaluation artifact not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    return value


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


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


def _atomic_write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _verify_file_artifact(artifacts: dict[str, Any], name: str) -> dict[str, str]:
    descriptor = _require_dict(artifacts.get(name), f"contract.artifacts.{name}")
    path = Path(_require_nonempty_str(descriptor.get("path"), f"contract.artifacts.{name}.path"))
    if not path.is_absolute():
        raise ValueError(f"contract.artifacts.{name}.path must be absolute: {path}")
    expected = _require_sha256(
        descriptor.get("sha256"),
        f"contract.artifacts.{name}.sha256",
    )
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA256 mismatch: expected={expected}, actual={actual}, path={path}")
    return {"path": str(path), "sha256": actual}


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
    source_manifest: Path,
    cosmos_server_module: Path,
    checkpoint_path: Path,
    checkpoint_manifest: Path,
    checkpoint_verification_receipt: Path,
    training_config: Path,
    action_stats: Path,
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
) -> dict[str, Any]:
    """Create the post-server-start immutable eval contract."""
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing eval contract: {output}")
    source_manifest = source_manifest.expanduser().resolve()
    cosmos_server_module = cosmos_server_module.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint_manifest = checkpoint_manifest.expanduser().resolve()
    checkpoint_verification_receipt = checkpoint_verification_receipt.expanduser().resolve()
    training_config = training_config.expanduser().resolve()
    action_stats = action_stats.expanduser().resolve()
    server_runtime_receipts = [path.expanduser().resolve() for path in server_runtime_receipts]
    demian_repo = demian_repo.expanduser().resolve()
    demian_evaluator = demian_evaluator.expanduser().resolve()
    demian_task_registry = demian_task_registry.expanduser().resolve()
    robocasa_repo = robocasa_repo.expanduser().resolve()
    robosuite_repo = robosuite_repo.expanduser().resolve()
    results_dir = results_dir.expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
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
    contract = {
        "schema_version": 1,
        "kind": _CONTRACT_KIND,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "logical_run_id": logical_run_id,
        "source_commit": source_commit,
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
            "checkpoint_manifest": artifact(checkpoint_manifest),
            "checkpoint_verification_receipt": artifact(checkpoint_verification_receipt),
            "training_config": artifact(training_config),
            "action_stats": artifact(action_stats),
            "demian_evaluator": evaluator_artifact,
            "demian_task_registry": task_registry_artifact,
            "server_runtime_receipts": [artifact(path) for path in server_runtime_receipts],
        },
    }
    _atomic_write(output, json.dumps(contract, indent=2, sort_keys=True) + "\n")
    try:
        validated = _validate_contract(output)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {key: value for key, value in validated.items() if not key.startswith("_")}


def _validate_contract(contract_path: Path) -> dict[str, Any]:
    contract = _require_dict(_load_json(contract_path), "contract")
    if contract.get("schema_version") != 1 or contract.get("kind") != _CONTRACT_KIND:
        raise ValueError(
            f"Unsupported eval contract schema/kind: {contract.get('schema_version')!r}/{contract.get('kind')!r}"
        )
    _require_nonempty_str(contract.get("experiment_id"), "contract.experiment_id")
    _require_nonempty_str(contract.get("logical_run_id"), "contract.logical_run_id")
    _require_commit(contract.get("source_commit"), "contract.source_commit")
    _require_sha256(contract.get("source_manifest_sha256"), "contract.source_manifest_sha256")
    checkpoint_path = Path(_require_nonempty_str(contract.get("checkpoint_path"), "contract.checkpoint_path"))
    results_dir = Path(_require_nonempty_str(contract.get("results_dir"), "contract.results_dir"))
    run_name = _require_nonempty_str(contract.get("run_name"), "contract.run_name")
    if not checkpoint_path.is_absolute() or not results_dir.is_absolute():
        raise ValueError("contract checkpoint_path and results_dir must be absolute")
    if results_dir.name != run_name:
        raise ValueError(
            f"contract results_dir must end in run_name: results_dir={results_dir}, run_name={run_name!r}"
        )

    demian = _require_dict(contract.get("demian"), "contract.demian")
    _require_commit(demian.get("commit"), "contract.demian.commit")
    _require_sha256(demian.get("evaluator_sha256"), "contract.demian.evaluator_sha256")
    _require_sha256(demian.get("task_registry_sha256"), "contract.demian.task_registry_sha256")
    demian_repo = Path(_require_nonempty_str(demian.get("repo"), "contract.demian.repo"))
    if not demian_repo.is_absolute() or not demian_repo.is_dir():
        raise ValueError(f"contract.demian.repo must be an existing absolute directory: {demian_repo}")

    simulator = _require_dict(contract.get("simulator"), "contract.simulator")
    if simulator.get("split") != "target":
        raise ValueError("contract.simulator.split must be target")
    _require_commit(simulator.get("robocasa_commit"), "contract.simulator.robocasa_commit")
    _require_commit(simulator.get("robosuite_commit"), "contract.simulator.robosuite_commit")
    for key in ("robocasa_repo", "robosuite_repo"):
        repo = Path(_require_nonempty_str(simulator.get(key), f"contract.simulator.{key}"))
        if not repo.is_absolute() or not repo.is_dir():
            raise ValueError(f"contract.simulator.{key} must be an existing absolute directory: {repo}")

    evaluation = _require_dict(contract.get("evaluation"), "contract.evaluation")
    stage = evaluation.get("stage")
    if stage not in {"one_episode_gate", "formal"}:
        raise ValueError("contract.evaluation.stage must be one_episode_gate or formal")
    split = evaluation.get("split")
    if split not in {"atomic", "composite"}:
        raise ValueError("contract.evaluation.split must be atomic or composite")
    expected_role = "primary" if split == "atomic" else "zero_shot_secondary"
    if evaluation.get("role") != expected_role:
        raise ValueError(
            f"contract.evaluation.role must be {expected_role!r} for split={split!r}"
        )
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
    elif episodes != 1:
        raise ValueError("one_episode_gate requires exactly one episode per task")

    _require_int(evaluation.get("base_seed"), "base_seed", minimum=0)
    _require_int(evaluation.get("max_steps"), "max_steps", minimum=1)
    _require_int(evaluation.get("open_loop_horizon"), "open_loop_horizon", minimum=1)
    num_shards = _require_int(evaluation.get("num_shards"), "num_shards", minimum=1)
    if num_shards > len(task_pool):
        raise ValueError("num_shards cannot exceed the number of tasks")
    if stage == "formal" and num_shards != 8:
        raise ValueError("formal evaluation requires exactly eight shards")
    if evaluation.get("seed_rule") != "base_seed + within_shard_task_index * 10000 + episode_index":
        raise ValueError("contract.evaluation.seed_rule does not match the pinned DeMiAn evaluator")
    if evaluation.get("causal_prompt_only") is not True:
        raise ValueError("contract.evaluation.causal_prompt_only must be true")
    if evaluation.get("initialization_mode") != "seeded_target_env":
        raise ValueError("contract.evaluation.initialization_mode must be seeded_target_env")
    sampling = _require_dict(evaluation.get("policy_sampling"), "policy_sampling")
    if sampling.get("sampler") not in {"unipc", "edm"}:
        raise ValueError("policy_sampling.sampler must be unipc or edm")
    _require_int(sampling.get("seed"), "policy_sampling.seed", minimum=0)
    guidance = sampling.get("guidance")
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(float(guidance))
        or not 0.0 <= float(guidance) <= 7.0
    ):
        raise TypeError("policy_sampling.guidance must be a finite number in [0,7]")
    _require_int(sampling.get("num_steps"), "policy_sampling.num_steps", minimum=1)
    if sampling.get("reuse_seed_per_request") is not True or sampling.get("use_ema_weights") is not True:
        raise ValueError("formal policy sampling must reuse its seed and load EMA weights")

    thresholds = _require_dict(evaluation.get("effect_thresholds"), "effect_thresholds")
    minimum_mean_sr = thresholds.get("minimum_mean_success_rate")
    minimum_successful_tasks = thresholds.get("minimum_successful_tasks")
    if isinstance(minimum_mean_sr, bool) or not isinstance(minimum_mean_sr, (int, float)):
        raise TypeError("minimum_mean_success_rate must be numeric")
    if not 0.0 <= float(minimum_mean_sr) <= 1.0:
        raise ValueError("minimum_mean_success_rate must be in [0,1]")
    _require_int(minimum_successful_tasks, "minimum_successful_tasks", minimum=0)
    if minimum_successful_tasks > len(task_pool):
        raise ValueError("minimum_successful_tasks exceeds task count")

    artifacts = _require_dict(contract.get("artifacts"), "contract.artifacts")
    verified: dict[str, Any] = {}
    for name in (
        "source_manifest",
        "cosmos_server_module",
        "checkpoint_manifest",
        "checkpoint_verification_receipt",
        "training_config",
        "action_stats",
        "demian_evaluator",
        "demian_task_registry",
    ):
        verified[name] = _verify_file_artifact(artifacts, name)
    receipt_descriptors = artifacts.get("server_runtime_receipts")
    if not isinstance(receipt_descriptors, list) or len(receipt_descriptors) != num_shards:
        raise ValueError("contract.artifacts.server_runtime_receipts must contain one entry per shard")
    verified_receipts = []
    for index, descriptor in enumerate(receipt_descriptors):
        synthetic = {"receipt": descriptor}
        verified_receipts.append(_verify_file_artifact(synthetic, "receipt"))
    verified["server_runtime_receipts"] = verified_receipts
    if verified["source_manifest"]["sha256"] != contract["source_manifest_sha256"]:
        raise ValueError("source_manifest artifact hash does not match contract.source_manifest_sha256")
    checkpoint_manifest = _require_dict(
        _load_json(Path(verified["checkpoint_manifest"]["path"])),
        "checkpoint_manifest",
    )
    if (
        checkpoint_manifest.get("schema_version") != 1
        or checkpoint_manifest.get("kind") != "cwam_serving_checkpoint_manifest"
    ):
        raise ValueError("checkpoint_manifest has an unsupported schema/kind")
    if Path(
        _require_nonempty_str(
            checkpoint_manifest.get("checkpoint_path"),
            "checkpoint_manifest.checkpoint_path",
        )
    ).resolve() != checkpoint_path.resolve():
        raise ValueError("checkpoint_manifest checkpoint_path does not match the eval contract")
    checkpoint_tree_sha256 = _require_sha256(
        checkpoint_manifest.get("tree_sha256"),
        "checkpoint_manifest.tree_sha256",
    )
    checkpoint_verification = _require_dict(
        _load_json(Path(verified["checkpoint_verification_receipt"]["path"])),
        "checkpoint_verification_receipt",
    )
    if checkpoint_verification.get("status") != "passed":
        raise ValueError("checkpoint_verification_receipt status must be passed")
    if Path(
        _require_nonempty_str(
            checkpoint_verification.get("checkpoint_path"),
            "checkpoint_verification_receipt.checkpoint_path",
        )
    ).resolve() != checkpoint_path.resolve():
        raise ValueError("checkpoint_verification_receipt path does not match the eval contract")
    _require_int(
        checkpoint_verification.get("file_count"),
        "checkpoint_verification_receipt.file_count",
        minimum=1,
    )
    if (
        _require_sha256(
            checkpoint_verification.get("tree_sha256"),
            "checkpoint_verification_receipt.tree_sha256",
        )
        != checkpoint_tree_sha256
    ):
        raise ValueError("checkpoint verification tree SHA does not match the checkpoint manifest")
    if verified["demian_evaluator"]["sha256"] != demian["evaluator_sha256"]:
        raise ValueError("demian_evaluator artifact hash does not match contract.demian.evaluator_sha256")
    if verified["demian_task_registry"]["sha256"] != demian["task_registry_sha256"]:
        raise ValueError("demian_task_registry artifact hash does not match contract.demian.task_registry_sha256")
    contract["_verified_artifacts"] = verified
    contract["_contract_sha256"] = _sha256(contract_path)
    return contract


def _validate_server_receipt(contract: dict[str, Any]) -> None:
    artifacts = contract["_verified_artifacts"]
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
        "model_action": "[dpos3,rot6d_columns6,gripper1]",
        "output_action": "[dpos3,axisangle3,gripper1]",
        "format_prompt_as_json": True,
        "source_module_sha256": artifacts["cosmos_server_module"]["sha256"],
    }
    seen_shards: set[int] = set()
    seen_nonces: set[str] = set()
    for expected_shard, receipt_descriptor in enumerate(artifacts["server_runtime_receipts"]):
        receipt_path = Path(receipt_descriptor["path"])
        receipt = _require_dict(_load_json(receipt_path), f"server_runtime_receipt[{expected_shard}]")
        if receipt.get("schema_version") != 1 or receipt.get("service") != "cwam_cosmos3_robocasa365":
            raise ValueError("server runtime receipt has an unsupported service/schema")
        if receipt.get("shard_index") != expected_shard:
            raise ValueError(
                f"server runtime receipt shard mismatch: expected={expected_shard}, "
                f"actual={receipt.get('shard_index')}"
            )
        seen_shards.add(expected_shard)
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(
                    f"server runtime receipt {key} mismatch: expected={value!r}, actual={receipt.get(key)!r}"
                )
        source_module_path = Path(
            _require_nonempty_str(
                receipt.get("source_module_path"),
                f"server_runtime_receipt[{expected_shard}].source_module_path",
            )
        ).expanduser().resolve()
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
            raise ValueError(
                f"args shard {shard} {key} mismatch: expected={value!r}, actual={args_json.get(key)!r}"
            )
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
            task
            for value in requested_tasks
            for task in (part.strip() for part in value.split(","))
            if task
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
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
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
        args_sha256_before = _sha256(args_path)
        args_json = _require_dict(_load_json(args_path), f"args shard {shard}")
        expected_tasks = _validate_shard_args(args_json, contract=contract, shard=shard)
        results_path = results_dir / f"results_shard{shard}.json"
        results_sha256_before = _sha256(results_path)
        results = _require_dict(_load_json(results_path), f"results shard {shard}")
        if set(results) != set(expected_tasks):
            raise ValueError(
                f"results shard {shard} task coverage mismatch: "
                f"missing={sorted(set(expected_tasks) - set(results))}, "
                f"unexpected={sorted(set(results) - set(expected_tasks))}"
            )
        args_sha256_after = _sha256(args_path)
        results_sha256_after = _sha256(results_path)
        if args_sha256_before != args_sha256_after or results_sha256_before != results_sha256_after:
            raise RuntimeError(f"Evaluation shard {shard} changed while it was being validated")
        raw_shard_artifacts.append(
            {
                "shard": shard,
                "args": {"path": str(args_path), "sha256": args_sha256_after},
                "results": {"path": str(results_path), "sha256": results_sha256_after},
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
                raise ValueError(
                    f"Task {task} must contain exactly {evaluation['episodes_per_task']} episode results"
                )

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


def summarize(
    *,
    contract_path: Path,
    results_dir: Path,
    summary_csv: Path,
    task_success_rates_csv: Path,
    evidence_json: Path,
    require_effect_pass: bool,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve()
    results_dir = results_dir.expanduser().resolve()
    contract = _validate_contract(contract_path)
    _validate_server_receipt(contract)
    expected_results_dir = Path(contract["results_dir"]).expanduser().resolve()
    if results_dir != expected_results_dir:
        raise ValueError(f"results_dir mismatch: contract={expected_results_dir}, requested={results_dir}")

    episode_rows, task_rows, raw_shard_artifacts = _collect_rows(contract, results_dir)
    _atomic_write_csv(summary_csv, _CSV_EPISODE_FIELDS, episode_rows)
    _atomic_write_csv(task_success_rates_csv, _CSV_TASK_FIELDS, task_rows)

    task_success_rates = [float(row["success_rate"]) for row in task_rows]
    successful_tasks = sum(rate > 0.0 for rate in task_success_rates)
    mean_success_rate = sum(task_success_rates) / len(task_success_rates)
    thresholds = contract["evaluation"]["effect_thresholds"]
    effect_pass = (
        mean_success_rate >= float(thresholds["minimum_mean_success_rate"])
        and successful_tasks >= int(thresholds["minimum_successful_tasks"])
    )
    evidence = {
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
        "evaluation": {
            "stage": contract["evaluation"]["stage"],
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
            "summary_csv": {"path": str(summary_csv), "sha256": _sha256(summary_csv)},
            "task_success_rates_csv": {
                "path": str(task_success_rates_csv),
                "sha256": _sha256(task_success_rates_csv),
            },
            "raw_shards": raw_shard_artifacts,
            **contract["_verified_artifacts"],
        },
        "seed_provenance": (
            "The pinned evaluator ran with eval_dataset=null, so target-environment "
            "resets used the seeds derived from verified DeMiAn shard args: "
            "base_seed + within_shard_task_index * 10000 + episode_index; "
            "the pinned evaluator does not serialize seed per episode."
        ),
    }
    _atomic_write(evidence_json, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    evidence["artifacts"]["evaluation_evidence"] = {
        "path": str(evidence_json),
        "sha256": _sha256(evidence_json),
    }
    if require_effect_pass and not effect_pass:
        raise RuntimeError(
            "Formal evaluation execution is complete, but the preregistered effect threshold failed: "
            f"mean_success_rate={mean_success_rate:.6f}, successful_tasks={successful_tasks}"
        )
    return evidence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract_parser = subparsers.add_parser("create-contract")
    contract_parser.add_argument("--output", type=Path, required=True)
    contract_parser.add_argument("--experiment-id", required=True)
    contract_parser.add_argument("--logical-run-id", required=True)
    contract_parser.add_argument("--source-commit", required=True)
    contract_parser.add_argument("--source-manifest", type=Path, required=True)
    contract_parser.add_argument("--cosmos-server-module", type=Path, required=True)
    contract_parser.add_argument("--checkpoint-path", type=Path, required=True)
    contract_parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    contract_parser.add_argument("--checkpoint-verification-receipt", type=Path, required=True)
    contract_parser.add_argument("--training-config", type=Path, required=True)
    contract_parser.add_argument("--action-stats", type=Path, required=True)
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
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--contract", type=Path, required=True)
    summarize_parser.add_argument("--results-dir", type=Path, required=True)
    summarize_parser.add_argument("--summary-csv", type=Path, required=True)
    summarize_parser.add_argument("--task-success-rates-csv", type=Path, required=True)
    summarize_parser.add_argument("--evidence-json", type=Path, required=True)
    summarize_parser.add_argument("--require-effect-pass", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "create-contract":
        contract = create_contract(
            output=args.output,
            experiment_id=args.experiment_id,
            logical_run_id=args.logical_run_id,
            source_commit=args.source_commit,
            source_manifest=args.source_manifest,
            cosmos_server_module=args.cosmos_server_module,
            checkpoint_path=args.checkpoint_path,
            checkpoint_manifest=args.checkpoint_manifest,
            checkpoint_verification_receipt=args.checkpoint_verification_receipt,
            training_config=args.training_config,
            action_stats=args.action_stats,
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
        )
        print(json.dumps(contract, indent=2, sort_keys=True))
        return
    if args.command == "summarize":
        evidence = summarize(
            contract_path=args.contract,
            results_dir=args.results_dir,
            summary_csv=args.summary_csv,
            task_success_rates_csv=args.task_success_rates_csv,
            evidence_json=args.evidence_json,
            require_effect_pass=args.require_effect_pass,
        )
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
