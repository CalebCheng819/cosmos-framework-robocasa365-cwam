# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for formal RoboCasa365 evaluation evidence generation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from examples.h_cluster import cwam_robocasa365_eval_evidence as evidence


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(tmp_path: Path, name: str, content: object) -> dict[str, str]:
    path = _write_json(tmp_path / f"{name}.json", content)
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    results_dir = tmp_path / "results" / "gate"
    results_dir.mkdir(parents=True)
    source_manifest = _artifact(tmp_path, "source-manifest", {"files": 1})
    checkpoint = _artifact(tmp_path, "checkpoint-manifest", {"iteration": 10000})
    config = _artifact(tmp_path, "training-config", {"recipe": "robocasa365"})
    stats = _artifact(tmp_path, "action-stats", {"global_raw": {}})
    demian_evaluator = _artifact(tmp_path, "demian-evaluator", {"source": "evaluator"})
    demian_task_registry = _artifact(tmp_path, "demian-task-registry", {"tasks": ["OpenDrawer"]})
    source_manifest_sha = source_manifest["sha256"]
    demian_commit = "2" * 40
    receipt = {
        "schema_version": 1,
        "service": "cwam_cosmos3_robocasa365",
        "checkpoint": "/checkpoints/iter_000010000",
        "action_stats_path": stats["path"],
        "shard_index": 0,
        "checkpoint_manifest_sha256": checkpoint["sha256"],
        "training_config_sha256": config["sha256"],
        "action_stats_sha256": stats["sha256"],
        "source_manifest_sha256": source_manifest_sha,
        "demian_commit": demian_commit,
        "wire_inputs": [
            "observation/agentview_left",
            "observation/eye_in_hand",
            "observation/agentview_right",
            "observation/state",
            "prompt",
        ],
        "conditioning_views": [
            "observation/agentview_left",
            "observation/eye_in_hand",
        ],
        "state_conditioning": False,
        "gripper_transform": "identity",
        "fps": 20,
        "action_chunk_size": 16,
    }
    receipt_artifact = _artifact(tmp_path, "server-runtime-receipt", receipt)
    contract = {
        "schema_version": 1,
        "kind": "cwam_robocasa365_formal_eval_contract",
        "experiment_id": "CWAM-E35-R12-GATE",
        "logical_run_id": "CWAM-E35-R12-GATE-S42",
        "source_commit": "3" * 40,
        "source_manifest_sha256": source_manifest_sha,
        "checkpoint_path": "/checkpoints/iter_000010000",
        "run_name": "gate",
        "results_dir": str(results_dir.resolve()),
        "demian": {
            "commit": demian_commit,
            "evaluator_sha256": demian_evaluator["sha256"],
            "task_registry_sha256": demian_task_registry["sha256"],
        },
        "evaluation": {
            "stage": "one_episode_gate",
            "role": "primary",
            "split": "atomic",
            "tasks": ["OpenDrawer"],
            "episodes_per_task": 1,
            "base_seed": 42,
            "max_steps": 400,
            "open_loop_horizon": 12,
            "num_shards": 1,
            "seed_rule": "base_seed + within_shard_task_index * 10000 + episode_index",
            "causal_prompt_only": True,
            "effect_thresholds": {
                "minimum_mean_success_rate": 0.0,
                "minimum_successful_tasks": 0,
            },
        },
        "artifacts": {
            "source_manifest": source_manifest,
            "checkpoint_manifest": checkpoint,
            "training_config": config,
            "action_stats": stats,
            "demian_evaluator": demian_evaluator,
            "demian_task_registry": demian_task_registry,
            "server_runtime_receipts": [receipt_artifact],
        },
    }
    contract_path = _write_json(tmp_path / "contract.json", contract)
    args_json = {
        "tasks": ["OpenDrawer"],
        "split": "atomic",
        "task_shard": 0,
        "num_shards": 1,
        "episodes": 1,
        "episodes_start": 0,
        "max_steps": 400,
        "open_loop_horizon": 12,
        "seed": 42,
        "checkpoint": "/checkpoints/iter_000010000",
        "results_dir": str((tmp_path / "results").resolve()),
        "run_name": "gate",
        "save_video": False,
        "headless": True,
        "dynamic_instructions_file": None,
        "gt_reasoning_file": None,
        "vlm_captions_file": None,
        "fg_captions_path": None,
    }
    _write_json(results_dir / "args.json", args_json)
    result = {
        "OpenDrawer": {
            "success_rate": 1.0,
            "successes": 1,
            "place_anytime_rate": 1.0,
            "place_anytime_count": 1,
            "episodes": 1,
            "episode_results": [
                {
                    "success": True,
                    "place_anytime": True,
                    "place_first_step": 7,
                    "steps": 8,
                    "subtask_successes": [],
                    "action_log": [[0.0] * 7 for _ in range(8)],
                    "instruction": "open the drawer",
                }
            ],
            "split": "atomic",
            "complete": True,
        }
    }
    _write_json(results_dir / "results_shard0.json", result)
    return contract_path, results_dir


def test_summarize_writes_complete_episode_and_task_csvs(tmp_path: Path) -> None:
    contract, results_dir = _fixture(tmp_path)
    summary_csv = tmp_path / "summary.csv"
    task_csv = tmp_path / "task_success_rates.csv"
    evidence_json = tmp_path / "evaluation_evidence.json"

    result = evidence.summarize(
        contract_path=contract,
        results_dir=results_dir,
        summary_csv=summary_csv,
        task_success_rates_csv=task_csv,
        evidence_json=evidence_json,
        require_effect_pass=True,
    )

    assert result["execution_complete"] is True
    assert result["effect_pass"] is True
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["task"] == "OpenDrawer"
    assert rows[0]["episode_index"] == "0"
    assert rows[0]["seed"] == "42"
    assert rows[0]["success"] == "true"
    with task_csv.open(newline="", encoding="utf-8") as handle:
        task_rows = list(csv.DictReader(handle))
    assert task_rows[0]["success_rate"] == "1.000000000000"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda result: result["OpenDrawer"].update(complete=False), "not complete"),
        (lambda result: result["OpenDrawer"].update(error="sim failed"), "evaluation error"),
        (lambda result: result["OpenDrawer"]["episode_results"].clear(), "exactly 1"),
        (
            lambda result: result["OpenDrawer"]["episode_results"][0]["action_log"][0].append(0.0),
            "width 7",
        ),
    ],
)
def test_summarize_rejects_incomplete_or_malformed_results(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    contract, results_dir = _fixture(tmp_path)
    result_path = results_dir / "results_shard0.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mutation(result)
    _write_json(result_path, result)

    with pytest.raises((RuntimeError, ValueError), match=match):
        evidence.summarize(
            contract_path=contract,
            results_dir=results_dir,
            summary_csv=tmp_path / "summary.csv",
            task_success_rates_csv=tmp_path / "task_success_rates.csv",
            evidence_json=tmp_path / "evaluation_evidence.json",
            require_effect_pass=False,
        )


def test_formal_contract_requires_full_canonical_atomic_suite(tmp_path: Path) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["evaluation"]["stage"] = "formal"
    contract["evaluation"]["episodes_per_task"] = 20
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="canonical ordered task list"):
        evidence._validate_contract(contract_path)


def test_effect_threshold_failure_preserves_evidence_then_fails(tmp_path: Path) -> None:
    contract_path, results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["evaluation"]["effect_thresholds"] = {
        "minimum_mean_success_rate": 1.0,
        "minimum_successful_tasks": 1,
    }
    _write_json(contract_path, contract)
    result_path = results_dir / "results_shard0.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["OpenDrawer"]["success_rate"] = 0.0
    result["OpenDrawer"]["successes"] = 0
    result["OpenDrawer"]["place_anytime_rate"] = 0.0
    result["OpenDrawer"]["place_anytime_count"] = 0
    result["OpenDrawer"]["episode_results"][0]["success"] = False
    result["OpenDrawer"]["episode_results"][0]["place_anytime"] = False
    _write_json(result_path, result)
    evidence_json = tmp_path / "evaluation_evidence.json"

    with pytest.raises(RuntimeError, match="effect threshold failed"):
        evidence.summarize(
            contract_path=contract_path,
            results_dir=results_dir,
            summary_csv=tmp_path / "summary.csv",
            task_success_rates_csv=tmp_path / "task_success_rates.csv",
            evidence_json=evidence_json,
            require_effect_pass=True,
        )

    persisted = json.loads(evidence_json.read_text(encoding="utf-8"))
    assert persisted["execution_complete"] is True
    assert persisted["effect_pass"] is False
