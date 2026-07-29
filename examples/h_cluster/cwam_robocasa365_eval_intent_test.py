# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for immutable RoboCasa365 evaluation intent manifests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from examples.h_cluster import cwam_robocasa365_eval_intent as intent


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path.resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    _write(path / "tracked.json", {"tracked": True})
    subprocess.run(["git", "-C", str(path), "add", "tracked.json"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return path.resolve(), commit


def _environment(tmp_path: Path, name: str) -> tuple[Path, Path]:
    root = (tmp_path / name).resolve()
    interpreter = _write(root / "bin" / "python", b"fixture interpreter\n")
    manifest = _write(
        tmp_path / f"{name}-manifest.json",
        {
            "schema_version": 1,
            "kind": "cwam_python_environment_manifest",
            "interpreter": {
                "resolved_path": str(interpreter),
                "size_bytes": interpreter.stat().st_size,
                "sha256": _sha(interpreter),
            },
        },
    )
    return root, manifest


def _fixture(tmp_path: Path, stage: str = "one_episode_gate") -> dict[str, object]:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    source_commit = "1" * 40
    source_manifest = _write(
        source / "SNAPSHOT_MANIFEST.json",
        {
            "schema_version": 1,
            "source_head": source_commit,
            "source_dirty": False,
            "file_count": 0,
            "files": [],
        },
    )
    source_checkpoint = (tmp_path / "iter_000010000").resolve()
    checkpoint = (source_checkpoint / "model").resolve()
    weight = _write(checkpoint / "weights.bin", b"weights\n")
    checkpoint_files = [
        {
            "relative_path": "weights.bin",
            "size_bytes": weight.stat().st_size,
            "sha256": _sha(weight),
        }
    ]
    tree_sha = hashlib.sha256(
        json.dumps(
            checkpoint_files,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    checkpoint_manifest = _write(
        tmp_path / "checkpoint-manifest.json",
        {
            "schema_version": 1,
            "kind": "cwam_serving_checkpoint_manifest",
            "checkpoint_path": str(checkpoint),
            "files": checkpoint_files,
            "tree_sha256": tree_sha,
        },
    )
    cosmos_env, cosmos_env_manifest = _environment(tmp_path, "cosmos-env")
    robocasa_env, robocasa_env_manifest = _environment(tmp_path, "robocasa-env")
    demian, demian_commit = _git_repo(tmp_path / "demian")
    robocasa, robocasa_commit = _git_repo(tmp_path / "robocasa")
    robosuite, robosuite_commit = _git_repo(tmp_path / "robosuite")
    assets = (robocasa / "assets").resolve()
    assets.mkdir()

    files = {
        "export_receipt": _write(
            tmp_path / "export-receipt.json",
            {
                "schema_version": 1,
                "kind": "cwam_serving_dcp_selection_receipt",
                "status": "passed",
                "source_checkpoint_path": str(source_checkpoint),
                "source_checkpoint_iteration": 10_000,
                "serving_checkpoint_path": str(checkpoint),
                "checkpoint_manifest_sha256": _sha(checkpoint_manifest),
                "checkpoint_tree_sha256": tree_sha,
                "validation": {
                    "source_iteration_path_matches": True,
                    "serving_checkpoint_is_exact_model_child": True,
                    "serving_checkpoint_manifest_matches": True,
                },
            },
        ),
        "training_config": _write(tmp_path / "config.json", {"recipe": "cwam"}),
        "action_stats": _write(tmp_path / "stats.json", {"global_raw": {}}),
        "asset_manifest": _write(tmp_path / "asset-manifest.json", {"files": []}),
        "asset_receipt": _write(tmp_path / "asset-receipt.json", {"status": "passed"}),
        "egl": _write(tmp_path / "10_nvidia.json", {"file_format_version": "1.0.0"}),
    }
    files["export_receipt"].chmod(0o444)
    prior_gate = _write(tmp_path / "gate-evidence.json", {"status": "passed"})
    prior_gate.chmod(0o444)

    formal = stage == "formal"
    values: dict[str, object] = {
        "experiment_id": "CWAM-E35-EVAL",
        "logical_run_id": "CWAM-E35-EVAL-R1",
        "run_name": "CWAM-E35-EVAL-R1",
        "stage": stage,
        "evaluation_class": "exploratory",
        "git_pushed": False,
        "split": "atomic",
        "tasks_csv": "" if formal else "OpenDrawer",
        "episodes_per_task": 20 if formal else 1,
        "num_shards": 8 if formal else 1,
        "base_seed": 42,
        "max_steps": 400,
        "open_loop_horizon": 12,
        "policy_sampler": "unipc",
        "policy_seed": 42,
        "policy_guidance": 1.0,
        "policy_num_steps": 30,
        "base_port": 16000,
        "server_ready_timeout_seconds": 1800,
        "minimum_mean_success_rate": 0.25 if formal else 0.0,
        "minimum_successful_tasks": 5 if formal else 0,
        "source_root": source,
        "source_commit": source_commit,
        "source_manifest": source_manifest,
        "expected_source_manifest_sha256": _sha(source_manifest),
        "serving_checkpoint_path": checkpoint,
        "checkpoint_manifest": checkpoint_manifest,
        "expected_checkpoint_manifest_sha256": _sha(checkpoint_manifest),
        "source_dcp_export_receipt": files["export_receipt"],
        "expected_source_dcp_export_receipt_sha256": _sha(files["export_receipt"]),
        "training_config": files["training_config"],
        "expected_training_config_sha256": _sha(files["training_config"]),
        "action_stats": files["action_stats"],
        "expected_action_stats_sha256": _sha(files["action_stats"]),
        "cosmos_env_root": cosmos_env,
        "cosmos_environment_manifest": cosmos_env_manifest,
        "expected_cosmos_environment_manifest_sha256": _sha(cosmos_env_manifest),
        "robocasa_env_root": robocasa_env,
        "robocasa_environment_manifest": robocasa_env_manifest,
        "expected_robocasa_environment_manifest_sha256": _sha(robocasa_env_manifest),
        "robocasa_asset_root": assets,
        "robocasa_asset_manifest": files["asset_manifest"],
        "expected_robocasa_asset_manifest_sha256": _sha(files["asset_manifest"]),
        "robocasa_asset_receipt": files["asset_receipt"],
        "expected_robocasa_asset_receipt_sha256": _sha(files["asset_receipt"]),
        "demian_root": demian,
        "demian_commit": demian_commit,
        "robocasa_root": robocasa,
        "robocasa_commit": robocasa_commit,
        "robosuite_root": robosuite,
        "robosuite_commit": robosuite_commit,
        "container_image": f"registry.example/cwam@sha256:{'a' * 64}",
        "egl_vendor_json": files["egl"],
        "expected_egl_vendor_json_sha256": _sha(files["egl"]),
        "prior_gate_evidence": prior_gate if formal else None,
        "expected_prior_gate_evidence_sha256": _sha(prior_gate) if formal else None,
        "cpu": 128 if formal else 16,
        "memory_mib": 1_500_000 if formal else 187_500,
        "gpu": 8 if formal else 1,
        "quota_group": "eailabagent_gpu",
        "negative_tag": "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn",
        "task_type": "normal",
        "private_machine": "group",
        "priority": 9,
        "restart_policy": "never",
        "backoff_limit": 1,
        "replicas": 1,
    }
    return values


def _args(values: dict[str, object]) -> object:
    return type("Args", (), values)()


def _create_and_verify(tmp_path: Path, values: dict[str, object]) -> tuple[Path, dict[str, object]]:
    built = intent.build_intent(_args(values))
    output = (tmp_path / "intent.json").resolve()
    receipt = intent.create_intent_file(output, built)
    assert receipt["status"] == "created"
    assert output.stat().st_mode & 0o777 == 0o444
    verified = intent.verify_intent_file(output, str(receipt["sha256"]), intent.build_intent(_args(values)))
    assert verified["status"] == "passed"
    return output, receipt


def test_positive_one_episode_gate(tmp_path: Path) -> None:
    output, _ = _create_and_verify(tmp_path, _fixture(tmp_path))
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["identity"]["stage"] == "one_episode_gate"
    assert manifest["identity"]["evaluation_class"] == "exploratory"
    assert manifest["prior_gate"] is None
    assert manifest["protocol"]["expected_episode_count"] == 1


def test_positive_exploratory_formal_keeps_stage_and_class_orthogonal(tmp_path: Path) -> None:
    output, _ = _create_and_verify(tmp_path, _fixture(tmp_path, "formal"))
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["identity"]["stage"] == "formal"
    assert manifest["identity"]["evaluation_class"] == "exploratory"
    assert manifest["source"]["git_pushed"] is False
    assert manifest["prior_gate"]["sha256"]
    assert manifest["protocol"]["expected_episode_count"] == 340


def test_confirmatory_unpushed_is_rejected(tmp_path: Path) -> None:
    values = _fixture(tmp_path, "formal")
    values["evaluation_class"] = "confirmatory"
    with pytest.raises(intent.EvalIntentError, match="git_pushed=true"):
        intent.build_intent(_args(values))


def test_changed_artifact_is_rejected_during_live_verification(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    output, receipt = _create_and_verify(tmp_path, values)
    Path(values["training_config"]).chmod(0o644)
    Path(values["training_config"]).write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(intent.EvalIntentError, match="training config SHA-256 mismatch"):
        intent.verify_intent_file(
            output,
            str(receipt["sha256"]),
            intent.build_intent(_args(values)),
        )


def test_symlinked_artifact_is_rejected(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    target = Path(values["training_config"])
    symlink = (tmp_path / "config-link.json").resolve()
    os.symlink(target, symlink)
    values["training_config"] = symlink
    with pytest.raises(intent.EvalIntentError, match="symlink components"):
        intent.build_intent(_args(values))


def test_create_refuses_to_overwrite_existing_manifest(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    built = intent.build_intent(_args(values))
    output = (tmp_path / "intent.json").resolve()
    intent.create_intent_file(output, built)
    with pytest.raises(intent.EvalIntentError, match="exclusively create"):
        intent.create_intent_file(output, built)


def test_source_dcp_receipt_must_be_valid_read_only_json(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    receipt = Path(values["source_dcp_export_receipt"])
    receipt.chmod(0o644)
    receipt.write_bytes(b"")
    receipt.chmod(0o444)
    values["expected_source_dcp_export_receipt_sha256"] = _sha(receipt)

    with pytest.raises(intent.EvalIntentError, match="not valid UTF-8 JSON"):
        intent.build_intent(_args(values))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda receipt: receipt.update(status="failed"), "schema/kind/status mismatch"),
        (
            lambda receipt: receipt.update(checkpoint_tree_sha256="9" * 64),
            "manifest/tree binding mismatch",
        ),
        (
            lambda receipt: receipt["validation"].update(source_iteration_path_matches=False),
            "exact all-true verdict",
        ),
        (
            lambda receipt: receipt["validation"].update(source_iteration_path_matches=1),
            "exact all-true verdict",
        ),
    ],
)
def test_source_dcp_receipt_semantic_drift_is_rejected(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    values = _fixture(tmp_path)
    receipt_path = Path(values["source_dcp_export_receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutation(receipt)
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    receipt_path.chmod(0o444)
    values["expected_source_dcp_export_receipt_sha256"] = _sha(receipt_path)

    with pytest.raises(intent.EvalIntentError, match=match):
        intent.build_intent(_args(values))
