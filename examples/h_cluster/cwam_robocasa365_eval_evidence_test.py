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


@pytest.fixture(autouse=True)
def _use_fixture_demian_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    def serialized_sha256(value: object) -> str:
        return hashlib.sha256(json.dumps(value, indent=2).encode("utf-8")).hexdigest()

    monkeypatch.setattr(evidence, "_EXPECTED_DEMIAN_COMMIT", "2" * 40)
    monkeypatch.setattr(
        evidence,
        "_EXPECTED_DEMIAN_EVALUATOR_SHA256",
        serialized_sha256({"source": "evaluator"}),
    )
    monkeypatch.setattr(
        evidence,
        "_EXPECTED_DEMIAN_TASK_REGISTRY_SHA256",
        serialized_sha256({"tasks": ["OpenDrawer"]}),
    )


def _artifact(tmp_path: Path, name: str, content: object) -> dict[str, str]:
    path = _write_json(tmp_path / f"{name}.json", content)
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _path_artifact(path: Path, content: object) -> dict[str, str]:
    _write_json(path, content)
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _environment_manifest(
    prefix: Path,
    base_prefix: Path,
    versions: dict[str, str],
) -> dict[str, object]:
    interpreter = prefix / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)
    distributions = {
        name: {
            "name": name,
            "version": version,
        }
        for name, version in sorted(versions.items())
    }
    distributions_sha256 = hashlib.sha256(
        json.dumps(
            distributions,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": "cwam_python_environment_manifest",
        "interpreter": {
            "resolved_path": str(interpreter.resolve()),
            "size_bytes": interpreter.stat().st_size,
            "sha256": _sha256(interpreter),
            "python_version": "3.11.14",
            "version_info": {
                "major": 3,
                "minor": 11,
                "micro": 14,
                "releaselevel": "final",
                "serial": 0,
            },
            "implementation": "cpython",
            "cache_tag": "cpython-311",
            "prefix": str(prefix.resolve()),
            "base_prefix": str(base_prefix.resolve()),
        },
        "distribution_count": len(distributions),
        "distributions": distributions,
        "distributions_sha256": distributions_sha256,
    }


def _mutate_contract_artifact(
    contract_path: Path,
    artifact_name: str,
    mutation,
) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    artifact = contract["artifacts"][artifact_name]
    artifact_path = Path(artifact["path"])
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    mutation(value)
    _write_json(artifact_path, value)
    artifact["sha256"] = _sha256(artifact_path)
    _write_json(contract_path, contract)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    results_dir = tmp_path / "results" / "gate"
    results_dir.mkdir(parents=True)
    cosmos_repo = tmp_path / "cosmos"
    demian_repo = tmp_path / "demian"
    robocasa_repo = tmp_path / "robocasa"
    robosuite_repo = tmp_path / "robosuite"
    base_prefix = tmp_path / "base-python"
    base_prefix.mkdir()
    for repo in (cosmos_repo, demian_repo, robocasa_repo, robosuite_repo):
        repo.mkdir()
    source_manifest = _artifact(tmp_path, "source-manifest", {"files": 1})
    checkpoint_path = "/checkpoints/iter_000010000"
    checkpoint_tree_sha = "7" * 64
    checkpoint = _artifact(
        tmp_path,
        "checkpoint-manifest",
        {
            "schema_version": 1,
            "kind": "cwam_serving_checkpoint_manifest",
            "checkpoint_path": checkpoint_path,
            "files": [
                {
                    "relative_path": "weights.bin",
                    "size_bytes": 1,
                    "sha256": "8" * 64,
                }
            ],
            "tree_sha256": checkpoint_tree_sha,
        },
    )
    checkpoint_verification = _artifact(
        tmp_path,
        "checkpoint-verification",
        {
            "status": "passed",
            "checkpoint_path": checkpoint_path,
            "file_count": 1,
            "tree_sha256": checkpoint_tree_sha,
        },
    )
    config = _artifact(tmp_path, "training-config", {"recipe": "robocasa365"})
    stats = _artifact(tmp_path, "action-stats", {"global_raw": {}})
    demian_evaluator = _path_artifact(
        demian_repo / "demian_wam" / "eval_utils" / "run_robocasa_eval.py",
        {"source": "evaluator"},
    )
    demian_task_registry = _path_artifact(
        demian_repo / "demian_wam" / "eval_utils" / "robocasa_task_list.py",
        {"tasks": ["OpenDrawer"]},
    )
    cosmos_server_module = _path_artifact(
        cosmos_repo / "cosmos_framework" / "scripts" / "action_policy_server_robocasa365.py",
        {"source": "server"},
    )
    cosmos_module = _write_json(
        cosmos_repo / "cosmos_framework" / "__init__.py",
        {"source": "cosmos"},
    )
    robocasa_module = _write_json(
        robocasa_repo / "robocasa" / "__init__.py",
        {"source": "robocasa"},
    )
    robosuite_module = _write_json(
        robosuite_repo / "robosuite" / "__init__.py",
        {"source": "robosuite"},
    )
    cosmos_prefix = tmp_path / "cosmos-env"
    robocasa_prefix = tmp_path / "robocasa-env"
    cosmos_environment = _artifact(
        tmp_path,
        "cosmos-environment",
        _environment_manifest(
            cosmos_prefix,
            base_prefix,
            {
                "cosmos-framework": "1.0",
                "torch": "2.10.0+cu128",
            },
        ),
    )
    robocasa_environment = _artifact(
        tmp_path,
        "robocasa-environment",
        _environment_manifest(
            robocasa_prefix,
            base_prefix,
            {
                "gymnasium": "1.0",
                "mujoco": "1.0",
                "numpy": "1.0",
                "robocasa": "1.0",
                "robosuite": "1.0",
            },
        ),
    )
    environment_modules = {
        "torch": _write_json(
            cosmos_prefix / "lib" / "python3.11" / "site-packages" / "torch" / "__init__.py",
            {"module": "torch"},
        ),
        "numpy": _write_json(
            robocasa_prefix / "lib" / "python3.11" / "site-packages" / "numpy" / "__init__.py",
            {"module": "numpy"},
        ),
        "mujoco": _write_json(
            robocasa_prefix / "lib" / "python3.11" / "site-packages" / "mujoco" / "__init__.py",
            {"module": "mujoco"},
        ),
        "gymnasium": _write_json(
            robocasa_prefix / "lib" / "python3.11" / "site-packages" / "gymnasium" / "__init__.py",
            {"module": "gymnasium"},
        ),
    }
    source_manifest_sha = source_manifest["sha256"]
    source_commit = "3" * 40
    demian_commit = "2" * 40
    robocasa_commit = "4" * 40
    robosuite_commit = "5" * 40
    receipt = {
        "schema_version": 1,
        "service": "cwam_cosmos3_robocasa365",
        "checkpoint": checkpoint_path,
        "action_stats_path": stats["path"],
        "shard_index": 0,
        "checkpoint_manifest_sha256": checkpoint["sha256"],
        "training_config_sha256": config["sha256"],
        "action_stats_sha256": stats["sha256"],
        "source_manifest_sha256": source_manifest_sha,
        "demian_commit": demian_commit,
        "source_module_path": cosmos_server_module["path"],
        "source_module_sha256": cosmos_server_module["sha256"],
        "server_pid": 12345,
        "launch_nonce": "6" * 64,
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
        "model_action": "[dpos3,rot6d_columns6,gripper1]",
        "output_action": "[dpos3,axisangle3,gripper1]",
        "gripper_transform": "identity",
        "fps": 20,
        "action_chunk_size": 16,
        "format_prompt_as_json": True,
        "sampler": "unipc",
        "policy_seed": 42,
        "guidance": 1.0,
        "num_steps": 30,
        "reuse_policy_seed_per_request": True,
        "use_ema_weights": True,
    }
    receipt_artifact = _artifact(tmp_path, "server-runtime-receipt", receipt)
    cosmos_environment_value = json.loads(Path(cosmos_environment["path"]).read_text(encoding="utf-8"))
    robocasa_environment_value = json.loads(Path(robocasa_environment["path"]).read_text(encoding="utf-8"))

    def environment_descriptor(artifact: dict[str, str]) -> dict[str, object]:
        return {
            "path": artifact["path"],
            "size_bytes": Path(artifact["path"]).stat().st_size,
            "sha256": artifact["sha256"],
            "expected_sha256": artifact["sha256"],
            "kind": "cwam_python_environment_manifest",
            "schema_version": 1,
        }

    def environment_report(manifest: dict[str, object]) -> dict[str, object]:
        interpreter = manifest["interpreter"]
        assert isinstance(interpreter, dict)
        return {
            "status": "passed",
            "interpreter_resolved_path": interpreter["resolved_path"],
            "interpreter_sha256": interpreter["sha256"],
            "python_version": interpreter["python_version"],
            "distribution_count": manifest["distribution_count"],
            "distributions_sha256": manifest["distributions_sha256"],
        }

    cosmos_python = cosmos_environment_value["interpreter"]["resolved_path"]
    robocasa_python = robocasa_environment_value["interpreter"]["resolved_path"]
    gpu_uuid = "00000000000000000000000000000001"
    source_records = {
        "cosmos": (cosmos_repo, source_commit),
        "demian": (demian_repo, demian_commit),
        "robocasa": (robocasa_repo, robocasa_commit),
        "robosuite": (robosuite_repo, robosuite_commit),
    }
    nvidia_smi = tmp_path / "tools" / "nvidia-smi"
    nvidia_smi.parent.mkdir()
    nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)
    eval_runtime_receipt = _artifact(
        tmp_path,
        "eval-runtime-receipt",
        {
            "schema_version": 1,
            "kind": "cwam_robocasa365_eval_runtime_receipt",
            "status": "passed",
            "captured_at_utc": "2026-07-29T01:02:03Z",
            "driver_environment": {
                "ld_library_path": evidence._EXPECTED_H_DRIVER_LIBRARY_PATH,
                "egl_vendor_library_dirs": None,
            },
            "inputs": {
                "experiment_id": "CWAM-E35-R12-GATE",
                "logical_run_id": "CWAM-E35-R12-GATE-S42",
                "capture_nonce": "e" * 64,
                "num_gpus": 1,
                "interpreters": {
                    "cosmos": {
                        "launch_path": cosmos_python,
                        "resolved_path": cosmos_python,
                    },
                    "robocasa": {
                        "launch_path": robocasa_python,
                        "resolved_path": robocasa_python,
                    },
                },
                "environment_manifests": {
                    "cosmos": environment_descriptor(cosmos_environment),
                    "robocasa": environment_descriptor(robocasa_environment),
                },
                "sources": {
                    name: {
                        "root": str(root.resolve()),
                        "expected_commit": commit,
                        "actual_commit": commit,
                        "clean": True,
                    }
                    for name, (root, commit) in source_records.items()
                },
            },
            "environment_verification": {
                "cosmos": environment_report(cosmos_environment_value),
                "robocasa": environment_report(robocasa_environment_value),
            },
            "cosmos_runtime": {
                "python": {
                    "version": "3.11.14 (test)",
                    "executable": cosmos_python,
                    "prefix": cosmos_environment_value["interpreter"]["prefix"],
                    "base_prefix": cosmos_environment_value["interpreter"]["base_prefix"],
                },
                "torch": {
                    "version": "2.10.0+cu128",
                    "cuda_version": "12.8",
                    "cuda_available": True,
                    "cuda_device_count": 1,
                    "module_path": str(environment_modules["torch"].resolve()),
                },
                "cosmos_framework": {"module_path": str(cosmos_module.resolve())},
                "gpus": [
                    {
                        "index": 0,
                        "name": "NVIDIA H200",
                        "uuid": f"GPU-{gpu_uuid}",
                        "normalized_uuid": gpu_uuid,
                        "capability": [9, 0],
                        "total_memory_bytes": 150_000_000_000,
                    }
                ],
            },
            "robocasa_runtime": {
                "python": {
                    "version": "3.11.14 (test)",
                    "executable": robocasa_python,
                    "prefix": robocasa_environment_value["interpreter"]["prefix"],
                    "base_prefix": robocasa_environment_value["interpreter"]["base_prefix"],
                },
                "packages": {
                    name: {
                        "distribution_version": "1.0",
                        "module_reported_version": "1.0",
                        "version_source": "importlib.metadata",
                        "module_path": str(module.resolve()),
                    }
                    for name, module in {
                        "numpy": environment_modules["numpy"],
                        "mujoco": environment_modules["mujoco"],
                        "gymnasium": environment_modules["gymnasium"],
                        "robocasa": robocasa_module,
                        "robosuite": robosuite_module,
                    }.items()
                },
                "demian_evaluator": {"module_path": demian_evaluator["path"]},
            },
            "robocasa_egl": [
                {
                    "status": "passed",
                    "shard_index": 0,
                    "cuda_visible_devices": "0",
                    "mujoco_egl_device_id": "0",
                    "mujoco_gl": "egl",
                    "pyopengl_platform": "egl",
                    "ld_library_path": evidence._EXPECTED_H_DRIVER_LIBRARY_PATH,
                    "egl_vendor_library_dirs": None,
                    "robosuite_module_path": str(robosuite_module.resolve()),
                }
            ],
            "nvidia_smi": {
                "executable": str(nvidia_smi.resolve()),
                "driver_version": "570.133.20",
                "gpus": [
                    {
                        "index": 0,
                        "uuid": f"GPU-{gpu_uuid}",
                        "normalized_uuid": gpu_uuid,
                        "name": "NVIDIA H200",
                        "driver_version": "570.133.20",
                    }
                ],
            },
            "validation": {
                "environment_manifests_match_live_interpreters": True,
                "source_commits_match_and_are_clean": True,
                "source_module_paths_match_exact_roots": True,
                "cuda_available": True,
                "gpu_count_matches": True,
                "all_gpus_are_h200_compute_capability_9_0": True,
                "torch_and_nvidia_smi_uuid_sets_match": True,
                "h_driver_library_path_is_exact": True,
                "robocasa_egl_contexts_created_per_shard": True,
            },
        },
    )
    contract = {
        "schema_version": 1,
        "kind": "cwam_robocasa365_formal_eval_contract",
        "experiment_id": "CWAM-E35-R12-GATE",
        "logical_run_id": "CWAM-E35-R12-GATE-S42",
        "source_commit": source_commit,
        "source_root": str(cosmos_repo.resolve()),
        "source_manifest_sha256": source_manifest_sha,
        "checkpoint_path": checkpoint_path,
        "run_name": "gate",
        "results_dir": str(results_dir.resolve()),
        "demian": {
            "repo": str(demian_repo.resolve()),
            "commit": demian_commit,
            "evaluator_sha256": demian_evaluator["sha256"],
            "task_registry_sha256": demian_task_registry["sha256"],
        },
        "simulator": {
            "split": "target",
            "robocasa_repo": str(robocasa_repo.resolve()),
            "robocasa_commit": robocasa_commit,
            "robosuite_repo": str(robosuite_repo.resolve()),
            "robosuite_commit": robosuite_commit,
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
            "initialization_mode": "seeded_target_env",
            "seed_rule": "base_seed + within_shard_task_index * 10000 + episode_index",
            "causal_prompt_only": True,
            "policy_sampling": {
                "sampler": "unipc",
                "seed": 42,
                "guidance": 1.0,
                "num_steps": 30,
                "reuse_seed_per_request": True,
                "use_ema_weights": True,
            },
            "effect_thresholds": {
                "minimum_mean_success_rate": 0.0,
                "minimum_successful_tasks": 0,
            },
        },
        "artifacts": {
            "source_manifest": source_manifest,
            "cosmos_server_module": cosmos_server_module,
            "checkpoint_manifest": checkpoint,
            "checkpoint_verification_receipt": checkpoint_verification,
            "training_config": config,
            "action_stats": stats,
            "cosmos_environment_manifest": cosmos_environment,
            "robocasa_environment_manifest": robocasa_environment,
            "eval_runtime_receipt": eval_runtime_receipt,
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
        "checkpoint": checkpoint_path,
        "results_dir": str((tmp_path / "results").resolve()),
        "run_name": "gate",
        "eval_dataset": None,
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
    assert result["artifacts"]["raw_shards"][0]["shard"] == 0
    assert result["artifacts"]["raw_shards"][0]["args"]["sha256"] == _sha256(results_dir / "args.json")
    assert result["artifacts"]["raw_shards"][0]["results"]["sha256"] == _sha256(results_dir / "results_shard0.json")
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


def test_evidence_source_hard_pins_the_audited_demian_revision() -> None:
    source = Path(evidence.__file__).read_text(encoding="utf-8")

    for value in (
        "2600ece5bf0eeacf010a95f084dcda2de6d30d59",
        "c38ed1f4049d079b84c185cb572667a47eaf984c2c22b492efc8985401119836",
        "e902311cf435986859ae01920548aba169e085ad6d26187cad9fb0d1f9c315de",
    ):
        assert value in source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("commit", "9" * 40),
        ("evaluator_sha256", "9" * 64),
        ("task_registry_sha256", "9" * 64),
    ],
)
def test_contract_rejects_non_audited_demian_identity(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["demian"][field] = value
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="audited DeMiAn revision"):
        evidence._validate_contract(contract_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda receipt: receipt.update(unexpected=True), "keys mismatch"),
        (lambda receipt: receipt.update(file_count=2), "file_count does not match"),
    ],
)
def test_checkpoint_verification_receipt_is_exact_and_matches_manifest_count(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    _mutate_contract_artifact(
        contract_path,
        "checkpoint_verification_receipt",
        mutation,
    )

    with pytest.raises(ValueError, match=match):
        evidence._validate_contract(contract_path)


def test_runtime_receipt_must_bind_the_exact_clean_source_commit(tmp_path: Path) -> None:
    contract_path, _results_dir = _fixture(tmp_path)

    def mutate(receipt: dict[str, object]) -> None:
        inputs = receipt["inputs"]
        assert isinstance(inputs, dict)
        sources = inputs["sources"]
        assert isinstance(sources, dict)
        demian = sources["demian"]
        assert isinstance(demian, dict)
        demian["actual_commit"] = "9" * 40

    _mutate_contract_artifact(contract_path, "eval_runtime_receipt", mutate)

    with pytest.raises(ValueError, match="clean expected source commit"):
        evidence._validate_contract(contract_path)


def test_runtime_receipt_must_bind_the_exact_environment_manifest(tmp_path: Path) -> None:
    contract_path, _results_dir = _fixture(tmp_path)

    def mutate(receipt: dict[str, object]) -> None:
        inputs = receipt["inputs"]
        assert isinstance(inputs, dict)
        manifests = inputs["environment_manifests"]
        assert isinstance(manifests, dict)
        cosmos = manifests["cosmos"]
        assert isinstance(cosmos, dict)
        cosmos["expected_sha256"] = "9" * 64

    _mutate_contract_artifact(contract_path, "eval_runtime_receipt", mutate)

    with pytest.raises(ValueError, match="SHA256 does not match"):
        evidence._validate_contract(contract_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda receipt: receipt["inputs"].update(logical_run_id="different-run"),
            "run identity does not match",
        ),
        (
            lambda receipt: receipt["driver_environment"].update(ld_library_path="/tmp/untrusted"),
            "exact audited H driver library path",
        ),
        (
            lambda receipt: receipt["robocasa_egl"][0].update(mujoco_egl_device_id="1"),
            "exact per-shard EGL environment",
        ),
        (
            lambda receipt: receipt["robocasa_runtime"]["packages"]["numpy"].update(distribution_version="9.9"),
            "does not match the environment manifest",
        ),
    ],
)
def test_runtime_receipt_binds_run_driver_egl_and_package_inventory(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    _mutate_contract_artifact(contract_path, "eval_runtime_receipt", mutation)

    with pytest.raises(ValueError, match=match):
        evidence._validate_contract(contract_path)


def test_runtime_module_paths_must_stay_in_their_bound_environment_prefix(
    tmp_path: Path,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    outside = _write_json(tmp_path / "outside" / "torch" / "__init__.py", {"module": "torch"})

    def mutate(receipt: dict[str, object]) -> None:
        cosmos_runtime = receipt["cosmos_runtime"]
        assert isinstance(cosmos_runtime, dict)
        torch = cosmos_runtime["torch"]
        assert isinstance(torch, dict)
        torch["module_path"] = str(outside.resolve())

    _mutate_contract_artifact(contract_path, "eval_runtime_receipt", mutate)

    with pytest.raises(ValueError, match="outside its environment prefix"):
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
