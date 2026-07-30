# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for formal RoboCasa365 evaluation evidence generation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from examples.h_cluster import cwam_robocasa365_eval_evidence as evidence


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    prior_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    if prior_mode is not None and not prior_mode & stat.S_IWUSR:
        path.chmod(prior_mode | stat.S_IWUSR)
    try:
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    finally:
        if prior_mode is not None:
            path.chmod(prior_mode)
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
    interpreter = base_prefix / f"{prefix.name}-toolchain" / "python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)
    launcher = prefix / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    os.symlink(interpreter, launcher)
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


def _rewrite_submission_and_runtime_binding(
    contract_path: Path,
    mutation,
) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    submission_path = Path(contract["artifacts"]["submission_verification_receipt"]["path"])
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    mutation(submission)
    _write_json(submission_path, submission)
    submission_sha256 = _sha256(submission_path)
    contract["artifacts"]["submission_verification_receipt"]["sha256"] = submission_sha256

    runtime_path = Path(contract["artifacts"]["eval_runtime_receipt"]["path"])
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    descriptor = runtime["inputs"]["submission_verification_receipt"]
    descriptor["size_bytes"] = submission_path.stat().st_size
    descriptor["sha256"] = submission_sha256
    descriptor["expected_sha256"] = submission_sha256
    _write_json(runtime_path, runtime)
    contract["artifacts"]["eval_runtime_receipt"]["sha256"] = _sha256(runtime_path)
    _write_json(contract_path, contract)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    logical_run_id = "CWAM-E35-R12-GATE-S42"
    results_dir = tmp_path / "results" / logical_run_id
    results_dir.mkdir(parents=True)
    cosmos_repo = tmp_path / "cosmos"
    demian_repo = tmp_path / "demian"
    robocasa_repo = tmp_path / "robocasa"
    robosuite_repo = tmp_path / "robosuite"
    base_prefix = tmp_path / "base-python"
    base_prefix.mkdir()
    for repo in (cosmos_repo, demian_repo, robocasa_repo, robosuite_repo):
        repo.mkdir()
    source_manifest = _path_artifact(
        cosmos_repo / "SNAPSHOT_MANIFEST.json",
        {"files": 1},
    )
    source_checkpoint_root = tmp_path / "checkpoints" / "iter_000010000"
    checkpoint_root = source_checkpoint_root / "model"
    checkpoint_root.mkdir(parents=True)
    checkpoint_path = str(checkpoint_root.resolve())
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
    cosmos_python_launch = str((cosmos_prefix / "bin" / "python").absolute())
    robocasa_python_launch = str((robocasa_prefix / "bin" / "python").absolute())
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

    submitted_at_utc = "2026-07-28T18:55:00Z"
    capture_nonce = "e" * 64
    submission_verification = _artifact(
        tmp_path,
        "submission-verification",
        {
            "schema_version": 1,
            "kind": "cwam_robocasa365_eval_submission_verification",
            "status": "passed",
            "verified_at_utc": "2026-07-28T19:01:00Z",
            "identity": {
                "experiment_id": "CWAM-E35-R12-GATE",
                "logical_run_id": logical_run_id,
                "run_name": logical_run_id,
                "submission_nonce": capture_nonce,
                "submitted_at_utc": submitted_at_utc,
            },
            "freshness": {
                "age_seconds": 360.0,
                "expires_at_utc": "2026-07-31T18:55:00Z",
                "max_age_seconds": 259200,
                "max_future_skew_seconds": 300,
                "scheduler_started_at_utc": "2026-07-28T19:00:00Z",
                "start_age_seconds": 60.0,
                "max_start_to_verification_seconds": 7200,
            },
            "evaluation": {
                "stage": "one_episode_gate",
                "split": "atomic",
                "tasks_csv": "OpenDrawer",
                "episodes_per_task": 1,
                "num_shards": 1,
                "base_seed": 42,
                "prior_gate_evidence": None,
            },
            "scheduler": {
                "rjob_name": "cwam-e35-r12-gate-abc123",
                "rjob_uid": "12345678-1234-4123-8123-123456789abc",
                "showname": "cwam-e35-r12-gate",
                "worker_command_sha256": "a" * 64,
                "expected": {
                    "cpu": 16,
                    "memory_mib": 187_500,
                    "gpu": 1,
                },
            },
            "reservation": {},
            "scheduler_readback_receipt": {},
            "scheduler_artifacts": {},
            "validation": {
                "submit_identity_matches": True,
                "freshness_window_passed": True,
                "scheduler_readback_matches": True,
                "scheduler_artifact_hashes_match": True,
                "submission_unknown_absent": True,
            },
        },
    )

    provenance_dir = robocasa_repo / ".cwam_provenance"
    asset_manifest_path = _write_json(
        provenance_dir / "asset-tree-manifest.jsonl",
        {"payload": "manifest"},
    )
    publisher_receipt_path = _write_json(
        provenance_dir / "asset-tree-receipt.json",
        {"payload": "publisher-receipt"},
    )
    asset_manifest_path.chmod(evidence.cwam_robocasa_asset_manifest.PROVENANCE_FILE_MODE)
    publisher_receipt_path.chmod(evidence.cwam_robocasa_asset_manifest.PROVENANCE_FILE_MODE)
    asset_verification = _artifact(
        tmp_path,
        "asset-verification",
        {
            "schema_version": 1,
            "kind": "cwam_robocasa_asset_tree_verification_receipt",
            "status": "passed",
            "asset_root": str(robocasa_repo.resolve()),
            "asset_root_mode": 0o755,
            "excluded_top_level_paths": [".git", ".cwam_provenance"],
            "manifest_path": str(asset_manifest_path.resolve()),
            "manifest_size_bytes": asset_manifest_path.stat().st_size,
            "manifest_sha256": _sha256(asset_manifest_path),
            "publisher_receipt_path": str(publisher_receipt_path.resolve()),
            "publisher_receipt_size_bytes": publisher_receipt_path.stat().st_size,
            "publisher_receipt_sha256": _sha256(publisher_receipt_path),
            "publisher_generator_id": "fixture",
            "publisher_generated_at_utc": "2026-07-27T00:00:00Z",
            "robocasa_git_commit": robocasa_commit,
            "payload_directory_count": 1,
            "payload_file_count": 1,
            "payload_symlink_count": 0,
            "payload_total_bytes": 1,
            "source_archive_directory": "/source/archive",
            "source_sha256sums_sha256": "b" * 64,
            "source_sizes_sha256": "c" * 64,
        },
    )

    def bound_receipt_descriptor(
        artifact: dict[str, str],
        kind: str,
    ) -> dict[str, object]:
        return {
            "path": artifact["path"],
            "size_bytes": Path(artifact["path"]).stat().st_size,
            "sha256": artifact["sha256"],
            "expected_sha256": artifact["sha256"],
            "kind": kind,
            "schema_version": 1,
        }

    graphics_root = tmp_path / "graphics"
    vendor_json_path = _write_json(
        graphics_root / "10_nvidia.json",
        {
            "file_format_version": "1.0.0",
            "ICD": {"library_path": "libEGL_nvidia.so.0"},
        },
    )
    evidence._EXPECTED_EGL_VENDOR_JSON = str(vendor_json_path.resolve())

    def graphics_file_descriptor(path: Path) -> dict[str, object]:
        return {
            "supplied_path": str(path.resolve()),
            "canonical_path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    library_contracts = {
        "libEGL": ("libEGL.so.1", "eglGetDisplay", "egl_context_mapped", True),
        "libEGL_nvidia": (
            "libEGL_nvidia.so.0",
            "eglQueryString",
            "egl_context_mapped",
            True,
        ),
        "libGL": (
            "libGL.so.1",
            "glGetString",
            "resolved_and_dlopen_verified",
            False,
        ),
        "libGLX": (
            "libGLX.so.0",
            "glXGetProcAddress",
            "resolved_and_dlopen_verified",
            False,
        ),
        "libGLX_nvidia": (
            "libGLX_nvidia.so.0",
            "glXGetProcAddress",
            "resolved_and_dlopen_verified",
            False,
        ),
    }
    graphics_libraries: dict[str, dict[str, object]] = {}
    for name, (soname, symbol, evidence_class, mapped) in library_contracts.items():
        library_path = graphics_root / soname
        library_path.write_bytes(f"{name}-fixture".encode())
        graphics_libraries[name] = {
            **graphics_file_descriptor(library_path),
            "name": name,
            "requested_soname": soname,
            "symbol": symbol,
            "evidence_class": evidence_class,
            "was_mapped_after_egl_context": mapped,
        }
    graphics_userspace = {
        "egl_vendor_json": {
            **graphics_file_descriptor(vendor_json_path),
            "file_format_version": "1.0.0",
            "icd_library_path": "libEGL_nvidia.so.0",
        },
        "libraries": graphics_libraries,
        "gl_identity": {
            "vendor": "NVIDIA Corporation",
            "renderer": "NVIDIA H200",
        },
    }

    source_dcp_export_receipt = _artifact(
        tmp_path,
        "source-dcp-export-receipt",
        {
            "schema_version": 1,
            "kind": "cwam_serving_dcp_selection_receipt",
            "status": "passed",
            "source_checkpoint_path": str(source_checkpoint_root.resolve()),
            "source_checkpoint_iteration": 10_000,
            "serving_checkpoint_path": checkpoint_path,
            "checkpoint_manifest_sha256": checkpoint["sha256"],
            "checkpoint_tree_sha256": checkpoint_tree_sha,
            "validation": {
                "source_iteration_path_matches": True,
                "serving_checkpoint_is_exact_model_child": True,
                "serving_checkpoint_manifest_matches": True,
            },
        },
    )
    Path(source_dcp_export_receipt["path"]).chmod(0o444)

    def file_descriptor(artifact: dict[str, str]) -> dict[str, object]:
        return {
            "path": artifact["path"],
            "size_bytes": Path(artifact["path"]).stat().st_size,
            "sha256": artifact["sha256"],
        }

    def tracked_repository_report(root: Path, commit: str) -> dict[str, object]:
        return {
            "root": str(root.resolve()),
            "expected_commit": commit,
            "actual_commit": commit,
            "clean": True,
            "tracked_files": {
                "object_format": "sha1",
                "head_tree_oid": "d" * 40,
                "tracked_entry_count": 1,
                "tracked_regular_file_count": 1,
                "tracked_non_regular_entry_count": 0,
                "tracked_regular_total_bytes": 1,
                "tracked_regular_manifest_sha256": "f" * 64,
                "all_tracked_regular_files_match_head_blobs": True,
            },
        }

    repository_preflight = {
        "demian": tracked_repository_report(demian_repo, demian_commit),
        "robocasa": tracked_repository_report(robocasa_repo, robocasa_commit),
        "robosuite": tracked_repository_report(robosuite_repo, robosuite_commit),
    }
    asset_verification_value = json.loads(Path(asset_verification["path"]).read_text(encoding="utf-8"))
    preflight = {
        "asset_tree": asset_verification_value,
        "repositories": repository_preflight,
        "validation": {
            "asset_tree_full_hash_match": True,
            "repository_heads_clean": True,
            "tracked_regular_files_match_head_blobs": True,
        },
    }
    preflight_sha256 = hashlib.sha256(
        json.dumps(
            preflight,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    intent = _artifact(
        tmp_path,
        "eval-intent",
        {
            "schema_version": 1,
            "kind": "cwam_robocasa365_eval_intent",
            "identity": {
                "experiment_id": "CWAM-E35-R12-GATE",
                "logical_run_id": logical_run_id,
                "run_name": logical_run_id,
                "stage": "one_episode_gate",
                "evaluation_class": "exploratory",
            },
            "protocol": {
                "split": "atomic",
                "tasks": ["OpenDrawer"],
                "task_count": 1,
                "episodes_per_task": 1,
                "expected_episode_count": 1,
                "num_shards": 1,
                "base_seed": 42,
                "max_steps": 400,
                "open_loop_horizon": 12,
                "policy_sampler": "unipc",
                "policy_seed": 42,
                "policy_guidance": 1.0,
                "policy_num_steps": 30,
                "base_port": 16000,
                "server_ready_timeout_seconds": 1800,
            },
            "thresholds": {
                "minimum_mean_success_rate": 0.0,
                "minimum_successful_tasks": 0,
            },
            "source": {
                "root": str(cosmos_repo.resolve()),
                "commit": source_commit,
                "git_pushed": False,
                "manifest": file_descriptor(source_manifest),
            },
            "serving_checkpoint": {
                "path": checkpoint_path,
                "manifest": file_descriptor(checkpoint),
                "tree_sha256": checkpoint_tree_sha,
                "source_dcp_export_receipt": file_descriptor(source_dcp_export_receipt),
            },
            "artifacts": {
                "training_config": file_descriptor(config),
                "action_stats": file_descriptor(stats),
                "cosmos_environment": {
                    "root": str(cosmos_prefix.resolve()),
                    "manifest": file_descriptor(cosmos_environment),
                },
                "robocasa_environment": {
                    "root": str(robocasa_prefix.resolve()),
                    "manifest": file_descriptor(robocasa_environment),
                },
                "robocasa_assets": {
                    "root": str(robocasa_repo.resolve()),
                    "manifest": {
                        "path": str(asset_manifest_path.resolve()),
                        "size_bytes": asset_manifest_path.stat().st_size,
                        "sha256": _sha256(asset_manifest_path),
                    },
                    "receipt": {
                        "path": str(publisher_receipt_path.resolve()),
                        "size_bytes": publisher_receipt_path.stat().st_size,
                        "sha256": _sha256(publisher_receipt_path),
                    },
                },
            },
            "repositories": {
                "demian": {"root": str(demian_repo.resolve()), "commit": demian_commit},
                "robocasa": {
                    "root": str(robocasa_repo.resolve()),
                    "commit": robocasa_commit,
                },
                "robosuite": {
                    "root": str(robosuite_repo.resolve()),
                    "commit": robosuite_commit,
                },
            },
            "runtime": {
                "container_image": evidence._EXPECTED_CONTAINER_IMAGE,
                "egl_vendor_json": {
                    "path": str(vendor_json_path.resolve()),
                    "size_bytes": vendor_json_path.stat().st_size,
                    "sha256": _sha256(vendor_json_path),
                },
            },
            "resources": {
                "cpu": 16,
                "memory_mib": 187_500,
                "gpu": 1,
                "host_network": False,
                "quota_group": "eailabagent_gpu",
                "negative_tag": "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn",
                "task_type": "normal",
                "private_machine": "group",
                "priority": 9,
                "restart_policy": "never",
                "backoff_limit": 1,
                "replicas": 1,
            },
            "prior_gate": None,
        },
    )
    Path(intent["path"]).chmod(0o444)

    submission_value = json.loads(Path(submission_verification["path"]).read_text(encoding="utf-8"))
    submission_value["eval_intent_manifest"] = file_descriptor(intent)
    submission_value["preflight"] = preflight
    submission_value["preflight_sha256"] = preflight_sha256
    submission_value["scheduler"]["expected"] = {
        "showname": "cwam-e35-r12-gate",
        "quota_group": "eailabagent_gpu",
        "private_machine": "group",
        "task_type": "normal",
        "priority": 9,
        "restart_policy": "Never",
        "termination_grace_period_seconds": 60,
        "backoff_limit": 1,
        "replicas": 1,
        "image": evidence._EXPECTED_CONTAINER_IMAGE,
        "cpu": 16,
        "memory_mib": 187_500,
        "gpu": 1,
        "host_network": False,
        "negative_tags": ["node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"],
        "store_host_nvme": True,
        "mounts": evidence._EXPECTED_SUBMISSION_MOUNTS,
    }
    submission_value["validation"]["eval_intent_manifest_matches"] = True
    submission_value["validation"]["pre_submission_full_inputs_passed"] = True
    _write_json(Path(submission_verification["path"]), submission_value)
    submission_verification["sha256"] = _sha256(Path(submission_verification["path"]))

    eval_runtime_receipt = _artifact(
        tmp_path,
        "eval-runtime-receipt",
        {
            "schema_version": 2,
            "kind": "cwam_robocasa365_eval_runtime_receipt",
            "status": "passed",
            "capture_started_at_utc": "2026-07-28T19:02:00Z",
            "captured_at_utc": "2026-07-28T19:02:03Z",
            "driver_environment": {
                "ld_library_path": evidence._EXPECTED_H_DRIVER_LIBRARY_PATH,
                "egl_vendor_library_filenames": str(vendor_json_path.resolve()),
                "glx_vendor_library_name": "nvidia",
                "forbidden_environment_cleared": list(evidence._EXPECTED_FORBIDDEN_GRAPHICS_ENVIRONMENT),
            },
            "inputs": {
                "experiment_id": "CWAM-E35-R12-GATE",
                "logical_run_id": logical_run_id,
                "capture_nonce": capture_nonce,
                "submitted_at_utc": submitted_at_utc,
                "num_gpus": 1,
                "submission_verification_receipt": bound_receipt_descriptor(
                    submission_verification,
                    "cwam_robocasa365_eval_submission_verification",
                ),
                "asset_verification_receipt": bound_receipt_descriptor(
                    asset_verification,
                    "cwam_robocasa_asset_tree_verification_receipt",
                ),
                "interpreters": {
                    "cosmos": {
                        "launch_path": cosmos_python_launch,
                        "resolved_path": cosmos_python,
                    },
                    "robocasa": {
                        "launch_path": robocasa_python_launch,
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
                    "egl_vendor_library_filenames": str(vendor_json_path.resolve()),
                    "glx_vendor_library_name": "nvidia",
                    "robosuite_module_path": str(robosuite_module.resolve()),
                    "graphics_userspace": graphics_userspace,
                }
            ],
            "graphics_userspace": graphics_userspace,
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
                "submission_identity_and_freshness_match": True,
                "asset_tree_verification_is_bound": True,
                "graphics_userspace_files_match_live_hashes": True,
                "nvidia_gl_vendor_and_renderer_match": True,
            },
        },
    )
    contract = {
        "schema_version": 1,
        "kind": "cwam_robocasa365_formal_eval_contract",
        "created_at": "2026-07-28T19:02:04Z",
        "experiment_id": "CWAM-E35-R12-GATE",
        "logical_run_id": logical_run_id,
        "source_commit": source_commit,
        "source_root": str(cosmos_repo.resolve()),
        "source_manifest_sha256": source_manifest_sha,
        "checkpoint_path": checkpoint_path,
        "run_name": logical_run_id,
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
            "prior_gate_evidence": None,
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
            "source_dcp_export_receipt": source_dcp_export_receipt,
            "training_config": config,
            "action_stats": stats,
            "cosmos_environment_manifest": cosmos_environment,
            "robocasa_environment_manifest": robocasa_environment,
            "eval_intent_manifest": intent,
            "submission_verification_receipt": submission_verification,
            "asset_verification_receipt": asset_verification,
            "eval_runtime_receipt": eval_runtime_receipt,
            "demian_evaluator": demian_evaluator,
            "demian_task_registry": demian_task_registry,
            "server_runtime_receipts": [receipt_artifact],
        },
    }
    contract_path = _write_json(tmp_path / "contract.json", contract)
    contract_path.chmod(0o444)
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
        "run_name": logical_run_id,
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


def _patch_post_rollout_verifiers(
    monkeypatch: pytest.MonkeyPatch,
    contract_path: Path,
) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        evidence,
        "_POST_ROLLOUT_SNAPSHOT_ALLOWED_ROOT",
        Path(contract["source_root"]).parent,
    )
    checkpoint_manifest_path = Path(contract["artifacts"]["checkpoint_manifest"]["path"])
    checkpoint_receipt_path = Path(contract["artifacts"]["checkpoint_verification_receipt"]["path"])
    asset_receipt_path = Path(contract["artifacts"]["asset_verification_receipt"]["path"])

    def verify_snapshot(bound_contract: dict[str, object]) -> dict[str, object]:
        assert bound_contract["source_root"] == contract["source_root"]
        assert bound_contract["source_manifest_sha256"] == contract["source_manifest_sha256"]
        return {
            "schema_version": 1,
            "status": "passed",
            "snapshot_root": str(Path(contract["source_root"]).resolve()),
            "snapshot_manifest": contract["artifacts"]["source_manifest"]["path"],
            "snapshot_manifest_sha256": contract["source_manifest_sha256"],
            "file_count": 1,
            "file_bytes": 1,
        }

    def load_checkpoint_manifest(path: str | Path) -> dict[str, object]:
        assert Path(path) == checkpoint_manifest_path
        return json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))

    def verify_checkpoint(
        manifest: dict[str, object],
        checkpoint_path: str | Path,
    ) -> dict[str, object]:
        assert manifest == json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        assert Path(checkpoint_path) == Path(contract["checkpoint_path"])
        return json.loads(checkpoint_receipt_path.read_text(encoding="utf-8"))

    def verify_assets(bound_contract: dict[str, object]) -> dict[str, object]:
        assert bound_contract["simulator"] == contract["simulator"]
        return json.loads(asset_receipt_path.read_text(encoding="utf-8"))

    def verify_git_repository(
        *,
        name: str,
        root: Path,
        expected_commit: str,
        verify_tracked_files: bool,
    ) -> dict[str, object]:
        assert name != "Cosmos"
        assert verify_tracked_files is True
        tracked_files: dict[str, object] = {
            "object_format": "sha1",
            "head_tree_oid": "a" * 40,
            "tracked_entry_count": 1,
            "tracked_regular_file_count": 1,
            "tracked_non_regular_entry_count": 0,
            "tracked_regular_total_bytes": 1,
            "tracked_regular_manifest_sha256": "b" * 64,
            "all_tracked_regular_files_match_head_blobs": True,
        }
        return {
            "root": str(root.resolve()),
            "expected_commit": expected_commit,
            "actual_commit": expected_commit,
            "clean": True,
            "tracked_files": tracked_files,
        }

    monkeypatch.setattr(
        evidence,
        "_verify_snapshot_with_bound_manifest",
        verify_snapshot,
    )
    monkeypatch.setattr(
        evidence.cwam_serving_checkpoint_manifest,
        "load_serving_checkpoint_manifest",
        load_checkpoint_manifest,
    )
    monkeypatch.setattr(
        evidence.cwam_serving_checkpoint_manifest,
        "verify_serving_checkpoint_manifest",
        verify_checkpoint,
    )
    monkeypatch.setattr(
        evidence,
        "_verify_asset_tree_with_bound_provenance",
        verify_assets,
    )
    monkeypatch.setattr(evidence, "_verify_git_repository", verify_git_repository)


def _mock_post_rollout_receipt(
    monkeypatch: pytest.MonkeyPatch,
    contract_path: Path,
    output: Path,
) -> Path:
    _patch_post_rollout_verifiers(monkeypatch, contract_path)
    evidence.verify_post_rollout(contract_path=contract_path, output=output)
    return output


def _gate_evidence_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object]]:
    contract_path, results_dir = _fixture(tmp_path)
    post_receipt = _mock_post_rollout_receipt(
        monkeypatch,
        contract_path,
        tmp_path / "post-rollout-receipt.json",
    )
    gate_evidence_path = results_dir / "evaluation_evidence.json"
    result = evidence.summarize(
        contract_path=contract_path,
        post_rollout_verification_receipt=post_receipt,
        results_dir=results_dir,
        summary_csv=results_dir / "summary.csv",
        task_success_rates_csv=results_dir / "task_success_rates.csv",
        evidence_json=gate_evidence_path,
        require_effect_pass=True,
    )
    assert result["prior_gate_evidence"] is None
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    asset_verification = json.loads(
        Path(contract["artifacts"]["asset_verification_receipt"]["path"]).read_text(encoding="utf-8")
    )
    runtime_path = Path(contract["artifacts"]["eval_runtime_receipt"]["path"])
    for path in (
        contract_path,
        runtime_path,
        gate_evidence_path,
        results_dir / "summary.csv",
        results_dir / "task_success_rates.csv",
        results_dir / "args.json",
        results_dir / "results_shard0.json",
    ):
        path.chmod(0o444)
    arguments: dict[str, object] = {
        "gate_evidence_path": gate_evidence_path,
        "expected_gate_evidence_sha256": _sha256(gate_evidence_path),
        "current_source_root": Path(contract["source_root"]),
        "current_source_commit": contract["source_commit"],
        "current_source_manifest_sha256": contract["source_manifest_sha256"],
        "current_checkpoint_path": Path(contract["checkpoint_path"]),
        "current_checkpoint_manifest_sha256": contract["artifacts"]["checkpoint_manifest"]["sha256"],
        "current_source_dcp_export_receipt_sha256": contract["artifacts"]["source_dcp_export_receipt"]["sha256"],
        "current_asset_manifest_sha256": asset_verification["manifest_sha256"],
        "current_asset_publisher_receipt_sha256": asset_verification["publisher_receipt_sha256"],
        "current_training_config_sha256": contract["artifacts"]["training_config"]["sha256"],
        "current_action_stats_sha256": contract["artifacts"]["action_stats"]["sha256"],
        "current_cosmos_environment_manifest_sha256": contract["artifacts"]["cosmos_environment_manifest"]["sha256"],
        "current_robocasa_environment_manifest_sha256": contract["artifacts"]["robocasa_environment_manifest"][
            "sha256"
        ],
        "current_demian_commit": contract["demian"]["commit"],
        "current_robocasa_commit": contract["simulator"]["robocasa_commit"],
        "current_robosuite_commit": contract["simulator"]["robosuite_commit"],
    }
    return contract_path, gate_evidence_path, arguments


def test_summarize_writes_complete_episode_and_task_csvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, results_dir = _fixture(tmp_path)
    post_receipt = _mock_post_rollout_receipt(
        monkeypatch,
        contract,
        tmp_path / "post-rollout-receipt.json",
    )
    summary_csv = tmp_path / "summary.csv"
    task_csv = tmp_path / "task_success_rates.csv"
    evidence_json = tmp_path / "evaluation_evidence.json"

    result = evidence.summarize(
        contract_path=contract,
        post_rollout_verification_receipt=post_receipt,
        results_dir=results_dir,
        summary_csv=summary_csv,
        task_success_rates_csv=task_csv,
        evidence_json=evidence_json,
        require_effect_pass=True,
    )

    assert result["execution_complete"] is True
    assert result["effect_pass"] is True
    assert result["evaluation"]["evaluation_class"] == "exploratory"
    assert result["evaluation"]["source_git_pushed_assertion"] is False
    assert result["external_execution_binding"]["submission"]["scheduler_rjob_name"] == "cwam-e35-r12-gate-abc123"
    assert result["external_execution_binding"]["assets"]["payload_symlink_count"] == 0
    assert result["external_execution_binding"]["graphics_userspace"]["gl_identity"]["vendor"] == "NVIDIA Corporation"
    assert (
        result["external_execution_binding"]["post_rollout_live_inputs"]["validation"][
            "tracked_source_files_match_head_blobs"
        ]
        is True
    )
    assert result["external_execution_binding"]["post_rollout_live_inputs"]["receipt"]["sha256"] == _sha256(
        post_receipt
    )
    assert result["artifacts"]["raw_shards"][0]["shard"] == 0
    assert result["artifacts"]["raw_shards"][0]["args"]["sha256"] == _sha256(results_dir / "args.json")
    assert result["artifacts"]["raw_shards"][0]["results"]["sha256"] == _sha256(results_dir / "results_shard0.json")
    assert stat.S_IMODE(summary_csv.stat().st_mode) == 0o444
    assert stat.S_IMODE(task_csv.stat().st_mode) == 0o444
    assert stat.S_IMODE(evidence_json.stat().st_mode) == 0o444
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
    "existing_name",
    ("summary.csv", "task_success_rates.csv", "evaluation_evidence.json"),
)
def test_summarize_refuses_to_overwrite_terminal_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_name: str,
) -> None:
    contract, results_dir = _fixture(tmp_path)
    post_receipt = _mock_post_rollout_receipt(
        monkeypatch,
        contract,
        tmp_path / "post-rollout-receipt.json",
    )
    outputs = {
        "summary.csv": tmp_path / "summary.csv",
        "task_success_rates.csv": tmp_path / "task_success_rates.csv",
        "evaluation_evidence.json": tmp_path / "evaluation_evidence.json",
    }
    existing = outputs[existing_name]
    existing.write_bytes(b"do-not-overwrite\n")

    with pytest.raises(FileExistsError, match="overwrite"):
        evidence.summarize(
            contract_path=contract,
            post_rollout_verification_receipt=post_receipt,
            results_dir=results_dir,
            summary_csv=outputs["summary.csv"],
            task_success_rates_csv=outputs["task_success_rates.csv"],
            evidence_json=outputs["evaluation_evidence.json"],
            require_effect_pass=False,
        )
    assert existing.read_bytes() == b"do-not-overwrite\n"


def test_summarize_rejects_symlinked_raw_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, results_dir = _fixture(tmp_path)
    post_receipt = _mock_post_rollout_receipt(
        monkeypatch,
        contract,
        tmp_path / "post-rollout-receipt.json",
    )
    args_path = results_dir / "args.json"
    args_target = results_dir / "args-target.json"
    args_path.rename(args_target)
    args_path.symlink_to(args_target.name)

    with pytest.raises(ValueError, match="non-symlink"):
        evidence.summarize(
            contract_path=contract,
            post_rollout_verification_receipt=post_receipt,
            results_dir=results_dir,
            summary_csv=tmp_path / "summary.csv",
            task_success_rates_csv=tmp_path / "task_success_rates.csv",
            evidence_json=tmp_path / "evaluation_evidence.json",
            require_effect_pass=False,
        )


def test_summarize_rejects_raw_path_replacement_during_fd_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, results_dir = _fixture(tmp_path)
    post_receipt = _mock_post_rollout_receipt(
        monkeypatch,
        contract,
        tmp_path / "post-rollout-receipt.json",
    )
    args_path = results_dir / "args.json"
    replacement = results_dir / "replacement.json"
    replacement.write_bytes(args_path.read_bytes())
    real_read = os.read
    replaced = False
    args_identity = (args_path.stat().st_dev, args_path.stat().st_ino)

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        payload = real_read(descriptor, count)
        opened = os.fstat(descriptor)
        if not replaced and (opened.st_dev, opened.st_ino) == args_identity:
            replaced = True
            os.replace(replacement, args_path)
        return payload

    monkeypatch.setattr(evidence.os, "read", racing_read)
    with pytest.raises(RuntimeError, match="changed or was replaced"):
        evidence.summarize(
            contract_path=contract,
            post_rollout_verification_receipt=post_receipt,
            results_dir=results_dir,
            summary_csv=tmp_path / "summary.csv",
            task_success_rates_csv=tmp_path / "task_success_rates.csv",
            evidence_json=tmp_path / "evaluation_evidence.json",
            require_effect_pass=False,
        )


def test_verify_sealed_evidence_reaggregates_and_publishes_exclusive_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, evidence_json, _arguments = _gate_evidence_fixture(
        tmp_path,
        monkeypatch,
    )
    results_dir = evidence_json.parent
    results_dir.chmod(0o555)
    output = tmp_path / "sealed-evidence-verification.json"
    keyword_args = {
        "contract_path": contract_path,
        "post_rollout_verification_receipt": tmp_path / "post-rollout-receipt.json",
        "results_dir": results_dir,
        "summary_csv": results_dir / "summary.csv",
        "task_success_rates_csv": results_dir / "task_success_rates.csv",
        "evidence_json": evidence_json,
        "output": output,
    }

    receipt = evidence.verify_sealed_evidence(**keyword_args)

    assert receipt["status"] == "passed"
    assert receipt["kind"] == evidence._SEALED_EVIDENCE_VERIFICATION_KIND
    assert receipt["evaluation"]["execution_complete"] is True
    assert receipt["evaluation"]["effect_pass"] is True
    assert all(receipt["validation"].values())
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    assert stat.S_IMODE(output.stat().st_mode) == 0o444

    original_sha256 = _sha256(output)
    with pytest.raises(FileExistsError, match="overwrite"):
        evidence.verify_sealed_evidence(**keyword_args)
    assert _sha256(output) == original_sha256


def test_verify_sealed_evidence_validates_the_contract_bytes_opened_before_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, evidence_json, _arguments = _gate_evidence_fixture(tmp_path, monkeypatch)
    results_dir = evidence_json.parent
    results_dir.chmod(0o555)
    valid_contract = contract_path.read_bytes()
    invalid_contract = json.loads(valid_contract)
    invalid_contract["unexpected"] = True
    _write_json(contract_path, invalid_contract)
    replacement = tmp_path / "valid-contract-replacement.json"
    replacement.write_bytes(valid_contract)
    replacement.chmod(0o444)
    opened_inode = contract_path.stat().st_ino
    real_stable_read = evidence._read_stable_regular_file
    replaced = False

    def swap_contract_after_stable_read(path: Path, label: str, **kwargs):
        nonlocal replaced
        payload, descriptor = real_stable_read(path, label, **kwargs)
        if label == "contract" and not replaced:
            replaced = True
            os.replace(replacement, contract_path)
        return payload, descriptor

    monkeypatch.setattr(evidence, "_read_stable_regular_file", swap_contract_after_stable_read)
    with pytest.raises(ValueError, match="contract keys mismatch"):
        evidence.verify_sealed_evidence(
            contract_path=contract_path,
            post_rollout_verification_receipt=tmp_path / "post-rollout-receipt.json",
            results_dir=results_dir,
            summary_csv=results_dir / "summary.csv",
            task_success_rates_csv=results_dir / "task_success_rates.csv",
            evidence_json=evidence_json,
            output=tmp_path / "sealed-evidence-verification.json",
        )
    assert replaced
    assert contract_path.stat().st_ino != opened_inode
    assert contract_path.read_bytes() == valid_contract


def test_verify_prior_gate_binds_terminal_gate_and_current_formal_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _contract_path, gate_evidence_path, arguments = _gate_evidence_fixture(
        tmp_path,
        monkeypatch,
    )

    def forbidden_full_tree_verification(*_args, **_kwargs):
        raise AssertionError("prior-gate verification must not repeat full checkpoint/asset hashing")

    monkeypatch.setattr(
        evidence.cwam_serving_checkpoint_manifest,
        "verify_serving_checkpoint_manifest",
        forbidden_full_tree_verification,
    )
    monkeypatch.setattr(
        evidence.cwam_robocasa_asset_manifest,
        "verify_robocasa_asset_tree",
        forbidden_full_tree_verification,
    )
    result = evidence.verify_prior_gate(**arguments)

    assert result["status"] == "passed"
    assert result["prior_gate_evidence"] == {
        "path": str(gate_evidence_path.resolve()),
        "size_bytes": gate_evidence_path.stat().st_size,
        "sha256": _sha256(gate_evidence_path),
    }
    assert result["gate_evaluation"] == {
        "stage": "one_episode_gate",
        "evaluation_class": "exploratory",
        "source_git_pushed_assertion": False,
        "task": "OpenDrawer",
        "tasks": 1,
        "episodes": 1,
        "episodes_per_task": 1,
        "num_shards": 1,
        "execution_complete": True,
        "effect_pass": True,
    }
    assert all(result["validation"].values())


def test_verify_prior_gate_validates_the_evidence_bytes_opened_before_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _contract_path, gate_evidence_path, arguments = _gate_evidence_fixture(tmp_path, monkeypatch)
    passing_evidence = gate_evidence_path.read_bytes()
    failing_evidence = json.loads(passing_evidence)
    failing_evidence["effect_pass"] = False
    _write_json(gate_evidence_path, failing_evidence)
    replacement = tmp_path / "passing-gate-evidence-replacement.json"
    replacement.write_bytes(passing_evidence)
    replacement.chmod(0o444)
    arguments["expected_gate_evidence_sha256"] = _sha256(gate_evidence_path)
    opened_inode = gate_evidence_path.stat().st_ino
    real_stable_read = evidence._read_stable_regular_file
    replaced = False

    def swap_evidence_after_stable_read(path: Path, label: str, **kwargs):
        nonlocal replaced
        payload, descriptor = real_stable_read(path, label, **kwargs)
        if label == "prior gate evidence" and not replaced:
            replaced = True
            os.replace(replacement, gate_evidence_path)
        return payload, descriptor

    monkeypatch.setattr(evidence, "_read_stable_regular_file", swap_evidence_after_stable_read)
    with pytest.raises(ValueError, match="effect_pass=true"):
        evidence.verify_prior_gate(**arguments)
    assert replaced
    assert gate_evidence_path.stat().st_ino != opened_inode
    assert gate_evidence_path.read_bytes() == passing_evidence


def test_contract_validates_the_bound_runtime_bytes_opened_before_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    runtime_path = Path(contract["artifacts"]["eval_runtime_receipt"]["path"])
    valid_runtime = runtime_path.read_bytes()
    invalid_runtime = json.loads(valid_runtime)
    invalid_runtime["status"] = "failed"
    _write_json(runtime_path, invalid_runtime)
    contract["artifacts"]["eval_runtime_receipt"]["sha256"] = _sha256(runtime_path)
    _write_json(contract_path, contract)
    replacement = tmp_path / "valid-runtime-replacement.json"
    replacement.write_bytes(valid_runtime)
    opened_inode = runtime_path.stat().st_ino
    real_stable_read = evidence._read_stable_regular_file
    replaced = False

    def swap_runtime_after_stable_read(path: Path, label: str, **kwargs):
        nonlocal replaced
        payload, descriptor = real_stable_read(path, label, **kwargs)
        if label == "contract.artifacts.eval_runtime_receipt" and not replaced:
            replaced = True
            os.replace(replacement, runtime_path)
        return payload, descriptor

    monkeypatch.setattr(evidence, "_read_stable_regular_file", swap_runtime_after_stable_read)
    with pytest.raises(ValueError, match="unsupported schema/kind/status"):
        evidence._validate_contract(contract_path)
    assert replaced
    assert runtime_path.stat().st_ino != opened_inode
    assert runtime_path.read_bytes() == valid_runtime


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda gate: gate.update(execution_complete=False),
            "execution_complete=true",
        ),
        (
            lambda gate: gate["evaluation"].update(stage="formal"),
            "not exactly one atomic task",
        ),
        (
            lambda gate: gate.update(prior_gate_evidence={"unexpected": True}),
            "must not recursively bind",
        ),
    ],
)
def test_verify_prior_gate_rejects_nonterminal_or_recursive_gate_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    match: str,
) -> None:
    _contract_path, gate_evidence_path, arguments = _gate_evidence_fixture(
        tmp_path,
        monkeypatch,
    )
    gate_evidence_path.chmod(0o644)
    gate = json.loads(gate_evidence_path.read_text(encoding="utf-8"))
    mutation(gate)
    _write_json(gate_evidence_path, gate)
    gate_evidence_path.chmod(0o444)
    arguments["expected_gate_evidence_sha256"] = _sha256(gate_evidence_path)

    with pytest.raises(ValueError, match=match):
        evidence.verify_prior_gate(**arguments)


def test_verify_prior_gate_rejects_hash_mode_and_current_input_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _contract_path, gate_evidence_path, arguments = _gate_evidence_fixture(
        tmp_path,
        monkeypatch,
    )

    with pytest.raises(ValueError, match="does not match"):
        evidence.verify_prior_gate(
            **{
                **arguments,
                "expected_gate_evidence_sha256": "0" * 64,
            }
        )

    gate_evidence_path.chmod(0o644)
    with pytest.raises(ValueError, match="mode mismatch"):
        evidence.verify_prior_gate(**arguments)
    gate_evidence_path.chmod(0o444)

    with pytest.raises(ValueError, match="inputs differ"):
        evidence.verify_prior_gate(
            **{
                **arguments,
                "current_training_config_sha256": "9" * 64,
            }
        )

    with pytest.raises(ValueError, match="inputs differ"):
        evidence.verify_prior_gate(
            **{
                **arguments,
                "current_asset_manifest_sha256": "8" * 64,
            }
        )


def test_contract_rejects_rehashed_intent_threshold_drift(tmp_path: Path) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    intent_path = Path(contract["artifacts"]["eval_intent_manifest"]["path"])
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["thresholds"]["minimum_mean_success_rate"] = 0.5
    _write_json(intent_path, intent)
    intent_path.chmod(0o444)
    intent_sha256 = _sha256(intent_path)
    contract["artifacts"]["eval_intent_manifest"]["sha256"] = intent_sha256
    _write_json(contract_path, contract)

    def rebind_intent(submission: dict[str, object]) -> None:
        descriptor = submission["eval_intent_manifest"]
        assert isinstance(descriptor, dict)
        descriptor["size_bytes"] = intent_path.stat().st_size
        descriptor["sha256"] = intent_sha256

    _rewrite_submission_and_runtime_binding(contract_path, rebind_intent)
    with pytest.raises(ValueError, match="intent thresholds differ"):
        evidence._validate_contract(contract_path)


def test_contract_rejects_non_json_source_dcp_receipt_even_when_rehashed(
    tmp_path: Path,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    receipt_path = Path(contract["artifacts"]["source_dcp_export_receipt"]["path"])
    prior_mode = stat.S_IMODE(receipt_path.stat().st_mode)
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(b"")
    receipt_path.chmod(prior_mode)
    contract["artifacts"]["source_dcp_export_receipt"]["sha256"] = _sha256(receipt_path)
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="Invalid JSON"):
        evidence._validate_contract(contract_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda receipt: receipt.update(status="failed"), "unsupported schema/kind/status"),
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
def test_contract_rejects_rehashed_source_dcp_semantic_drift(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    receipt_path = Path(contract["artifacts"]["source_dcp_export_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutation(receipt)
    _write_json(receipt_path, receipt)
    contract["artifacts"]["source_dcp_export_receipt"]["sha256"] = _sha256(receipt_path)
    _write_json(contract_path, contract)

    with pytest.raises((TypeError, ValueError), match=match):
        evidence._validate_contract(contract_path)


def test_contract_rejects_rehashed_preflight_false_verdict(tmp_path: Path) -> None:
    contract_path, _results_dir = _fixture(tmp_path)

    def invalidate_preflight(submission: dict[str, object]) -> None:
        preflight = submission["preflight"]
        assert isinstance(preflight, dict)
        validation = preflight["validation"]
        assert isinstance(validation, dict)
        validation["tracked_regular_files_match_head_blobs"] = False
        submission["preflight_sha256"] = hashlib.sha256(
            json.dumps(
                preflight,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    _rewrite_submission_and_runtime_binding(contract_path, invalidate_preflight)
    with pytest.raises(ValueError, match="preflight validation"):
        evidence._validate_contract(contract_path)


def test_contract_consumes_stable_intent_bytes_after_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    intent_path = Path(contract["artifacts"]["eval_intent_manifest"]["path"])
    valid_intent = intent_path.read_bytes()
    invalid_intent = json.loads(valid_intent)
    invalid_intent["thresholds"]["minimum_mean_success_rate"] = 0.5
    replacement = tmp_path / "invalid-intent-replacement.json"
    _write_json(replacement, invalid_intent)
    replacement.chmod(0o444)
    opened_inode = intent_path.stat().st_ino
    real_stable_read = evidence._read_stable_regular_file
    replaced = False

    def swap_intent_after_stable_read(path: Path, label: str, **kwargs):
        nonlocal replaced
        payload, descriptor = real_stable_read(path, label, **kwargs)
        if label == "contract.artifacts.eval_intent_manifest" and not replaced:
            replaced = True
            os.replace(replacement, intent_path)
        return payload, descriptor

    monkeypatch.setattr(evidence, "_read_stable_regular_file", swap_intent_after_stable_read)
    validated = evidence._validate_contract(contract_path)

    assert replaced
    assert intent_path.stat().st_ino != opened_inode
    assert validated["_runtime_bindings"]["evaluation_intent"]["thresholds"]["minimum_mean_success_rate"] == 0.0
    assert json.loads(intent_path.read_text(encoding="utf-8"))["thresholds"]["minimum_mean_success_rate"] == 0.5


def test_contract_prior_gate_field_is_stage_aware(tmp_path: Path) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["evaluation"]["prior_gate_evidence"] = {
        "path": str(contract_path),
        "size_bytes": contract_path.stat().st_size,
        "sha256": _sha256(contract_path),
    }
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="must be null"):
        evidence._validate_contract(contract_path)

    contract["evaluation"]["stage"] = "formal"
    contract["evaluation"]["prior_gate_evidence"] = None
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="is required"):
        evidence._validate_contract(contract_path)


def test_formal_contract_revalidates_and_binds_prior_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_contract_path, gate_evidence_path, _arguments = _gate_evidence_fixture(
        tmp_path,
        monkeypatch,
    )
    formal_contract = json.loads(gate_contract_path.read_text(encoding="utf-8"))
    formal_contract["experiment_id"] = "CWAM-E35-R12-FORMAL"
    formal_contract["logical_run_id"] = "CWAM-E35-R12-FORMAL-S42"
    formal_contract["run_name"] = formal_contract["logical_run_id"]
    formal_results = tmp_path / "formal-results" / formal_contract["logical_run_id"]
    formal_results.mkdir(parents=True)
    formal_contract["results_dir"] = str(formal_results.resolve())
    formal_contract["evaluation"].update(
        {
            "stage": "formal",
            "tasks": list(evidence.ATOMIC_TASKS),
            "episodes_per_task": 20,
            "num_shards": 8,
            "prior_gate_evidence": {
                "path": str(gate_evidence_path.resolve()),
                "size_bytes": gate_evidence_path.stat().st_size,
                "sha256": _sha256(gate_evidence_path),
            },
        }
    )
    receipt = formal_contract["artifacts"]["server_runtime_receipts"][0]
    formal_contract["artifacts"]["server_runtime_receipts"] = [receipt] * 8
    formal_contract_path = tmp_path / "formal-contract.json"
    _write_json(formal_contract_path, formal_contract)
    formal_contract_path.chmod(0o444)
    asset_verification = json.loads(
        Path(formal_contract["artifacts"]["asset_verification_receipt"]["path"]).read_text(encoding="utf-8")
    )

    monkeypatch.setattr(
        evidence,
        "_validate_eval_runtime_receipt",
        lambda **_kwargs: {
            "capture_started_at_utc": "2026-07-28T19:02:00Z",
            "captured_at_utc": "2026-07-28T19:02:03Z",
            "assets": {
                "manifest_sha256": asset_verification["manifest_sha256"],
                "publisher_receipt_sha256": asset_verification["publisher_receipt_sha256"],
            },
            "evaluation_intent": {
                "evaluation_class": "exploratory",
                "source_git_pushed": False,
            },
        },
    )
    validated = evidence._validate_contract(formal_contract_path)

    assert validated["_prior_gate_binding"]["status"] == "passed"
    assert (
        validated["_prior_gate_binding"]["prior_gate_evidence"] == formal_contract["evaluation"]["prior_gate_evidence"]
    )


def test_post_rollout_receipt_is_strict_exclusive_read_only_and_reverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    _patch_post_rollout_verifiers(monkeypatch, contract_path)
    receipt_path = tmp_path / "post-rollout-receipt.json"
    other_parent = tmp_path / "other"
    other_parent.mkdir()

    with pytest.raises(ValueError, match="share the eval contract"):
        evidence.verify_post_rollout(
            contract_path=contract_path,
            output=other_parent / "post-rollout-receipt.json",
        )

    payload = evidence.verify_post_rollout(
        contract_path=contract_path,
        output=receipt_path,
    )

    assert payload["kind"] == evidence._POST_ROLLOUT_VERIFICATION_KIND
    assert payload["scope"] == evidence._POST_ROLLOUT_SCOPE
    assert all(payload["validation"].values())
    assert all(payload["semantic_matches"].values())
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    assert payload["elapsed_seconds"]["total"] >= sum(
        payload["elapsed_seconds"][key]
        for key in (
            "cosmos_snapshot",
            "serving_checkpoint",
            "robocasa_assets",
            "source_repositories",
        )
    )

    contract = evidence._validate_contract(contract_path)
    binding = evidence._validate_post_rollout_verification_receipt(
        receipt_path=receipt_path,
        contract_path=contract_path,
        contract=contract,
    )
    assert binding["receipt"]["sha256"] == _sha256(receipt_path)
    assert binding["validation"]["tracked_source_files_match_head_blobs"] is True

    with pytest.raises(FileExistsError, match="overwrite"):
        evidence.verify_post_rollout(
            contract_path=contract_path,
            output=receipt_path,
        )


def test_post_rollout_snapshot_identity_uses_manifest_without_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = tmp_path / "snapshots" / "run"
    snapshot_root.mkdir(parents=True)
    payload = snapshot_root / "payload.txt"
    payload.write_text("immutable\n", encoding="utf-8")
    payload.chmod(0o444)
    source_commit = "a" * 40
    source_manifest = snapshot_root / "SNAPSHOT_MANIFEST.json"
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "source_head": source_commit,
            "source_dirty": False,
            "file_count": 1,
            "files": [
                {
                    "path": "payload.txt",
                    "sha256": _sha256(payload),
                    "bytes": payload.stat().st_size,
                }
            ],
        },
    )
    source_manifest.chmod(0o444)
    snapshot_root.chmod(0o555)
    monkeypatch.setattr(evidence, "_POST_ROLLOUT_SNAPSHOT_ALLOWED_ROOT", snapshot_root.parent)
    manifest_sha256 = _sha256(source_manifest)
    contract = {
        "source_root": str(snapshot_root),
        "source_commit": source_commit,
        "source_manifest_sha256": manifest_sha256,
        "_stable_artifacts": {
            "source_manifest": {
                "path": str(source_manifest),
                "sha256": manifest_sha256,
            }
        },
        "_bound_json": {
            "source_manifest": json.loads(source_manifest.read_text(encoding="utf-8")),
        },
    }

    report = evidence._verify_snapshot_with_bound_manifest(contract)
    assert report["status"] == "passed"
    assert report["snapshot_root"] == str(snapshot_root)

    contract["_bound_json"]["source_manifest"]["source_head"] = "b" * 40
    with pytest.raises(ValueError, match="source_head differs"):
        evidence._verify_snapshot_with_bound_manifest(contract)
    snapshot_root.chmod(0o755)
    payload.chmod(0o644)
    source_manifest.chmod(0o644)


@pytest.mark.parametrize("input_name", ["checkpoint", "assets"])
def test_post_rollout_requires_exact_pre_and_post_semantic_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    _patch_post_rollout_verifiers(monkeypatch, contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    if input_name == "checkpoint":
        receipt_path = Path(contract["artifacts"]["checkpoint_verification_receipt"]["path"])

        def mismatched_checkpoint(
            _manifest: dict[str, object],
            _checkpoint_path: str | Path,
        ) -> dict[str, object]:
            result = json.loads(receipt_path.read_text(encoding="utf-8"))
            result["file_count"] += 1
            return result

        monkeypatch.setattr(
            evidence.cwam_serving_checkpoint_manifest,
            "verify_serving_checkpoint_manifest",
            mismatched_checkpoint,
        )
    else:
        receipt_path = Path(contract["artifacts"]["asset_verification_receipt"]["path"])

        def mismatched_assets(_contract: dict[str, object]) -> dict[str, object]:
            result = json.loads(receipt_path.read_text(encoding="utf-8"))
            result["payload_file_count"] += 1
            return result

        monkeypatch.setattr(
            evidence,
            "_verify_asset_tree_with_bound_provenance",
            mismatched_assets,
        )

    output = tmp_path / "post-rollout-receipt.json"
    with pytest.raises(ValueError, match="does not exactly match"):
        evidence.verify_post_rollout(contract_path=contract_path, output=output)
    assert not output.exists()


def test_post_rollout_receipt_rejects_unknown_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    receipt_path = _mock_post_rollout_receipt(
        monkeypatch,
        contract_path,
        tmp_path / "post-rollout-receipt.json",
    )
    receipt_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["unexpected"] = True
    _write_json(receipt_path, receipt)
    receipt_path.chmod(0o444)

    with pytest.raises(ValueError, match="keys mismatch"):
        evidence._validate_post_rollout_verification_receipt(
            receipt_path=receipt_path,
            contract_path=contract_path,
            contract=evidence._validate_contract(contract_path),
        )


def test_git_tracked_regular_file_attestation_matches_head_and_detects_drift(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(repo), *arguments],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "--quiet")
    alpha = repo / "alpha.txt"
    alpha.write_text("alpha\n", encoding="utf-8")
    alpha.chmod(0o644)
    nested = repo / "nested"
    nested.mkdir()
    executable = nested / "tool.sh"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    git("add", "--", "alpha.txt", "nested/tool.sh")
    git(
        "-c",
        "user.name=Evidence Test",
        "-c",
        "user.email=evidence@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    commit = git("rev-parse", "HEAD")

    report = evidence._verify_git_repository(
        name="fixture",
        root=repo.resolve(),
        expected_commit=commit,
        verify_tracked_files=True,
    )
    tracked = report["tracked_files"]
    assert tracked["tracked_regular_file_count"] == 2
    assert tracked["tracked_regular_total_bytes"] == (alpha.stat().st_size + executable.stat().st_size)
    assert len(tracked["tracked_regular_manifest_sha256"]) == 64
    assert tracked["all_tracked_regular_files_match_head_blobs"] is True

    alpha.chmod(0o600)
    permission_drift = evidence._verify_tracked_regular_files(repo.resolve(), "fixture")
    assert permission_drift["tracked_regular_manifest_sha256"] != tracked["tracked_regular_manifest_sha256"]
    alpha.chmod(0o700)
    with pytest.raises(ValueError, match="executable-bit semantics differ from HEAD"):
        evidence._verify_tracked_regular_files(repo.resolve(), "fixture")
    alpha.chmod(0o644)
    alpha.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from HEAD"):
        evidence._verify_tracked_regular_files(repo.resolve(), "fixture")


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
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    match: str,
) -> None:
    contract, results_dir = _fixture(tmp_path)
    post_receipt = _mock_post_rollout_receipt(
        monkeypatch,
        contract,
        tmp_path / "post-rollout-receipt.json",
    )
    result_path = results_dir / "results_shard0.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mutation(result)
    _write_json(result_path, result)

    with pytest.raises((RuntimeError, ValueError), match=match):
        evidence.summarize(
            contract_path=contract,
            post_rollout_verification_receipt=post_receipt,
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
    placeholder = Path(contract["artifacts"]["source_manifest"]["path"])
    placeholder.chmod(0o444)
    contract["evaluation"]["prior_gate_evidence"] = {
        "path": str(placeholder),
        "size_bytes": placeholder.stat().st_size,
        "sha256": _sha256(placeholder),
    }
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="canonical ordered task list"):
        evidence._validate_contract(contract_path)


def test_contract_and_artifact_schemas_reject_unknown_fields(tmp_path: Path) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["unexpected"] = True
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="contract keys mismatch"):
        evidence._validate_contract(contract_path)

    contract.pop("unexpected")
    contract["artifacts"]["source_manifest"]["unexpected"] = True
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="contract.artifacts.source_manifest keys mismatch"):
        evidence._validate_contract(contract_path)

    contract_path.chmod(0o600)
    contract_path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    contract_path.chmod(0o444)
    with pytest.raises(ValueError, match="duplicate key"):
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


def test_runtime_receipt_must_preserve_the_exact_environment_launcher(tmp_path: Path) -> None:
    contract_path, _results_dir = _fixture(tmp_path)

    def mutate(receipt: dict[str, object]) -> None:
        inputs = receipt["inputs"]
        assert isinstance(inputs, dict)
        interpreters = inputs["interpreters"]
        assert isinstance(interpreters, dict)
        cosmos = interpreters["cosmos"]
        assert isinstance(cosmos, dict)
        cosmos["launch_path"] = cosmos["resolved_path"]

    _mutate_contract_artifact(contract_path, "eval_runtime_receipt", mutate)

    with pytest.raises(ValueError, match="not the exact environment launcher"):
        evidence._validate_contract(contract_path)


def test_environment_interpreter_must_remain_executable(tmp_path: Path) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    environment_path = Path(contract["artifacts"]["cosmos_environment_manifest"]["path"])
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    interpreter = Path(environment["interpreter"]["resolved_path"])
    interpreter.chmod(0o644)

    with pytest.raises(ValueError, match="must be executable"):
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


def test_final_evidence_rehashes_graphics_and_asset_provenance_files(
    tmp_path: Path,
) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    runtime_path = Path(contract["artifacts"]["eval_runtime_receipt"]["path"])
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    libegl = Path(runtime["graphics_userspace"]["libraries"]["libEGL"]["canonical_path"])
    original = libegl.read_bytes()
    libegl.write_bytes(b"X" * len(original))
    with pytest.raises(ValueError, match="live graphics file"):
        evidence._validate_contract(contract_path)

    libegl.write_bytes(original)
    asset_path = Path(contract["artifacts"]["asset_verification_receipt"]["path"])
    asset = json.loads(asset_path.read_text(encoding="utf-8"))
    manifest_path = Path(asset["manifest_path"])
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(manifest_path.read_bytes() + b"drift")
    manifest_path.chmod(evidence.cwam_robocasa_asset_manifest.PROVENANCE_FILE_MODE)
    with pytest.raises(ValueError, match="asset provenance SHA256"):
        evidence._validate_contract(contract_path)


def test_runtime_capture_must_remain_in_scheduler_start_window(tmp_path: Path) -> None:
    contract_path, _results_dir = _fixture(tmp_path)

    def mutate(receipt: dict[str, object]) -> None:
        receipt["capture_started_at_utc"] = "2026-07-28T22:00:01Z"
        receipt["captured_at_utc"] = "2026-07-28T22:00:02Z"

    _mutate_contract_artifact(contract_path, "eval_runtime_receipt", mutate)
    with pytest.raises(ValueError, match="scheduler-start freshness window"):
        evidence._validate_contract(contract_path)


def test_gate_rejects_nonzero_effect_thresholds(tmp_path: Path) -> None:
    contract_path, _results_dir = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["evaluation"]["effect_thresholds"] = {
        "minimum_mean_success_rate": 1.0,
        "minimum_successful_tasks": 1,
    }
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="effect thresholds must be exactly"):
        evidence._validate_contract(contract_path)
