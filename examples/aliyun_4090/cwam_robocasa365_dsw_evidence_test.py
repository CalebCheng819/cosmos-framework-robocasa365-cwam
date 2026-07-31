# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the Alibaba DSW RoboCasa365 runner and evidence adapter.

Workflow iteration: ITER-2026-07-31-CWAM-DSW974-FSDP4-EVAL-ADAPTER.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
import stat
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

HERE = Path(__file__).resolve().parent
H_HELPERS = HERE.parent / "h_cluster"
sys.path.insert(0, str(H_HELPERS))
sys.path.insert(0, str(HERE))

import cwam_robocasa365_dsw_evidence as dsw_evidence  # noqa: E402
import cwam_robocasa365_eval_evidence as existing_aggregator  # noqa: E402

RUNNER = HERE / "run_cwam_cosmos3_robocasa365_eval_dsw.sh"
README = HERE / "README.md"
SHA = "a" * 64
SOURCE_COMMIT = "1" * 40
DEMIAN_COMMIT = "2" * 40
VAE_PATH = "/inputs/wan/Wan2.2_VAE.pth"
QWEN_PATH = "/inputs/qwen/Qwen3-VL-8B"
CHECKPOINT_PATH = "/inputs/checkpoint/model"
CONFIG_PATH = "/inputs/checkpoint/config.yaml"
STATS_PATH = "/inputs/stats/action_stats.json"
SOURCE_MANIFEST_PATH = "/inputs/source/SNAPSHOT_MANIFEST.json"
QWEN_MANIFEST_PATH = "/inputs/qwen/Qwen3-VL-8B-manifest.json"
DEMIAN_OPENPI_SERVER_PATH = (
    "/inputs/demian/openpi/src/openpi/serving/websocket_policy_server.py"
)
ROBOCASA_ENVIRONMENT_PATH = "/inputs/envs/robocasa"
ROBOCASA_READY_MARKER_PATH = f"{ROBOCASA_ENVIRONMENT_PATH}/.cwam_env_ready.json"
EGL_VENDOR_JSON_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
EGL_MAPPING_RECEIPT_PATH = "/inputs/config/robocasa-egl-device-mapping-dsw974.json"

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _descriptor(path: str, sha256: str = SHA, size_bytes: int = 1) -> dict:
    return {"path": path, "size_bytes": size_bytes, "sha256": sha256}


def _weight_receipt(metadata_sha256: str = SHA) -> dict:
    return {
        "schema_version": 1,
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_metadata_sha256": metadata_sha256,
        "selected_weight_tree": "net_ema",
        "selected_source_prefix": "net_ema.",
        "selected_key_count": 809,
        "selected_source_dtype_census": {"torch.bfloat16": 809},
        "selected_logical_bytes": 1,
        "selected_fqn_shape_dtype_sha256": SHA,
        "regular_key_count": 0,
        "ema_key_count": 809,
        "other_key_count": 0,
        "serving_only_selected_tree": True,
        "load_planner": "CustomLoadPlanner(load_ema_to_reg=True)",
    }


def _checkpoint_publication_receipt(
    checkpoint_path: Path,
    tree_sha256: str,
) -> dict:
    final_root = checkpoint_path.parent
    stage_root = final_root.parent / (
        f".{final_root.name}.incomplete-transfer-20260731"
    )
    model_files = [
        {
            "relative_path": ".metadata",
            "size_bytes": 1,
            "sha256": hashlib.sha256(b"metadata").hexdigest(),
        },
        *[
            {
                "relative_path": f"__{index}_0.distcp",
                "size_bytes": index + 1,
                "sha256": hashlib.sha256(f"shard-{index}".encode()).hexdigest(),
            }
            for index in range(8)
        ],
    ]
    top_level_entries = sorted(
        {
            "H-SHA256SUMS",
            "config.yaml",
            "ema-bf16-export-receipt.json",
            "ema-bf16-key-manifest.json",
            "ema-bf16-validation.json",
            "ema-serving-dcp-transfer-manifest.json",
            "h-serving-checkpoint-manifest.json",
            "source-dcp-full-verification.json",
            "model",
        }
    )
    top_level_files = [
        {
            "relative_path": name,
            "size_bytes": index + 1,
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
            "mode": "0444",
        }
        for index, name in enumerate(
            name for name in top_level_entries if name != "model"
        )
    ]
    validation = {
        "exact_top_level_allowlist": True,
        "exact_model_allowlist": True,
        "full_model_tree_rehashed": True,
    }

    def audit(root: Path, model: Path) -> dict:
        return {
            "root": str(root),
            "top_level_entries": top_level_entries,
            "top_level_files": top_level_files,
            "model": {
                "path": str(model),
                "file_count": 9,
                "total_bytes": sum(record["size_bytes"] for record in model_files),
                "files": model_files,
                "tree_sha256": tree_sha256,
            },
            "validation": validation,
        }

    return {
        "schema_version": 2,
        "kind": "cwam_dsw_ema_checkpoint_publication",
        "status": "published",
        "mode": "cpfs_transaction_publish",
        "stage_root": str(stage_root),
        "final_root": str(final_root),
        "final_root_absent_before_publication": True,
        "recovered_uncommitted_final_before_publication": False,
        "prepublication": audit(stage_root, stage_root / "model"),
        "completed_at": "2026-07-31T09:00:00Z",
        "publication": {
            "primitive": (
                "mkdir(final)+per-entry-rename+exclusive-receipt-hardlink"
            ),
            "same_parent_same_filesystem": True,
            "parent_fsync_before_and_after": True,
            "stage_absent_after": True,
            "final_present_after": True,
            "directory_visibility_atomic": False,
            "committed_visibility_atomic": True,
            "final_root_exclusive_reservation": True,
            "final_root_inode_verified_throughout": True,
            "receipt_is_commit_point": True,
            "receipt_required_for_success": True,
            "rollback_on_precommit_failure": True,
            "no_rollback_after_receipt_commit": True,
        },
        "postpublication": audit(final_root, checkpoint_path),
    }


def _checkpoint_publication_gate_fixture(
    tmp_path: Path,
    *,
    checkpoint_path: Path,
    tree_sha256: str,
) -> tuple[Path, Path, dict]:
    publication_path = tmp_path / "checkpoint_atomic_publication.json"
    _write_json(
        publication_path,
        _checkpoint_publication_receipt(checkpoint_path, tree_sha256),
        mode=0o444,
    )
    publication_sha256 = hashlib.sha256(publication_path.read_bytes()).hexdigest()
    gate_path = tmp_path / "checkpoint_publication_gate.json"
    gate = dsw_evidence.verify_checkpoint_publication(
        publication_receipt_path=publication_path,
        expected_publication_receipt_sha256=publication_sha256,
        checkpoint_path=checkpoint_path,
        expected_tree_sha256=tree_sha256,
        output=gate_path,
    )
    return publication_path, gate_path, gate


def _runtime_receipt(
    *,
    checkpoint_path: str = CHECKPOINT_PATH,
    checkpoint_manifest_path: str = "/inputs/checkpoint/dsw-serving-checkpoint-manifest.json",
    config_path: str = CONFIG_PATH,
    stats_path: str = STATS_PATH,
    vae_path: str = VAE_PATH,
    qwen_path: str = QWEN_PATH,
    source_manifest_path: str = SOURCE_MANIFEST_PATH,
    qwen_manifest_path: str = QWEN_MANIFEST_PATH,
    provenance: dict[str, dict] | None = None,
    file_sha256: str = SHA,
    manifest_sha256: str = SHA,
    metadata_sha256: str = SHA,
) -> dict:
    if provenance is None:
        provenance = {
            name: _descriptor(f"/receipts/{name}.json")
            for name in dsw_evidence._REQUIRED_RUNTIME_PROVENANCE
        }
    weight = _weight_receipt(metadata_sha256)
    weight["checkpoint_path"] = checkpoint_path
    return {
        "schema_version": 2,
        "kind": "cwam_robocasa365_alibaba_dsw_runtime_receipt",
        "captured_at_utc": "2026-07-31T05:00:00+00:00",
        "platform": {
            "provider": "Alibaba Cloud",
            "service": "PAI DSW",
            "region": "cn-beijing",
        },
        "topology": {
            "hostname": "dsw-test",
            "physical_server_count": 1,
            "physical_policy_server_count": 1,
            "physical_gpu_count": 4,
            "distributed_world_size": 4,
            "logical_num_shards": 8,
            "evaluator_schedule": "sequential_shards_0_through_7",
            "policy_server_uri": "ws://127.0.0.1:16000",
        },
        "resources": {
            "cpu_affinity_count": 128,
            "cpu_affinity_ids": list(range(128)),
            "memory_limit_bytes": 400 * (1024**3),
            "memory_limit_gib": 400.0,
            "memory_limit_source": "cgroup_v2_memory.max",
            "gpus": [
                {
                    "visible_index": index,
                    "name": "NVIDIA GeForce RTX 4090",
                    "compute_capability": "8.9",
                    "sm": "sm89",
                    "total_memory_bytes": 48 * (1024**3),
                }
                for index in range(4)
            ],
            "torch_version": "2.test",
            "torch_cuda_version": "12.test",
        },
        "policy_server": {
            "checkpoint": checkpoint_path,
            "checkpoint_metadata_sha256": metadata_sha256,
            "checkpoint_weight_tree": "net_ema",
            "checkpoint_weight_receipt": weight,
            "config_file": config_path,
            "config_file_sha256": file_sha256,
            "action_stats_path": stats_path,
            "action_stats_sha256": file_sha256,
            "distributed_world_size": 4,
            "wan2pt2_vae_path": vae_path,
            "qwen_tokenizer_path": qwen_path,
            "parallelism": {
                "dp_shard_size": 4,
                "dp_replicate_size": 1,
                "cp_size": 1,
                "cfgp_size": 1,
            },
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
            "websocket_server": {
                "module": "openpi.serving.websocket_policy_server",
                "qualname": "WebsocketPolicyServer",
                "source_path": DEMIAN_OPENPI_SERVER_PATH,
                "source_sha256": file_sha256,
            },
        },
        "bindings": {
            "checkpoint": {
                "path": checkpoint_path,
                "manifest": _descriptor(
                    checkpoint_manifest_path,
                    manifest_sha256,
                ),
                "tree_sha256": SHA,
            },
            "training_config": _descriptor(config_path, file_sha256),
            "action_stats": _descriptor(stats_path, file_sha256),
            "wan2pt2_vae": _descriptor(vae_path, file_sha256),
            "qwen_tokenizer_path": qwen_path,
            "qwen_tokenizer_manifest": _descriptor(
                qwen_manifest_path,
                file_sha256,
            ),
            "source_manifest": _descriptor(source_manifest_path, file_sha256),
            "source_commit": SOURCE_COMMIT,
            "demian_commit": DEMIAN_COMMIT,
            "demian_openpi_server_source": _descriptor(
                DEMIAN_OPENPI_SERVER_PATH,
                file_sha256,
            ),
            "egl_mapping_receipt": _descriptor(
                EGL_MAPPING_RECEIPT_PATH,
                file_sha256,
            ),
            "egl_vendor_json": _descriptor(EGL_VENDOR_JSON_PATH, file_sha256),
            "launch_nonce": SHA,
            "robocasa_environment_path": ROBOCASA_ENVIRONMENT_PATH,
            "robocasa_ready_marker": _descriptor(
                ROBOCASA_READY_MARKER_PATH,
                file_sha256,
            ),
            "torchrun_launcher_pid": 10,
            "rank0_server_pid": 11,
        },
        "provenance": {"receipts": provenance},
        "watchdogs": {
            "nccl_server_environment": dict(dsw_evidence._NCCL_WATCHDOG_ENV),
            "server_ready_timeout_seconds": 1800,
            "one_episode_gate_timeout_seconds": 3600,
            "evaluator_timeout_seconds": 43200,
            "server_exit_timeout_seconds": 180,
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
            "egl_vendor_json_path_and_sha256_match": True,
            "egl_mapping_receipt_reverified_against_live_gpu_identity": True,
            "robocasa_environment_and_ready_marker_match_egl_probe": True,
            "server_selected_serving_only_net_ema_bf16_tree": True,
            "checkpoint_atomic_publication_semantics_match": True,
            "checkpoint_publication_pre_read_gate_matches": True,
            "all_required_pre_rollout_provenance_bound": True,
            "rank0_server_inherited_exact_nccl_watchdog_environment": True,
            "all_external_waits_have_finite_timeouts": True,
        },
    }


def _write_json(path: Path, value: object, *, mode: int = 0o600) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(mode)


def _mode_descriptor(path: Path, *, mode: int) -> dict[str, object]:
    path.chmod(mode)
    descriptor = _live_descriptor(path)
    return {**descriptor, "mode": f"{mode:04o}"}


def _egl_mapping_fixture(
    tmp_path: Path,
    *,
    environment: Path,
    ready_marker_sha256: str,
    vendor_json: Path,
    gpu_identity: list[dict[str, object]],
) -> tuple[Path, str]:
    evidence_root = tmp_path / "egl-evidence"
    evidence_root.mkdir()
    script = tmp_path / "egl-probe.py"
    script.write_text("print('probe')\n", encoding="utf-8")
    script_descriptor = _mode_descriptor(script, mode=0o555)

    def row(
        scheme: str,
        physical_index: int,
        *,
        observed_index: int,
        mapping_matches: bool,
    ) -> dict[str, object]:
        probe_pid = 1000 + physical_index
        cuda_visible_devices = (
            "0,1,2,3"
            if scheme == "visible_all_physical_index"
            else str(physical_index)
        )
        mujoco_egl_device_id = (
            "0" if scheme == "isolated_relative_zero" else str(physical_index)
        )

        probe_log = evidence_root / f"{scheme}-{physical_index}-probe_log.txt"
        probe_log.write_text(
            json.dumps(
                {
                    "event": "egl_ready",
                    "pid": probe_pid,
                    "python": "3.11.14",
                    "mujoco": "3.3.1",
                    "mujoco_gl": "egl",
                    "cuda_visible_devices": cuda_visible_devices,
                    "mujoco_egl_device_id": mujoco_egl_device_id,
                    "frame_shape": [64, 64, 3],
                    "frame_sha256": hashlib.sha256(
                        f"{scheme}-{physical_index}".encode()
                    ).hexdigest(),
                },
                sort_keys=True,
            )
            + "\n"
            + json.dumps({"event": "egl_closed", "frames": 12}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

        pmon = evidence_root / f"{scheme}-{physical_index}-nvidia_smi_pmon.txt"
        pmon_lines = [
            "# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command",
            "# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name",
        ]
        for _sample in range(2):
            for gpu_index in range(4):
                if gpu_index == observed_index:
                    pmon_lines.append(
                        f"{gpu_index:5d} {probe_pid:10d}      G      1      1"
                        "      -      -      -      -     12      -    python"
                    )
                else:
                    pmon_lines.append(
                        f"{gpu_index:5d}          -      -      -      -"
                        "      -      -      -      -      -      -    -"
                    )
        pmon.write_text("\n".join(pmon_lines) + "\n", encoding="utf-8")

        snapshot = (
            evidence_root
            / f"{scheme}-{physical_index}-nvidia_smi_snapshot.txt"
        )
        snapshot_lines = [
            "Fri Jul 31 08:00:00 2026",
            "| NVIDIA-SMI 570.153.02   Driver Version: 570.153.02   CUDA Version: 12.8 |",
        ]
        for identity in gpu_identity:
            snapshot_lines.append(
                f"| {int(identity['physical_index']):3d}  "
                f"{identity['name']}        Off |   "
                f"{identity['pci_bus_id']} Off |                    0 |"
            )
        snapshot_lines.extend(
            [
                "| Processes: |",
                "|  GPU   GI   CI              PID   Type   Process name"
                "                        GPU Memory |",
                f"| {observed_index:4d}   N/A  N/A {probe_pid:9d}"
                "      G   /env/bin/python                         12MiB |",
            ]
        )
        snapshot.write_text("\n".join(snapshot_lines) + "\n", encoding="utf-8")

        evidence = {
            "probe_log": _mode_descriptor(probe_log, mode=0o444),
            "nvidia_smi_pmon": _mode_descriptor(pmon, mode=0o444),
            "nvidia_smi_snapshot": _mode_descriptor(snapshot, mode=0o444),
        }
        observed_uuid = gpu_identity[observed_index]["uuid"]
        return {
            "physical_index": physical_index,
            "cuda_visible_devices": cuda_visible_devices,
            "mujoco_egl_device_id": mujoco_egl_device_id,
            "probe_pid": probe_pid,
            "renderer_status": "passed",
            "expected_gpu_uuid": gpu_identity[physical_index]["uuid"],
            "observed_physical_index": observed_index,
            "observed_gpu_uuid": observed_uuid,
            "pmon_matched_pid": True,
            "mapping_matches_expected": mapping_matches,
            "evidence": evidence,
        }

    isolated_rows = [
        row(
            "isolated_physical_identity",
            index,
            observed_index=index,
            mapping_matches=True,
        )
        for index in range(4)
    ]
    visible_all_rows = [
        row(
            "visible_all_physical_index",
            index,
            observed_index=index,
            mapping_matches=True,
        )
        for index in range(4)
    ]
    relative_rows = [
        row(
            "isolated_relative_zero",
            index,
            observed_index=0,
            mapping_matches=index == 0,
        )
        for index in range(4)
    ]
    vendor_descriptor = _mode_descriptor(vendor_json, mode=0o644)
    receipt = {
        "schema_version": 1,
        "kind": "cwam_dsw974_robocasa_egl_device_mapping",
        "status": "passed",
        "created_at_utc": "2026-07-31T08:00:00Z",
        "host": {
            "hostname": socket.gethostname(),
            "gpu_count": 4,
            "gpu_inventory": gpu_identity,
        },
        "environment": {
            "path": str(environment),
            "ready_marker_sha256": ready_marker_sha256,
            "python": "3.11.14",
            "mujoco": "3.3.1",
            "egl_vendor_json": vendor_descriptor,
        },
        "probe": {
            "script": script_descriptor,
            "hold_seconds": 12.0,
            "evidence_root": str(evidence_root),
        },
        "schemes": {
            "isolated_physical_identity": {
                "status": "passed",
                "rows": isolated_rows,
            },
            "visible_all_physical_index": {
                "status": "passed",
                "rows": visible_all_rows,
            },
            "isolated_relative_zero": {
                "status": "failed",
                "rows": relative_rows,
            },
        },
        "recommendation": {
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
        },
    }
    receipt_path = tmp_path / "robocasa-egl-device-mapping-dsw974.json"
    _write_json(receipt_path, receipt, mode=0o444)
    return receipt_path, hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def _attach_runtime_egl_fixture(
    tmp_path: Path,
    runtime: dict,
) -> tuple[list[dict[str, object]], Path]:
    egl_root = tmp_path / "egl-runtime-closure"
    egl_root.mkdir()
    environment = egl_root / "robocasa-environment"
    environment.mkdir()
    ready_marker = environment / ".cwam_env_ready.json"
    _write_json(ready_marker, {"status": "ready"}, mode=0o444)
    ready_marker_sha = hashlib.sha256(ready_marker.read_bytes()).hexdigest()
    vendor_json = egl_root / "10_nvidia.json"
    vendor_json.write_text(
        '{"file_format_version":"1.0.0"}\n',
        encoding="utf-8",
    )
    vendor_json.chmod(0o644)
    gpu_identity: list[dict[str, object]] = [
        {
            "physical_index": index,
            "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
            "name": "NVIDIA GeForce RTX 4090",
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
        }
        for index in range(4)
    ]
    mapping_receipt, _mapping_sha = _egl_mapping_fixture(
        egl_root,
        environment=environment,
        ready_marker_sha256=ready_marker_sha,
        vendor_json=vendor_json,
        gpu_identity=gpu_identity,
    )
    bindings = runtime["bindings"]
    bindings["robocasa_environment_path"] = str(environment)
    bindings["robocasa_ready_marker"] = _live_descriptor(ready_marker)
    bindings["egl_vendor_json"] = _live_descriptor(vendor_json)
    bindings["egl_mapping_receipt"] = _live_descriptor(mapping_receipt)
    return gpu_identity, mapping_receipt


def _qwen_manifest_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    tokenizer = tmp_path / "Qwen3-VL-8B"
    tokenizer.mkdir(parents=True)
    contents = {
        "config.json": b'{"model_type":"qwen3_vl"}\n',
        "preprocessor_config.json": b'{"image_processor_type":"Qwen"}\n',
        "tokenizer.json": b'{"version":"1.0"}\n',
        "tokenizer_config.json": b'{"tokenizer_class":"Qwen"}\n',
        "vocab/merges.txt": b"#version: 0.2\n",
    }
    files = []
    for relative_path, content in contents.items():
        path = tokenizer / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(
            {
                "relative_path": relative_path,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = tmp_path / "qwen-tokenizer-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "kind": "cwam_qwen_tokenizer_manifest",
            "files": files,
        },
    )
    return tokenizer, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _raw_results(results_dir: Path, *, checkpoint_path: Path, run_name: str) -> None:
    results_dir.mkdir(parents=True)
    results_root = results_dir.parent
    for shard in range(8):
        args = {
            "task_shard": shard,
            "num_shards": 8,
            "episodes": 20,
            "episodes_start": 0,
            "max_steps": 400,
            "open_loop_horizon": 12,
            "seed": 42,
            "checkpoint": str(checkpoint_path),
            "run_name": run_name,
            "results_dir": str(results_root),
            "eval_dataset": None,
            "headless": True,
            "save_video": False,
            "tasks": [],
            "split": "atomic",
            "dynamic_instructions_file": None,
            "gt_reasoning_file": None,
            "vlm_captions_file": None,
            "fg_captions_path": None,
        }
        shard_results = {}
        tasks = [task for task_index, task in enumerate(existing_aggregator.ATOMIC_TASKS) if task_index % 8 == shard]
        for within_shard_index, task in enumerate(tasks):
            episodes = []
            for episode_index in range(20):
                success = (within_shard_index + episode_index) % 2 == 0
                episodes.append(
                    {
                        "success": success,
                        "place_anytime": success,
                        "steps": 1,
                        "instruction": f"test {task}",
                        "action_log": [[0.0] * 7],
                    }
                )
            successes = sum(int(episode["success"]) for episode in episodes)
            shard_results[task] = {
                "complete": True,
                "split": "atomic",
                "episodes": 20,
                "episode_results": episodes,
                "successes": successes,
                "place_anytime_count": successes,
                "success_rate": successes / 20,
                "place_anytime_rate": successes / 20,
            }
        _write_json(results_dir / f"args_shard{shard}.json", args)
        _write_json(results_dir / f"results_shard{shard}.json", shard_results)


def _live_descriptor(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _provenance_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, dict]]:
    tmp_path.mkdir(parents=True)
    pre_paths: dict[str, Path] = {}
    post_paths: dict[str, Path] = {}
    pair_bases = (
        "source_snapshot",
        "static_inputs",
        "cosmos_environment",
        "robocasa_environment",
        "robocasa_assets",
    )
    for base in pair_bases:
        pre = tmp_path / f"{base}-pre.json"
        post = tmp_path / f"{base}-post.json"
        common = {"schema_version": 1, "kind": f"test_{base}", "status": "passed"}
        _write_json(pre, {**common, "phase": "pre"}, mode=0o444)
        _write_json(post, {**common, "phase": "post"}, mode=0o444)
        pre_paths[f"{base}_pre"] = pre
        post_paths[f"{base}_post"] = post
    qwen_pre = tmp_path / "qwen-pre.json"
    qwen_post = tmp_path / "qwen-post.json"
    qwen_value = {"schema_version": 1, "kind": "test_qwen", "status": "passed"}
    _write_json(qwen_pre, qwen_value, mode=0o444)
    _write_json(qwen_post, qwen_value, mode=0o444)
    pre_paths["qwen_tokenizer_pre"] = qwen_pre
    post_paths["qwen_tokenizer_post"] = qwen_post
    gate = tmp_path / "one-episode-gate.json"
    _write_json(
        gate,
        {
            "schema_version": 1,
            "kind": "cwam_robocasa365_alibaba_dsw_one_episode_gate",
            "status": "passed",
            "role": "preflight_only_not_part_of_340_primary_metric",
            "episode_count": 1,
        },
        mode=0o444,
    )
    post_paths["one_episode_gate"] = gate
    for name in sorted(dsw_evidence._REQUIRED_RUNTIME_PROVENANCE - set(pre_paths)):
        path = tmp_path / f"{name}.json"
        _write_json(path, {"schema_version": 1, "name": name}, mode=0o444)
        pre_paths[name] = path
    assert set(pre_paths) == dsw_evidence._REQUIRED_RUNTIME_PROVENANCE
    assert set(post_paths) == dsw_evidence._REQUIRED_POST_PROVENANCE
    return pre_paths, post_paths, {
        name: _live_descriptor(path) for name, path in pre_paths.items()
    }


def _terminal_fixture(
    tmp_path: Path,
    *,
    runtime_path: Path,
    runtime: dict,
) -> Path:
    runtime_descriptor = _live_descriptor(runtime_path)
    shutdown_path = tmp_path / "shutdown.json"
    shutdown = {
        "schema_version": 1,
        "kind": "cwam_robocasa365_alibaba_dsw_controlled_shutdown_receipt",
        "status": "passed",
        "completed_at_utc": "2026-07-31T05:01:00+00:00",
        "uri": "ws://127.0.0.1:16000",
        "launch_nonce": SHA,
        "server_pid": 11,
        "runtime_receipt": runtime_descriptor,
        "request": {"__cwam_control__": "shutdown", "launch_nonce": SHA},
        "ack": {
            "actions": [],
            "cwam_control": "shutdown_ack",
            "launch_nonce": SHA,
            "server_timing": {"infer_ms": 1.25},
        },
        "validation": {"ack": True},
    }
    _write_json(shutdown_path, shutdown, mode=0o444)
    server_log = tmp_path / "policy-server.log"
    server_log.write_text("clean exit\n", encoding="utf-8")
    server_log.chmod(0o444)
    terminal_path = tmp_path / "terminal.json"
    terminal = {
        "schema_version": 1,
        "kind": "cwam_robocasa365_alibaba_dsw_server_terminal_receipt",
        "status": "passed",
        "captured_at_utc": "2026-07-31T05:02:00+00:00",
        "launch_nonce": SHA,
        "torchrun_launcher_pid": runtime["bindings"]["torchrun_launcher_pid"],
        "torchrun_exit_code": 0,
        "runtime_receipt": runtime_descriptor,
        "shutdown_receipt": _live_descriptor(shutdown_path),
        "server_log": _live_descriptor(server_log),
        "validation": {"clean_exit": True},
    }
    _write_json(terminal_path, terminal, mode=0o444)
    return terminal_path


def test_runner_keeps_eight_logical_shards_on_one_fsdp4_server() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    for token in (
        "readonly PHYSICAL_GPU_COUNT=4",
        "readonly PHYSICAL_SERVER_COUNT=1",
        "readonly LOGICAL_NUM_SHARDS=8",
        "readonly TASK_COUNT=17",
        "readonly EPISODES_PER_TASK=20",
        "readonly EXPECTED_EPISODE_COUNT=340",
        "readonly EXPECTED_CPU_COUNT=128",
        '--nproc-per-node="$PHYSICAL_GPU_COUNT"',
        '--num-shards "$LOGICAL_NUM_SHARDS"',
        '--episodes "$EPISODES_PER_TASK"',
        "--split atomic",
        "for ((shard = 0; shard < LOGICAL_NUM_SHARDS; shard++)); do",
        "capture-runtime",
        "verify-checkpoint-publication",
        "relocate-checkpoint",
        "verify-source-snapshot",
        "verify-static-inputs",
        "verify-one-episode-gate",
        "shutdown-server",
        "capture-terminal",
        "verify-qwen-tokenizer",
        "summarize",
        "verify-sealed",
        "summary.csv",
        "task_success_rates.csv",
        '--wan2pt2-vae-path "$WAN2PT2_VAE"',
        '--qwen-tokenizer-path "$QWEN_TOKENIZER"',
        '--egl-mapping-receipt-path "$EGL_MAPPING_RECEIPT"',
        '--egl-vendor-json-path "$EVAL_EGL_VENDOR_JSON"',
        '--robocasa-environment-path "$ROBOCASA_ENV"',
        '--qwen-tokenizer-manifest-sha256 "$EXPECTED_QWEN_TOKENIZER_MANIFEST_SHA256"',
        "--installation-mode native_exact_rebuild",
        'COSMOS_ENV_READY_MARKER" == "$COSMOS_ENV/.cwam_env_ready.json',
        'ROBOCASA_ENV_READY_MARKER" == "$ROBOCASA_ENV/.cwam_env_ready.json',
        "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
        "export TORCH_NCCL_ENABLE_MONITORING=1",
        "export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900",
        'readonly ONE_EPISODE_GATE_TIMEOUT_SECONDS=3600',
        'CUDA_VISIBLE_DEVICES="$evaluator_gpu"',
        'MUJOCO_EGL_DEVICE_ID="$evaluator_gpu"',
        '--tasks "$ONE_EPISODE_GATE_TASK"',
        "--episodes 1",
        '--post-provenance-receipt "one_episode_gate=$ONE_EPISODE_GATE_RECEIPT"',
        'readonly FORMAL_INPUT_ROOT="/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs"',
        'readonly FORMAL_NVME_INPUT_ROOT="/tmp/cwam_cosmos3_robocasa365_eval_nvme/inputs"',
        'readonly FORMAL_COSMOS_ENV="$FORMAL_NVME_INPUT_ROOT/envs/cosmos-framework-09f23119-cu128-train"',
        'readonly FORMAL_ROBOCASA_REPO="$FORMAL_NVME_INPUT_ROOT/repos/robocasa-29f7ce8814c1"',
        'require_exact_value \\\n    cosmos_environment_path \\\n    "$COSMOS_ENV" \\\n    "$FORMAL_COSMOS_ENV"',
        'require_exact_value \\\n    robocasa_repository_path \\\n    "$ROBOCASA_REPO" \\\n    "$FORMAL_ROBOCASA_REPO"',
        'require_exact_value \\\n    robocasa_asset_root \\\n    "$ROBOCASA_ASSET_ROOT" \\\n    "$FORMAL_ROBOCASA_REPO"',
        'require_exact_value \\\n    dsw_local_scratch_parent \\\n    "$DSW_LOCAL_SCRATCH_PARENT" \\\n    /tmp/cwam_cosmos3_robocasa365_eval',
        'readonly PINNED_COSMOS_ENV_SOURCE_COMMIT="09f23119ea92c707207bba55565e7a09d16896a2"',
        'readonly PINNED_DEMIAN_COMMIT="2600ece5bf0eeacf010a95f084dcda2de6d30d59"',
        'readonly PINNED_ROBOCASA_COMMIT="29f7ce8814c1547f5af762a0997fbd4b64848dd7"',
        'readonly PINNED_ROBOSUITE_COMMIT="232ce7d4a6ed89c949a9aba024a05c8c32fdd08b"',
        'readonly PINNED_ACTION_STATS_SHA256="2b48eb8a42e0c9c7befc45034d152282dad328f36389c10cafdc5be2635b67df"',
        'readonly PINNED_WAN2PT2_VAE_SHA256="20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"',
        'exec "$SETSID" env \\',
        "PENDING_LAUNCH_SIGNAL_STATUS=",
        "trap 'PENDING_LAUNCH_SIGNAL_STATUS=130' INT",
        "trap 'PENDING_LAUNCH_SIGNAL_STATUS=143' TERM",
        'SERVER_PROCESS_GROUP="$SERVER_LAUNCHER_PID"',
        'cleanup "$PENDING_LAUNCH_SIGNAL_STATUS"',
        'kill -TERM -- "-$SERVER_PROCESS_GROUP"',
        'kill -KILL -- "-$SERVER_PROCESS_GROUP"',
        "--demian-openpi-server-source-path",
        'robocasa_torchcodec_relay=$ROBOCASA_TORCHCODEC_RELAY_RECEIPT',
        ': "${CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT:',
        ': "${EXPECTED_CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT_SHA256:',
        'readonly FORMAL_CHECKPOINT_PUBLICATION_RECEIPT="/mnt/workspace/cwam_cosmos3_robocasa365_eval/control/receipts/ema-checkpoint-atomic-publication.json"',
        'checkpoint_atomic_publication=$CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT',
        'checkpoint_publication_gate=$CHECKPOINT_PUBLICATION_GATE_RECEIPT',
        'readonly CHECKPOINT_PUBLICATION_GATE_RECEIPT="$CONTROL_DIR/checkpoint-publication-pre-read-gate.json"',
        '"$EXPECTED_CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT_SHA256"',
        '"$EXPECTED_SERVING_CHECKPOINT_TREE_SHA256"',
    ):
        assert token in script
    checkpoint_gate = script.index(
        "\n    verify-checkpoint-publication \\\n"
    )
    first_checkpoint_content_hash = script.index(
        "verify_sha256 \\\n"
        "    h_serving_checkpoint_manifest \\\n"
    )
    relocation = script.index(
        "\n    relocate-checkpoint \\\n"
    )
    assert checkpoint_gate < first_checkpoint_content_hash < relocation
    assert script.index(
        'checkpoint_publication_gate=$CHECKPOINT_PUBLICATION_GATE_RECEIPT'
    ) > checkpoint_gate
    formal_loop = script.index(
        "for ((shard = 0; shard < LOGICAL_NUM_SHARDS; shard++)); do"
    )
    assert script.index("capture-runtime") < script.index("verify-one-episode-gate") < formal_loop
    assert script.index("shutdown-server", script.index("FINISH_LOGICAL_SHARD")) < script.index("summarize \\")
    for forbidden in ("KUBEBRAIN", "rjob create", "eailabagent_gpu", "gpu-lg-cmc", "H200"):
        assert forbidden not in script
    assert "readonly EXPECTED_CPU_COUNT=56" not in script
    assert script.count("verify-qwen-tokenizer") >= 2
    assert script.index("verify-qwen-tokenizer") < script.index("capture-runtime")


def test_runner_and_launch_contract_pin_only_selected_inputs_to_nvme() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert (
        'readonly FORMAL_COSMOS_ENV="$FORMAL_NVME_INPUT_ROOT/envs/'
        'cosmos-framework-09f23119-cu128-train"'
    ) in script
    assert (
        'readonly FORMAL_ROBOCASA_REPO="$FORMAL_NVME_INPUT_ROOT/repos/'
        'robocasa-29f7ce8814c1"'
    ) in script
    assert (
        'export COSMOS_ENV="$NVME_INPUTS_ROOT/envs/'
        'cosmos-framework-09f23119-cu128-train"'
    ) in readme
    assert (
        'export ROBOCASA_REPO="$NVME_INPUTS_ROOT/repos/'
        'robocasa-29f7ce8814c1"'
    ) in readme
    assert 'export ROBOCASA_ASSET_ROOT="$ROBOCASA_REPO"' in readme
    assert (
        'require_exact_value \\\n    robocasa_asset_root \\\n'
        '    "$ROBOCASA_ASSET_ROOT" \\\n    "$FORMAL_ROBOCASA_REPO"'
    ) in script

    for cpfs_binding in (
        '"$FORMAL_INPUT_ROOT/envs/cosmos-framework-09f23119-cu128-train"',
        '"$FORMAL_INPUT_ROOT/repos/robocasa-29f7ce8814c1"',
    ):
        assert cpfs_binding not in script

    for cpfs_resident_binding in (
        '"$FORMAL_INPUT_ROOT/envs/robocasa-1.0.1-py311-mj331-rs232ce7-shared"',
        '"$FORMAL_CHECKPOINT_ARTIFACT_ROOT/model"',
        '/mnt/workspace/cwam_cosmos3_robocasa365_eval/runs',
        '/tmp/cwam_cosmos3_robocasa365_eval',
    ):
        assert cpfs_resident_binding in script


def test_runner_binds_manifest_only_snapshot_to_expected_source_commit() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    for token in (
        '[[ "$SOURCE_MANIFEST" == "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json" ]]',
        '[[ "$(stat -c %a -- "$SOURCE_MANIFEST")" == 444 ]]',
        'manifest.get("schema_version") != 1',
        'manifest.get("source_head") != sys.argv[2]',
        'manifest.get("source_dirty") is not False',
        'manifest.get("file_count") != len(files)',
        'verify_sha256 source_manifest "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_MANIFEST_SHA256"',
        "verify-source-snapshot",
        '--expected-source-commit "$EXPECTED_SOURCE_COMMIT"',
    ):
        assert token in script
    manifest_path_check = script.index('[[ "$SOURCE_MANIFEST" == "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json" ]]')
    manifest_hash_check = script.index(
        'verify_sha256 source_manifest "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_MANIFEST_SHA256"'
    )
    optional_git_check = script.index('if [[ -e "$COSMOS_SOURCE/.git" || -L "$COSMOS_SOURCE/.git" ]]')
    assert manifest_path_check < manifest_hash_check < optional_git_check


def test_readme_matches_live_dsw_cpu_and_memory_evidence() -> None:
    documentation = README.read_text(encoding="utf-8")

    assert "128 effective CPUs" in documentation
    assert "400 GiB cgroup memory limit" in documentation
    assert "four visible NVIDIA 4090 GPUs" in documentation
    assert "eight logical evaluator shards" in documentation
    assert "56 effective CPUs" not in documentation


def test_runtime_receipt_rejects_physical_or_logical_topology_fabrication() -> None:
    valid = _runtime_receipt()
    assert dsw_evidence.validate_runtime_receipt(valid)["topology"]["logical_num_shards"] == 8

    eight_servers = deepcopy(valid)
    eight_servers["topology"]["physical_server_count"] = 8
    with pytest.raises(ValueError, match="physical_server_count"):
        dsw_evidence.validate_runtime_receipt(eight_servers)

    four_logical_shards = deepcopy(valid)
    four_logical_shards["topology"]["logical_num_shards"] = 4
    with pytest.raises(ValueError, match="logical_num_shards"):
        dsw_evidence.validate_runtime_receipt(four_logical_shards)

    stale_56_cpu_receipt = deepcopy(valid)
    stale_56_cpu_receipt["resources"]["cpu_affinity_count"] = 56
    stale_56_cpu_receipt["resources"]["cpu_affinity_ids"] = list(range(56))
    with pytest.raises(ValueError, match="exactly 128 effective CPUs"):
        dsw_evidence.validate_runtime_receipt(stale_56_cpu_receipt)

    wrong_qwen_path = deepcopy(valid)
    wrong_qwen_path["policy_server"]["qwen_tokenizer_path"] = "/inputs/qwen/other"
    with pytest.raises(ValueError, match="Qwen tokenizer path"):
        dsw_evidence.validate_runtime_receipt(wrong_qwen_path)

    malformed_qwen_manifest_sha = deepcopy(valid)
    malformed_qwen_manifest_sha["bindings"]["qwen_tokenizer_manifest"]["sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="artifact descriptor sha256"):
        dsw_evidence.validate_runtime_receipt(malformed_qwen_manifest_sha)

    missing_nccl_watchdog = deepcopy(valid)
    del missing_nccl_watchdog["watchdogs"]["nccl_server_environment"][
        "TORCH_NCCL_ENABLE_MONITORING"
    ]
    with pytest.raises(ValueError, match="NCCL watchdog environment"):
        dsw_evidence.validate_runtime_receipt(missing_nccl_watchdog)

    unbounded_evaluator = deepcopy(valid)
    unbounded_evaluator["watchdogs"]["evaluator_timeout_seconds"] = 0
    with pytest.raises(ValueError, match="watchdog timeout"):
        dsw_evidence.validate_runtime_receipt(unbounded_evaluator)

    changed_sampler = deepcopy(valid)
    changed_sampler["policy_server"]["sampler"] = "edm"
    with pytest.raises(ValueError, match="formal setting mismatch for sampler"):
        dsw_evidence.validate_runtime_receipt(changed_sampler)


def test_capture_runtime_binds_asset_hashes_and_exact_server_cli_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model"
    checkpoint_path.mkdir()
    metadata_path = checkpoint_path / ".metadata"
    metadata_path.write_bytes(b"metadata")
    metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    checkpoint_manifest = (
        dsw_evidence.cwam_serving_checkpoint_manifest.create_serving_checkpoint_manifest(
            checkpoint_path
        )
    )
    checkpoint_manifest_path = tmp_path / "dsw-serving-checkpoint-manifest.json"
    _write_json(checkpoint_manifest_path, checkpoint_manifest, mode=0o444)
    checkpoint_manifest_sha = hashlib.sha256(checkpoint_manifest_path.read_bytes()).hexdigest()
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes(b"config")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    stats_path = tmp_path / "action_stats.json"
    stats_path.write_bytes(b"stats")
    stats_sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    vae_path = tmp_path / "Wan2.2_VAE.pth"
    vae_path.write_bytes(b"vae")
    vae_sha = hashlib.sha256(vae_path.read_bytes()).hexdigest()
    qwen_path = tmp_path / "Qwen3-VL-8B"
    qwen_path.mkdir()
    qwen_manifest_path = tmp_path / "qwen-manifest.json"
    _write_json(qwen_manifest_path, {"schema_version": 1})
    qwen_manifest_sha = hashlib.sha256(qwen_manifest_path.read_bytes()).hexdigest()
    source_manifest_path = tmp_path / "SNAPSHOT_MANIFEST.json"
    _write_json(source_manifest_path, {"schema_version": 1}, mode=0o444)
    source_manifest_sha = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
    demian_openpi_server_source_path = tmp_path / "websocket_policy_server.py"
    demian_openpi_server_source_path.write_bytes(b"pinned demian websocket server\n")
    demian_openpi_server_source_sha = hashlib.sha256(
        demian_openpi_server_source_path.read_bytes()
    ).hexdigest()
    robocasa_environment_path = tmp_path / "robocasa-environment"
    robocasa_environment_path.mkdir()
    robocasa_ready_marker_path = (
        robocasa_environment_path / ".cwam_env_ready.json"
    )
    _write_json(robocasa_ready_marker_path, {"status": "ready"}, mode=0o444)
    robocasa_ready_marker_sha = hashlib.sha256(
        robocasa_ready_marker_path.read_bytes()
    ).hexdigest()
    egl_vendor_json_path = tmp_path / "10_nvidia.json"
    egl_vendor_json_path.write_text(
        '{"file_format_version":"1.0.0"}\n',
        encoding="utf-8",
    )
    egl_vendor_json_path.chmod(0o644)
    egl_vendor_json_sha = hashlib.sha256(
        egl_vendor_json_path.read_bytes()
    ).hexdigest()
    gpu_identity = [
        {
            "physical_index": index,
            "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
            "name": "NVIDIA GeForce RTX 4090",
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
        }
        for index in range(4)
    ]
    egl_mapping_receipt_path, egl_mapping_receipt_sha = _egl_mapping_fixture(
        tmp_path,
        environment=robocasa_environment_path,
        ready_marker_sha256=robocasa_ready_marker_sha,
        vendor_json=egl_vendor_json_path,
        gpu_identity=gpu_identity,
    )
    publication_path, publication_gate_path, _ = (
        _checkpoint_publication_gate_fixture(
            tmp_path,
            checkpoint_path=checkpoint_path,
            tree_sha256=checkpoint_manifest["tree_sha256"],
        )
    )
    provenance_receipts = {
        "checkpoint_atomic_publication": publication_path,
        "checkpoint_publication_gate": publication_gate_path,
    }
    for name in (
        dsw_evidence._REQUIRED_RUNTIME_PROVENANCE - set(provenance_receipts)
    ):
        path = tmp_path / f"{name}.json"
        _write_json(path, {}, mode=0o444)
        provenance_receipts[name] = path
    weight = _weight_receipt(metadata_sha)
    weight["checkpoint_path"] = str(checkpoint_path)
    metadata = {
        "launch_nonce": SHA,
        "server_pid": os.getpid(),
        "distributed_world_size": 4,
        "parallelism": {
            "dp_shard_size": 4,
            "dp_replicate_size": 1,
            "cp_size": 1,
            "cfgp_size": 1,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata_sha256": metadata_sha,
        "checkpoint_weight_tree": "net_ema",
        "checkpoint_weight_receipt": weight,
        "config_file": str(config_path),
        "config_file_sha256": config_sha,
        "action_stats_path": str(stats_path),
        "action_stats_sha256": stats_sha,
        "wan2pt2_vae_path": str(vae_path),
        "qwen_tokenizer_path": str(qwen_path),
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
        "websocket_server": {
            "module": "openpi.serving.websocket_policy_server",
            "qualname": "WebsocketPolicyServer",
            "source_path": str(demian_openpi_server_source_path),
            "source_sha256": demian_openpi_server_source_sha,
        },
    }
    fake_server_receipt = ModuleType("capture_robocasa365_server_receipt")
    fake_server_receipt._connect_once = lambda _uri, _timeout: metadata
    fake_server_receipt._validate_metadata = lambda _metadata: None
    monkeypatch.setitem(
        sys.modules,
        "capture_robocasa365_server_receipt",
        fake_server_receipt,
    )
    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "2.test"
    fake_torch.version = ModuleType("torch.version")
    fake_torch.version.cuda = "12.test"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(dsw_evidence, "_effective_cpu_ids", lambda: list(range(128)))
    monkeypatch.setattr(
        dsw_evidence,
        "_effective_memory_limit",
        lambda: (400 * (1024**3), "test"),
    )
    monkeypatch.setattr(
        dsw_evidence,
        "_gpu_inventory",
        lambda: [
            {
                "visible_index": index,
                "name": "NVIDIA GeForce RTX 4090",
                "compute_capability": "8.9",
                "sm": "sm89",
                "total_memory_bytes": 24 * (1024**3),
            }
            for index in range(4)
        ],
    )
    monkeypatch.setattr(
        dsw_evidence,
        "_nvidia_smi_gpu_identity",
        lambda: gpu_identity,
    )
    monkeypatch.setattr(
        dsw_evidence,
        "_read_process_environment",
        lambda _pid, _keys: dict(dsw_evidence._NCCL_WATCHDOG_ENV),
    )

    receipt = dsw_evidence.capture_runtime(
        uri="ws://127.0.0.1:16000",
        output=tmp_path / "runtime.json",
        timeout_seconds=1800.0,
        launcher_pid=os.getpid(),
        expected_launch_nonce=SHA,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        checkpoint_manifest_sha256=checkpoint_manifest_sha,
        checkpoint_tree_sha256=checkpoint_manifest["tree_sha256"],
        training_config_path=config_path,
        training_config_sha256=config_sha,
        action_stats_path=stats_path,
        action_stats_sha256=stats_sha,
        wan2pt2_vae_path=vae_path,
        wan2pt2_vae_sha256=vae_sha,
        qwen_tokenizer_path=qwen_path,
        qwen_tokenizer_manifest_path=qwen_manifest_path,
        qwen_tokenizer_manifest_sha256=qwen_manifest_sha,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha,
        source_commit=SOURCE_COMMIT,
        demian_commit=DEMIAN_COMMIT,
        demian_openpi_server_source_path=demian_openpi_server_source_path,
        robocasa_environment_path=robocasa_environment_path,
        robocasa_ready_marker_path=robocasa_ready_marker_path,
        robocasa_ready_marker_sha256=robocasa_ready_marker_sha,
        egl_vendor_json_path=egl_vendor_json_path,
        egl_vendor_json_sha256=egl_vendor_json_sha,
        egl_mapping_receipt_path=egl_mapping_receipt_path,
        egl_mapping_receipt_sha256=egl_mapping_receipt_sha,
        provenance_receipts=provenance_receipts,
        evaluator_timeout_seconds=43200,
        one_episode_gate_timeout_seconds=3600,
        server_exit_timeout_seconds=180,
    )

    assert receipt["bindings"]["wan2pt2_vae"]["sha256"] == vae_sha
    assert receipt["bindings"]["qwen_tokenizer_manifest"]["sha256"] == qwen_manifest_sha
    assert receipt["policy_server"]["wan2pt2_vae_path"] == str(vae_path)
    assert receipt["policy_server"]["qwen_tokenizer_path"] == str(qwen_path)
    assert receipt["bindings"]["egl_vendor_json"]["sha256"] == egl_vendor_json_sha
    assert (
        receipt["bindings"]["egl_mapping_receipt"]["sha256"]
        == egl_mapping_receipt_sha
    )
    assert receipt["validation"][
        "checkpoint_atomic_publication_semantics_match"
    ] is True
    assert receipt["validation"][
        "checkpoint_publication_pre_read_gate_matches"
    ] is True
    assert "checkpoint_atomic_publication" in receipt["provenance"]["receipts"]
    assert "checkpoint_publication_gate" in receipt["provenance"]["receipts"]

    metadata["qwen_tokenizer_path"] = str(tmp_path / "wrong-qwen")
    with pytest.raises(ValueError, match="qwen_tokenizer_path"):
        dsw_evidence.capture_runtime(
            uri="ws://127.0.0.1:16000",
            output=tmp_path / "wrong-runtime.json",
            timeout_seconds=1800.0,
            launcher_pid=os.getpid(),
            expected_launch_nonce=SHA,
            checkpoint_path=checkpoint_path,
            checkpoint_manifest_path=checkpoint_manifest_path,
            checkpoint_manifest_sha256=checkpoint_manifest_sha,
            checkpoint_tree_sha256=checkpoint_manifest["tree_sha256"],
            training_config_path=config_path,
            training_config_sha256=config_sha,
            action_stats_path=stats_path,
            action_stats_sha256=stats_sha,
            wan2pt2_vae_path=vae_path,
            wan2pt2_vae_sha256=vae_sha,
            qwen_tokenizer_path=qwen_path,
            qwen_tokenizer_manifest_path=qwen_manifest_path,
            qwen_tokenizer_manifest_sha256=qwen_manifest_sha,
            source_manifest_path=source_manifest_path,
            source_manifest_sha256=source_manifest_sha,
            source_commit=SOURCE_COMMIT,
            demian_commit=DEMIAN_COMMIT,
            demian_openpi_server_source_path=demian_openpi_server_source_path,
            robocasa_environment_path=robocasa_environment_path,
            robocasa_ready_marker_path=robocasa_ready_marker_path,
            robocasa_ready_marker_sha256=robocasa_ready_marker_sha,
            egl_vendor_json_path=egl_vendor_json_path,
            egl_vendor_json_sha256=egl_vendor_json_sha,
            egl_mapping_receipt_path=egl_mapping_receipt_path,
            egl_mapping_receipt_sha256=egl_mapping_receipt_sha,
            provenance_receipts=provenance_receipts,
            evaluator_timeout_seconds=43200,
            one_episode_gate_timeout_seconds=3600,
            server_exit_timeout_seconds=180,
        )


def test_checkpoint_publication_pre_read_gate_binds_commit_receipt(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "formal" / "model"
    checkpoint_path.mkdir(parents=True)
    tree_sha256 = hashlib.sha256(b"tree").hexdigest()
    publication_path, gate_path, gate = _checkpoint_publication_gate_fixture(
        tmp_path,
        checkpoint_path=checkpoint_path,
        tree_sha256=tree_sha256,
    )

    publication_descriptor = dsw_evidence._stable_sha256(
        publication_path,
        expected_mode=0o444,
    )
    assert gate["status"] == "passed"
    assert gate["publication_receipt"] == publication_descriptor
    assert gate["checkpoint_path"] == str(checkpoint_path)
    assert gate["publication_binding"]["schema_version"] == 2
    assert gate["publication_binding"]["mode"] == "cpfs_transaction_publish"
    assert stat.S_IMODE(gate_path.stat().st_mode) == 0o444
    publication_receipt = json.loads(
        publication_path.read_text(encoding="utf-8")
    )
    dsw_evidence._validate_checkpoint_publication_gate_receipt(
        gate,
        publication_descriptor=publication_descriptor,
        publication_receipt=publication_receipt,
        checkpoint_path=checkpoint_path,
        expected_tree_sha256=tree_sha256,
    )
    mutated = deepcopy(gate)
    mutated["publication_binding"]["completed_at"] = (
        "2026-07-31T09:00:01Z"
    )
    with pytest.raises(ValueError, match="publication semantics mismatch"):
        dsw_evidence._validate_checkpoint_publication_gate_receipt(
            mutated,
            publication_descriptor=publication_descriptor,
            publication_receipt=publication_receipt,
            checkpoint_path=checkpoint_path,
            expected_tree_sha256=tree_sha256,
        )


def test_checkpoint_publication_pre_read_gate_rejects_wrong_receipt_sha(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "formal" / "model"
    checkpoint_path.mkdir(parents=True)
    tree_sha256 = hashlib.sha256(b"tree").hexdigest()
    publication_path = tmp_path / "checkpoint_atomic_publication.json"
    _write_json(
        publication_path,
        _checkpoint_publication_receipt(checkpoint_path, tree_sha256),
        mode=0o444,
    )

    with pytest.raises(ValueError, match="receipt SHA-256 mismatch"):
        dsw_evidence.verify_checkpoint_publication(
            publication_receipt_path=publication_path,
            expected_publication_receipt_sha256=SHA,
            checkpoint_path=checkpoint_path,
            expected_tree_sha256=tree_sha256,
            output=tmp_path / "gate.json",
        )
    assert not (tmp_path / "gate.json").exists()


@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    [
        ("empty_probe_log", "must be non-empty"),
        ("forged_probe_pid", "probe PID"),
        ("forged_observed_uuid", "claims differ from parsed raw evidence"),
    ],
)
def test_egl_mapping_receipt_rejects_empty_or_forged_raw_evidence(
    tmp_path: Path,
    mutation: str,
    error_pattern: str,
) -> None:
    runtime = _runtime_receipt()
    gpu_identity, mapping_path = _attach_runtime_egl_fixture(tmp_path, runtime)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    row = mapping["schemes"]["isolated_physical_identity"]["rows"][0]
    probe_descriptor = row["evidence"]["probe_log"]
    if mutation in {"empty_probe_log", "forged_probe_pid"}:
        probe_path = Path(probe_descriptor["path"])
        if mutation == "empty_probe_log":
            replacement = b""
        else:
            lines = probe_path.read_text(encoding="utf-8").splitlines()
            ready = json.loads(lines[0])
            ready["pid"] += 999
            replacement = (
                json.dumps(ready, sort_keys=True)
                + "\n"
                + lines[1]
                + "\n"
            ).encode()
        probe_path.chmod(0o644)
        probe_path.write_bytes(replacement)
        probe_path.chmod(0o444)
        probe_descriptor["size_bytes"] = len(replacement)
        probe_descriptor["sha256"] = hashlib.sha256(replacement).hexdigest()
    else:
        row["observed_gpu_uuid"] = gpu_identity[1]["uuid"]

    with pytest.raises(ValueError, match=error_pattern):
        dsw_evidence._validate_egl_mapping_receipt(
            mapping,
            robocasa_environment=Path(
                runtime["bindings"]["robocasa_environment_path"]
            ),
            robocasa_ready_marker_sha256=runtime["bindings"][
                "robocasa_ready_marker"
            ]["sha256"],
            egl_vendor_descriptor=runtime["bindings"]["egl_vendor_json"],
            live_gpu_identity=gpu_identity,
        )


@pytest.mark.parametrize(
    ("target", "operation"),
    [
        ("mapping_receipt", "mutate"),
        ("vendor_json", "mutate"),
        ("probe_script", "mutate"),
        ("raw_evidence", "mutate"),
        ("raw_evidence", "delete"),
    ],
)
def test_runtime_egl_closure_reverification_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: str,
    operation: str,
) -> None:
    runtime = _runtime_receipt()
    gpu_identity, mapping_path = _attach_runtime_egl_fixture(tmp_path, runtime)
    monkeypatch.setattr(
        dsw_evidence,
        "_nvidia_smi_gpu_identity",
        lambda: gpu_identity,
    )
    dsw_evidence._reverify_runtime_egl_closure(runtime)
    if target == "mapping_receipt":
        target_path = mapping_path
    elif target == "vendor_json":
        target_path = Path(runtime["bindings"]["egl_vendor_json"]["path"])
    else:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if target == "probe_script":
            target_path = Path(mapping["probe"]["script"]["path"])
        else:
            target_path = Path(
                mapping["schemes"]["isolated_physical_identity"]["rows"][0][
                    "evidence"
                ]["nvidia_smi_pmon"]["path"]
            )
    if operation == "delete":
        target_path.unlink()
    else:
        original_mode = stat.S_IMODE(target_path.stat().st_mode)
        target_path.chmod(0o644)
        target_path.write_bytes(target_path.read_bytes() + b"drift\n")
        target_path.chmod(original_mode)

    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        dsw_evidence._reverify_runtime_egl_closure(runtime)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_final_root",
        "tree_drift",
        "file_drift",
        "missing_rollback_guarantee",
        "missing_receipt_guarantee",
        "directory_visibility_claim",
        "missing_commit_atomicity",
        "missing_reservation",
        "missing_commit_point",
        "rollback_after_commit",
    ],
)
def test_checkpoint_atomic_publication_receipt_semantics_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    checkpoint_path = tmp_path / "formal" / "model"
    checkpoint_path.mkdir(parents=True)
    tree_sha256 = hashlib.sha256(b"tree").hexdigest()
    receipt = _checkpoint_publication_receipt(
        checkpoint_path,
        tree_sha256,
    )
    dsw_evidence._validate_checkpoint_atomic_publication_receipt(
        receipt,
        checkpoint_path=checkpoint_path,
        expected_tree_sha256=tree_sha256,
    )
    mutated = deepcopy(receipt)
    if mutation == "wrong_final_root":
        mutated["final_root"] = str(tmp_path / "other")
    elif mutation == "tree_drift":
        mutated["postpublication"]["model"]["tree_sha256"] = SHA
    elif mutation == "file_drift":
        mutated["postpublication"]["model"]["files"] = deepcopy(
            mutated["postpublication"]["model"]["files"]
        )
        mutated["postpublication"]["model"]["files"][0]["sha256"] = SHA
    elif mutation == "missing_rollback_guarantee":
        mutated["publication"]["rollback_on_precommit_failure"] = False
    elif mutation == "missing_receipt_guarantee":
        mutated["publication"]["receipt_required_for_success"] = False
    elif mutation == "directory_visibility_claim":
        mutated["publication"]["directory_visibility_atomic"] = True
    elif mutation == "missing_commit_atomicity":
        mutated["publication"]["committed_visibility_atomic"] = False
    elif mutation == "missing_reservation":
        mutated["publication"]["final_root_exclusive_reservation"] = False
    elif mutation == "missing_commit_point":
        mutated["publication"]["receipt_is_commit_point"] = False
    else:
        mutated["publication"]["no_rollback_after_receipt_commit"] = False

    with pytest.raises(ValueError):
        dsw_evidence._validate_checkpoint_atomic_publication_receipt(
            mutated,
            checkpoint_path=checkpoint_path,
            expected_tree_sha256=tree_sha256,
        )


def test_checkpoint_publication_receipt_rejects_reappeared_hidden_stage(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "formal" / "model"
    checkpoint_path.mkdir(parents=True)
    tree_sha256 = hashlib.sha256(b"tree").hexdigest()
    receipt = _checkpoint_publication_receipt(
        checkpoint_path,
        tree_sha256,
    )
    Path(receipt["stage_root"]).mkdir()

    with pytest.raises(ValueError, match="hidden stage must be absent"):
        dsw_evidence._validate_checkpoint_atomic_publication_receipt(
            receipt,
            checkpoint_path=checkpoint_path,
            expected_tree_sha256=tree_sha256,
        )


def test_one_episode_gate_is_closed_loop_validated_and_excluded_from_primary_340(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model"
    checkpoint_path.mkdir()
    runtime_path = tmp_path / "runtime.json"
    _write_json(
        runtime_path,
        _runtime_receipt(checkpoint_path=str(checkpoint_path)),
        mode=0o444,
    )
    run_name = "formal-one-episode-gate"
    task = "CloseBlenderLid"
    gate_results = tmp_path / "gate-results"
    gate_results.mkdir()
    args = {
        "tasks": [task],
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
    results = {
        task: {
            "complete": True,
            "split": "atomic",
            "episodes": 1,
            "episode_results": [
                {
                    "success": False,
                    "place_anytime": False,
                    "steps": 2,
                    "action_log": [[0.0] * 7, [0.25] * 7],
                }
            ],
        }
    }
    _write_json(gate_results / "args.json", args, mode=0o444)
    _write_json(gate_results / "results_shard0.json", results, mode=0o444)

    receipt = dsw_evidence.verify_one_episode_gate(
        gate_results_dir=gate_results,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        task=task,
        runtime_receipt=runtime_path,
        output=tmp_path / "gate-receipt.json",
    )

    assert receipt["status"] == "passed"
    assert receipt["episode_count"] == 1
    assert receipt["role"] == "preflight_only_not_part_of_340_primary_metric"
    assert receipt["validation"]["every_action_is_finite_width_7"] is True

    results[task]["episode_results"][0]["action_log"][0] = [0.0] * 6
    gate_result_path = gate_results / "results_shard0.json"
    gate_result_path.chmod(0o600)
    _write_json(gate_result_path, results, mode=0o444)
    with pytest.raises(ValueError, match="finite width-7"):
        dsw_evidence.verify_one_episode_gate(
            gate_results_dir=gate_results,
            checkpoint_path=checkpoint_path,
            run_name=run_name,
            task=task,
            runtime_receipt=runtime_path,
            output=tmp_path / "invalid-gate-receipt.json",
        )


def test_qwen_manifest_verifier_checks_complete_tree_and_publishes_receipt(
    tmp_path: Path,
) -> None:
    tokenizer, manifest, manifest_sha = _qwen_manifest_fixture(tmp_path)
    output = tmp_path / "qwen-verification.json"

    receipt = dsw_evidence.verify_qwen_tokenizer_manifest(
        tokenizer_path=tokenizer,
        manifest_path=manifest,
        expected_manifest_sha256=manifest_sha,
        output=output,
    )

    assert receipt["kind"] == "cwam_qwen_tokenizer_manifest_verification"
    assert receipt["manifest"]["sha256"] == manifest_sha
    assert receipt["file_count"] == 5
    assert {entry["relative_path"] for entry in receipt["files"]} == {
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab/merges.txt",
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o444


def test_qwen_manifest_verifier_rejects_missing_required_and_extra_files(
    tmp_path: Path,
) -> None:
    tokenizer, manifest, _manifest_sha = _qwen_manifest_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["files"] = [entry for entry in value["files"] if entry["relative_path"] != "tokenizer_config.json"]
    _write_json(manifest, value)
    missing_required_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="missing required files"):
        dsw_evidence.verify_qwen_tokenizer_manifest(
            tokenizer_path=tokenizer,
            manifest_path=manifest,
            expected_manifest_sha256=missing_required_sha,
            output=tmp_path / "missing-required.json",
        )

    tokenizer, manifest, manifest_sha = _qwen_manifest_fixture(tmp_path / "extra")
    (tokenizer / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="does not exactly match"):
        dsw_evidence.verify_qwen_tokenizer_manifest(
            tokenizer_path=tokenizer,
            manifest_path=manifest,
            expected_manifest_sha256=manifest_sha,
            output=tmp_path / "extra-file.json",
        )


def test_qwen_manifest_verifier_rejects_traversal_hash_drift_and_symlinks(
    tmp_path: Path,
) -> None:
    tokenizer, manifest, _manifest_sha = _qwen_manifest_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["files"].append(
        {
            "relative_path": "../escape.json",
            "size_bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
    )
    _write_json(manifest, value)
    traversal_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not canonical"):
        dsw_evidence.verify_qwen_tokenizer_manifest(
            tokenizer_path=tokenizer,
            manifest_path=manifest,
            expected_manifest_sha256=traversal_sha,
            output=tmp_path / "traversal.json",
        )

    tokenizer, manifest, manifest_sha = _qwen_manifest_fixture(tmp_path / "hash")
    (tokenizer / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size/hash mismatch"):
        dsw_evidence.verify_qwen_tokenizer_manifest(
            tokenizer_path=tokenizer,
            manifest_path=manifest,
            expected_manifest_sha256=manifest_sha,
            output=tmp_path / "hash-drift.json",
        )

    tokenizer, manifest, manifest_sha = _qwen_manifest_fixture(tmp_path / "symlink")
    (tokenizer / "linked-config.json").symlink_to(tokenizer / "config.json")
    with pytest.raises(ValueError, match="symlink or non-regular"):
        dsw_evidence.verify_qwen_tokenizer_manifest(
            tokenizer_path=tokenizer,
            manifest_path=manifest,
            expected_manifest_sha256=manifest_sha,
            output=tmp_path / "symlink.json",
        )


def test_existing_aggregator_emits_and_reverifies_exact_340_episode_csvs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    run_name = "formal-dsw-test"
    results_dir = tmp_path / "results" / run_name
    _raw_results(results_dir, checkpoint_path=checkpoint_path, run_name=run_name)
    pre_paths, post_paths, provenance_descriptors = _provenance_fixture(
        tmp_path / "provenance"
    )
    common_paths = {}
    for name in (
        "config.yaml",
        "action_stats.json",
        "Wan2.2_VAE.pth",
        "qwen-manifest.json",
        "SNAPSHOT_MANIFEST.json",
    ):
        path = tmp_path / name
        path.write_bytes(b"same-input")
        common_paths[name] = path
    common_sha = hashlib.sha256(b"same-input").hexdigest()
    checkpoint_manifest_path = tmp_path / "dsw-serving-checkpoint-manifest.json"
    checkpoint_manifest_path.write_bytes(b"checkpoint-manifest")
    checkpoint_manifest_sha = hashlib.sha256(
        checkpoint_manifest_path.read_bytes()
    ).hexdigest()
    qwen_path = tmp_path / "qwen"
    qwen_path.mkdir()
    runtime = _runtime_receipt(
        checkpoint_path=str(checkpoint_path),
        checkpoint_manifest_path=str(checkpoint_manifest_path),
        config_path=str(common_paths["config.yaml"]),
        stats_path=str(common_paths["action_stats.json"]),
        vae_path=str(common_paths["Wan2.2_VAE.pth"]),
        qwen_path=str(qwen_path),
        source_manifest_path=str(common_paths["SNAPSHOT_MANIFEST.json"]),
        qwen_manifest_path=str(common_paths["qwen-manifest.json"]),
        provenance=provenance_descriptors,
        file_sha256=common_sha,
        manifest_sha256=checkpoint_manifest_sha,
    )
    gpu_identity, mapping_receipt = _attach_runtime_egl_fixture(
        tmp_path,
        runtime,
    )
    monkeypatch.setattr(
        dsw_evidence,
        "_nvidia_smi_gpu_identity",
        lambda: gpu_identity,
    )
    runtime_receipt = tmp_path / "dsw-runtime-receipt.json"
    _write_json(runtime_receipt, runtime, mode=0o444)
    terminal_receipt = _terminal_fixture(
        tmp_path,
        runtime_path=runtime_receipt,
        runtime=runtime,
    )
    summary_csv = results_dir / "summary.csv"
    task_csv = results_dir / "task_success_rates.csv"
    evidence_json = results_dir / "evaluation_evidence.json"

    evidence = dsw_evidence.summarize(
        experiment_id="CWAM-DSW-TEST",
        logical_run_id="CWAM-DSW-TEST-A1",
        source_commit=SOURCE_COMMIT,
        run_name=run_name,
        results_dir=results_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_sha256=checkpoint_manifest_sha,
        training_config_sha256=common_sha,
        action_stats_sha256=common_sha,
        wan2pt2_vae_sha256=common_sha,
        qwen_tokenizer_manifest_sha256=common_sha,
        source_manifest_sha256=common_sha,
        demian_commit=DEMIAN_COMMIT,
        base_seed=42,
        runtime_receipt=runtime_receipt,
        terminal_receipt=terminal_receipt,
        post_provenance_receipts=post_paths,
        summary_csv=summary_csv,
        task_success_rates_csv=task_csv,
        evidence_json=evidence_json,
    )
    assert evidence["protocol"]["episode_count"] == 340
    assert evidence["inputs"]["wan2pt2_vae_sha256"] == common_sha
    assert evidence["inputs"]["qwen_tokenizer_manifest_sha256"] == common_sha
    assert evidence["execution"] == {
        "platform": {
            "provider": "Alibaba Cloud",
            "service": "PAI DSW",
            "region": "cn-beijing",
        },
        "physical_server_count": 1,
        "physical_policy_server_count": 1,
        "physical_gpu_count": 4,
        "distributed_world_size": 4,
        "logical_num_shards": 8,
        "evaluator_schedule": "sequential_shards_0_through_7",
    }
    assert stat.S_IMODE(summary_csv.stat().st_mode) == 0o444
    assert stat.S_IMODE(task_csv.stat().st_mode) == 0o444
    assert stat.S_IMODE(evidence_json.stat().st_mode) == 0o444
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 340
    with task_csv.open(newline="", encoding="utf-8") as handle:
        task_rows = list(csv.DictReader(handle))
    assert len(task_rows) == 17
    assert {row["episodes"] for row in task_rows} == {"20"}
    assert {row["execution_complete"] for row in task_rows} == {"true"}

    for shard in range(8):
        (results_dir / f"args_shard{shard}.json").chmod(0o444)
        (results_dir / f"results_shard{shard}.json").chmod(0o444)
    sealed_output = tmp_path / "sealed-evidence-verification.json"
    sealed = dsw_evidence.verify_sealed(
        evidence_json=evidence_json,
        runtime_receipt=runtime_receipt,
        terminal_receipt=terminal_receipt,
        results_dir=results_dir,
        summary_csv=summary_csv,
        task_success_rates_csv=task_csv,
        output=sealed_output,
    )
    assert sealed["status"] == "passed"
    assert sealed["counts"] == {
        "physical_server_count": 1,
        "physical_gpu_count": 4,
        "logical_num_shards": 8,
        "tasks": 17,
        "episodes": 340,
    }
    assert stat.S_IMODE(sealed_output.stat().st_mode) == 0o444
    assert sealed["validation"][
        "egl_receipt_vendor_probe_and_36_evidence_files_reverified"
    ] is True

    mapping = json.loads(mapping_receipt.read_text(encoding="utf-8"))
    raw_evidence_path = Path(
        mapping["schemes"]["isolated_physical_identity"]["rows"][0]["evidence"][
            "nvidia_smi_snapshot"
        ]["path"]
    )
    original_raw_evidence = raw_evidence_path.read_bytes()
    raw_evidence_path.chmod(0o644)
    raw_evidence_path.write_bytes(original_raw_evidence + b"tampered\n")
    raw_evidence_path.chmod(0o444)
    with pytest.raises((ValueError, RuntimeError)):
        dsw_evidence.verify_sealed(
            evidence_json=evidence_json,
            runtime_receipt=runtime_receipt,
            terminal_receipt=terminal_receipt,
            results_dir=results_dir,
            summary_csv=summary_csv,
            task_success_rates_csv=task_csv,
            output=tmp_path / "sealed-after-egl-tamper.json",
        )
    raw_evidence_path.chmod(0o644)
    raw_evidence_path.write_bytes(original_raw_evidence)
    raw_evidence_path.chmod(0o444)
    raw_evidence_path.unlink()
    with pytest.raises(FileNotFoundError):
        dsw_evidence.verify_sealed(
            evidence_json=evidence_json,
            runtime_receipt=runtime_receipt,
            terminal_receipt=terminal_receipt,
            results_dir=results_dir,
            summary_csv=summary_csv,
            task_success_rates_csv=task_csv,
            output=tmp_path / "sealed-after-egl-delete.json",
        )


def test_process_descendant_check_accepts_self_only_for_self() -> None:
    assert dsw_evidence._process_is_descendant(os.getpid(), os.getpid())
    assert not dsw_evidence._process_is_descendant(-1, os.getpid())
