# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
import yaml

from examples.h_cluster.cwam_formal_training_evidence import (
    _PROTOCOLS,
    EvidenceError,
    TrainingProtocol,
    _expected_gradient_iterations,
    _validate_dcp,
    _validate_validation_summaries,
    protocol_for_world_size,
    verify_formal,
    verify_resume,
    verify_storage,
)


def _config(*, world_size: int, max_iter: int, gradient_every_n: int = 1) -> dict:
    return {
        "trainer": {
            "distributed_parallelism": "fsdp",
            "max_iter": max_iter,
            "max_val_iter": 1,
            "grad_accum_iter": 2,
            "run_validation": True,
            "run_validation_on_start": True,
            "validation_iter": 667 if world_size == 6 else 500,
            "callbacks": {
                "action_gradient_evidence": {
                    "families": ["action2llm", "llm2action", "action_modality_embed"],
                    "every_n": gradient_every_n,
                    "fail_on_first_step": True,
                },
                "production_batch_memory_gate": {
                    "max_peak_reserved_fraction": 0.94,
                    "max_driver_used_fraction": 0.94,
                    "expected_samples_per_microbatch": 128,
                    "expected_grad_accum_iter": 2,
                    "expected_action_tokens_per_sample": 16,
                    "allowed_world_sizes": [6, 8],
                    "required": True,
                },
            },
        },
        "checkpoint": {
            "strict_resume": True,
            "save_iter": 1,
            "dcp_async_mode_enabled": False,
        },
        "optimizer": {"lr": 3.75e-5 if world_size == 6 else 5.0e-5},
        "scheduler": {
            "warm_up_steps": [667 if world_size == 6 else 500],
            "cycle_lengths": [21_344 if world_size == 6 else 16_000],
        },
        "model": {
            "config": {
                "compile": {"enabled": True},
                "ema": {"enabled": True},
                "parallelism": {
                    "data_parallel_shard_degree": world_size,
                    "data_parallel_replicate_degree": 1,
                    "context_parallel_shard_degree": 1,
                    "cfg_parallel_shard_degree": 1,
                },
            }
        },
        "dataloader_train": {
            "max_samples_per_batch": 128,
            "max_sequence_length": None,
            "dataloader": {
                "num_workers": 4,
                "persistent_workers": True,
                "prefetch_factor": 4,
            },
        },
        "dataloader_val": {
            "max_samples_per_batch": 128,
            "dataloader": {
                "num_workers": 2,
                "persistent_workers": True,
                "prefetch_factor": 4,
            },
        },
    }


def _write_dcp(root: Path, iteration: int) -> None:
    checkpoint = root / "checkpoints" / f"iter_{iteration:09d}"
    for component in ("model", "optim", "scheduler", "trainer"):
        component_dir = checkpoint / component
        dcp.save(
            {"value": torch.tensor([iteration], dtype=torch.int64)},
            checkpoint_id=component_dir,
        )


def _memory_report(
    *,
    world_size: int,
    iteration: int,
    initial_validation: bool,
    checkpoint_save: bool,
) -> dict:
    rows = []
    for rank in range(world_size):
        row = {
            "rank": rank,
            "device_index": rank,
            "total_memory_bytes": 1_000,
            "peak_allocated_bytes": 400,
            "peak_reserved_bytes": 500,
            "current_allocated_bytes": 300,
            "current_reserved_bytes": 400,
            "driver_free_bytes": 400,
            "driver_used_bytes": 600,
            "peak_allocated_fraction": 0.4,
            "peak_reserved_fraction": 0.5,
            "driver_used_fraction": 0.6,
            "observed_microbatch_count": 2,
        }
        for microbatch in range(2):
            prefix = f"microbatch_{microbatch}"
            row.update(
                {
                    f"{prefix}_input_sample_count": 128,
                    f"{prefix}_unique_identity_count": 128,
                    f"{prefix}_model_batch_size": 128,
                    f"{prefix}_und_token_length": 1,
                    f"{prefix}_gen_token_length": 1,
                    f"{prefix}_vision_token_length": 1,
                    f"{prefix}_action_token_length": 2048,
                    f"{prefix}_input_tensor_bytes": 1024,
                }
            )
        rows.append(row)
    included_phases = [
        "resident_model_optimizer_checkpoint_state",
        "2_training_microbatches",
        "first_optimizer_scheduler_ema_step",
    ]
    if initial_validation:
        included_phases.insert(1, "initial_ema_validation")
    if checkpoint_save:
        included_phases.append("first_step_dcp_save")
    return {
        "schema_version": 1,
        "status": "passed",
        "recorded_at": "2026-07-28T00:00:00+00:00",
        "scope": "first_production_optimizer_boundary_after_checkpoint_load",
        "world_size": world_size,
        "start_iteration": iteration - 1,
        "observed_iteration": iteration,
        "included_phases": included_phases,
        "violations": [],
        "production_batch_contract": {
            "expected_samples_per_microbatch": 128,
            "expected_grad_accum_iter": 2,
            "expected_action_tokens_per_sample": 16,
            "allowed_world_sizes": [6, 8],
        },
        "thresholds": {
            "max_peak_reserved_fraction": 0.94,
            "max_driver_used_fraction": 0.94,
        },
        "per_rank": rows,
    }


def _gradient_report(*, world_size: int, iteration: int) -> dict:
    stats = {
        "parameter_elements": 10,
        "trainable_elements": 10,
        "optimizer_elements": 10,
        "gradient_elements": 10,
        "nonfinite_gradient_elements": 0,
        "nonzero_gradient_elements": 10,
        "gradient_l2_norm": 1.0,
        "gradient_abs_max": 0.5,
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "world_size": world_size,
        "iteration": iteration,
        "gradient_stage": "after_backward_before_callback_clipping",
        "families": {
            "action2llm": stats,
            "llm2action": stats,
            "action_modality_embed": stats,
        },
        "losses": {
            "scope": "final_microbatch_at_optimizer_boundary",
            "final_microbatch_total_loss_rank_mean": 1.0,
            "final_microbatch_flow_matching_loss_action_rank_mean": 0.5,
        },
    }


def _sampled_memory(*, world_size: int) -> dict:
    return {
        "schema_version": 1,
        "status": "passed",
        "gpu_count": world_size,
        "sample_count": world_size * 2,
        "max_memory_used_fraction": 0.6,
        "maximum_allowed_fraction": 0.94,
        "per_gpu": {
            f"GPU-{index}": {
                "index": index,
                "name": "NVIDIA H200",
                "sample_count": 2,
                "max_memory_used_fraction": 0.6,
                "max_memory_used_mib": 90_000,
                "min_memory_free_mib": 50_000,
                "max_utilization_gpu_percent": 100.0,
            }
            for index in range(world_size)
        },
    }


def _prepare_resume_run(
    tmp_path: Path,
    *,
    world_size: int = 6,
) -> tuple[Path, Path, str, str]:
    experiment_id = "CWAM-E35A-TEST"
    formal_run_name = experiment_id
    run_dir = tmp_path / f"{formal_run_name}-prod128-probe"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(_config(world_size=world_size, max_iter=2)))
    _write_dcp(run_dir, 1)
    _write_dcp(run_dir, 2)
    (run_dir / "checkpoints" / "latest_checkpoint.txt").write_text("iter_000000002\n")

    memory_dir = run_dir / "production_batch_memory_gate"
    memory_dir.mkdir()
    (memory_dir / "iteration_000000001.json").write_text(
        json.dumps(
            _memory_report(
                world_size=world_size,
                iteration=1,
                initial_validation=True,
                checkpoint_save=True,
            )
        )
    )
    (memory_dir / "iteration_000000002.json").write_text(
        json.dumps(
            _memory_report(
                world_size=world_size,
                iteration=2,
                initial_validation=False,
                checkpoint_save=True,
            )
        )
    )
    gradient_dir = run_dir / "action_gradient_evidence"
    gradient_dir.mkdir()
    (gradient_dir / "iteration_000000001.json").write_text(
        json.dumps(_gradient_report(world_size=world_size, iteration=1))
    )
    (gradient_dir / "iteration_000000002.json").write_text(
        json.dumps(_gradient_report(world_size=world_size, iteration=2))
    )

    resumed_from = run_dir / "checkpoints" / "iter_000000001"
    log = tmp_path / "resume.log"
    log.write_text(
        "\n".join(
            (
                "WARNING: resuming model/optim/scheduler/trainer with a restarted data stream.",
                f"Resuming ckpt {resumed_from} with keys: ['dataloader', 'model', 'optim', 'scheduler', 'trainer']",
                *(
                    f"[RANK {rank}] Checkpoint {resumed_from}/dataloader does not exist, skip loading dataloader."
                    for rank in range(world_size)
                ),
                "- Loading the model...",
                "- Loading the optimizer...",
                "- Loading the scheduler...",
                "- Loading the trainer...",
                f"Loaded checkpoint from {resumed_from} in iteration 1",
                "Distributed parallelism mode: fsdp",
                "Done with training.",
            )
        )
    )
    sampled_memory = tmp_path / "resume-sampled-memory.json"
    sampled_memory.write_text(json.dumps(_sampled_memory(world_size=world_size)))
    return run_dir, log, sampled_memory, experiment_id, formal_run_name


def test_validation_contract_requires_zero_unstable_and_exact_iterations() -> None:
    text = "\n".join(
        (
            "[val] number of unique url and key: 6144 / 6144; avg_final_loss: 1.25; unstable_count: 0; iteration: 0",
            "[val] number of unique url and key: 6144 / 6144; avg_final_loss: 0.75; unstable_count: 0; iteration: 500",
        )
    )
    rows = _validate_validation_summaries(
        text,
        expected_iterations=[0, 500],
        expected_samples=6144,
    )
    assert [row["iteration"] for row in rows] == [0, 500]

    with pytest.raises(EvidenceError, match="unstable validation loss"):
        _validate_validation_summaries(
            text.replace("unstable_count: 0; iteration: 500", "unstable_count: 1; iteration: 500"),
            expected_iterations=[0, 500],
            expected_samples=6144,
        )


def test_topology_protocols_preserve_twenty_validation_and_save_boundaries() -> None:
    e35 = protocol_for_world_size(8)
    e35a = protocol_for_world_size(6)
    assert e35.global_batch == 2048
    assert e35a.global_batch == 1536
    assert e35.max_iter * e35.global_batch == 20_480_000
    assert e35a.max_iter * e35a.global_batch == 20_490_240
    assert len(e35.validation_iterations) == len(e35a.validation_iterations) == 21
    assert len(e35.checkpoint_iterations) == len(e35a.checkpoint_iterations) == 20
    assert e35.validation_iterations[-1] == 10_000
    assert e35a.validation_iterations[-1] == 13_340
    assert _expected_gradient_iterations(e35a.max_iter)[-1] == 13_340


def test_resume_contract_proves_full_state_reload_and_second_step(tmp_path) -> None:
    run_dir, log, sampled_memory, experiment_id, formal_run_name = _prepare_resume_run(tmp_path)
    result = verify_resume(
        run_dir=run_dir,
        log_path=log,
        world_size=6,
        experiment_id=experiment_id,
        formal_run_name=formal_run_name,
        sampled_memory_path=sampled_memory,
    )
    assert result["status"] == "passed"
    assert result["checkpoint"].endswith("iter_000000002")

    log.write_text(log.read_text() + "\nTraining from scratch.\n")
    with pytest.raises(EvidenceError, match="Training from scratch"):
        verify_resume(
            run_dir=run_dir,
            log_path=log,
            world_size=6,
            experiment_id=experiment_id,
            formal_run_name=formal_run_name,
            sampled_memory_path=sampled_memory,
        )


def test_resume_rejects_out_of_order_load_markers(tmp_path) -> None:
    run_dir, log, sampled_memory, experiment_id, formal_run_name = _prepare_resume_run(tmp_path)
    lines = log.read_text().splitlines()
    loaded_index = next(index for index, line in enumerate(lines) if line.startswith("Loaded checkpoint"))
    trainer_index = lines.index("- Loading the trainer...")
    lines[loaded_index], lines[trainer_index] = lines[trainer_index], lines[loaded_index]
    log.write_text("\n".join(lines) + "\n")
    with pytest.raises(EvidenceError, match="out of order"):
        verify_resume(
            run_dir=run_dir,
            log_path=log,
            world_size=6,
            experiment_id=experiment_id,
            formal_run_name=formal_run_name,
            sampled_memory_path=sampled_memory,
        )


def test_resume_rejects_duplicate_rank_and_nonfinite_memory(tmp_path) -> None:
    run_dir, log, sampled_memory, experiment_id, formal_run_name = _prepare_resume_run(tmp_path)
    memory_path = run_dir / "production_batch_memory_gate/iteration_000000002.json"
    memory = json.loads(memory_path.read_text())
    memory["per_rank"][1]["rank"] = 0
    memory["per_rank"][1]["device_index"] = 0
    memory_path.write_text(json.dumps(memory))
    with pytest.raises(EvidenceError, match="rank set"):
        verify_resume(
            run_dir=run_dir,
            log_path=log,
            world_size=6,
            experiment_id=experiment_id,
            formal_run_name=formal_run_name,
            sampled_memory_path=sampled_memory,
        )

    memory = _memory_report(
        world_size=6,
        iteration=2,
        initial_validation=False,
        checkpoint_save=True,
    )
    memory["per_rank"][0]["peak_reserved_fraction"] = float("nan")
    memory_path.write_text(json.dumps(memory))
    with pytest.raises(EvidenceError, match="invalid peak_reserved_fraction"):
        verify_resume(
            run_dir=run_dir,
            log_path=log,
            world_size=6,
            experiment_id=experiment_id,
            formal_run_name=formal_run_name,
            sampled_memory_path=sampled_memory,
        )


def test_resume_rejects_extra_gradient_family(tmp_path) -> None:
    run_dir, log, sampled_memory, experiment_id, formal_run_name = _prepare_resume_run(tmp_path)
    gradient_path = run_dir / "action_gradient_evidence/iteration_000000002.json"
    gradient = json.loads(gradient_path.read_text())
    gradient["families"]["unexpected"] = gradient["families"]["action2llm"]
    gradient_path.write_text(json.dumps(gradient))
    with pytest.raises(EvidenceError, match="family set"):
        verify_resume(
            run_dir=run_dir,
            log_path=log,
            world_size=6,
            experiment_id=experiment_id,
            formal_run_name=formal_run_name,
            sampled_memory_path=sampled_memory,
        )


def test_dcp_rejects_nonparseable_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "iter_000000001"
    for component in ("model", "optim", "scheduler", "trainer"):
        component_dir = checkpoint / component
        component_dir.mkdir(parents=True)
        (component_dir / ".metadata").write_bytes(b"not-a-dcp")
        (component_dir / "__0_0.distcp").write_bytes(b"not-a-shard")
    with pytest.raises(EvidenceError, match="not parseable"):
        _validate_dcp(checkpoint)


def test_storage_gate_rejects_threshold_below_estimate(tmp_path) -> None:
    with pytest.raises(EvidenceError, match="formal minimum free storage"):
        verify_storage(
            path=tmp_path,
            minimum_free_bytes=1_000,
            estimated_required_bytes=2_000,
        )


def test_formal_contract_binds_storage_reservation_and_provenance(tmp_path, monkeypatch) -> None:
    protocol = TrainingProtocol(
        experiment_family="CWAM-E35",
        world_size=8,
        learning_rate=5.0e-5,
        max_iter=100,
        warm_up_steps=5,
        cycle_length=160,
        validation_iter=50,
        max_val_iter=1,
        save_iter=50,
    )
    monkeypatch.setitem(_PROTOCOLS, 8, protocol)
    experiment_id = "CWAM-E35-TEST"
    formal_run_name = experiment_id
    output_root = tmp_path / "formal-output"
    monkeypatch.setattr(
        "examples.h_cluster.cwam_formal_training_evidence._FORMAL_OUTPUT_ROOT",
        output_root,
    )
    formal_log_root = tmp_path / "formal-logs"
    formal_log_root.mkdir()
    monkeypatch.setattr(
        "examples.h_cluster.cwam_formal_training_evidence._FORMAL_LOG_ROOT",
        formal_log_root,
    )
    run_dir = output_root / "cosmos3_action_robocasa365" / "action_sft" / formal_run_name
    run_dir.mkdir(parents=True)
    config = _config(world_size=8, max_iter=protocol.max_iter, gradient_every_n=50)
    config["trainer"]["max_val_iter"] = protocol.max_val_iter
    config["trainer"]["validation_iter"] = protocol.validation_iter
    config["checkpoint"]["save_iter"] = protocol.save_iter
    config["scheduler"]["warm_up_steps"] = [protocol.warm_up_steps]
    config["scheduler"]["cycle_lengths"] = [protocol.cycle_length]
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))

    memory_dir = run_dir / "production_batch_memory_gate"
    memory_dir.mkdir()
    (memory_dir / "iteration_000000001.json").write_text(
        json.dumps(
            _memory_report(
                world_size=8,
                iteration=1,
                initial_validation=True,
                checkpoint_save=False,
            )
        )
    )
    gradient_dir = run_dir / "action_gradient_evidence"
    gradient_dir.mkdir()
    for iteration in (1, 50, 100):
        (gradient_dir / f"iteration_{iteration:09d}.json").write_text(
            json.dumps(_gradient_report(world_size=8, iteration=iteration))
        )
    _write_dcp(run_dir, 50)
    _write_dcp(run_dir, 100)
    (run_dir / "checkpoints" / "latest_checkpoint.txt").write_text("iter_000000100\n")

    formal_log = formal_log_root / f"{formal_run_name}.formal.log"
    validation_lines = [
        f"[val] number of unique url and key: 1024 / 1024; "
        f"avg_final_loss: 1.0; unstable_count: 0; iteration: {iteration}"
        for iteration in (0, 50, 100)
    ]
    formal_log.write_text(
        "\n".join(
            (
                "Distributed parallelism mode: fsdp",
                *validation_lines,
                "Done with training.",
            )
        )
        + "\n"
    )

    evidence_dir = output_root / "formal_evidence" / formal_run_name
    evidence_dir.mkdir(parents=True)
    evidence_metadata = evidence_dir.stat()
    path_security_path = evidence_dir / "path-security.json"
    path_security_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "operation": "create_directory_exclusive",
                "root": str(output_root),
                "target": str(evidence_dir),
                "relative_path": f"formal_evidence/{formal_run_name}",
                "created": True,
                "created_directories": [f"formal_evidence/{formal_run_name}"],
                "exclusive": True,
                "atomic_leaf_creation": True,
                "strictly_within_root": True,
                "resolved_strict": True,
                "symlink_policy": "reject_every_existing_component",
                "target_kind": "directory",
                "target_state": "existing",
                "root_identity": {},
                "target_identity": {
                    "device": int(evidence_metadata.st_dev),
                    "inode": int(evidence_metadata.st_ino),
                    "kind": "directory",
                    "mode": f"{evidence_metadata.st_mode & 0o777:04o}",
                },
            }
        )
    )
    expected_probe_dir = str(run_dir.parent / f"{formal_run_name}-prod128-probe")
    probe = {
        "schema_version": 1,
        "status": "passed",
        "experiment_family": "CWAM-E35",
        "experiment_id": experiment_id,
        "formal_run_name": formal_run_name,
        "probe_run_name": f"{formal_run_name}-prod128-probe",
        "world_size": 8,
        "run_dir": expected_probe_dir,
    }
    resume = {
        **probe,
        "resumed_from": f"{expected_probe_dir}/checkpoints/iter_000000001",
        "data_stream_resume": "restarted_from_beginning_not_exact_loader_cursor",
    }
    probe_path = evidence_dir / "probe-assertions.json"
    resume_path = evidence_dir / "resume-assertions.json"
    probe_path.write_text(json.dumps(probe))
    resume_path.write_text(json.dumps(resume))

    formal_manifest = tmp_path / "formal-manifest.json"
    formal_manifest.write_text(json.dumps({"experiment_id": experiment_id}))
    formal_manifest_sha256 = hashlib.sha256(formal_manifest.read_bytes()).hexdigest()
    source_manifest_sha256 = "a" * 64
    workflow_snapshot_sha256 = "b" * 64
    source_commit = "c" * 40
    container_image = f"registry.example/cwam@sha256:{'d' * 64}"
    rjob_name = "cwam-e35-test-rjob"
    scheduler_rjob_name = f"{rjob_name}-12345678"
    scheduler_rjob_uid = "11111111-1111-1111-1111-111111111111"
    scheduler_replica_name = f"{scheduler_rjob_name}-abcde"
    scheduler_pod_name = "rjob-fixture-0"
    snapshot_root = str(tmp_path / "snapshot")
    now = datetime.now(timezone.utc)
    created_at = now - timedelta(seconds=30)
    accepted_at = now - timedelta(seconds=20)
    owner_acquired_at = now - timedelta(seconds=10)
    storage_recorded_at = now - timedelta(seconds=5)
    owner = {
        "schema_version": 1,
        "status": "owned",
        "acquired_at": owner_acquired_at.isoformat(),
        "experiment_id": experiment_id,
        "run_name": formal_run_name,
        "rjob_name": rjob_name,
        "scheduler_rjob_name": scheduler_rjob_name,
        "scheduler_rjob_uid": scheduler_rjob_uid,
        "scheduler_replica_name": scheduler_replica_name,
        "scheduler_pod_name": scheduler_pod_name,
        "scheduler_receipt": "",
        "scheduler_receipt_sha256": "",
        "hostname": scheduler_pod_name,
        "node_name": "lg-cmc-h-h200-fixture.host.h.pjlab.org.cn",
        "shell_pid": 1234,
        "world_size": 8,
        "container_image": container_image,
        "source_manifest_sha256": source_manifest_sha256,
        "formal_manifest_sha256": formal_manifest_sha256,
    }
    provenance = {
        "schema_version": 1,
        "status": "passed",
        "experiment_id": experiment_id,
        "snapshot_root": snapshot_root,
        "formal_manifest": str(formal_manifest),
        "formal_manifest_sha256": formal_manifest_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_commit": source_commit,
        "workflow_snapshot_sha256": workflow_snapshot_sha256,
        "container_image": container_image,
        "nproc_per_node": 8,
        "storage": {
            "output_root": str(output_root),
            "minimum_free_bytes": 5 * 1024**4,
            "estimated_required_bytes": 4 * 1024**4,
        },
        "rjob": {
            "quota_group": "eailabagent_gpu",
            "cpu": 128,
            "memory_mib": 1_500_000,
            "negative_tag": "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn",
            "task_type": "normal",
            "private_machine": "group",
            "priority": 9,
            "restart_policy": "never",
            "backoff_limit": 1,
        },
    }
    storage = {
        "schema_version": 1,
        "status": "passed",
        "path": str(output_root),
        "filesystem_total_bytes": 50 * 1024**4,
        "filesystem_available_bytes": 29 * 1024**4,
        "minimum_free_bytes": 5 * 1024**4,
        "estimated_required_bytes": 4 * 1024**4,
        "headroom_after_estimate_bytes": 25 * 1024**4,
        "recorded_at": storage_recorded_at.isoformat(),
    }
    reservation = {
        "schema_version": 2,
        "status": "accepted",
        "created_at": created_at.isoformat(),
        "accepted_at": accepted_at.isoformat(),
        "experiment_id": experiment_id,
        "run_name": formal_run_name,
        "rjob_name": rjob_name,
        "scheduler_rjob_name": scheduler_rjob_name,
        "world_size": 8,
        "container_image": container_image,
        "formal_manifest_sha256": formal_manifest_sha256,
        "source_commit": source_commit,
        "workflow_snapshot_sha256": workflow_snapshot_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "formal_output_root": str(output_root),
        "minimum_free_bytes": 5 * 1024**4,
        "estimated_required_bytes": 4 * 1024**4,
        "quota_group": "eailabagent_gpu",
        "cpu": 128,
        "memory_mib": 1_500_000,
        "negative_tag": "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn",
        "task_type": "normal",
        "private_machine": "group",
        "priority": 9,
        "restart_policy": "never",
        "backoff_limit": 1,
    }
    owner_path = evidence_dir / "owner.json"
    provenance_path = evidence_dir / "provenance.json"
    storage_path = evidence_dir / "formal-storage-preflight.json"
    sampled_memory_path = evidence_dir / "formal-gpu-memory.json"
    reservation_path = output_root / "formal_submission_reservations" / formal_run_name / "submission.json"
    reservation_path.parent.mkdir(parents=True)
    submission_log_path = reservation_path.parent / "rjob-submit.log"
    scheduler_expected_contract_path = reservation_path.parent / "rjob-expected-contract.json"
    scheduler_capture_log_path = reservation_path.parent / "rjob-scheduler-capture.log"
    scheduler_receipt_path = reservation_path.parent / "rjob-scheduler-receipt.json"
    scheduler_text_receipt_path = reservation_path.parent / "rjob-get.log"
    submission_log_path.write_text(f"created rjob_name: {scheduler_rjob_name}\n")
    scheduler_expected_contract = {
        "schema_version": 1,
        "scheduler_rjob_name": scheduler_rjob_name,
        "showname": rjob_name,
        "quota_group": "eailabagent_gpu",
        "task_type": "normal",
        "priority": 9,
        "backoff_limit": 1,
        "task_name": "t0",
        "replicas": 1,
        "image": container_image,
        "cpu": 128,
        "memory_mib": 1_500_000,
        "gpu": 8,
        "negative_tags": ["node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"],
        "private_machine": "group",
        "restart_policy": "Never",
        "termination_grace_period_seconds": 60,
        "mounts": [
            "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs",
            "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public",
        ],
        "snapshot_root": snapshot_root,
        "wrapper_path": (f"{snapshot_root}/examples/h_cluster/run_cwam_cosmos3_nano_production_then_formal.sh"),
        "experiment_id": experiment_id,
        "command_sha256": hashlib.sha256(b"fixture").hexdigest(),
    }
    scheduler_expected_contract_path.write_text(json.dumps(scheduler_expected_contract))
    scheduler_capture_log_path.write_text("CWAM_FORMAL_RJOB_RECEIPT_CAPTURED\n")
    scheduler_receipt = {
        "schema_version": 1,
        "kind": "cwam_formal_rjob_scheduler_receipt",
        "captured_at": accepted_at.isoformat(),
        "validation": {"status": "passed", "expected_contract_schema_version": 1},
        "metadata": {
            "name": scheduler_rjob_name,
            "showname": rjob_name,
            "uid": scheduler_rjob_uid,
            "resource_version": "1",
            "creation_timestamp": created_at.isoformat(),
        },
        "spec": {
            "quota_group": "eailabagent_gpu",
            "task_type": "normal",
            "priority": 9,
            "backoff_limit": 1,
            "task": {
                "name": "t0",
                "replicas": 1,
                "negative_tags": ["node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"],
                "private_machine": "group",
                "restart_policy": "Never",
                "termination_grace_period_seconds": 60,
                "mounts": scheduler_expected_contract["mounts"],
                "container": {
                    "name": "t0",
                    "image": container_image,
                    "resources": {
                        "requests_equal_core_limits": True,
                        "cpu": 128,
                        "memory_mib": 1_500_000,
                        "gpu": 8,
                        "platform_injected_limit_keys": [],
                    },
                    "command": ["bash", "-lc", "fixture"],
                    "snapshot_root": snapshot_root,
                    "wrapper_path": scheduler_expected_contract["wrapper_path"],
                    "experiment_id": experiment_id,
                },
            },
        },
        "status": {"phase": "Inqueue"},
        "replica_names": [scheduler_replica_name],
        "pod_names": [scheduler_pod_name],
        "replicas": [
            {
                "replica_name": scheduler_replica_name,
                "pod_name": scheduler_pod_name,
                "pod_hostname": scheduler_pod_name,
                "worker_spec": {
                    "image": container_image,
                    "resources": {
                        "requests_equal_core_limits": True,
                        "cpu": 128,
                        "memory_mib": 1_500_000,
                        "gpu": 8,
                        "platform_injected_limit_keys": [
                            "mellanox.com/mlnx_rdma",
                            "rdma/mlnx_shared",
                        ],
                    },
                    "restart_policy": "Never",
                    "termination_grace_period_seconds": 60,
                    "mounts": scheduler_expected_contract["mounts"],
                    "command": [
                        "/kubebrain/worker-init",
                        "run",
                        "--enable-pid-monitor",
                        "--metrics-addr=",
                        "--",
                        "bash",
                        "-lc",
                        "fixture",
                    ],
                    "worker_init_prefix": [
                        "/kubebrain/worker-init",
                        "run",
                        "--enable-pid-monitor",
                        "--metrics-addr=",
                        "--",
                    ],
                },
            }
        ],
        "source_objects": {
            "rjob_raw_sha256": "e" * 64,
            "replicas_raw_sha256": "f" * 64,
        },
    }
    scheduler_receipt_path.write_text(json.dumps(scheduler_receipt))
    scheduler_text_receipt_path.write_text(f"rjob {scheduler_rjob_name} (showname={rjob_name}): Inqueue\n")
    reservation.update(
        {
            "submission_log": str(submission_log_path),
            "submission_log_sha256": hashlib.sha256(submission_log_path.read_bytes()).hexdigest(),
            "scheduler_expected_contract": str(scheduler_expected_contract_path),
            "scheduler_expected_contract_sha256": hashlib.sha256(
                scheduler_expected_contract_path.read_bytes()
            ).hexdigest(),
            "scheduler_capture_log": str(scheduler_capture_log_path),
            "scheduler_capture_log_sha256": hashlib.sha256(scheduler_capture_log_path.read_bytes()).hexdigest(),
            "scheduler_receipt": str(scheduler_receipt_path),
            "scheduler_receipt_sha256": hashlib.sha256(scheduler_receipt_path.read_bytes()).hexdigest(),
            "scheduler_text_receipt": str(scheduler_text_receipt_path),
            "scheduler_text_receipt_sha256": hashlib.sha256(scheduler_text_receipt_path.read_bytes()).hexdigest(),
        }
    )
    owner["scheduler_receipt"] = str(scheduler_receipt_path)
    owner["scheduler_receipt_sha256"] = reservation["scheduler_receipt_sha256"]
    owner_path.write_text(json.dumps(owner))
    provenance_path.write_text(json.dumps(provenance))
    storage_path.write_text(json.dumps(storage))
    sampled_memory_path.write_text(json.dumps(_sampled_memory(world_size=8)))
    reservation_path.write_text(json.dumps(reservation))
    submission_log_path.chmod(0o444)
    scheduler_expected_contract_path.chmod(0o444)
    scheduler_capture_log_path.chmod(0o444)
    scheduler_receipt_path.chmod(0o444)
    scheduler_text_receipt_path.chmod(0o444)
    reservation_path.chmod(0o444)

    result = verify_formal(
        run_dir=run_dir,
        log_path=formal_log,
        world_size=8,
        probe_assertions=probe_path,
        resume_assertions=resume_path,
        experiment_id=experiment_id,
        formal_manifest=formal_manifest,
        formal_manifest_sha256=formal_manifest_sha256,
        owner_path=owner_path,
        provenance_path=provenance_path,
        formal_run_name=formal_run_name,
        storage_preflight_path=storage_path,
        submission_reservation_path=reservation_path,
        sampled_memory_path=sampled_memory_path,
    )
    assert result["status"] == "training_completed"
    assert result["checkpoint_count"] == 2
    assert result["submission_reservation_sha256"] == hashlib.sha256(reservation_path.read_bytes()).hexdigest()

    reservation["world_size"] = 6
    reservation_path.chmod(0o644)
    reservation_path.write_text(json.dumps(reservation))
    reservation_path.chmod(0o444)
    with pytest.raises(EvidenceError, match="submission-reservation world_size"):
        verify_formal(
            run_dir=run_dir,
            log_path=formal_log,
            world_size=8,
            probe_assertions=probe_path,
            resume_assertions=resume_path,
            experiment_id=experiment_id,
            formal_manifest=formal_manifest,
            formal_manifest_sha256=formal_manifest_sha256,
            owner_path=owner_path,
            provenance_path=provenance_path,
            formal_run_name=formal_run_name,
            storage_preflight_path=storage_path,
            submission_reservation_path=reservation_path,
            sampled_memory_path=sampled_memory_path,
        )
