# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import argparse
import hashlib
import json
from pathlib import Path

import pytest

import examples.h_cluster.cwam_cosmos3_formal_continuation as continuation
from examples.h_cluster.cwam_cosmos3_formal_continuation import (
    OFFICIAL_WAN22_VAE_SHA256,
    ContinuationError,
    build_contract,
    fingerprint_checkpoint,
    prepare_run,
    validate_throughput_matrix,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache(root: Path) -> tuple[str, str]:
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "cache_id": "cache-fixture",
        "equivalence": {"status": "passed", "torch_equal": True},
        "shards": [{"shard_id": 0}],
        "splits": {"train": {"mapping_path": "mapping.npy"}},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    manifest_sha = _sha256(manifest_path)
    success_path = root / "_SUCCESS.json"
    success_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "cache_id": "cache-fixture",
                "manifest_path": "manifest.json",
                "manifest_sha256": manifest_sha,
            },
            sort_keys=True,
        )
    )
    return _sha256(success_path), manifest_sha


def _matrix(
    path: Path,
    *,
    selected_status: str = "passed",
    cache_root: Path | None = None,
    cache_manifest_sha: str | None = None,
    checkpoint: Path | None = None,
) -> str:
    run_root = path.parent
    cache_root = cache_root or run_root / "benchmark-cache"
    cache_root.mkdir(exist_ok=True)
    cache_manifest_sha = cache_manifest_sha or "a" * 64
    checkpoint = checkpoint or run_root / "benchmark-r6" / "iter_000001000"
    (run_root / "snapshot").mkdir(exist_ok=True)
    (run_root / "Wan2.2_VAE.pth").write_bytes(b"wan-vae-fixture")
    fixed_trace = run_root / "fixed-index-trace.npy"
    fixed_trace.write_bytes(b"fixed-trace")
    fixed_trace_manifest = fixed_trace.with_suffix(".manifest.json")
    fixed_trace_manifest.write_text('{"status":"complete"}')
    node_evidence = run_root / "node-evidence.json"
    node_evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "expected_gpu_count": 8,
                "visible_gpu_count": 8,
                "training_rank_count": 8,
                "gpus": [{"name": "NVIDIA H200"} for _ in range(8)],
            },
            sort_keys=True,
        )
    )
    trace_steps = [
        {
            "relative_optimizer_step": step,
            "completed_iteration": 1_000 + step,
            "identity_count": 2_048,
            "identity_sha256": hashlib.sha256(f"fixed-step-{step}".encode()).hexdigest(),
        }
        for step in range(1, 61)
    ]
    trace_sha = hashlib.sha256(
        json.dumps(
            trace_steps,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    def passed_cell(
        workers: int,
        storage: str,
        activation_checkpointing: str,
        mean_step_seconds: float,
        *,
        status: str = "passed",
    ) -> dict[str, object]:
        start_ns = 1_000_000_000
        end_ns = 2_000_000_000
        rows = [
            {
                "rank": rank,
                "peak_allocated_fraction": 0.30 + rank / 1_000,
                "peak_reserved_fraction": 0.40 + rank / 1_000,
                "driver_used_fraction": 0.50 + rank / 1_000,
            }
            for rank in range(8)
        ]
        cell_id = f"workers-{workers}__storage-{storage}__ac-{activation_checkpointing}"
        return {
            "schema_version": 1,
            "status": status,
            "summarized_at": "2026-07-29T00:00:00+00:00",
            "cell_id": cell_id,
            "workers_per_rank": workers,
            "node_total_workers": workers * 8,
            "storage": storage,
            "activation_checkpointing": activation_checkpointing,
            "training_exit_code": 0 if status == "passed" else 1,
            "log_sha256": "1" * 64,
            "optimizer_step_ndjson_sha256": "2" * 64,
            "timing": {
                "observed_optimizer_steps": 60,
                "contiguous": True,
                "warmup_steps": 10,
                "measure_steps": 50,
                "first_measured_iteration": 1_011,
                "last_measured_iteration": 1_060,
                "measured_start_unix_ns": start_ns,
                "measured_end_unix_ns": end_ns,
                "mean_step_seconds": mean_step_seconds,
                "median_step_seconds": mean_step_seconds,
                "p90_step_seconds": mean_step_seconds,
                "p95_step_seconds": mean_step_seconds,
                "min_step_seconds": mean_step_seconds,
                "max_step_seconds": mean_step_seconds,
                "optimizer_steps_per_second": 1.0 / mean_step_seconds,
                "mean_timer_dataloader_train_seconds": (mean_step_seconds / 10),
                "p95_timer_dataloader_train_seconds": (mean_step_seconds / 10),
                "mean_callback_loader_wait_seconds": (mean_step_seconds / 10),
                "mean_timer_dataloader_fraction": 0.1,
            },
            "gpu": {
                "window_start_unix_ns": start_ns,
                "window_end_unix_ns": end_ns,
                "sample_batches": 2,
                "sample_rows": 16,
                "gpu_count": 8,
                "gpu_uuids": [f"GPU-{rank}" for rank in range(8)],
                "average_utilization_percent": 90.0,
                "p50_utilization_percent": 95.0,
                "average_power_draw_w": 600.0,
                "peak_memory_used_mib": 50_000.0,
                "peak_memory_used_fraction": 0.40,
            },
            "production_batch_memory_gate": {
                "status": "passed",
                "path": f"/fixture/{cell_id}/memory.json",
                "sha256": "3" * 64,
                "start_iteration": 1_000,
                "observed_iteration": 1_001,
                "world_size": 8,
                "thresholds": {
                    "max_peak_reserved_fraction": 0.94,
                    "max_driver_used_fraction": 0.94,
                },
                "production_batch_contract": {
                    "expected_samples_per_microbatch": 128,
                    "expected_grad_accum_iter": 2,
                    "expected_action_tokens_per_sample": 16,
                    "allowed_world_sizes": [8],
                },
                "violations": [],
                "per_rank": rows,
                "max_peak_allocated_fraction": 0.307,
                "max_peak_reserved_fraction": 0.407,
                "max_driver_used_fraction": 0.507,
            },
            "sample_trace": {
                "algorithm": ("sha256-canonical-ranked-microbatch-identities-text-token-contents-and-token-lengths-v1"),
                "aggregate_sha256": trace_sha,
                "step_count": 60,
                "per_step": trace_steps,
            },
        }

    cells = [
        passed_cell(4, "gpfs", "selective", 3.0),
        passed_cell(8, "gpfs", "selective", 3.2),
        passed_cell(12, "gpfs", "selective", 3.4),
        passed_cell(16, "gpfs", "selective", 3.6),
        passed_cell(
            4,
            "nvme",
            "selective",
            2.5,
            status=selected_status,
        ),
        {
            "schema_version": 1,
            "status": "infeasible_oom",
            "summarized_at": "2026-07-29T00:01:00+00:00",
            "cell_id": "workers-4__storage-nvme__ac-none",
            "workers_per_rank": 4,
            "node_total_workers": 32,
            "storage": "nvme",
            "activation_checkpointing": "none",
            "training_exit_code": 1,
            "infeasibility_reason": "no_ac_cuda_oom",
            "canonical_marker": "CUDA out of memory",
            "evidence_parse_errors": [],
            "log_sha256": "4" * 64,
            "optimizer_step_ndjson_sha256": hashlib.sha256(b"").hexdigest(),
            "gpu_samples_sha256": "5" * 64,
            "timing": {
                "observed_optimizer_steps": 0,
                "observed_start_unix_ns": None,
                "observed_end_unix_ns": None,
            },
            "gpu": {
                "sample_batches": 0,
                "sample_rows": 0,
                "gpu_count": 0,
            },
            "production_batch_memory_gate": {
                "status": "missing",
                "path": "/fixture/no-ac/memory.json",
            },
            "sample_trace": {
                "algorithm": ("sha256-canonical-ranked-microbatch-identities-text-token-contents-and-token-lengths-v1"),
                "aggregate_sha256": None,
                "step_count": 0,
                "per_step": [],
            },
        },
    ]
    run_id = run_root.name
    for cell in cells:
        cell_root = run_root / "cells" / cell["cell_id"]
        cell_root.mkdir(parents=True, exist_ok=True)
        (cell_root / "metrics.json").write_text(json.dumps(cell, sort_keys=True))
        storage = cell["storage"]
        latent_cache_root = (
            str(cache_root) if storage == "gpfs" else str(Path("/nvme/.cwam-cosmos3-latent-cache") / run_id / "cache")
        )
        contract = {
            "schema_version": 1,
            "status": "planned",
            "experiment_id": "CWAM-E35-THROUGHPUT-TEST",
            "run_id": run_id,
            "cell_id": cell["cell_id"],
            "world_size": 8,
            "workers_per_rank": cell["workers_per_rank"],
            "node_total_workers": cell["node_total_workers"],
            "storage": storage,
            "activation_checkpointing": cell["activation_checkpointing"],
            "video_tokenizer": {
                "target": (
                    "cosmos_framework.model.generator.tokenizers.precomputed_wan2pt2.PrecomputedWan2pt2LatentTokenizer"
                ),
                "metadata_only": True,
            },
            "cell_timeout_seconds": 600 if cell["activation_checkpointing"] == "none" else 0,
            "warmup_steps": 10,
            "measure_steps": 50,
            "total_steps": 60,
            "latent_cache_root": latent_cache_root,
            "latent_cache_manifest_sha256": cache_manifest_sha,
            "fixed_index_trace_path": str(fixed_trace),
            "fixed_index_trace_sha256": _sha256(fixed_trace),
            "fixed_index_trace_manifest_sha256": _sha256(fixed_trace_manifest),
            "wan_vae_path": str(run_root / "Wan2.2_VAE.pth"),
            "wan_vae_sha256": OFFICIAL_WAN22_VAE_SHA256,
            "resume_mode": "r6_iter1000_model_optimizer",
            "resume_checkpoint": str(checkpoint),
            "data_stream_resume": "restarted_not_cursor_resumed",
            "source_root": str(run_root / "snapshot"),
            "source_commit": "c" * 40,
            "snapshot_manifest_sha256": "d" * 64,
            "runner_sha256": "e" * 64,
            "helper_sha256": "f" * 64,
            "node_evidence": str(node_evidence),
        }
        (cell_root / "contract.json").write_text(json.dumps(contract, sort_keys=True))
    (run_root / "run-exit.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "exit_code": 0,
                "last_phase": "complete",
            },
            sort_keys=True,
        )
    )
    passed = [cell for cell in cells if cell["status"] == "passed"]
    infeasible = [cell for cell in cells if str(cell["status"]).startswith("infeasible_")]
    ranked = sorted(
        passed,
        key=lambda cell: (
            float(cell["timing"]["mean_step_seconds"]),  # type: ignore[index]
            str(cell["cell_id"]),
        ),
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": ("passed_with_no_ac_fallback" if infeasible else "passed"),
                "outcome": ("optional_no_ac_infeasible_selective_fallback" if infeasible else "all_cells_passed"),
                "summarized_at": "2026-07-29T00:02:00+00:00",
                "run_root": str(run_root),
                "expected_cell_count": 6,
                "observed_cell_count": 6,
                "passed_cell_count": len(passed),
                "infeasible_cell_count": len(infeasible),
                "infeasible_cell_ids": [cell["cell_id"] for cell in infeasible],
                "failed_cell_count": 0,
                "invalid_infeasible_cell_ids": [],
                "missing_selective_fallback_cell_ids": [],
                "nonpassed_selective_cell_ids": [],
                "missing_cell_ids": [],
                "cell_identity_mismatches": [],
                "sample_trace_consistent_across_cells": True,
                "sample_trace_aggregate_sha256": trace_sha,
                "best_observed_cell_id": (ranked[0]["cell_id"] if ranked else None),
                "ranking_by_mean_step_seconds": [
                    {
                        "cell_id": cell["cell_id"],
                        "mean_step_seconds": cell["timing"][  # type: ignore[index]
                            "mean_step_seconds"
                        ],
                        "average_gpu_utilization_percent": cell["gpu"][  # type: ignore[index]
                            "average_utilization_percent"
                        ],
                        "peak_memory_used_fraction": cell["gpu"][  # type: ignore[index]
                            "peak_memory_used_fraction"
                        ],
                    }
                    for cell in ranked
                ],
                "cells": cells,
            },
            sort_keys=True,
        )
    )
    return _sha256(path)


def _checkpoint(root: Path) -> Path:
    checkpoint = root / "iter_000001000"
    for component in ("model", "optim", "scheduler", "trainer"):
        component_dir = checkpoint / component
        component_dir.mkdir(parents=True)
        (component_dir / ".metadata").write_bytes(f"{component}-metadata".encode())
        for rank in range(8):
            payload = b"" if component == "scheduler" and rank > 0 else f"{component}-rank-{rank}-payload".encode()
            (component_dir / f"__{rank}_0.distcp").write_bytes(payload)
    return checkpoint


def _args(
    *,
    cache_root: Path,
    success_sha: str,
    manifest_sha: str,
    matrix: Path,
    matrix_sha: str,
    checkpoint: Path,
    checkpoint_sha: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        experiment_id="CWAM-E35-CONTINUATION-TEST",
        run_name="CWAM-E35-CONTINUATION-TEST",
        world_size=8,
        cache_root=cache_root,
        cache_success_sha256=success_sha,
        cache_manifest_sha256=manifest_sha,
        matrix_summary=matrix,
        matrix_summary_sha256=matrix_sha,
        resume_checkpoint=checkpoint,
        resume_checkpoint_fingerprint_sha256=checkpoint_sha,
        expected_benchmark_source_root=matrix.parent / "snapshot",
        expected_benchmark_source_commit="c" * 40,
        expected_benchmark_snapshot_manifest_sha256="d" * 64,
        wan_vae_path=matrix.parent / "Wan2.2_VAE.pth",
        expected_wan_vae_sha256=OFFICIAL_WAN22_VAE_SHA256,
    )


def test_contract_binds_cache_matrix_selection_and_complete_checkpoint(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    success_sha, manifest_sha = _cache(cache_root)
    checkpoint = _checkpoint(tmp_path / "r6")
    matrix = tmp_path / "matrix-summary.json"
    matrix_sha = _matrix(
        matrix,
        cache_root=cache_root,
        cache_manifest_sha=manifest_sha,
        checkpoint=checkpoint,
    )
    fingerprint = fingerprint_checkpoint(checkpoint)

    contract = build_contract(
        _args(
            cache_root=cache_root,
            success_sha=success_sha,
            manifest_sha=manifest_sha,
            matrix=matrix,
            matrix_sha=matrix_sha,
            checkpoint=checkpoint,
            checkpoint_sha=fingerprint["fingerprint_sha256"],
        )
    )

    assert contract["status"] == "passed"
    assert contract["throughput"]["best_observed_cell_id"] == ("workers-4__storage-nvme__ac-selective")
    assert contract["throughput"]["workers_per_rank"] == 4
    assert contract["throughput"]["storage"] == "nvme"
    assert contract["throughput"]["activation_checkpointing"] == "selective"
    artifacts = contract["throughput"]["benchmark_artifacts"]
    assert artifacts["gpfs_latent_cache_root"] == str(cache_root)
    assert artifacts["latent_cache_manifest_sha256"] == manifest_sha
    assert artifacts["resume_checkpoint"] == str(checkpoint)
    assert len(artifacts["cells"]) == 6
    assert contract["resume"]["fingerprint_sha256"] == fingerprint["fingerprint_sha256"]
    assert contract["protocol"]["fixed_index_trace_path"] is None


def test_matrix_rejects_nonpassed_best_cell_and_digest_drift(tmp_path) -> None:
    matrix = tmp_path / "matrix-summary.json"
    failed_sha = _matrix(matrix, selected_status="failed")
    with pytest.raises(ContinuationError, match="did not pass or prove"):
        validate_throughput_matrix(matrix, expected_sha256=failed_sha)

    passed_sha = _matrix(matrix)
    with pytest.raises(ContinuationError, match="SHA-256 mismatch"):
        validate_throughput_matrix(matrix, expected_sha256="0" * 64)
    assert validate_throughput_matrix(matrix, expected_sha256=passed_sha)["workers_per_rank"] == 4


def test_matrix_recomputes_best_cell_from_passed_timing(tmp_path) -> None:
    matrix = tmp_path / "matrix-summary.json"
    _matrix(matrix)
    payload = json.loads(matrix.read_text())
    payload["best_observed_cell_id"] = "workers-4__storage-gpfs__ac-selective"
    matrix.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(ContinuationError, match="best cell/ranking"):
        validate_throughput_matrix(matrix, expected_sha256=_sha256(matrix))


def test_matrix_rejects_two_cell_partial_protocol(tmp_path) -> None:
    matrix = tmp_path / "matrix-summary.json"
    _matrix(matrix)
    payload = json.loads(matrix.read_text())
    payload["cells"] = payload["cells"][:2]
    matrix.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ContinuationError, match="exactly 6 adaptive or 16"):
        validate_throughput_matrix(matrix, expected_sha256=_sha256(matrix))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("timing", "observed_optimizer_steps", 59, "optimizer-step contract"),
        ("gpu", "gpu_count", 7, "eight-GPU telemetry"),
        (
            "production_batch_memory_gate",
            "status",
            "failed",
            "memory gate status",
        ),
    ],
)
def test_matrix_rejects_forged_passed_cell_semantics(
    tmp_path,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    matrix = tmp_path / "matrix-summary.json"
    _matrix(matrix)
    payload = json.loads(matrix.read_text())
    payload["cells"][0][section][key] = value
    matrix.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ContinuationError, match=message):
        validate_throughput_matrix(matrix, expected_sha256=_sha256(matrix))


def test_matrix_recomputes_sample_trace_digest(tmp_path) -> None:
    matrix = tmp_path / "matrix-summary.json"
    _matrix(matrix)
    payload = json.loads(matrix.read_text())
    payload["cells"][0]["sample_trace"]["per_step"][0]["identity_sha256"] = "0" * 64
    matrix.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ContinuationError, match="sample-trace digest mismatch"):
        validate_throughput_matrix(matrix, expected_sha256=_sha256(matrix))


def test_matrix_recomputes_outcome_counters(tmp_path) -> None:
    matrix = tmp_path / "matrix-summary.json"
    _matrix(matrix)
    payload = json.loads(matrix.read_text())
    payload["passed_cell_count"] = 6
    matrix.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ContinuationError, match="outcome/counter mismatch"):
        validate_throughput_matrix(matrix, expected_sha256=_sha256(matrix))


def test_matrix_rejects_metrics_artifact_drift(tmp_path) -> None:
    matrix = tmp_path / "matrix-summary.json"
    matrix_sha = _matrix(matrix)
    metrics = tmp_path / "cells" / "workers-4__storage-nvme__ac-selective" / "metrics.json"
    payload = json.loads(metrics.read_text())
    payload["timing"]["mean_step_seconds"] = 1.0
    metrics.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ContinuationError, match="does not match its metrics artifact"):
        validate_throughput_matrix(matrix, expected_sha256=matrix_sha)


def test_matrix_rejects_fixed_trace_artifact_drift(tmp_path) -> None:
    matrix = tmp_path / "matrix-summary.json"
    matrix_sha = _matrix(matrix)
    (tmp_path / "fixed-index-trace.npy").write_bytes(b"drifted-trace")

    with pytest.raises(ContinuationError, match="fixed-index trace SHA-256 drifted"):
        validate_throughput_matrix(matrix, expected_sha256=matrix_sha)


def test_matrix_rejects_nonterminal_run_receipt(tmp_path) -> None:
    matrix = tmp_path / "matrix-summary.json"
    matrix_sha = _matrix(matrix)
    (tmp_path / "run-exit.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "exit_code": 1,
                "last_phase": "matrix_summary",
            },
            sort_keys=True,
        )
    )

    with pytest.raises(ContinuationError, match="no successful terminal receipt"):
        validate_throughput_matrix(matrix, expected_sha256=matrix_sha)


def test_contract_rejects_matrix_from_another_cache(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    success_sha, manifest_sha = _cache(cache_root)
    checkpoint = _checkpoint(tmp_path / "r6")
    matrix = tmp_path / "matrix-summary.json"
    matrix_sha = _matrix(
        matrix,
        cache_root=cache_root,
        cache_manifest_sha="0" * 64,
        checkpoint=checkpoint,
    )
    fingerprint = fingerprint_checkpoint(checkpoint)

    with pytest.raises(ContinuationError, match="cache manifest does not match"):
        build_contract(
            _args(
                cache_root=cache_root,
                success_sha=success_sha,
                manifest_sha=manifest_sha,
                matrix=matrix,
                matrix_sha=matrix_sha,
                checkpoint=checkpoint,
                checkpoint_sha=fingerprint["fingerprint_sha256"],
            )
        )


def test_checkpoint_fingerprint_rejects_unexpected_dataloader_state(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    dataloader = checkpoint / "dataloader"
    dataloader.mkdir()
    (dataloader / ".metadata").write_bytes(b"metadata")
    (dataloader / "__0_0.distcp").write_bytes(b"shard")
    with pytest.raises(ContinuationError, match="component set mismatch"):
        fingerprint_checkpoint(checkpoint)


def test_checkpoint_fingerprint_accepts_scheduler_empty_rank_shards(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)

    fingerprint = fingerprint_checkpoint(checkpoint)

    assert fingerprint["zero_byte_distcp_file_count"] == 7
    scheduler = fingerprint["component_reports"]["scheduler"]
    assert scheduler["nonempty_payload_shards"] == ["__0_0.distcp"]
    assert scheduler["zero_byte_rank_shards"] == [f"__{rank}_0.distcp" for rank in range(1, 8)]
    empty_digest = hashlib.sha256(b"").hexdigest()
    empty_scheduler_files = [
        item for item in fingerprint["files"] if item["path"].startswith("scheduler/") and item["size_bytes"] == 0
    ]
    assert len(empty_scheduler_files) == 7
    assert {item["sha256"] for item in empty_scheduler_files} == {empty_digest}


def test_checkpoint_fingerprint_is_portable_across_stat_identity(
    tmp_path,
) -> None:
    first = _checkpoint(tmp_path / "first")
    second = _checkpoint(tmp_path / "second")

    first_fingerprint = fingerprint_checkpoint(first)
    second_fingerprint = fingerprint_checkpoint(second)

    assert first_fingerprint["fingerprint_algorithm"] == ("sha256-full-dcp-content-inventory-v3")
    assert any(
        left["inode"] != right["inode"]
        for left, right in zip(
            first_fingerprint["files"],
            second_fingerprint["files"],
            strict=True,
        )
    )
    assert first_fingerprint["fingerprint_sha256"] == second_fingerprint["fingerprint_sha256"]

    shard = second / "model" / "__0_0.distcp"
    shard.write_bytes(b"x" * shard.stat().st_size)
    assert fingerprint_checkpoint(second)["fingerprint_sha256"] != first_fingerprint["fingerprint_sha256"]


def test_checkpoint_fingerprint_rejects_empty_metadata(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "scheduler" / ".metadata").write_bytes(b"")

    with pytest.raises(ContinuationError, match="metadata must be non-empty"):
        fingerprint_checkpoint(checkpoint)


def test_checkpoint_fingerprint_rejects_empty_non_distcp_file(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "scheduler" / "empty.txt").write_bytes(b"")

    with pytest.raises(ContinuationError, match="component file set mismatch"):
        fingerprint_checkpoint(checkpoint)


def test_checkpoint_fingerprint_rejects_component_without_payload(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    for rank in range(8):
        (checkpoint / "trainer" / f"__{rank}_0.distcp").write_bytes(b"")

    with pytest.raises(ContinuationError, match="no non-empty DCP payload shard"):
        fingerprint_checkpoint(checkpoint)


def test_submission_binding_rejects_accepted_unknown_ambiguity(
    tmp_path,
    monkeypatch,
) -> None:
    experiment_id = "CWAM-E35-SUBMISSION-BINDING-RACE-TEST"
    monkeypatch.setattr(continuation, "PINNED_FORMAL_OUTPUT_ROOT", tmp_path)
    reservation_dir = tmp_path / "formal_continuation_submission_reservations" / experiment_id
    reservation_dir.mkdir(parents=True)
    accepted = reservation_dir / "submission.json"
    accepted.write_text('{"status":"accepted"}\n')
    accepted.chmod(0o444)
    unknown = reservation_dir / "submission-unknown.json"
    unknown.write_text('{"status":"submission_unknown"}\n')
    unknown.chmod(0o444)
    reservation_dir.chmod(0o555)
    try:
        with pytest.raises(
            ContinuationError,
            match="submission-unknown sibling",
        ):
            continuation.validate_submission_binding(
                reservation_path=accepted,
                submission_nonce="a" * 64,
                expected_command_sha256="b" * 64,
                expected_cpu=128,
                experiment_id=experiment_id,
                cache={},
                throughput={},
                resume={},
            )
    finally:
        reservation_dir.chmod(0o755)


def test_prepare_run_rejects_same_size_checkpoint_mutation(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    success_sha, manifest_sha = _cache(cache_root)
    checkpoint = _checkpoint(tmp_path / "r6")
    matrix = tmp_path / "matrix-summary.json"
    matrix_sha = _matrix(
        matrix,
        cache_root=cache_root,
        cache_manifest_sha=manifest_sha,
        checkpoint=checkpoint,
    )
    fingerprint = fingerprint_checkpoint(checkpoint)
    contract = build_contract(
        _args(
            cache_root=cache_root,
            success_sha=success_sha,
            manifest_sha=manifest_sha,
            matrix=matrix,
            matrix_sha=matrix_sha,
            checkpoint=checkpoint,
            checkpoint_sha=fingerprint["fingerprint_sha256"],
        )
    )
    contract_path = tmp_path / "continuation-inputs.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    output_root = tmp_path / "formal-output"
    output_root.mkdir()
    shard = checkpoint / "scheduler" / "__0_0.distcp"
    original = shard.read_bytes()
    shard.write_bytes(b"x" * len(original))

    with pytest.raises(ContinuationError, match="file/stat inventory changed"):
        prepare_run(
            output_root=output_root,
            run_name=contract["run_name"],
            contract_path=contract_path,
            expected_contract_sha256=_sha256(contract_path),
        )


def test_prepare_run_rejects_submission_artifact_drift(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    success_sha, manifest_sha = _cache(cache_root)
    checkpoint = _checkpoint(tmp_path / "r6")
    matrix = tmp_path / "matrix-summary.json"
    matrix_sha = _matrix(
        matrix,
        cache_root=cache_root,
        cache_manifest_sha=manifest_sha,
        checkpoint=checkpoint,
    )
    fingerprint = fingerprint_checkpoint(checkpoint)
    contract = build_contract(
        _args(
            cache_root=cache_root,
            success_sha=success_sha,
            manifest_sha=manifest_sha,
            matrix=matrix,
            matrix_sha=matrix_sha,
            checkpoint=checkpoint,
            checkpoint_sha=fingerprint["fingerprint_sha256"],
        )
    )
    reservation = tmp_path / "submission.json"
    reservation.write_text('{"status":"accepted"}\n')
    reservation.chmod(0o444)
    scheduler_receipt = tmp_path / "scheduler-receipt.json"
    scheduler_receipt.write_text('{"status":"passed"}\n')
    scheduler_receipt.chmod(0o444)
    contract["submission"] = {
        "reservation_path": str(reservation),
        "reservation_sha256": _sha256(reservation),
        "artifacts": {
            "scheduler_structured_receipt": {
                "path": str(scheduler_receipt),
                "size_bytes": scheduler_receipt.stat().st_size,
                "sha256": _sha256(scheduler_receipt),
            }
        },
    }
    contract_path = tmp_path / "continuation-inputs.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    output_root = tmp_path / "formal-output"
    output_root.mkdir()
    scheduler_receipt.chmod(0o644)
    scheduler_receipt.write_text('{"status":"failed"}\n')
    scheduler_receipt.chmod(0o444)

    with pytest.raises(
        ContinuationError,
        match="submission artifact changed",
    ):
        prepare_run(
            output_root=output_root,
            run_name=contract["run_name"],
            contract_path=contract_path,
            expected_contract_sha256=_sha256(contract_path),
        )


def test_prepare_run_atomically_links_only_the_bound_seed(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    success_sha, manifest_sha = _cache(cache_root)
    checkpoint = _checkpoint(tmp_path / "r6")
    matrix = tmp_path / "matrix-summary.json"
    matrix_sha = _matrix(
        matrix,
        cache_root=cache_root,
        cache_manifest_sha=manifest_sha,
        checkpoint=checkpoint,
    )
    fingerprint = fingerprint_checkpoint(checkpoint)
    contract = build_contract(
        _args(
            cache_root=cache_root,
            success_sha=success_sha,
            manifest_sha=manifest_sha,
            matrix=matrix,
            matrix_sha=matrix_sha,
            checkpoint=checkpoint,
            checkpoint_sha=fingerprint["fingerprint_sha256"],
        )
    )
    contract_path = tmp_path / "continuation-inputs.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    output_root = tmp_path / "formal-output"
    output_root.mkdir()

    receipt = prepare_run(
        output_root=output_root,
        run_name=contract["run_name"],
        contract_path=contract_path,
        expected_contract_sha256=_sha256(contract_path),
    )

    run_dir = Path(receipt["run_dir"])
    seed = run_dir / "checkpoints" / "iter_000001000"
    assert seed.is_symlink()
    assert seed.resolve() == checkpoint
    assert (run_dir / "checkpoints" / "latest_checkpoint.txt").read_text() == "iter_000001000\n"
    assert sorted(path.name for path in (run_dir / "checkpoints").iterdir()) == [
        "iter_000001000",
        "latest_checkpoint.txt",
    ]
    with pytest.raises(ContinuationError, match="already exists"):
        prepare_run(
            output_root=output_root,
            run_name=contract["run_name"],
            contract_path=contract_path,
            expected_contract_sha256=_sha256(contract_path),
        )
