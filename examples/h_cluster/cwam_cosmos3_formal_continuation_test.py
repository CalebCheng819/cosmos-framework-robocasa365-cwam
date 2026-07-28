# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from examples.h_cluster.cwam_cosmos3_formal_continuation import (
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


def _matrix(path: Path, *, selected_status: str = "passed") -> str:
    selected = {
        "cell_id": "workers-12__storage-nvme__ac-none",
        "status": selected_status,
        "workers_per_rank": 12,
        "node_total_workers": 96,
        "storage": "nvme",
        "activation_checkpointing": "none",
        "timing": {"mean_step_seconds": 2.5},
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "sample_trace_consistent_across_cells": True,
                "best_observed_cell_id": selected["cell_id"],
                "ranking_by_mean_step_seconds": [
                    {
                        "cell_id": selected["cell_id"],
                        "mean_step_seconds": selected["timing"]["mean_step_seconds"],
                    }
                ],
                "cells": [selected],
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
        (component_dir / "__0_0.distcp").write_bytes(f"{component}-shard".encode())
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
    )


def test_contract_binds_cache_matrix_selection_and_complete_checkpoint(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    success_sha, manifest_sha = _cache(cache_root)
    matrix = tmp_path / "matrix-summary.json"
    matrix_sha = _matrix(matrix)
    checkpoint = _checkpoint(tmp_path / "r6")
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
    assert contract["throughput"]["best_observed_cell_id"] == "workers-12__storage-nvme__ac-none"
    assert contract["throughput"]["workers_per_rank"] == 12
    assert contract["throughput"]["storage"] == "nvme"
    assert contract["throughput"]["activation_checkpointing"] == "none"
    assert contract["resume"]["fingerprint_sha256"] == fingerprint["fingerprint_sha256"]
    assert contract["protocol"]["fixed_index_trace_path"] is None


def test_matrix_rejects_nonpassed_best_cell_and_digest_drift(tmp_path) -> None:
    matrix = tmp_path / "matrix-summary.json"
    failed_sha = _matrix(matrix, selected_status="failed")
    with pytest.raises(ContinuationError, match="selected throughput cell status"):
        validate_throughput_matrix(matrix, expected_sha256=failed_sha)

    passed_sha = _matrix(matrix)
    with pytest.raises(ContinuationError, match="SHA-256 mismatch"):
        validate_throughput_matrix(matrix, expected_sha256="0" * 64)
    assert validate_throughput_matrix(matrix, expected_sha256=passed_sha)["workers_per_rank"] == 12


def test_matrix_recomputes_best_cell_from_passed_timing(tmp_path) -> None:
    matrix = tmp_path / "matrix-summary.json"
    slow = {
        "cell_id": "workers-12__storage-nvme__ac-none",
        "status": "passed",
        "workers_per_rank": 12,
        "node_total_workers": 96,
        "storage": "nvme",
        "activation_checkpointing": "none",
        "timing": {"mean_step_seconds": 2.5},
    }
    fast = {
        "cell_id": "workers-8__storage-gpfs__ac-selective",
        "status": "passed",
        "workers_per_rank": 8,
        "node_total_workers": 64,
        "storage": "gpfs",
        "activation_checkpointing": "selective",
        "timing": {"mean_step_seconds": 2.0},
    }
    matrix.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "sample_trace_consistent_across_cells": True,
                "best_observed_cell_id": slow["cell_id"],
                "ranking_by_mean_step_seconds": [
                    {"cell_id": slow["cell_id"], "mean_step_seconds": 2.5},
                    {"cell_id": fast["cell_id"], "mean_step_seconds": 2.0},
                ],
                "cells": [slow, fast],
            },
            sort_keys=True,
        )
    )
    with pytest.raises(ContinuationError, match="does not match passed-cell timing"):
        validate_throughput_matrix(matrix, expected_sha256=_sha256(matrix))


def test_checkpoint_fingerprint_rejects_unexpected_dataloader_state(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    dataloader = checkpoint / "dataloader"
    dataloader.mkdir()
    (dataloader / ".metadata").write_bytes(b"metadata")
    (dataloader / "__0_0.distcp").write_bytes(b"shard")
    with pytest.raises(ContinuationError, match="component set mismatch"):
        fingerprint_checkpoint(checkpoint)


def test_prepare_run_atomically_links_only_the_bound_seed(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    success_sha, manifest_sha = _cache(cache_root)
    matrix = tmp_path / "matrix-summary.json"
    matrix_sha = _matrix(matrix)
    checkpoint = _checkpoint(tmp_path / "r6")
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
