# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pure-CPU tests for the CWAM latent-cache throughput operations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
HELPER_PATH = HERE / "cwam_cosmos3_latent_cache_throughput.py"
RUNNER_PATH = HERE / "run_cwam_cosmos3_latent_cache_throughput_probe.sh"
SUBMITTER_PATH = HERE / "submit_cwam_cosmos3_latent_cache_throughput_rjob.sh"
CACHE_SUBMITTER_PATH = HERE / "submit_robocasa365_wan_latent_cache_rjob.sh"
SPEC = importlib.util.spec_from_file_location("cwam_cosmos3_latent_cache_throughput", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
OPS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPS)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_synthetic_cell(root: Path, *, cell_id: str, mean_bias: float = 0.0) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    log = root / "train.log"
    log.write_text("synthetic CPU-only throughput dry run\n")
    steps = root / "optimizer-step-wall-clock.ndjson"
    base_unix_ns = 10_000_000_000_000
    with steps.open("w") as handle:
        for relative_step in range(1, 61):
            start_ns = base_unix_ns + relative_step * 10_000_000_000
            step_seconds = 2.0 + mean_bias + relative_step / 1000.0
            trace_digest = hashlib.sha256(f"trace-step-{relative_step}".encode()).hexdigest()
            record = {
                "schema_version": 1,
                "completed_iteration": 1000 + relative_step,
                "relative_optimizer_step": relative_step,
                "window": "warmup" if relative_step <= 10 else "measure",
                "start_unix_ns": start_ns,
                "end_unix_ns": start_ns + int(step_seconds * 1e9),
                "outer_step_seconds": step_seconds,
                "timer_dataloader_train_seconds": 0.2 + relative_step / 10000.0,
                "callback_loader_wait_seconds": 0.21 + relative_step / 10000.0,
                "trace_identity_count": 2048,
                "trace_identity_sha256": trace_digest,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    gpu_samples = root / "gpu-samples.ndjson"
    measured_start_ns = base_unix_ns + 11 * 10_000_000_000
    sample_times = [measured_start_ns - 1, measured_start_ns + 1_000_000_000]
    with gpu_samples.open("w") as handle:
        for sample_time in sample_times:
            gpus = [
                {
                    "index": index,
                    "uuid": f"GPU-{index}",
                    "name": "NVIDIA H200",
                    "utilization_gpu_percent": 70.0 + index,
                    "memory_used_mib": 10000.0 + index,
                    "memory_total_mib": 140000.0,
                    "power_draw_w": 400.0 + index,
                }
                for index in range(8)
            ]
            handle.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sample_started_unix_ns": sample_time - 10,
                        "sample_finished_unix_ns": sample_time + 10,
                        "sample_unix_ns": sample_time,
                        "gpus": gpus,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    memory_gate = root / "production_batch_memory_gate" / "iteration_000001001.json"
    rows = []
    for rank in range(8):
        rows.append(
            {
                "rank": rank,
                "total_memory_bytes": 1000,
                "peak_allocated_bytes": 700,
                "peak_reserved_bytes": 800,
                "current_allocated_bytes": 600,
                "current_reserved_bytes": 700,
                "driver_free_bytes": 150,
                "driver_used_bytes": 850,
                "peak_allocated_fraction": 0.7,
                "peak_reserved_fraction": 0.8,
                "driver_used_fraction": 0.85,
            }
        )
    _write_json(
        memory_gate,
        {
            "schema_version": 1,
            "status": "passed",
            "start_iteration": 1000,
            "observed_iteration": 1001,
            "world_size": 8,
            "production_batch_contract": {
                "expected_samples_per_microbatch": 128,
                "expected_grad_accum_iter": 2,
                "expected_action_tokens_per_sample": 16,
                "allowed_world_sizes": [8],
            },
            "thresholds": {
                "max_peak_reserved_fraction": 0.94,
                "max_driver_used_fraction": 0.94,
            },
            "per_rank": rows,
            "violations": [],
        },
    )
    return {
        "log": log,
        "steps": steps,
        "gpu": gpu_samples,
        "memory": memory_gate,
        "metrics": root / "metrics.json",
        "cell_id": Path(cell_id),
    }


def _summary_args(
    paths: dict[str, Path],
    *,
    cell_id: str,
    workers: int = 4,
    activation_checkpointing: str = "selective",
    training_exit_code: int = 0,
) -> argparse.Namespace:
    return argparse.Namespace(
        cell_id=cell_id,
        workers=workers,
        storage="gpfs",
        activation_checkpointing=activation_checkpointing,
        warmup_steps=10,
        measure_steps=50,
        log=str(paths["log"]),
        step_ndjson=str(paths["steps"]),
        gpu_samples=str(paths["gpu"]),
        memory_gate_json=str(paths["memory"]),
        training_exit_code=training_exit_code,
        expected_gpus=8,
        output=str(paths["metrics"]),
    )


class ThroughputHelperTest(unittest.TestCase):
    def test_synthetic_cell_summary_uses_outer_wall_clock_and_strict_gpu_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell_id = "workers-4__storage-gpfs__ac-selective"
            paths = _write_synthetic_cell(root, cell_id=cell_id)
            payload = OPS.summarize_cell(_summary_args(paths, cell_id=cell_id))

        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["workers_per_rank"], 4)
        self.assertEqual(payload["node_total_workers"], 32)
        self.assertEqual(payload["timing"]["first_measured_iteration"], 1011)
        self.assertEqual(payload["timing"]["last_measured_iteration"], 1060)
        self.assertEqual(payload["timing"]["observed_optimizer_steps"], 60)
        self.assertIn("dataloader fetch", payload["timing"]["measurement_scope"])
        self.assertGreater(payload["timing"]["mean_timer_dataloader_train_seconds"], 0)
        self.assertEqual(payload["gpu"]["sample_batches"], 1)
        self.assertEqual(payload["gpu"]["gpu_count"], 8)
        self.assertEqual(payload["production_batch_memory_gate"]["status"], "passed")
        self.assertEqual(payload["sample_trace"]["step_count"], 60)

    def test_cli_synthetic_cell_end_to_end_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell_id = "workers-8__storage-gpfs__ac-selective"
            paths = _write_synthetic_cell(root, cell_id=cell_id)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(HELPER_PATH),
                    "summarize-cell",
                    "--cell-id",
                    cell_id,
                    "--workers",
                    "8",
                    "--storage",
                    "gpfs",
                    "--activation-checkpointing",
                    "selective",
                    "--warmup-steps",
                    "10",
                    "--measure-steps",
                    "50",
                    "--log",
                    str(paths["log"]),
                    "--step-ndjson",
                    str(paths["steps"]),
                    "--gpu-samples",
                    str(paths["gpu"]),
                    "--memory-gate-json",
                    str(paths["memory"]),
                    "--training-exit-code",
                    "0",
                    "--expected-gpus",
                    "8",
                    "--output",
                    str(paths["metrics"]),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(paths["metrics"].read_text())
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["node_total_workers"], 64)

    def test_only_no_ac_cuda_oom_is_classified_as_optional_infeasible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            none_paths = _write_synthetic_cell(
                root / "none",
                cell_id="workers-4__storage-gpfs__ac-none",
            )
            none_paths["log"].write_text(
                "rank 3: torch.cuda.OutOfMemoryError: CUDA out of memory\n"
            )
            none_payload = OPS.summarize_cell_fail_closed(
                _summary_args(
                    none_paths,
                    cell_id="workers-4__storage-gpfs__ac-none",
                    activation_checkpointing="none",
                    training_exit_code=1,
                )
            )

            selective_paths = _write_synthetic_cell(
                root / "selective",
                cell_id="workers-4__storage-gpfs__ac-selective",
            )
            selective_paths["log"].write_text("CUDA out of memory\n")
            selective_payload = OPS.summarize_cell_fail_closed(
                _summary_args(
                    selective_paths,
                    cell_id="workers-4__storage-gpfs__ac-selective",
                    activation_checkpointing="selective",
                    training_exit_code=1,
                )
            )

            other_paths = _write_synthetic_cell(
                root / "other",
                cell_id="workers-8__storage-gpfs__ac-none",
            )
            other_paths["log"].write_text("NCCL transport failure\n")
            other_payload = OPS.summarize_cell_fail_closed(
                _summary_args(
                    other_paths,
                    cell_id="workers-8__storage-gpfs__ac-none",
                    workers=8,
                    activation_checkpointing="none",
                    training_exit_code=1,
                )
            )

            cpu_oom_paths = _write_synthetic_cell(
                root / "cpu-oom",
                cell_id="workers-12__storage-gpfs__ac-none",
            )
            cpu_oom_paths["log"].write_text("OutOfMemoryError: CPU allocator failed\n")
            cpu_oom_payload = OPS.summarize_cell_fail_closed(
                _summary_args(
                    cpu_oom_paths,
                    cell_id="workers-12__storage-gpfs__ac-none",
                    workers=12,
                    activation_checkpointing="none",
                    training_exit_code=1,
                )
            )

        self.assertEqual(none_payload["status"], "infeasible_oom")
        self.assertEqual(none_payload["activation_checkpointing"], "none")
        self.assertEqual(none_payload["sample_trace"]["step_count"], 60)
        self.assertEqual(none_payload["production_batch_memory_gate"]["status"], "passed")
        self.assertEqual(selective_payload["status"], "failed")
        self.assertEqual(other_payload["status"], "failed")
        self.assertEqual(cpu_oom_payload["status"], "failed")

    def test_matrix_accepts_only_no_ac_infeasible_with_matching_selective_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            selective_id = "workers-4__storage-gpfs__ac-selective"
            selective_paths = _write_synthetic_cell(
                run_root / "cells" / selective_id,
                cell_id=selective_id,
            )
            OPS.summarize_cell(_summary_args(selective_paths, cell_id=selective_id))

            none_id = "workers-4__storage-gpfs__ac-none"
            none_paths = _write_synthetic_cell(
                run_root / "cells" / none_id,
                cell_id=none_id,
            )
            none_paths["log"].write_text("torch.cuda.OutOfMemoryError: CUDA out of memory\n")
            OPS.summarize_cell_fail_closed(
                _summary_args(
                    none_paths,
                    cell_id=none_id,
                    activation_checkpointing="none",
                    training_exit_code=1,
                )
            )
            payload = OPS.summarize_matrix(
                run_root,
                run_root / "matrix.json",
                [selective_id, none_id],
            )

        self.assertEqual(payload["status"], "passed_with_no_ac_fallback")
        self.assertEqual(payload["infeasible_cell_count"], 1)
        self.assertTrue(payload["sample_trace_consistent_across_cells"])

    def test_no_ac_structurally_valid_memory_gate_violation_is_infeasible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell_id = "workers-4__storage-gpfs__ac-none"
            paths = _write_synthetic_cell(root, cell_id=cell_id)
            gate = json.loads(paths["memory"].read_text())
            gate["status"] = "failed"
            gate["violations"] = ["rank 0 peak_reserved_fraction=0.95 exceeds 0.94"]
            gate["per_rank"][0]["peak_reserved_fraction"] = 0.95
            _write_json(paths["memory"], gate)
            paths["log"].write_text("production batch memory gate rejected the configuration\n")
            payload = OPS.summarize_cell_fail_closed(
                _summary_args(
                    paths,
                    cell_id=cell_id,
                    activation_checkpointing="none",
                    training_exit_code=1,
                )
            )

        self.assertEqual(payload["status"], "infeasible_memory_headroom")
        self.assertEqual(payload["production_batch_memory_gate"]["world_size"], 8)
        self.assertEqual(
            payload["production_batch_memory_gate"]["max_peak_reserved_fraction"],
            0.95,
        )

    def test_no_ac_memory_gate_boundary_is_infeasible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell_id = "workers-4__storage-gpfs__ac-none"
            paths = _write_synthetic_cell(root, cell_id=cell_id)
            gate = json.loads(paths["memory"].read_text())
            gate["status"] = "failed"
            gate["violations"] = ["rank 0 peak_reserved_fraction=0.94 >= 0.94"]
            gate["per_rank"][0]["peak_reserved_fraction"] = 0.94
            _write_json(paths["memory"], gate)
            paths["log"].write_text("production batch memory gate rejected the configuration\n")
            payload = OPS.summarize_cell_fail_closed(
                _summary_args(
                    paths,
                    cell_id=cell_id,
                    activation_checkpointing="none",
                    training_exit_code=1,
                )
            )

        self.assertEqual(payload["status"], "infeasible_memory_headroom")
        self.assertEqual(
            payload["production_batch_memory_gate"]["max_peak_reserved_fraction"],
            0.94,
        )

    def test_matrix_rejects_missing_selective_fallback_and_forged_selective_infeasible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            none_id = "workers-4__storage-gpfs__ac-none"
            none_paths = _write_synthetic_cell(
                run_root / "cells" / none_id,
                cell_id=none_id,
            )
            none_paths["log"].write_text("CUDA out of memory\n")
            OPS.summarize_cell_fail_closed(
                _summary_args(
                    none_paths,
                    cell_id=none_id,
                    activation_checkpointing="none",
                    training_exit_code=1,
                )
            )
            missing_fallback = OPS.summarize_matrix(
                run_root,
                run_root / "missing-fallback.json",
                [none_id],
            )
            self.assertEqual(missing_fallback["status"], "partial_or_failed")
            self.assertEqual(missing_fallback["missing_selective_fallback_cell_ids"], [none_id])

            forged_root = run_root / "forged"
            selective_id = "workers-4__storage-gpfs__ac-selective"
            selective_metrics = forged_root / "cells" / selective_id / "metrics.json"
            forged = json.loads(none_paths["metrics"].read_text())
            forged["cell_id"] = selective_id
            forged["activation_checkpointing"] = "selective"
            _write_json(selective_metrics, forged)
            forged_matrix = OPS.summarize_matrix(
                forged_root,
                forged_root / "matrix.json",
                [selective_id],
            )

        self.assertEqual(forged_matrix["status"], "partial_or_failed")
        self.assertEqual(forged_matrix["invalid_infeasible_cell_ids"], [selective_id])
        self.assertEqual(forged_matrix["nonpassed_selective_cell_ids"], [selective_id])

    def test_matrix_rejects_metrics_copied_from_another_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            selective_id = "workers-4__storage-gpfs__ac-selective"
            selective_paths = _write_synthetic_cell(
                run_root / "cells" / selective_id,
                cell_id=selective_id,
            )
            OPS.summarize_cell(_summary_args(selective_paths, cell_id=selective_id))

            none_id = "workers-4__storage-gpfs__ac-none"
            copied_metrics = run_root / "cells" / none_id / "metrics.json"
            _write_json(copied_metrics, json.loads(selective_paths["metrics"].read_text()))
            payload = OPS.summarize_matrix(
                run_root,
                run_root / "matrix.json",
                [selective_id, none_id],
            )

        self.assertEqual(payload["status"], "partial_or_failed")
        self.assertEqual(len(payload["cell_identity_mismatches"]), 1)
        mismatch = payload["cell_identity_mismatches"][0]
        self.assertEqual(mismatch["expected"]["cell_id"], none_id)
        self.assertEqual(mismatch["observed"]["cell_id"], selective_id)

    def test_matrix_requires_identical_step_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            cell_ids = [
                "workers-4__storage-gpfs__ac-selective",
                "workers-8__storage-gpfs__ac-selective",
            ]
            for index, cell_id in enumerate(cell_ids):
                root = run_root / "cells" / cell_id
                paths = _write_synthetic_cell(root, cell_id=cell_id, mean_bias=index / 10.0)
                OPS.summarize_cell(_summary_args(paths, cell_id=cell_id, workers=4 + index * 4))
            passed = OPS.summarize_matrix(run_root, run_root / "passed.json", cell_ids)
            self.assertEqual(passed["status"], "passed")
            self.assertTrue(passed["sample_trace_consistent_across_cells"])

            changed_path = run_root / "cells" / cell_ids[1] / "metrics.json"
            changed = json.loads(changed_path.read_text())
            changed["sample_trace"]["aggregate_sha256"] = "0" * 64
            _write_json(changed_path, changed)
            failed = OPS.summarize_matrix(run_root, run_root / "failed.json", cell_ids)
            self.assertEqual(failed["status"], "partial_or_failed")
            self.assertFalse(failed["sample_trace_consistent_across_cells"])

    def test_adaptive_selection_rejects_trace_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metric_paths = []
            for index, workers in enumerate((4, 8)):
                cell_id = f"workers-{workers}__storage-gpfs__ac-selective"
                cell_root = root / cell_id
                paths = _write_synthetic_cell(cell_root, cell_id=cell_id, mean_bias=index / 10.0)
                OPS.summarize_cell(_summary_args(paths, cell_id=cell_id, workers=workers))
                metric_paths.append(paths["metrics"])
            selected = OPS.select_best_cell(
                metric_paths,
                decision="workers",
                output=root / "selection.json",
            )
            self.assertEqual(selected["selected_workers_per_rank"], 4)
            changed = json.loads(metric_paths[1].read_text())
            changed["sample_trace"]["aggregate_sha256"] = "f" * 64
            _write_json(metric_paths[1], changed)
            with self.assertRaises(OPS.ContractError):
                OPS.select_best_cell(
                    metric_paths,
                    decision="workers-mismatch",
                    output=root / "selection-mismatch.json",
                )

    def test_ops_imports_the_shared_cache_validator(self) -> None:
        source = HELPER_PATH.read_text()
        self.assertIn("validate_published_cache_files", source)
        self.assertIn(
            'dist.init_process_group(backend="gloo", timeout=timedelta(hours=48))',
            source,
        )
        self.assertNotIn("def _cache_contract_files", source)
        self.assertNotIn("shard_ids: set[str]", source)

    def test_fixed_trace_sidecar_is_fail_closed_before_numpy_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "fixed-index-trace.npy"
            trace.write_bytes(b"not-needed-for-contract-mismatch")
            _write_json(
                trace.with_suffix(".manifest.json"),
                {
                    "schema_version": 1,
                    "status": "complete",
                    "array_path": trace.name,
                    "array_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                    "dtype": "int64",
                    "shape": [8, 15551],
                    "dataset_length": 1,
                    "shuffle_blocks_sha256": "0" * 64,
                    "shuffle_seed": 42,
                    "source_world_size": 8,
                    "baseline_num_workers": 4,
                    "indices_per_rank": 15552,
                    "batch_size_per_rank": 128,
                    "grad_accum_iter": 2,
                    "warmup_steps": 10,
                    "measure_steps": 50,
                    "total_optimizer_steps": 60,
                    "prefetch_factor": 4,
                    "in_order": True,
                    "transform_seed": 42,
                    "algorithm": "baseline-worker-interleave-v1",
                },
            )
            with self.assertRaises(OPS.ContractError):
                OPS.validate_fixed_index_trace(trace)

    def test_fixed_trace_prepare_rejects_partial_publication_before_dataset_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "fixed-index-trace.npy"
            trace.write_bytes(b"partial")
            with self.assertRaisesRegex(OPS.ContractError, "partial fixed trace publication"):
                OPS.prepare_fixed_index_trace(
                    trace,
                    root / "unused-dataset",
                    root / "unused-source",
                )

    def test_runner_and_submitter_wire_the_complete_contract(self) -> None:
        runner = RUNNER_PATH.read_text()
        submitter = SUBMITTER_PATH.read_text()
        cache_submitter = CACHE_SUBMITTER_PATH.read_text()
        helper = HELPER_PATH.read_text()
        required_runner_fragments = (
            "readonly WARMUP_STEPS=10",
            "readonly MEASURE_STEPS=50",
            "readonly -a WORKER_MATRIX=(4 8 12 16)",
            "PROBE_MATRIX_MODE:=adaptive6",
            "PROBE_MATRIX_MODE\" == full16",
            "sample-gpu",
            "optimizer-step-wall-clock.ndjson",
            "make_throughput_probe_callback",
            "trainer.callbacks.production_batch_memory_gate.required=true",
            "checkpoint.load_training_state=true",
            "checkpoint.keys_to_skip_loading=[]",
            "checkpoint.keys_not_to_resume=[dataloader]",
            "trainer.max_iter=1060",
            "trainer.save_final_checkpoint=false",
            "CWAM_METADATA_ONLY_VIDEO_TOKENIZER=1",
            "PrecomputedWan2pt2LatentTokenizer",
            "NO_AC_CELL_TIMEOUT_SECONDS:=3600",
            "passed_with_no_ac_fallback",
            "EXPECTED_SNAPSHOT_MANIFEST_SHA256",
            'export PATH="$COSMOS_ENV/bin:$PATH"',
            'export PYTHONPATH="$COSMOS_SOURCE"',
            "minimum-cpus \"$MINIMUM_EFFECTIVE_CPUS\"",
            "NVME_MOUNT_ROOT:=/nvme",
            "NVME_MOUNT_SOURCE\" == /dev/nvme1n1",
            "ROBOCASA365_FIXED_INDEX_TRACE_PATH",
            "prepare-fixed-trace",
            "--dataset-root \"$ROBOCASA365_ROOT\"",
            "--source-root \"$ROBOCASA365_SOURCE_ROOT\"",
        )
        for fragment in required_runner_fragments:
            self.assertIn(fragment, runner)
        self.assertLess(
            runner.index('export PYTHONPATH="$COSMOS_SOURCE"'),
            runner.index('"$PYTHON" "$HELPER" validate-cache'),
        )
        self.assertIn("text_token_ids", helper)
        self.assertIn("readonly RJOB_CPU=160", submitter)
        self.assertIn("--store-host-nvme", submitter)
        self.assertNotIn("--store-host-nvme true", submitter)
        self.assertIn("--store-host-nvme", cache_submitter)
        self.assertNotIn("--store-host-nvme true", cache_submitter)
        self.assertIn("readonly RJOB_CPU=128", cache_submitter)
        self.assertIn("readonly SCHEDULER_RECEIPT_ATTEMPTS=86400", cache_submitter)
        self.assertIn("readonly SCHEDULER_RECEIPT_TIMEOUT_SECONDS=90000", cache_submitter)
        self.assertIn("--foreground --kill-after=10s 180s", cache_submitter)
        cache_wrapper = (
            Path(__file__).with_name("build_robocasa365_wan_latent_cache.sh").read_text()
        )
        self.assertIn(
            'export LD_LIBRARY_PATH="$H_NVIDIA_DRIVER_LIB:$H_NVIDIA_DRIVER_LIB64"',
            cache_wrapper,
        )
        self.assertIn(
            'readonly H_WORKER_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"',
            cache_wrapper,
        )
        self.assertIn(
            'readonly H_NVIDIA_SYSTEM_PATH="$H_NVIDIA_DRIVER_BIN:/usr/local/sbin:',
            cache_wrapper,
        )
        self.assertIn(
            '"$H_WORKER_LD_LIBRARY_PATH" != *::*',
            cache_wrapper,
        )
        self.assertIn(
            '"$H_NVIDIA_DRIVER_LIB"|"$H_NVIDIA_DRIVER_LIB64"',
            cache_wrapper,
        )
        self.assertNotIn('-d "$H_NVIDIA_DRIVER_LIB"', cache_wrapper)
        self.assertNotIn('-e "$H_NVIDIA_DRIVER_LIB64/libcuda.so.1"', cache_wrapper)
        self.assertIn(
            'NVIDIA_SMI_BIN="$(PATH="$H_NVIDIA_SYSTEM_PATH" type -P nvidia-smi || true)"',
            cache_wrapper,
        )
        self.assertIn('"$NVIDIA_SMI_BIN" -L', cache_wrapper)
        self.assertIn('ctypes.CDLL("libcuda.so.1")', cache_wrapper)
        self.assertIn('expected_torch_version = "2.10.0+cu128"', cache_wrapper)
        self.assertIn('expected_torch_cuda = "12.8"', cache_wrapper)
        self.assertIn("CUDA_RUNTIME_READY", cache_wrapper)
        self.assertIn("expected_device_count", cache_wrapper)
        self.assertIn("NVME_MOUNT_ROOT:=/nvme", submitter)
        self.assertIn("RJOB_PREDICT_ONLY", submitter)


if __name__ == "__main__":
    unittest.main()
