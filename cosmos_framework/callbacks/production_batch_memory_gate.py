# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed first-step GPU-memory evidence for production action batches."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


def _evaluate_memory_rows(
    rows: list[dict[str, int]],
    *,
    max_peak_reserved_fraction: float,
    max_driver_used_fraction: float,
) -> tuple[list[dict[str, int | float]], list[str]]:
    """Validate gathered per-rank counters and classify threshold violations."""
    if not 0.0 < max_peak_reserved_fraction < 1.0:
        raise ValueError(f"max_peak_reserved_fraction must be between zero and one, got {max_peak_reserved_fraction}.")
    if not 0.0 < max_driver_used_fraction < 1.0:
        raise ValueError(f"max_driver_used_fraction must be between zero and one, got {max_driver_used_fraction}.")
    if not rows:
        raise ValueError("ProductionBatchMemoryGate received no per-rank memory rows.")

    evaluated: list[dict[str, int | float]] = []
    violations: list[str] = []
    expected_ranks = list(range(len(rows)))
    actual_ranks = sorted(int(row["rank"]) for row in rows)
    if actual_ranks != expected_ranks:
        raise ValueError(f"Expected one memory row for ranks {expected_ranks}, got {actual_ranks}.")

    for row in sorted(rows, key=lambda item: int(item["rank"])):
        rank = int(row["rank"])
        total = int(row["total_memory_bytes"])
        peak_allocated = int(row["peak_allocated_bytes"])
        peak_reserved = int(row["peak_reserved_bytes"])
        current_allocated = int(row["current_allocated_bytes"])
        current_reserved = int(row["current_reserved_bytes"])
        driver_free = int(row["driver_free_bytes"])
        driver_used = int(row["driver_used_bytes"])
        counters = {
            "total_memory_bytes": total,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "current_allocated_bytes": current_allocated,
            "current_reserved_bytes": current_reserved,
            "driver_free_bytes": driver_free,
            "driver_used_bytes": driver_used,
        }
        if total <= 0:
            raise ValueError(f"rank {rank}: total_memory_bytes must be positive, got {total}.")
        for name, value in counters.items():
            if name != "total_memory_bytes" and not 0 <= value <= total:
                raise ValueError(f"rank {rank}: {name}={value} is outside [0, {total}].")
        if peak_allocated < current_allocated:
            raise ValueError(
                f"rank {rank}: peak_allocated_bytes={peak_allocated} "
                f"is below current_allocated_bytes={current_allocated}."
            )
        if peak_reserved < current_reserved:
            raise ValueError(
                f"rank {rank}: peak_reserved_bytes={peak_reserved} is below current_reserved_bytes={current_reserved}."
            )

        peak_allocated_fraction = peak_allocated / total
        peak_reserved_fraction = peak_reserved / total
        driver_used_fraction = driver_used / total
        for name, value in (
            ("peak_allocated_fraction", peak_allocated_fraction),
            ("peak_reserved_fraction", peak_reserved_fraction),
            ("driver_used_fraction", driver_used_fraction),
        ):
            if not math.isfinite(value):
                raise ValueError(f"rank {rank}: {name} is non-finite.")

        evaluated_row: dict[str, int | float] = {
            **{key: int(value) for key, value in row.items()},
            "peak_allocated_fraction": peak_allocated_fraction,
            "peak_reserved_fraction": peak_reserved_fraction,
            "driver_used_fraction": driver_used_fraction,
        }
        evaluated.append(evaluated_row)
        if peak_reserved_fraction >= max_peak_reserved_fraction:
            violations.append(
                f"rank {rank}: peak reserved fraction {peak_reserved_fraction:.6f} >= {max_peak_reserved_fraction:.6f}"
            )
        if driver_used_fraction >= max_driver_used_fraction:
            violations.append(
                f"rank {rank}: driver used fraction {driver_used_fraction:.6f} >= {max_driver_used_fraction:.6f}"
            )
    return evaluated, violations


def _evaluate_production_batch_rows(
    rows: list[dict[str, int | float]],
    *,
    expected_samples_per_microbatch: int,
    expected_grad_accum_iter: int,
    expected_action_tokens_per_sample: int,
) -> list[str]:
    """Validate the model-observed production batch on every rank/microbatch."""
    if expected_samples_per_microbatch < 1:
        raise ValueError("expected_samples_per_microbatch must be positive.")
    if expected_grad_accum_iter < 1:
        raise ValueError("expected_grad_accum_iter must be positive.")
    if expected_action_tokens_per_sample < 1:
        raise ValueError("expected_action_tokens_per_sample must be positive.")

    violations: list[str] = []
    expected_action_tokens = expected_samples_per_microbatch * expected_action_tokens_per_sample
    for row in sorted(rows, key=lambda item: int(item["rank"])):
        rank = int(row["rank"])
        observed_count = int(row["observed_microbatch_count"])
        if observed_count != expected_grad_accum_iter:
            violations.append(
                f"rank {rank}: observed {observed_count} microbatches; expected {expected_grad_accum_iter}"
            )
        for microbatch in range(expected_grad_accum_iter):
            prefix = f"microbatch_{microbatch}"
            input_samples = int(row[f"{prefix}_input_sample_count"])
            unique_identities = int(row[f"{prefix}_unique_identity_count"])
            model_batch_size = int(row[f"{prefix}_model_batch_size"])
            und_tokens = int(row[f"{prefix}_und_token_length"])
            gen_tokens = int(row[f"{prefix}_gen_token_length"])
            vision_tokens = int(row[f"{prefix}_vision_token_length"])
            action_tokens = int(row[f"{prefix}_action_token_length"])
            if input_samples != expected_samples_per_microbatch:
                violations.append(
                    f"rank {rank} microbatch {microbatch}: input sample count {input_samples}; "
                    f"expected {expected_samples_per_microbatch}"
                )
            if unique_identities != expected_samples_per_microbatch:
                violations.append(
                    f"rank {rank} microbatch {microbatch}: unique identity count {unique_identities}; "
                    f"expected {expected_samples_per_microbatch}"
                )
            if model_batch_size != expected_samples_per_microbatch:
                violations.append(
                    f"rank {rank} microbatch {microbatch}: model batch size {model_batch_size}; "
                    f"expected {expected_samples_per_microbatch}"
                )
            for name, value in (
                ("und_token_length", und_tokens),
                ("gen_token_length", gen_tokens),
                ("vision_token_length", vision_tokens),
            ):
                if value <= 0:
                    violations.append(f"rank {rank} microbatch {microbatch}: {name}={value} is not positive")
            if action_tokens != expected_action_tokens:
                violations.append(
                    f"rank {rank} microbatch {microbatch}: action_token_length={action_tokens}; "
                    f"expected {expected_action_tokens}"
                )
    return violations


def _logical_tensor_bytes(value: Any) -> int:
    """Sum logical tensor storage represented by a nested training batch."""
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_logical_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_logical_tensor_bytes(item) for item in value)
    return 0


def _sequence_length(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple)) else -1


def _unique_identity_count(data_batch: dict[str, Any]) -> int:
    urls = data_batch.get("__url__")
    keys = data_batch.get("__key__")
    if not isinstance(urls, (list, tuple)) or not isinstance(keys, (list, tuple)) or len(urls) != len(keys):
        return -1
    identities = list(zip(urls, keys))
    if any(not isinstance(url, str) or not isinstance(key, str) for url, key in identities):
        return -1
    return len(set(identities))


def _integer_metric(output_batch: dict[str, Any], key: str) -> int:
    value = output_batch.get(key)
    if isinstance(value, torch.Tensor):
        return int(value.item()) if value.numel() == 1 else -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


class ProductionBatchMemoryGate(Callback):
    """Record and enforce memory headroom after the first optimizer step."""

    _MEMORY_FIELDS = (
        "rank",
        "device_index",
        "total_memory_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "current_allocated_bytes",
        "current_reserved_bytes",
        "driver_free_bytes",
        "driver_used_bytes",
    )

    def __init__(
        self,
        max_peak_reserved_fraction: float = 0.94,
        max_driver_used_fraction: float = 0.94,
        expected_samples_per_microbatch: int = 128,
        expected_grad_accum_iter: int = 2,
        expected_action_tokens_per_sample: int = 16,
        allowed_world_sizes: list[int] | None = None,
        required: bool = True,
        output_subdir: str = "production_batch_memory_gate",
    ) -> None:
        self.max_peak_reserved_fraction = float(max_peak_reserved_fraction)
        self.max_driver_used_fraction = float(max_driver_used_fraction)
        self.expected_samples_per_microbatch = int(expected_samples_per_microbatch)
        self.expected_grad_accum_iter = int(expected_grad_accum_iter)
        self.expected_action_tokens_per_sample = int(expected_action_tokens_per_sample)
        configured_world_sizes = [6, 8] if allowed_world_sizes is None else allowed_world_sizes
        self.allowed_world_sizes = tuple(int(value) for value in configured_world_sizes)
        self.required = bool(required)
        self.output_subdir = str(output_subdir)
        _evaluate_memory_rows(
            [
                {
                    "rank": 0,
                    "device_index": 0,
                    "total_memory_bytes": 1,
                    "peak_allocated_bytes": 0,
                    "peak_reserved_bytes": 0,
                    "current_allocated_bytes": 0,
                    "current_reserved_bytes": 0,
                    "driver_free_bytes": 1,
                    "driver_used_bytes": 0,
                }
            ],
            max_peak_reserved_fraction=self.max_peak_reserved_fraction,
            max_driver_used_fraction=self.max_driver_used_fraction,
        )
        if not self.output_subdir:
            raise ValueError("ProductionBatchMemoryGate.output_subdir must be non-empty.")
        if self.expected_samples_per_microbatch < 1:
            raise ValueError("expected_samples_per_microbatch must be positive.")
        if self.expected_grad_accum_iter < 1:
            raise ValueError("expected_grad_accum_iter must be positive.")
        if self.expected_action_tokens_per_sample < 1:
            raise ValueError("expected_action_tokens_per_sample must be positive.")
        if not self.allowed_world_sizes or any(value < 1 for value in self.allowed_world_sizes):
            raise ValueError("allowed_world_sizes must contain positive integers.")
        self._start_iteration: int | None = None
        self._recorded = False
        self._microbatches: list[dict[str, int]] = []
        self._included_initial_validation = False
        self._first_step_includes_checkpoint_save = False

    @staticmethod
    def _distributed_rank() -> int:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    @staticmethod
    def _distributed_world_size() -> int:
        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def on_train_start(self, model: torch.nn.Module, iteration: int = 0) -> None:
        del model
        if not self.required:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("ProductionBatchMemoryGate requires CUDA.")
        world_size = self._distributed_world_size()
        if world_size not in self.allowed_world_sizes:
            raise RuntimeError(
                f"ProductionBatchMemoryGate world_size={world_size} is not allowed; "
                f"expected one of {list(self.allowed_world_sizes)}."
            )
        local_world_size_text = os.environ.get("LOCAL_WORLD_SIZE")
        if local_world_size_text is None or int(local_world_size_text) != world_size:
            raise RuntimeError(
                "ProductionBatchMemoryGate requires one torchrun node with "
                f"LOCAL_WORLD_SIZE=WORLD_SIZE={world_size}; got LOCAL_WORLD_SIZE={local_world_size_text!r}."
            )
        if str(self.config.trainer.distributed_parallelism) != "fsdp":
            raise RuntimeError(
                "ProductionBatchMemoryGate requires trainer.distributed_parallelism='fsdp', "
                f"got {self.config.trainer.distributed_parallelism!r}."
            )
        resolved_grad_accum = int(self.config.trainer.grad_accum_iter)
        if resolved_grad_accum != self.expected_grad_accum_iter:
            raise RuntimeError(
                f"ProductionBatchMemoryGate expected grad_accum_iter={self.expected_grad_accum_iter}, "
                f"resolved {resolved_grad_accum}."
            )
        resolved_sample_cap = int(self.config.dataloader_train.max_samples_per_batch)
        if resolved_sample_cap != self.expected_samples_per_microbatch:
            raise RuntimeError(
                "ProductionBatchMemoryGate expected "
                f"max_samples_per_batch={self.expected_samples_per_microbatch}, resolved {resolved_sample_cap}."
            )
        parallelism = self.config.model.config.parallelism
        shard_degree = int(parallelism.data_parallel_shard_degree)
        replicate_degree = int(parallelism.data_parallel_replicate_degree)
        context_degree = int(parallelism.context_parallel_shard_degree)
        cfg_degree = int(parallelism.cfg_parallel_shard_degree)
        if shard_degree != world_size or replicate_degree != 1 or context_degree != 1 or cfg_degree != 1:
            raise RuntimeError(
                "ProductionBatchMemoryGate requires single-node pure FSDP with "
                "dp_shard=world_size, replicate=1, context=1, and cfg=1; "
                f"got world={world_size}, shard={shard_degree}, replicate={replicate_degree}, "
                f"context={context_degree}, cfg={cfg_degree}."
            )
        torch.cuda.synchronize()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        self._start_iteration = int(iteration)
        self._recorded = False
        self._microbatches = []
        self._included_initial_validation = bool(
            int(iteration) == 0 and self.config.trainer.run_validation and self.config.trainer.run_validation_on_start
        )
        first_completed_iteration = int(iteration) + 1
        self._first_step_includes_checkpoint_save = (
            first_completed_iteration % int(self.config.checkpoint.save_iter) == 0
        )

    def on_training_step_batch_end(
        self,
        model: torch.nn.Module,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, loss
        if not self.required or self._recorded or self._start_iteration is None:
            return
        if int(iteration) != self._start_iteration:
            return
        self._microbatches.append(
            {
                "input_sample_count": _sequence_length(data_batch.get("video")),
                "unique_identity_count": _unique_identity_count(data_batch),
                "model_batch_size": _integer_metric(output_batch, "batch_size"),
                "und_token_length": _integer_metric(output_batch, "und_token_length"),
                "gen_token_length": _integer_metric(output_batch, "gen_token_length"),
                "vision_token_length": _integer_metric(output_batch, "vision_token_length"),
                "action_token_length": _integer_metric(output_batch, "action_token_length"),
                "input_tensor_bytes": _logical_tensor_bytes(data_batch),
            }
        )

    def _collect_rows(self) -> list[dict[str, int]]:
        torch.cuda.synchronize()
        device_index = torch.cuda.current_device()
        device = torch.device("cuda", device_index)
        driver_free, driver_total = torch.cuda.mem_get_info(device_index)
        fields = list(self._MEMORY_FIELDS) + ["observed_microbatch_count"]
        raw_values = [
            self._distributed_rank(),
            device_index,
            driver_total,
            torch.cuda.max_memory_allocated(device_index),
            torch.cuda.max_memory_reserved(device_index),
            torch.cuda.memory_allocated(device_index),
            torch.cuda.memory_reserved(device_index),
            driver_free,
            driver_total - driver_free,
            len(self._microbatches),
        ]
        observation_fields = (
            "input_sample_count",
            "unique_identity_count",
            "model_batch_size",
            "und_token_length",
            "gen_token_length",
            "vision_token_length",
            "action_token_length",
            "input_tensor_bytes",
        )
        for microbatch in range(self.expected_grad_accum_iter):
            observation = self._microbatches[microbatch] if microbatch < len(self._microbatches) else {}
            for name in observation_fields:
                fields.append(f"microbatch_{microbatch}_{name}")
                raw_values.append(int(observation.get(name, -1)))
        values = torch.tensor(raw_values, dtype=torch.int64, device=device)
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.empty_like(values) for _ in range(self._distributed_world_size())]
            dist.all_gather(gathered, values)
        else:
            gathered = [values]
        return [
            {name: int(value) for name, value in zip(fields, gathered_row.cpu().tolist())} for gathered_row in gathered
        ]

    def _report_path(self, iteration: int) -> Path:
        configured_root = os.environ.get("CWAM_MEMORY_GATE_DIR")
        root = Path(configured_root) if configured_root else Path(str(self.config.job.path_local)) / self.output_subdir
        return root / f"iteration_{iteration:09d}.json"

    def _write_report(self, iteration: int, payload: dict[str, Any]) -> str | None:
        if self._distributed_rank() != 0:
            return None
        try:
            target = self._report_path(iteration)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.replace(target)
        except Exception as error:  # noqa: BLE001 - propagated consistently to all ranks below.
            return f"{type(error).__name__}: {error}"
        return None

    def on_training_step_end(
        self,
        model: torch.nn.Module,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch, output_batch, loss
        if not self.required or self._recorded:
            return
        if self._start_iteration is None:
            raise RuntimeError("ProductionBatchMemoryGate.on_train_start was not called.")
        if int(iteration) != self._start_iteration + 1:
            return

        rows, violations = _evaluate_memory_rows(
            self._collect_rows(),
            max_peak_reserved_fraction=self.max_peak_reserved_fraction,
            max_driver_used_fraction=self.max_driver_used_fraction,
        )
        violations.extend(
            _evaluate_production_batch_rows(
                rows,
                expected_samples_per_microbatch=self.expected_samples_per_microbatch,
                expected_grad_accum_iter=self.expected_grad_accum_iter,
                expected_action_tokens_per_sample=self.expected_action_tokens_per_sample,
            )
        )
        included_phases = [
            "resident_model_optimizer_checkpoint_state",
            f"{self.expected_grad_accum_iter}_training_microbatches",
            "first_optimizer_scheduler_ema_step",
        ]
        if self._included_initial_validation:
            included_phases.insert(1, "initial_ema_validation")
        if self._first_step_includes_checkpoint_save:
            included_phases.append("first_step_dcp_save")
        payload = {
            "schema_version": 1,
            "status": "failed" if violations else "passed",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "scope": "first_production_optimizer_boundary_after_checkpoint_load",
            "included_phases": included_phases,
            "start_iteration": self._start_iteration,
            "observed_iteration": int(iteration),
            "world_size": self._distributed_world_size(),
            "production_batch_contract": {
                "expected_samples_per_microbatch": self.expected_samples_per_microbatch,
                "expected_grad_accum_iter": self.expected_grad_accum_iter,
                "expected_action_tokens_per_sample": self.expected_action_tokens_per_sample,
                "allowed_world_sizes": list(self.allowed_world_sizes),
            },
            "thresholds": {
                "max_peak_reserved_fraction": self.max_peak_reserved_fraction,
                "max_driver_used_fraction": self.max_driver_used_fraction,
            },
            "per_rank": rows,
            "violations": violations,
        }
        write_error = self._write_report(int(iteration), payload)
        write_failed = torch.tensor(
            [1 if write_error else 0],
            dtype=torch.int32,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(write_failed, op=dist.ReduceOp.MAX)
        if int(write_failed.item()) != 0:
            detail = write_error or "rank 0 could not persist the memory-gate report"
            raise RuntimeError(f"ProductionBatchMemoryGate evidence write failed: {detail}")

        self._recorded = True
        if violations:
            raise RuntimeError("ProductionBatchMemoryGate failed: " + "; ".join(violations))

    def on_train_end(self, model: torch.nn.Module, iteration: int = 0) -> None:
        del model
        if self.required and not self._recorded:
            raise RuntimeError(
                "ProductionBatchMemoryGate was required but no first-step report was recorded; "
                f"training ended at iteration {iteration} after starting at {self._start_iteration}."
            )
