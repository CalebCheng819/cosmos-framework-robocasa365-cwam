# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed gradient evidence for action-policy heads.

The generic norm monitor intentionally filters for generation-tower parameters.
Action-policy post-training also needs direct evidence that every freshly
initialized action interface is in the optimizer and receives a finite,
non-zero gradient. This callback aggregates local FSDP shards across ranks and
writes a small machine-readable report before the optimizer step.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.utils import distributed, log, misc
from cosmos_framework.utils.callback import Callback


class ActionGradientEvidence(Callback):
    """Validate and record gradients for named action-parameter families."""

    def __init__(
        self,
        families: list[str] | None = None,
        every_n: int = 50,
        fail_on_first_step: bool = True,
        output_subdir: str = "action_gradient_evidence",
        excluded_parameter_patterns: list[str] | None = None,
    ) -> None:
        self.families = tuple(families or ("action2llm", "llm2action", "action_modality_embed"))
        self.every_n = int(every_n)
        self.fail_on_first_step = bool(fail_on_first_step)
        self.output_subdir = output_subdir
        self.excluded_parameter_patterns = tuple(excluded_parameter_patterns or ())
        self._excluded_parameter_regexes = tuple(
            re.compile(pattern) for pattern in self.excluded_parameter_patterns
        )
        if not self.families:
            raise ValueError("ActionGradientEvidence requires at least one parameter family.")
        if self.every_n < 1:
            raise ValueError(f"ActionGradientEvidence.every_n must be positive, got {self.every_n}.")
        self._raw_report_by_iteration: dict[int, dict[str, dict[str, float | int]]] = {}
        self._report_path_by_iteration: dict[int, Path] = {}
        self._optimizer_boundary_iterations: set[int] = set()

    @staticmethod
    def _local(tensor: torch.Tensor) -> torch.Tensor:
        return misc.get_local_tensor_if_DTensor(tensor)

    def _collect(
        self,
        model: torch.nn.Module,
        optimizer: Any | None,
    ) -> dict[str, dict[str, float | int]]:
        named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if "net_ema" not in name and any(family in name for family in self.families)
        ]
        if not named_parameters:
            raise RuntimeError(f"ActionGradientEvidence found no action parameters for families {list(self.families)}.")

        device = self._local(named_parameters[0][1]).device
        optimizer_ids: set[int] = set()
        if optimizer is not None:
            if hasattr(optimizer, "param_groups"):
                inner_optimizers = [optimizer]
            elif hasattr(optimizer, "optimizers"):
                inner_optimizers = list(optimizer.optimizers)
            else:
                raise TypeError(
                    "ActionGradientEvidence expected a torch optimizer or an optimizer container "
                    f"with an 'optimizers' attribute, got {type(optimizer).__name__}."
                )
            for inner_optimizer in inner_optimizers:
                if not hasattr(inner_optimizer, "param_groups"):
                    raise TypeError(
                        "ActionGradientEvidence optimizer container includes an object without param_groups: "
                        f"{type(inner_optimizer).__name__}."
                    )
                optimizer_ids.update(
                    id(parameter) for group in inner_optimizer.param_groups for parameter in group["params"]
                )
        # Columns reduced by SUM:
        # total, trainable, optimizer, grad, nonfinite, nonzero, grad_sq_sum, excluded.
        sums = torch.zeros((len(self.families), 8), dtype=torch.float64, device=device)
        maxima = torch.zeros(len(self.families), dtype=torch.float64, device=device)

        for family_index, family in enumerate(self.families):
            for name, parameter in named_parameters:
                if family not in name:
                    continue
                local_parameter = self._local(parameter)
                local_elements = int(local_parameter.numel())
                if any(pattern.search(name) for pattern in self._excluded_parameter_regexes):
                    sums[family_index, 7] += local_elements
                    continue
                sums[family_index, 0] += local_elements
                if parameter.requires_grad:
                    sums[family_index, 1] += local_elements
                if optimizer is not None and id(parameter) in optimizer_ids:
                    sums[family_index, 2] += local_elements

                gradient = parameter.grad
                if gradient is None:
                    continue
                local_gradient = self._local(gradient)
                sums[family_index, 3] += int(local_gradient.numel())
                finite = torch.isfinite(local_gradient)
                sums[family_index, 4] += (~finite).sum(dtype=torch.float64)
                finite_gradient = torch.where(finite, local_gradient, torch.zeros_like(local_gradient))
                sums[family_index, 5] += torch.count_nonzero(finite_gradient)
                sums[family_index, 6] += finite_gradient.double().square().sum()
                if finite_gradient.numel() > 0:
                    maxima[family_index] = torch.maximum(
                        maxima[family_index],
                        finite_gradient.detach().abs().max().double(),
                    )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)

        report: dict[str, dict[str, float | int]] = {}
        for family_index, family in enumerate(self.families):
            values = sums[family_index].cpu().tolist()
            report[family] = {
                "parameter_elements": int(values[0]),
                "trainable_elements": int(values[1]),
                "optimizer_elements": int(values[2]),
                "gradient_elements": int(values[3]),
                "nonfinite_gradient_elements": int(values[4]),
                "nonzero_gradient_elements": int(values[5]),
                "gradient_l2_norm": float(values[6] ** 0.5),
                "gradient_abs_max": float(maxima[family_index].cpu().item()),
                "excluded_parameter_elements": int(values[7]),
            }
        return report

    def _validate_gradients(self, report: dict[str, dict[str, float | int]]) -> list[str]:
        failures: list[str] = []
        for family, stats in report.items():
            parameter_elements = int(stats["parameter_elements"])
            trainable_elements = int(stats["trainable_elements"])
            gradient_elements = int(stats["gradient_elements"])
            nonfinite = int(stats["nonfinite_gradient_elements"])
            nonzero = int(stats["nonzero_gradient_elements"])
            gradient_norm = float(stats["gradient_l2_norm"])
            if parameter_elements == 0:
                failures.append(f"{family}: parameter family is absent")
            if trainable_elements != parameter_elements:
                failures.append(
                    f"{family}: only {trainable_elements}/{parameter_elements} parameter elements require gradients"
                )
            if gradient_elements != parameter_elements:
                failures.append(
                    f"{family}: only {gradient_elements}/{parameter_elements} parameter elements have a gradient"
                )
            if nonfinite:
                failures.append(f"{family}: {nonfinite} gradient elements are non-finite")
            if nonzero == 0 or gradient_norm <= 0.0:
                failures.append(f"{family}: aggregate gradient norm is zero")
        return failures

    @staticmethod
    def _validate_optimizer(report: dict[str, dict[str, float | int]]) -> list[str]:
        failures: list[str] = []
        for family, stats in report.items():
            parameter_elements = int(stats["parameter_elements"])
            optimizer_elements = int(stats["optimizer_elements"])
            if optimizer_elements != parameter_elements:
                failures.append(
                    f"{family}: only {optimizer_elements}/{parameter_elements} parameter elements are in the optimizer"
                )
        return failures

    def _write_report(self, iteration: int, report: dict[str, Any], passed: bool) -> Path:
        configured_root = os.environ.get("CWAM_GRADIENT_EVIDENCE_DIR")
        root = Path(configured_root) if configured_root else Path(str(self.config.job.path_local)) / self.output_subdir
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"iteration_{iteration:09d}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "iteration": iteration,
                    "world_size": distributed.get_world_size(),
                    "gradient_stage": "after_backward_before_callback_clipping",
                    "families": report,
                    "status": "passed" if passed else "failed",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary.replace(target)
        return target

    @staticmethod
    def _raise_if_collective_write_failed(write_error: str | None, *, context: str) -> None:
        """Propagate rank-0 evidence I/O failures before ranks diverge."""
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        failed = torch.tensor([1 if write_error else 0], dtype=torch.int32, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if int(failed.item()) != 0:
            detail = write_error or "rank 0 could not persist action-gradient evidence"
            raise RuntimeError(f"ActionGradientEvidence {context} write failed: {detail}")

    def _should_record(self, iteration: int) -> bool:
        trainer = getattr(getattr(self, "config", None), "trainer", None)
        max_iter = getattr(trainer, "max_iter", None)
        is_final_iteration = isinstance(max_iter, int) and iteration + 1 == max_iter
        return iteration == 0 or (iteration + 1) % self.every_n == 0 or is_final_iteration

    def on_after_backward(self, model: torch.nn.Module, iteration: int = 0) -> None:
        if not self._should_record(iteration):
            return
        # The trainer calls this hook after model.on_after_backward(), but
        # before on_before_optimizer_step callbacks such as GradClip.
        report = self._collect(model, optimizer=None)
        failures = self._validate_gradients(report)
        self._raw_report_by_iteration[iteration] = report
        if failures and (self.fail_on_first_step or iteration != 0):
            raise RuntimeError(
                "Action-gradient evidence gate failed before optimizer callbacks: " + "; ".join(failures)
            )

    def on_before_optimizer_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del scheduler, grad_scaler
        completed_iteration = iteration + 1
        if not self._should_record(iteration):
            return

        raw_report = self._raw_report_by_iteration.pop(iteration, None)
        if raw_report is None:
            raise RuntimeError(
                f"ActionGradientEvidence has no pre-clip backward report for iteration {completed_iteration}."
            )
        optimizer_report = self._collect(model, optimizer)
        self._optimizer_boundary_iterations.add(iteration)
        for family in self.families:
            raw_report[family]["optimizer_elements"] = optimizer_report[family]["optimizer_elements"]
        failures = self._validate_gradients(raw_report) + self._validate_optimizer(raw_report)
        write_error: str | None = None
        if distributed.is_rank0():
            try:
                path = self._write_report(completed_iteration, raw_report, passed=not failures)
                self._report_path_by_iteration[iteration] = path
                log.info(
                    "ACTION_GRADIENT_EVIDENCE "
                    + json.dumps(
                        {
                            "iteration": completed_iteration,
                            "status": "passed" if not failures else "failed",
                            "path": str(path),
                            "families": raw_report,
                        },
                        sort_keys=True,
                    )
                )
            except Exception as error:  # noqa: BLE001 - synchronized to every rank below.
                write_error = f"{type(error).__name__}: {error}"
        self._raise_if_collective_write_failed(write_error, context="gradient")
        if failures and (self.fail_on_first_step or iteration != 0):
            raise RuntimeError("Action-gradient evidence gate failed: " + "; ".join(failures))

    def on_training_step_batch_end(
        self,
        model: torch.nn.Module,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch
        if not self._should_record(iteration):
            return
        # This hook runs after every microbatch, whereas
        # ``on_before_optimizer_step`` runs only on the final microbatch in a
        # gradient-accumulation window. Do not demand a report path until that
        # optimizer-boundary hook has fired.
        if iteration not in self._optimizer_boundary_iterations:
            return
        self._optimizer_boundary_iterations.remove(iteration)
        action_loss = output_batch.get("flow_matching_loss_action")
        if not isinstance(action_loss, torch.Tensor):
            raise RuntimeError("ActionGradientEvidence did not receive flow_matching_loss_action.")
        total_loss = self._local(loss.detach()).float().mean()
        action_loss_value = self._local(action_loss.detach()).float().mean()
        losses = torch.stack([total_loss, action_loss_value])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= dist.get_world_size()
        total_loss_number = float(losses[0].cpu().item())
        action_loss_number = float(losses[1].cpu().item())
        loss_failures: list[str] = []
        if not torch.isfinite(losses[0]):
            loss_failures.append(f"total loss is non-finite: {total_loss_number}")
        if not torch.isfinite(losses[1]):
            loss_failures.append(f"action flow-matching loss is non-finite: {action_loss_number}")
        elif action_loss_number <= 0.0:
            loss_failures.append(f"action flow-matching loss must be positive, got {action_loss_number}")

        write_error: str | None = None
        if distributed.is_rank0():
            try:
                path = self._report_path_by_iteration.pop(iteration, None)
                if path is None:
                    raise RuntimeError(f"ActionGradientEvidence report path is absent for iteration {iteration + 1}.")
                payload = json.loads(path.read_text())
                payload["losses"] = {
                    "scope": "final_microbatch_at_optimizer_boundary",
                    "final_microbatch_total_loss_rank_mean": total_loss_number,
                    "final_microbatch_flow_matching_loss_action_rank_mean": action_loss_number,
                }
                if loss_failures:
                    payload["status"] = "failed"
                    payload["loss_failures"] = loss_failures
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                temporary.replace(path)
            except Exception as error:  # noqa: BLE001 - synchronized to every rank below.
                write_error = f"{type(error).__name__}: {error}"
        self._raise_if_collective_write_failed(write_error, context="loss")
        if loss_failures:
            raise RuntimeError("Action-loss evidence gate failed: " + "; ".join(loss_failures))
