# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.utils.data
import wandb

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.easy_io import easy_io


_PROBE_METRIC_KEYS = (
    "flow_matching_loss_action",
    "strong_action_visual_memory_gate",
    "strong_action_visual_memory_rms",
    "strong_action_visual_memory_slot_cosine",
    "strong_action_visual_memory_slot_abs_cosine",
    "strong_action_visual_memory_diversity_loss",
    "strong_action_visual_memory_assignment_entropy_loss",
    "strong_action_visual_memory_assignment_balance_loss",
    "strong_action_visual_memory_spatial_compactness_loss",
    "strong_action_visual_memory_history_fraction",
    "strong_action_visual_memory_temporal_correspondence_loss",
    "strong_action_visual_memory_temporal_pair_fraction",
    "strong_action_visual_memory_object_contrastive_loss",
    "strong_action_visual_memory_episode_contrastive_loss",
    "strong_action_visual_memory_positive_similarity",
    "strong_action_visual_memory_object_negative_similarity",
    "strong_action_visual_memory_episode_negative_similarity",
    "strong_action_visual_memory_episode_negative_fraction",
    "strong_action_visual_memory_action_gradient_loss",
    "strong_action_visual_memory_action_gradient_valid_fraction",
    "strong_action_visual_memory_action_gradient_target_entropy",
    "strong_action_visual_memory_conditioning_scale",
    "strong_action_visual_memory_conditioned_residual_rms",
    "strong_action_visual_memory_action_aux_hidden_rms",
    "flow_matching_loss_action_visual_memory_only",
    "strong_action_visual_memory_action_aux_loss_weight",
    "strong_action_visual_memory_diversity_loss_weight",
    "strong_action_visual_memory_assignment_entropy_loss_weight",
    "strong_action_visual_memory_assignment_balance_loss_weight",
    "strong_action_visual_memory_spatial_compactness_loss_weight",
    "strong_action_visual_memory_temporal_loss_weight",
    "strong_action_visual_memory_object_contrastive_loss_weight",
    "strong_action_visual_memory_episode_contrastive_loss_weight",
    "strong_action_visual_memory_action_gradient_loss_weight",
)


def _normalize_dataset_name(dataset_name: object) -> object:
    """Collapse a packed batch's homogeneous dataset-name list to one key."""
    if not isinstance(dataset_name, list):
        return dataset_name
    if not dataset_name:
        raise ValueError("dataset_name must not be an empty list")
    first = dataset_name[0]
    if any(name != first for name in dataset_name[1:]):
        raise ValueError(f"Packed validation batch contains mixed dataset names: {dataset_name!r}")
    return first


def _extract_url_keys(data_batch: dict) -> list[str]:
    """Return every complete URL/key pair, or no coverage evidence if absent."""
    if "__url__" not in data_batch or "__key__" not in data_batch:
        return []
    urls = data_batch["__url__"]
    keys = data_batch["__key__"]
    urls = urls if isinstance(urls, list) else [urls]
    keys = keys if isinstance(keys, list) else [keys]
    if len(urls) != len(keys):
        raise ValueError(f"Validation URL/key cardinality mismatch: urls={len(urls)}, keys={len(keys)}")
    return [f"{url}, {key}" for url, key in zip(urls, keys)]


def _format_validation_summary(
    *,
    unique_count: int,
    total_count: int,
    average_loss: float,
    unstable_count: int,
    iteration: int,
    probe_metrics: dict[str, float] | None = None,
) -> str:
    """Build the machine-parseable validation health record used by formal gates."""
    summary = (
        f"[val] number of unique url and key: {unique_count} / {total_count}; "
        f"avg_final_loss: {average_loss}; unstable_count: {unstable_count}; iteration: {iteration}"
    )
    if probe_metrics:
        suffix = "; ".join(f"avg_{key}: {probe_metrics[key]}" for key in sorted(probe_metrics))
        summary = f"{summary}; {suffix}"
    return summary


@dataclass
class _LossRecord:
    loss: float = 0
    iter_count: int = 0

    def reset(self) -> None:
        self.loss = 0
        self.iter_count = 0

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            avg_loss_tensor = self.loss / self.iter_count
            # Create a mask (1 if valid, 0 if NaN or Inf)
            valid_mask = torch.tensor([torch.isfinite(avg_loss_tensor).float()], device="cuda")

            # Replace NaN/Inf with 0 to avoid affecting sum
            avg_loss_tensor = torch.where(
                torch.isfinite(avg_loss_tensor), avg_loss_tensor, torch.tensor([0.0], device="cuda")
            )

            # Reduce across all ranks
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)  # Sum of valid losses
            dist.all_reduce(valid_mask, op=dist.ReduceOp.SUM)  # Count of valid losses

            # Compute final average, avoiding division by zero
            if valid_mask.item() > 0:
                final_avg_loss = (avg_loss_tensor / valid_mask).item()
            else:
                final_avg_loss = 0.0  # Default to zero if all values were invalid

            avg_loss = final_avg_loss
        else:
            avg_loss = 0
        self.reset()
        return avg_loss


class WandbCallback(Callback):
    def __init__(
        self,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.final_loss_log = _LossRecord()
        self.final_loss_log_per_dataset = {}

        self.save_s3 = save_s3
        self.wandb_extra_tag = ""
        self.name = "wandb_loss_val_log"
        self.unstable_count = torch.zeros(1, device="cuda")
        self.url_key_list = []
        self.probe_metric_logs = {key: _LossRecord() for key in _PROBE_METRIC_KEYS}

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if torch.isnan(loss) or torch.isinf(loss):
            log.critical(
                f"Unstable val loss {loss} at iteration {iteration}",
                rank0_only=False,
            )
            self.unstable_count += 1

        dataset_name = _normalize_dataset_name(data_batch.get("dataset_name", "default"))

        if dataset_name not in self.final_loss_log_per_dataset:
            self.final_loss_log_per_dataset[dataset_name] = _LossRecord()

        self.final_loss_log_per_dataset[dataset_name].loss += loss.detach().float()
        self.final_loss_log_per_dataset[dataset_name].iter_count += 1
        self.final_loss_log.loss += loss.detach().float()
        self.final_loss_log.iter_count += 1

        for key, record in self.probe_metric_logs.items():
            value = output_batch.get(key)
            if value is None:
                continue
            if value.numel() != 1:
                raise ValueError(f"Validation probe metric {key!r} must be scalar, got {tuple(value.shape)}")
            record.loss += value.detach().float()
            record.iter_count += 1

        self.url_key_list.extend(_extract_url_keys(data_batch))

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        avg_final_loss = self.final_loss_log.get_stat()
        probe_metrics = {
            key: average
            for key, record in self.probe_metric_logs.items()
            if record.iter_count > 0 and (average := record.get_stat()) is not None
        }

        log.info(f"avg_final_loss: {avg_final_loss}")
        dist.all_reduce(self.unstable_count, op=dist.ReduceOp.SUM)
        # gather url and key list from all ranks
        url_key_list = [None] * dist.get_world_size()
        dist.all_gather_object(url_key_list, self.url_key_list)
        url_key_list = [item for sublist in url_key_list for item in sublist]

        unique_url_key_list = list(set(url_key_list))
        if distributed.is_rank0():
            info = {}
            unstable_count = int(self.unstable_count.item())
            log.info(
                _format_validation_summary(
                    unique_count=len(unique_url_key_list),
                    total_count=len(url_key_list),
                    average_loss=avg_final_loss,
                    unstable_count=unstable_count,
                    iteration=iteration,
                    probe_metrics=probe_metrics,
                )
            )
            info.update(
                {
                    f"val{self.wandb_extra_tag}/loss": avg_final_loss,
                    f"val{self.wandb_extra_tag}/unstable_count": unstable_count,
                    "iteration": iteration,
                    f"val{self.wandb_extra_tag}/num_unique_url_key": len(unique_url_key_list),
                    f"val{self.wandb_extra_tag}/total_url_key": len(url_key_list),
                }
            )
            info.update({f"val{self.wandb_extra_tag}/{key}": value for key, value in probe_metrics.items()})
            if self.save_s3:
                easy_io.dump(
                    info,
                    f"s3://rundir/{self.name}/Val_Iter{iteration:09d}.json",
                )

            if wandb.run is not None:
                wandb.log(info, step=iteration)

        # reset unstable count
        self.unstable_count.zero_()
        self.url_key_list = []
