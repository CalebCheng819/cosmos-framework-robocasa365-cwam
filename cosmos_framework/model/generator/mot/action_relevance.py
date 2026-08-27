# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action-conditioned routing over clean policy context tokens."""

from __future__ import annotations

import math
from itertools import accumulate
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from cosmos_framework.data.generator.sequence_packing import ModalityData


ACTION_RELEVANCE_INTERVENTIONS = (
    "normal",
    "selected",
    "complement",
    "random_matched",
    "zero",
    "shuffle",
    "cross_episode",
)


def conditioned_vision_token_indexes(vision: ModalityData) -> torch.Tensor:
    """Return packed indexes for clean vision frames, preserving item order."""
    if not isinstance(vision.sequence_indexes, torch.Tensor):
        raise TypeError("vision.sequence_indexes must be a finalized tensor")
    if len(vision.token_shapes) != len(vision.condition_mask):
        raise ValueError(
            "vision token-shape/condition-mask count mismatch: "
            f"{len(vision.token_shapes)} != {len(vision.condition_mask)}"
        )

    selected: list[torch.Tensor] = []
    cursor = 0
    for token_shape, condition_mask in zip(vision.token_shapes, vision.condition_mask):
        if len(token_shape) != 3:
            raise ValueError(f"vision token shape must be (T,H,W), got {token_shape!r}")
        latent_t, latent_h, latent_w = (int(value) for value in token_shape)
        item_tokens = latent_t * latent_h * latent_w
        item_indexes = vision.sequence_indexes[cursor : cursor + item_tokens]
        if item_indexes.numel() != item_tokens:
            raise ValueError("vision sequence indexes ended before token_shapes")

        frame_mask = condition_mask.reshape(latent_t, -1).amax(dim=1) > 0
        token_mask = frame_mask[:, None].expand(latent_t, latent_h * latent_w).reshape(-1)
        selected.append(item_indexes[token_mask])
        cursor += item_tokens

    if cursor != vision.sequence_indexes.numel():
        raise ValueError(
            "vision token shapes do not cover sequence indexes: "
            f"covered={cursor}, indexes={vision.sequence_indexes.numel()}"
        )
    if not selected:
        return vision.sequence_indexes.new_empty((0,))
    return torch.cat(selected)


class ActionRelevanceAdapter(nn.Module):
    """Soft relation-aware context routing for packed action tokens.

    The official MoT output remains the primary path. This module adds a gated
    residual only at action-token positions, using task tokens and clean visual
    tokens as context. It never removes or replaces the original visual path.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        routing_dim: int,
        num_heads: int,
        topk: int,
        initial_residual_gate: float,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or routing_dim <= 0:
            raise ValueError("hidden_size and routing_dim must be positive")
        if num_heads <= 0 or routing_dim % num_heads != 0:
            raise ValueError("routing_dim must be divisible by num_heads")
        if topk <= 0:
            raise ValueError("topk must be positive")
        if not 0.0 < initial_residual_gate < 1.0:
            raise ValueError("initial_residual_gate must be strictly between 0 and 1")

        self.hidden_size = hidden_size
        self.routing_dim = routing_dim
        self.num_heads = num_heads
        self.head_dim = routing_dim // num_heads
        self.topk = topk
        self.initial_residual_gate = initial_residual_gate

        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.query_proj = nn.Linear(hidden_size, routing_dim, bias=False)
        self.key_proj = nn.Linear(hidden_size, routing_dim, bias=False)
        self.value_proj = nn.Linear(hidden_size, routing_dim, bias=False)
        self.output_proj = nn.Linear(routing_dim, hidden_size, bias=False)
        self.relation_bias = nn.Sequential(
            nn.Linear(5, 64),
            nn.SiLU(),
            nn.Linear(64, num_heads),
        )
        initial_logit = math.log(initial_residual_gate / (1.0 - initial_residual_gate))
        # FSDP2 rejects zero-dimensional parameters.
        self.residual_gate_logit = nn.Parameter(torch.tensor([initial_logit]))
        self.intervention_mode = "normal"
        self.intervention_seed = 0
        self.capture_evidence = False
        self.capture_graph = False
        self.last_evidence: dict[str, Any] | None = None
        self.last_routed_c: torch.Tensor | None = None
        # Runtime-only donor used by strict episode-level replay interventions.
        # This is deliberately not a buffer, so it never enters a checkpoint.
        self.fixed_routed_c: torch.Tensor | None = None

    def configure_evidence(
        self,
        *,
        intervention_mode: str = "normal",
        intervention_seed: int = 0,
        capture_evidence: bool = False,
        capture_graph: bool = False,
    ) -> None:
        if intervention_mode not in ACTION_RELEVANCE_INTERVENTIONS:
            raise ValueError(
                f"Unknown action-relevance intervention {intervention_mode!r}; "
                f"expected one of {ACTION_RELEVANCE_INTERVENTIONS}"
            )
        self.intervention_mode = intervention_mode
        self.intervention_seed = intervention_seed
        self.capture_evidence = capture_evidence
        self.capture_graph = capture_graph
        self.last_evidence = None
        self.last_routed_c = None

    def set_fixed_routed_c(self, routed_c: torch.Tensor | None) -> None:
        """Set one exact routed-C donor for the next forward pass."""
        if routed_c is not None and not isinstance(routed_c, torch.Tensor):
            raise TypeError("fixed routed C must be a tensor or None")
        self.fixed_routed_c = routed_c

    def init_weights(self) -> None:
        nn.init.ones_(self.query_norm.weight)
        nn.init.zeros_(self.query_norm.bias)
        nn.init.ones_(self.context_norm.weight)
        nn.init.zeros_(self.context_norm.bias)

        input_std = 1.0 / math.sqrt(self.hidden_size)
        output_std = 1.0 / math.sqrt(self.routing_dim)
        for projection in (self.query_proj, self.key_proj, self.value_proj):
            nn.init.trunc_normal_(projection.weight, std=input_std, a=-3 * input_std, b=3 * input_std)
        nn.init.trunc_normal_(
            self.output_proj.weight,
            std=output_std,
            a=-3 * output_std,
            b=3 * output_std,
        )
        first_relation = self.relation_bias[0]
        final_relation = self.relation_bias[2]
        assert isinstance(first_relation, nn.Linear)
        assert isinstance(final_relation, nn.Linear)
        nn.init.trunc_normal_(first_relation.weight, std=0.02, a=-0.06, b=0.06)
        nn.init.zeros_(first_relation.bias)
        nn.init.zeros_(final_relation.weight)
        nn.init.zeros_(final_relation.bias)
        initial_logit = math.log(self.initial_residual_gate / (1.0 - self.initial_residual_gate))
        nn.init.constant_(self.residual_gate_logit, initial_logit)

    @staticmethod
    def _sample_ids(indexes: torch.Tensor, sample_lens: list[int]) -> torch.Tensor:
        if not sample_lens or any(length <= 0 for length in sample_lens):
            raise ValueError(f"sample_lens must contain positive lengths, got {sample_lens!r}")
        boundaries = torch.tensor(
            list(accumulate(sample_lens[:-1])),
            device=indexes.device,
            dtype=indexes.dtype,
        )
        if boundaries.numel() == 0:
            return torch.zeros_like(indexes)
        return torch.bucketize(indexes, boundaries, right=True)

    @staticmethod
    def _signed_log_position_delta(delta: torch.Tensor) -> torch.Tensor:
        return delta.sign() * torch.log1p(delta.abs()) / 10.0

    @staticmethod
    def _renormalize_weights(weights: torch.Tensor) -> torch.Tensor:
        denominator = weights.sum(dim=-1, keepdim=True)
        return torch.where(denominator > 0, weights / denominator.clamp_min(1.0e-12), weights)

    def _intervene_on_weights(
        self,
        weights: torch.Tensor,
        context_valid: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.intervention_mode
        if mode not in {"selected", "complement", "random_matched"}:
            return weights

        valid = context_valid[:, None, None, :].expand_as(weights)
        k = min(self.topk, weights.shape[-1])
        if mode in {"selected", "complement"}:
            pooled_scores = weights.mean(dim=1, keepdim=True)
            selected = torch.zeros_like(pooled_scores, dtype=torch.bool)
            selected.scatter_(-1, pooled_scores.topk(k, dim=-1).indices, True)
            selected = selected.expand_as(weights) & valid
            keep = selected if mode == "selected" else valid & ~selected
        else:
            generator = torch.Generator(device=weights.device)
            generator.manual_seed(self.intervention_seed)
            random_scores = torch.rand(
                (weights.shape[0], 1, weights.shape[2], weights.shape[3]),
                device=weights.device,
                dtype=torch.float32,
                generator=generator,
            ).masked_fill(~valid[:, :1], -torch.inf)
            keep = torch.zeros_like(random_scores, dtype=torch.bool)
            keep.scatter_(-1, random_scores.topk(k, dim=-1).indices, True)
            keep = keep.expand_as(weights) & valid
        return self._renormalize_weights(weights.masked_fill(~keep, 0.0))

    def _intervene_on_routed(
        self,
        routed: torch.Tensor,
        query_lengths: list[int],
    ) -> torch.Tensor:
        mode = self.intervention_mode
        if mode == "zero":
            return torch.zeros_like(routed)
        if mode in {"shuffle", "cross_episode"}:
            if self.fixed_routed_c is None:
                raise ValueError(
                    f"{mode} intervention requires an exact fixed routed-C donor"
                )
            if self.fixed_routed_c.shape != routed.shape:
                raise ValueError(
                    "fixed routed-C intervention shape mismatch: "
                    f"expected={tuple(routed.shape)}, got={tuple(self.fixed_routed_c.shape)}"
                )
            return self.fixed_routed_c.to(device=routed.device, dtype=routed.dtype)
        return routed

    def _record_evidence(
        self,
        *,
        base_weights: torch.Tensor,
        used_weights: torch.Tensor,
        routed_normal: torch.Tensor,
        routed: torch.Tensor,
        query_valid: torch.Tensor,
        context_valid: torch.Tensor,
        context_vision: torch.Tensor,
        query_position: torch.Tensor,
        context_position: torch.Tensor,
        query_lengths: list[int],
        context_lengths: list[int],
    ) -> None:
        self.last_routed_c = routed if self.capture_graph else None
        if self.capture_graph and routed.requires_grad:
            routed.retain_grad()
        if not self.capture_evidence:
            self.last_evidence = None
            return

        def cpu(value: torch.Tensor) -> torch.Tensor:
            return value.detach().to(device="cpu")

        self.last_evidence = {
            "intervention_mode": self.intervention_mode,
            "attention_normal": cpu(base_weights.mean(dim=1)),
            "attention_used": cpu(used_weights.mean(dim=1)),
            "routed_c_normal": cpu(routed_normal),
            "routed_c": cpu(routed),
            "query_valid": cpu(query_valid),
            "context_valid": cpu(context_valid),
            "context_is_vision": cpu(context_vision),
            "query_position": cpu(query_position),
            "context_position": cpu(context_position),
            "query_lengths": list(query_lengths),
            "context_lengths": list(context_lengths),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        action_indexes: torch.Tensor,
        context_indexes: torch.Tensor,
        context_is_vision: torch.Tensor,
        position_ids: torch.Tensor,
        sample_lens: list[int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError(f"hidden_states must be [N,{self.hidden_size}], got {tuple(hidden_states.shape)}")
        if action_indexes.ndim != 1 or context_indexes.ndim != 1:
            raise ValueError("action_indexes and context_indexes must be one-dimensional")
        if context_is_vision.shape != context_indexes.shape or context_is_vision.dtype != torch.bool:
            raise ValueError("context_is_vision must be a boolean vector aligned with context_indexes")
        if position_ids.shape != (3, hidden_states.shape[0]):
            raise ValueError(
                f"position_ids must be [3,N], got {tuple(position_ids.shape)} for N={hidden_states.shape[0]}"
            )
        if sum(sample_lens) != hidden_states.shape[0]:
            raise ValueError("sample_lens do not cover hidden_states")

        if action_indexes.numel() == 0:
            self.last_evidence = None
            self.last_routed_c = None
            zero = hidden_states.sum() * 0.0
            return hidden_states, {
                "sparsity_loss": zero,
                "entropy": zero,
                "topk_mass": zero,
                "effective_support": zero,
                "residual_gate": self.residual_gate_logit.sigmoid().squeeze(0),
            }
        if context_indexes.numel() == 0:
            raise ValueError("action relevance requires task or clean-vision context tokens")

        action_sample_ids = self._sample_ids(action_indexes, sample_lens)
        context_sample_ids = self._sample_ids(context_indexes, sample_lens)
        num_samples = len(sample_lens)

        query_all = self.query_proj(self.query_norm(hidden_states[action_indexes]))
        context_hidden = self.context_norm(hidden_states[context_indexes])
        key_all = self.key_proj(context_hidden)
        value_all = self.value_proj(context_hidden)
        query_position_all = position_ids[:, action_indexes].T
        context_position_all = position_ids[:, context_indexes].T

        action_index_groups: list[torch.Tensor] = []
        query_groups: list[torch.Tensor] = []
        key_groups: list[torch.Tensor] = []
        value_groups: list[torch.Tensor] = []
        query_position_groups: list[torch.Tensor] = []
        context_position_groups: list[torch.Tensor] = []
        context_vision_groups: list[torch.Tensor] = []
        query_lengths: list[int] = []
        context_lengths: list[int] = []

        for sample_id in range(num_samples):
            action_mask = action_sample_ids == sample_id
            context_mask = context_sample_ids == sample_id
            query_length = int(action_mask.sum().item())
            context_length = int(context_mask.sum().item())
            if query_length > 0 and context_length == 0:
                raise ValueError(f"sample {sample_id} has action tokens but no routed context")
            action_index_groups.append(action_indexes[action_mask])
            query_groups.append(query_all[action_mask])
            key_groups.append(key_all[context_mask])
            value_groups.append(value_all[context_mask])
            query_position_groups.append(query_position_all[action_mask])
            context_position_groups.append(context_position_all[context_mask])
            context_vision_groups.append(context_is_vision[context_mask])
            query_lengths.append(query_length)
            context_lengths.append(context_length)

        query = pad_sequence(query_groups, batch_first=True)
        key = pad_sequence(key_groups, batch_first=True)
        value = pad_sequence(value_groups, batch_first=True)
        query_position = pad_sequence(query_position_groups, batch_first=True)
        context_position = pad_sequence(context_position_groups, batch_first=True)
        context_vision = pad_sequence(context_vision_groups, batch_first=True)
        max_query = query.shape[1]
        max_context = key.shape[1]

        query_valid = (
            torch.arange(max_query, device=hidden_states.device)[None, :]
            < torch.tensor(query_lengths, device=hidden_states.device)[:, None]
        )
        context_valid = (
            torch.arange(max_context, device=hidden_states.device)[None, :]
            < torch.tensor(context_lengths, device=hidden_states.device)[:, None]
        )

        query = query.view(num_samples, max_query, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(num_samples, max_context, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(num_samples, max_context, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        position_delta = query_position[:, :, None, :].float() - context_position[:, None, :, :].float()
        position_delta = self._signed_log_position_delta(position_delta)
        action_progress = torch.arange(max_query, device=hidden_states.device, dtype=torch.float32)
        action_progress = action_progress / max(max_query - 1, 1)
        relation_features = torch.cat(
            (
                position_delta,
                context_vision[:, None, :, None].expand(-1, max_query, -1, -1).float(),
                action_progress[None, :, None, None].expand(num_samples, -1, max_context, -1),
            ),
            dim=-1,
        )
        relation_logits = self.relation_bias(relation_features.to(query.dtype))
        logits = logits + relation_logits.permute(0, 3, 1, 2).float()
        logits = logits.masked_fill(~context_valid[:, None, None, :], -torch.inf)
        base_weights = torch.softmax(logits, dim=-1)
        weights = self._intervene_on_weights(base_weights, context_valid)

        routed = torch.matmul(weights.to(value.dtype), value)
        routed = routed.transpose(1, 2).reshape(num_samples, max_query, self.routing_dim)
        if weights is base_weights:
            routed_normal = routed
        else:
            routed_normal = torch.matmul(base_weights.to(value.dtype), value)
            routed_normal = routed_normal.transpose(1, 2).reshape(
                num_samples, max_query, self.routing_dim
            )
        routed = self._intervene_on_routed(routed, query_lengths)
        routed = routed * query_valid[:, :, None]
        if self.capture_graph:
            # The protocol measures the local downstream action Jacobian at routed C.
            # Always detach here so this diagnostic cannot retain or accidentally
            # attribute gradients through the visual/context encoder.
            routed = routed.detach().requires_grad_(True)
        self._record_evidence(
            base_weights=base_weights,
            used_weights=weights,
            routed_normal=routed_normal,
            routed=routed,
            query_valid=query_valid,
            context_valid=context_valid,
            context_vision=context_vision,
            query_position=query_position,
            context_position=context_position,
            query_lengths=query_lengths,
            context_lengths=context_lengths,
        )
        routed = self.output_proj(routed)

        ordered_indexes = torch.cat(
            [indexes for indexes in action_index_groups if indexes.numel() > 0],
            dim=0,
        )
        ordered_residuals = torch.cat(
            [routed[sample_id, :query_length] for sample_id, query_length in enumerate(query_lengths) if query_length],
            dim=0,
        )
        residual_gate = self.residual_gate_logit.sigmoid().squeeze(0)
        updated_hidden = hidden_states.index_add(0, ordered_indexes, residual_gate * ordered_residuals)

        per_head_entropy = -(weights.clamp_min(1.0e-12).log() * weights).sum(dim=-1)
        context_count = torch.tensor(context_lengths, device=hidden_states.device, dtype=torch.float32)
        entropy_denominator = context_count.clamp_min(2.0).log()[:, None, None]
        normalized_entropy = per_head_entropy / entropy_denominator
        effective_support = weights.square().sum(dim=-1).clamp_min(1.0e-12).reciprocal()
        effective_support = effective_support / context_count.clamp_min(1.0)[:, None, None]

        k = min(self.topk, max_context)
        topk_mass = weights.topk(k, dim=-1).values.sum(dim=-1)
        valid_queries = query_valid[:, None, :].expand(-1, self.num_heads, -1)
        valid_count = valid_queries.sum().clamp_min(1)

        def masked_mean(value: torch.Tensor) -> torch.Tensor:
            return value.masked_fill(~valid_queries, 0.0).sum() / valid_count

        mean_topk_mass = masked_mean(topk_mass)
        metrics = {
            "sparsity_loss": 1.0 - mean_topk_mass,
            "entropy": masked_mean(normalized_entropy),
            "topk_mass": mean_topk_mass,
            "effective_support": masked_mean(effective_support),
            "residual_gate": residual_gate,
        }
        return updated_hidden, metrics


def configure_action_relevance_modules(
    root: nn.Module,
    *,
    intervention_mode: str = "normal",
    intervention_seed: int = 0,
    capture_evidence: bool = False,
    capture_graph: bool = False,
) -> int:
    """Configure every action-relevance adapter without changing its state dict."""
    count = 0
    for module in root.modules():
        if isinstance(module, ActionRelevanceAdapter):
            module.configure_evidence(
                intervention_mode=intervention_mode,
                intervention_seed=intervention_seed,
                capture_evidence=capture_evidence,
                capture_graph=capture_graph,
            )
            count += 1
    return count


def collect_action_relevance_evidence(root: nn.Module) -> list[dict[str, Any]]:
    """Collect the last captured routed-C record from each adapter layer."""
    records = []
    for name, module in root.named_modules():
        if isinstance(module, ActionRelevanceAdapter) and module.last_evidence is not None:
            records.append({"module": name, **module.last_evidence})
    return records


def set_action_relevance_fixed_routed_c(
    root: nn.Module,
    routed_by_module: dict[str, torch.Tensor] | None,
) -> int:
    """Install exact per-layer routed-C donors without changing model state."""
    adapters = {
        name: module
        for name, module in root.named_modules()
        if isinstance(module, ActionRelevanceAdapter)
    }
    if routed_by_module is None:
        for module in adapters.values():
            module.set_fixed_routed_c(None)
        return len(adapters)

    missing = sorted(set(adapters) - set(routed_by_module))
    unexpected = sorted(set(routed_by_module) - set(adapters))
    if missing or unexpected:
        raise ValueError(
            "fixed routed-C layer mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, module in adapters.items():
        module.set_fixed_routed_c(routed_by_module[name])
    return len(adapters)


def action_relevance_graph_capture_enabled(root: nn.Module) -> bool:
    """Return whether any routed-C adapter is configured for graph capture."""
    return any(
        isinstance(module, ActionRelevanceAdapter) and module.capture_graph
        for module in root.modules()
    )


def collect_action_relevance_graph_tensors(root: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Collect routed-C graph tensors in deterministic module order."""
    tensors: list[tuple[str, torch.Tensor]] = []
    for name, module in root.named_modules():
        if isinstance(module, ActionRelevanceAdapter) and module.last_routed_c is not None:
            tensors.append((name, module.last_routed_c))
    return tensors


def clear_action_relevance_graph_tensors(root: nn.Module) -> None:
    """Release routed-C graph references after a local Jacobian estimate."""
    for module in root.modules():
        if isinstance(module, ActionRelevanceAdapter):
            module.last_routed_c = None


def _flatten_action_predictions(prediction: Any) -> list[torch.Tensor]:
    if isinstance(prediction, torch.Tensor):
        return [prediction]
    if isinstance(prediction, (list, tuple)):
        return [tensor for item in prediction for tensor in _flatten_action_predictions(item)]
    if isinstance(prediction, dict):
        return [
            tensor
            for key in sorted(prediction)
            for tensor in _flatten_action_predictions(prediction[key])
        ]
    raise TypeError(f"Unsupported action prediction container: {type(prediction).__name__}")


def estimate_action_relevance_jacobian_energy(
    prediction: Any,
    routed_tensors: list[tuple[str, torch.Tensor]],
    *,
    num_projections: int = 8,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Estimate per-C-coordinate action Jacobian energy with Rademacher VJPs.

    For routed C shaped ``[B,Q,D]``, query positions are averaged and the
    returned energy is ``[B,D]``. Multiple adapter layers remain separate so
    downstream analysis can concatenate them without losing layer identity.
    """
    if num_projections <= 0:
        raise ValueError("num_projections must be positive")
    predictions = [tensor for tensor in _flatten_action_predictions(prediction) if tensor.numel()]
    if not predictions:
        raise ValueError("action prediction is empty")
    if not routed_tensors:
        raise ValueError("no routed-C graph tensors were captured")
    if not torch.is_grad_enabled():
        raise RuntimeError("Jacobian estimation requires torch.enable_grad()")

    names = [name for name, _ in routed_tensors]
    routed = [tensor for _, tensor in routed_tensors]
    batch_size = routed[0].shape[0]
    for name, tensor in routed_tensors:
        if tensor.ndim < 2 or tensor.shape[0] != batch_size:
            raise ValueError(f"routed C {name!r} must be batch-first with a shared batch size")
        if not tensor.requires_grad:
            raise ValueError(f"routed C {name!r} does not require gradients")

    output_count = sum(tensor.numel() for tensor in predictions)
    normalizer = math.sqrt(max(output_count, 1))
    accumulated = [torch.zeros_like(tensor, dtype=torch.float32) for tensor in routed]
    for projection in range(num_projections):
        objective = predictions[0].new_zeros(())
        for prediction_index, tensor in enumerate(predictions):
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(seed + projection * 104729 + prediction_index)
            direction = torch.randint(
                0,
                2,
                tensor.shape,
                device=tensor.device,
                generator=generator,
                dtype=torch.int64,
            ).mul_(2).sub_(1).to(dtype=tensor.dtype)
            objective = objective + (tensor * direction).sum() / normalizer
        gradients = torch.autograd.grad(
            objective,
            routed,
            retain_graph=projection + 1 < num_projections,
            allow_unused=True,
        )
        for index, gradient in enumerate(gradients):
            if gradient is not None:
                accumulated[index].add_(gradient.detach().float().square())

    result: dict[str, torch.Tensor] = {}
    for name, energy in zip(names, accumulated):
        energy = energy / num_projections
        if energy.ndim > 2:
            energy = energy.mean(dim=tuple(range(1, energy.ndim - 1)))
        result[name] = energy
    return result
