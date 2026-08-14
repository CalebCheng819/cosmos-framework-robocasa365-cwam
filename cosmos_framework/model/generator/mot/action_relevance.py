# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action-conditioned routing over clean policy context tokens."""

from __future__ import annotations

import math
from itertools import accumulate

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from cosmos_framework.data.generator.sequence_packing import ModalityData

_ROUTING_MODES = frozenset({"soft_attention", "dselect_k", "adaptive_dselect_k"})


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


class DSelectKSupportRouter(nn.Module):
    """Example-conditioned DSelect-k support over a fixed context capacity.

    Each selector predicts a binary code for one context position. SmoothStep
    keeps the code differentiable during training and becomes exactly binary
    outside its transition interval, so the resulting support has at most
    ``num_selectors`` non-zero positions after convergence.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        num_selectors: int,
        capacity: int,
        gamma: float,
        adaptive_counts: tuple[int, ...] | None = None,
        adaptive_temperature: float = 1.0,
        adaptive_target_selectors: float = 24.0,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or num_selectors <= 0:
            raise ValueError("head_dim and num_selectors must be positive")
        if capacity <= 1 or capacity & (capacity - 1):
            raise ValueError("DSelect-k capacity must be a power of two greater than one")
        if num_selectors > capacity:
            raise ValueError("DSelect-k num_selectors cannot exceed capacity")
        if gamma <= 0.0:
            raise ValueError("DSelect-k gamma must be positive")
        if adaptive_temperature <= 0.0:
            raise ValueError("adaptive DSelect-k temperature must be positive")
        if adaptive_counts is not None:
            if len(adaptive_counts) < 2:
                raise ValueError("adaptive DSelect-k requires at least two candidate counts")
            if tuple(sorted(set(adaptive_counts))) != adaptive_counts:
                raise ValueError("adaptive DSelect-k counts must be strictly increasing")
            if adaptive_counts[0] <= 0 or adaptive_counts[-1] > num_selectors:
                raise ValueError("adaptive DSelect-k counts must lie within the selector capacity")
            if not adaptive_counts[0] <= adaptive_target_selectors <= adaptive_counts[-1]:
                raise ValueError("adaptive DSelect-k target must lie within the candidate range")

        self.head_dim = head_dim
        self.num_selectors = num_selectors
        self.capacity = capacity
        self.gamma = gamma
        self.num_binary = int(math.log2(capacity))
        self.adaptive_counts = adaptive_counts
        self.adaptive_temperature = adaptive_temperature
        self.adaptive_target_selectors = adaptive_target_selectors
        routing_input_dim = 2 * head_dim
        self.selector_logits = nn.Linear(routing_input_dim, num_selectors * self.num_binary)
        self.selector_mix_logits = nn.Linear(routing_input_dim, num_selectors)
        self.selector_count_logits = (
            nn.Linear(routing_input_dim, len(adaptive_counts)) if adaptive_counts is not None else None
        )

        values = torch.arange(capacity, dtype=torch.long)[:, None]
        shifts = torch.arange(self.num_binary - 1, -1, -1, dtype=torch.long)[None, :]
        self.register_buffer("binary_codes", ((values >> shifts) & 1).bool(), persistent=False)
        if adaptive_counts is not None:
            count_values = torch.tensor(adaptive_counts, dtype=torch.float32)
            selector_ids = torch.arange(num_selectors, dtype=torch.long)[None, :]
            prefix_masks = selector_ids < count_values.long()[:, None]
            self.register_buffer("adaptive_count_values", count_values, persistent=False)
            self.register_buffer("adaptive_prefix_masks", prefix_masks.float(), persistent=False)

    def init_weights(self) -> None:
        nn.init.zeros_(self.selector_logits.weight)
        nn.init.uniform_(self.selector_logits.bias, -self.gamma / 100.0, self.gamma / 100.0)
        nn.init.zeros_(self.selector_mix_logits.weight)
        nn.init.zeros_(self.selector_mix_logits.bias)
        if self.selector_count_logits is not None:
            nn.init.zeros_(self.selector_count_logits.weight)
            nn.init.zeros_(self.selector_count_logits.bias)
            # Start as the fixed-32 B3 baseline; the action and budget losses
            # decide when a sample can use fewer selectors.
            with torch.no_grad():
                self.selector_count_logits.bias[-1] = 2.0

    def _smooth_step(self, logits: torch.Tensor) -> torch.Tensor:
        lower = -self.gamma / 2.0
        upper = self.gamma / 2.0
        polynomial = (
            -2.0 * logits.pow(3) / self.gamma**3
            + 3.0 * logits / (2.0 * self.gamma)
            + 0.5
        )
        return torch.where(
            logits <= lower,
            torch.zeros_like(logits),
            torch.where(logits >= upper, torch.ones_like(logits), polynomial),
        )

    @staticmethod
    def _masked_query_mean(value: torch.Tensor, query_valid: torch.Tensor) -> torch.Tensor:
        valid = query_valid[:, None, :].expand(-1, value.shape[1], -1)
        return value.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1)

    @staticmethod
    def _masked_sample_mean(value: torch.Tensor, query_valid: torch.Tensor) -> torch.Tensor:
        valid = query_valid.any(dim=-1)
        return value.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1)

    def _selector_gates(
        self,
        routing_input: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.selector_count_logits is None:
            gates = routing_input.new_ones((routing_input.shape[0], self.num_selectors), dtype=torch.float32)
            fixed_count = gates.new_full((routing_input.shape[0],), float(self.num_selectors))
            zero = routing_input.sum(dim=(1, 2, 3), dtype=torch.float32) * 0.0
            return gates, {
                "selector_expected_count": self._masked_sample_mean(fixed_count, query_valid),
                "selector_selected_count": self._masked_sample_mean(fixed_count, query_valid),
                "selector_count_entropy": self._masked_sample_mean(zero, query_valid),
                "selector_budget_loss": self._masked_sample_mean(zero, query_valid),
                "selector_utilization": self._masked_sample_mean(fixed_count, query_valid) / self.num_selectors,
            }

        query_mask = query_valid[:, None, :, None].to(routing_input.dtype)
        denominator = (query_mask.sum(dim=(1, 2)) * routing_input.shape[1]).clamp_min(1.0)
        sample_input = (routing_input * query_mask).sum(dim=(1, 2)) / denominator
        count_logits = self.selector_count_logits(sample_input).float() / self.adaptive_temperature
        count_probabilities = torch.softmax(count_logits, dim=-1)
        hard_indexes = count_probabilities.argmax(dim=-1)
        hard_probabilities = torch.nn.functional.one_hot(
            hard_indexes,
            num_classes=count_probabilities.shape[-1],
        ).to(count_probabilities.dtype)
        if self.training:
            count_selection = hard_probabilities + count_probabilities - count_probabilities.detach()
        else:
            count_selection = hard_probabilities
        gates = torch.matmul(count_selection, self.adaptive_prefix_masks)

        expected_count = torch.matmul(count_probabilities, self.adaptive_count_values)
        selected_count = self.adaptive_count_values[hard_indexes]
        selected_count_st = torch.matmul(count_selection, self.adaptive_count_values)
        count_entropy = -(count_probabilities.clamp_min(1.0e-12).log() * count_probabilities).sum(dim=-1)
        count_entropy = count_entropy / math.log(count_probabilities.shape[-1])
        valid_samples = query_valid.any(dim=-1)
        valid_expected_count = self._masked_sample_mean(expected_count, query_valid)
        valid_selected_count_st = self._masked_sample_mean(selected_count_st, query_valid)
        budget_loss = ((valid_selected_count_st - self.adaptive_target_selectors) / self.num_selectors).square()
        budget_loss = budget_loss + expected_count.masked_fill(valid_samples, 0.0).sum() * 0.0
        return gates, {
            "selector_expected_count": valid_expected_count,
            "selector_selected_count": self._masked_sample_mean(selected_count, query_valid),
            "selector_count_entropy": self._masked_sample_mean(count_entropy, query_valid),
            "selector_budget_loss": budget_loss,
            "selector_utilization": valid_expected_count / self.num_selectors,
        }

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        context_valid: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("DSelect-k query and key must be [B,H,N,D]")
        if query.shape[:2] != key.shape[:2] or query.shape[-1] != self.head_dim:
            raise ValueError("DSelect-k query/key head dimensions do not match")
        if key.shape[-1] != self.head_dim:
            raise ValueError("DSelect-k key head dimension does not match its configuration")
        if context_valid.shape != (key.shape[0], key.shape[2]):
            raise ValueError("DSelect-k context_valid is not aligned with key")
        if query_valid.shape != (query.shape[0], query.shape[2]):
            raise ValueError("DSelect-k query_valid is not aligned with query")
        if key.shape[2] > self.capacity:
            raise ValueError(
                f"DSelect-k context length {key.shape[2]} exceeds configured capacity {self.capacity}"
            )

        context_mask = context_valid[:, None, :, None].to(key.dtype)
        context_count = context_mask.sum(dim=2).clamp_min(1.0)
        context_summary = (key * context_mask).sum(dim=2) / context_count
        context_summary = context_summary[:, :, None, :].expand(-1, -1, query.shape[2], -1)
        routing_input = torch.cat((query, context_summary), dim=-1)

        selector_logits = self.selector_logits(routing_input).float()
        selector_logits = selector_logits.view(*query.shape[:3], self.num_selectors, self.num_binary)
        selector_probabilities = self._smooth_step(selector_logits)
        selector_gates, count_metrics = self._selector_gates(routing_input, query_valid)
        selector_mix_logits = self.selector_mix_logits(routing_input).float()
        selector_mix_unnormalized = torch.exp(selector_mix_logits - selector_mix_logits.amax(dim=-1, keepdim=True))
        selector_mix_unnormalized = selector_mix_unnormalized * selector_gates[:, None, None, :]
        selector_mix = selector_mix_unnormalized / selector_mix_unnormalized.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-12)

        codes = self.binary_codes[: key.shape[2]]
        valid_context = context_valid[:, None, None, :]
        support = query.new_zeros((*query.shape[:3], key.shape[2]), dtype=torch.float32)
        invalid_mass = query.new_zeros(query.shape[:3], dtype=torch.float32)
        for selector_id in range(self.num_selectors):
            selector_output = query.new_ones((*query.shape[:3], key.shape[2]), dtype=torch.float32)
            selector_bits = selector_probabilities[..., selector_id, :]
            for bit_id in range(self.num_binary):
                bit_probability = selector_bits[..., bit_id, None]
                selector_output = selector_output * torch.where(
                    codes[None, None, None, :, bit_id],
                    bit_probability,
                    1.0 - bit_probability,
                )
            valid_output = selector_output.masked_fill(~valid_context, 0.0)
            valid_mass = valid_output.sum(dim=-1)
            mix = selector_mix[..., selector_id]
            support = support + mix[..., None] * valid_output
            invalid_mass = invalid_mass + mix * (1.0 - valid_mass).clamp(0.0, 1.0)

        support_mass = support.sum(dim=-1, keepdim=True)
        uniform_support = valid_context.float() / valid_context.sum(dim=-1, keepdim=True).clamp_min(1)
        support = torch.where(
            support_mass > 1.0e-12,
            support / support_mass.clamp_min(1.0e-12),
            uniform_support,
        )

        probability = selector_probabilities.clamp(0.0, 1.0)
        bit_entropy = -(
            probability * probability.clamp_min(1.0e-12).log()
            + (1.0 - probability) * (1.0 - probability).clamp_min(1.0e-12).log()
        )
        selector_entropy_per_selector = bit_entropy.sum(dim=-1) / (self.num_binary * math.log(2.0))
        entropy_gates = selector_gates[:, None, None, :]
        selector_entropy = (selector_entropy_per_selector * entropy_gates).sum(dim=-1)
        selector_entropy = selector_entropy / entropy_gates.sum(dim=-1).clamp_min(1.0)

        if self.num_selectors > 1:
            probability_i = probability.unsqueeze(-2)
            probability_j = probability.unsqueeze(-3)
            overlap_per_bit = (
                (probability_i * probability_j).clamp_min(1.0e-12).sqrt()
                + ((1.0 - probability_i) * (1.0 - probability_j)).clamp_min(1.0e-12).sqrt()
            )
            pairwise_overlap = overlap_per_bit.prod(dim=-1)
            off_diagonal = ~torch.eye(
                self.num_selectors,
                device=query.device,
                dtype=torch.bool,
            )
            overlap_gates = selector_gates[:, None, None, :, None] * selector_gates[:, None, None, None, :]
            overlap_gates = overlap_gates * off_diagonal[None, None, None, :, :]
            selector_overlap = (pairwise_overlap * overlap_gates).sum(dim=(-1, -2))
            selector_overlap = selector_overlap / overlap_gates.sum(dim=(-1, -2)).clamp_min(1.0)
        else:
            selector_overlap = selector_entropy * 0.0

        selector_hard = ((probability <= 1.0e-6) | (probability >= 1.0 - 1.0e-6)).float()
        selector_hard_fraction = (selector_hard * entropy_gates[..., None]).sum(dim=(-1, -2))
        selector_hard_fraction = selector_hard_fraction / (entropy_gates.sum(dim=-1).clamp_min(1.0) * self.num_binary)
        metrics = {
            "selector_entropy": self._masked_query_mean(selector_entropy, query_valid),
            "selector_overlap": self._masked_query_mean(selector_overlap, query_valid),
            "selector_invalid_mass": self._masked_query_mean(invalid_mass, query_valid),
            "selector_hard_fraction": self._masked_query_mean(selector_hard_fraction, query_valid),
            **count_metrics,
        }
        return support, metrics


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
        routing_mode: str = "soft_attention",
        dselect_num_selectors: int = 32,
        dselect_capacity: int = 8192,
        dselect_gamma: float = 1.0,
        dselect_overlap_weight: float = 0.1,
        dselect_invalid_weight: float = 1.0,
        dselect_adaptive_counts: tuple[int, ...] = (4, 8, 12, 16, 24, 32),
        dselect_adaptive_temperature: float = 1.0,
        dselect_adaptive_target_selectors: float = 24.0,
        dselect_adaptive_budget_weight: float = 1.0,
        dselect_adaptive_entropy_weight: float = 0.01,
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
        if routing_mode not in _ROUTING_MODES:
            raise ValueError(f"unsupported action-relevance routing mode: {routing_mode!r}")
        if (
            dselect_overlap_weight < 0.0
            or dselect_invalid_weight < 0.0
            or dselect_adaptive_budget_weight < 0.0
            or dselect_adaptive_entropy_weight < 0.0
        ):
            raise ValueError("DSelect-k regularization weights must be non-negative")

        self.hidden_size = hidden_size
        self.routing_dim = routing_dim
        self.num_heads = num_heads
        self.head_dim = routing_dim // num_heads
        self.topk = topk
        self.initial_residual_gate = initial_residual_gate
        self.routing_mode = routing_mode
        self.dselect_overlap_weight = dselect_overlap_weight
        self.dselect_invalid_weight = dselect_invalid_weight
        self.dselect_adaptive_budget_weight = dselect_adaptive_budget_weight
        self.dselect_adaptive_entropy_weight = dselect_adaptive_entropy_weight

        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.auxiliary_norm = nn.LayerNorm(hidden_size)
        self.query_proj = nn.Linear(hidden_size, routing_dim, bias=False)
        self.key_proj = nn.Linear(hidden_size, routing_dim, bias=False)
        self.value_proj = nn.Linear(hidden_size, routing_dim, bias=False)
        self.output_proj = nn.Linear(routing_dim, hidden_size, bias=False)
        self.relation_bias = nn.Sequential(
            nn.Linear(5, 64),
            nn.SiLU(),
            nn.Linear(64, num_heads),
        )
        self.dselect_router = (
            DSelectKSupportRouter(
                head_dim=self.head_dim,
                num_selectors=dselect_num_selectors,
                capacity=dselect_capacity,
                gamma=dselect_gamma,
                adaptive_counts=(tuple(dselect_adaptive_counts) if routing_mode == "adaptive_dselect_k" else None),
                adaptive_temperature=dselect_adaptive_temperature,
                adaptive_target_selectors=dselect_adaptive_target_selectors,
            )
            if routing_mode in {"dselect_k", "adaptive_dselect_k"}
            else None
        )
        initial_logit = math.log(initial_residual_gate / (1.0 - initial_residual_gate))
        # FSDP2 rejects zero-dimensional parameters.
        self.residual_gate_logit = nn.Parameter(torch.tensor([initial_logit]))

    def init_weights(self) -> None:
        nn.init.ones_(self.query_norm.weight)
        nn.init.zeros_(self.query_norm.bias)
        nn.init.ones_(self.context_norm.weight)
        nn.init.zeros_(self.context_norm.bias)
        nn.init.ones_(self.auxiliary_norm.weight)
        nn.init.zeros_(self.auxiliary_norm.bias)

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
        if self.dselect_router is not None:
            self.dselect_router.init_weights()
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        action_indexes: torch.Tensor,
        context_indexes: torch.Tensor,
        context_is_vision: torch.Tensor,
        position_ids: torch.Tensor,
        sample_lens: list[int],
        action_seed_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
        if action_seed_hidden is None:
            action_seed_hidden = hidden_states[action_indexes]
        if action_seed_hidden.shape != (action_indexes.numel(), self.hidden_size):
            raise ValueError(
                "action_seed_hidden must be aligned with action_indexes: "
                f"expected {(action_indexes.numel(), self.hidden_size)}, got {tuple(action_seed_hidden.shape)}"
            )

        if action_indexes.numel() == 0:
            zero = hidden_states.sum() * 0.0
            return (
                hidden_states,
                action_seed_hidden,
                {
                    "sparsity_loss": zero,
                    "entropy": zero,
                    "topk_mass": zero,
                    "effective_support": zero,
                    "residual_gate": self.residual_gate_logit.sigmoid().squeeze(0),
                    "selector_entropy": zero,
                    "selector_overlap": zero,
                    "selector_invalid_mass": zero,
                    "selector_hard_fraction": zero,
                    "selector_expected_count": zero,
                    "selector_selected_count": zero,
                    "selector_count_entropy": zero,
                    "selector_budget_loss": zero,
                    "selector_utilization": zero,
                },
            )
        if context_indexes.numel() == 0:
            raise ValueError("action relevance requires task or clean-vision context tokens")

        action_sample_ids = self._sample_ids(action_indexes, sample_lens)
        context_sample_ids = self._sample_ids(context_indexes, sample_lens)
        num_samples = len(sample_lens)

        # The auxiliary route queries context from the pre-transformer noisy
        # action seed. This prevents full-context action states from leaking into
        # the relevance-only prediction through the query representation.
        query_all = self.query_proj(self.query_norm(action_seed_hidden))
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
        weights = torch.softmax(logits, dim=-1)
        dselect_metrics = {
            "selector_entropy": weights.sum() * 0.0,
            "selector_overlap": weights.sum() * 0.0,
            "selector_invalid_mass": weights.sum() * 0.0,
            "selector_hard_fraction": weights.sum() * 0.0,
            "selector_expected_count": weights.sum() * 0.0,
            "selector_selected_count": weights.sum() * 0.0,
            "selector_count_entropy": weights.sum() * 0.0,
            "selector_budget_loss": weights.sum() * 0.0,
            "selector_utilization": weights.sum() * 0.0,
        }
        if self.dselect_router is not None:
            support, dselect_metrics = self.dselect_router(
                query,
                key,
                context_valid=context_valid,
                query_valid=query_valid,
            )
            supported_weights = weights * support
            supported_mass = supported_weights.sum(dim=-1, keepdim=True)
            weights = torch.where(
                supported_mass > 1.0e-12,
                supported_weights / supported_mass.clamp_min(1.0e-12),
                weights,
            )

        routed = torch.matmul(weights.to(value.dtype), value)
        routed = routed.transpose(1, 2).reshape(num_samples, max_query, self.routing_dim)
        routed = self.output_proj(routed)
        routed = routed * query_valid[:, :, None]

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
        auxiliary_residuals = torch.zeros_like(action_seed_hidden)
        auxiliary_cursor = 0
        for action_mask, query_length in zip(
            (action_sample_ids == sample_id for sample_id in range(num_samples)),
            query_lengths,
        ):
            if query_length == 0:
                continue
            local_indexes = action_mask.nonzero(as_tuple=False).squeeze(1)
            auxiliary_residuals[local_indexes] = ordered_residuals[
                auxiliary_cursor : auxiliary_cursor + query_length
            ]
            auxiliary_cursor += query_length
        auxiliary_action_hidden = self.auxiliary_norm(action_seed_hidden + auxiliary_residuals)
        # Keep the B2-only normalization parameters in the backward graph when
        # this adapter is reused by the zero-auxiliary S1 recipe under FSDP.
        updated_hidden = updated_hidden + auxiliary_action_hidden.sum() * 0.0

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
        if self.dselect_router is None:
            sparsity_loss = 1.0 - mean_topk_mass
        else:
            sparsity_loss = (
                dselect_metrics["selector_entropy"]
                + self.dselect_overlap_weight * dselect_metrics["selector_overlap"]
                + self.dselect_invalid_weight * dselect_metrics["selector_invalid_mass"]
                + self.dselect_adaptive_budget_weight * dselect_metrics["selector_budget_loss"]
                + self.dselect_adaptive_entropy_weight * dselect_metrics["selector_count_entropy"]
            )
        metrics = {
            "sparsity_loss": sparsity_loss,
            "entropy": masked_mean(normalized_entropy),
            "topk_mass": mean_topk_mass,
            "effective_support": masked_mean(effective_support),
            "residual_gate": residual_gate,
            **dselect_metrics,
        }
        return updated_hidden, auxiliary_action_hidden, metrics
