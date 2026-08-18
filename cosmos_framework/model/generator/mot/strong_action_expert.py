# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DeMiAn-inspired action expert with explicit causal control memory."""

from __future__ import annotations

import math
from itertools import accumulate

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence


_TEXT_CONTEXT = 0
_VISION_CONTEXT = 1
_PROPRIO_CONTEXT = 2
_ACTION_HISTORY_CONTEXT = 3
_MEMORY_CONTEXT = 4
_NUM_CONTEXT_TYPES = 5


def _strip_optional_collator_axis(
    values: list[torch.Tensor],
    *,
    expected_ndim: int,
) -> list[torch.Tensor]:
    """Remove the per-sample singleton axis emitted by the packed collator."""
    return [
        value[0] if value.ndim == expected_ndim + 1 and value.shape[0] == 1 else value
        for value in values
    ]


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


class TemporalControlMemory(nn.Module):
    """Encode sparse proprio events and dense executed actions into memory slots."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        num_memory_tokens: int,
        proprio_stride: int,
    ) -> None:
        super().__init__()
        if min(state_dim, action_dim, model_dim, num_heads, num_layers, num_memory_tokens, proprio_stride) < 1:
            raise ValueError("Temporal control memory dimensions must be positive")
        if model_dim % num_heads:
            raise ValueError("Temporal control memory model_dim must be divisible by num_heads")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.num_memory_tokens = num_memory_tokens
        self.proprio_stride = proprio_stride

        self.state_norm = nn.LayerNorm(state_dim)
        self.action_norm = nn.LayerNorm(action_dim)
        self.state_proj = nn.Linear(state_dim, model_dim)
        self.action_proj = nn.Linear(action_dim, model_dim)
        self.history_type_embed = nn.Embedding(2, model_dim)
        self.age_mlp = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.memory_queries = nn.Parameter(torch.empty(num_memory_tokens, model_dim))
        self.memory_pool = nn.MultiheadAttention(model_dim, num_heads, dropout=0.0, batch_first=True)
        self.memory_norm = nn.LayerNorm(model_dim)

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = 1.0 / math.sqrt(max(module.in_features, 1))
                nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
        nn.init.normal_(self.memory_queries, mean=0.0, std=0.02)

    def forward(
        self,
        proprio_history: torch.Tensor,
        proprio_mask: torch.Tensor,
        action_history: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if proprio_history.ndim != 3 or proprio_history.shape[-1] != self.state_dim:
            raise ValueError(
                f"proprio_history must be [B,P,{self.state_dim}], got {tuple(proprio_history.shape)}"
            )
        if action_history.ndim != 3 or action_history.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_history must be [B,H,{self.action_dim}], got {tuple(action_history.shape)}"
            )
        if proprio_mask.shape != proprio_history.shape[:2] or action_mask.shape != action_history.shape[:2]:
            raise ValueError("Control history masks must align with their history tensors")
        if proprio_history.shape[0] != action_history.shape[0]:
            raise ValueError("Proprio and action histories must use the same batch size")
        proprio_mask = proprio_mask.bool()
        action_mask = action_mask.bool()
        torch._assert(
            torch.all(proprio_mask[:, -1]),
            "Every sample must contain a valid current proprioception",
        )

        batch_size, num_states = proprio_history.shape[:2]
        num_actions = action_history.shape[1]
        dtype = self.state_proj.weight.dtype
        device = proprio_history.device
        state_tokens = self.state_proj(self.state_norm(proprio_history.to(dtype)))
        action_tokens = self.action_proj(self.action_norm(action_history.to(dtype)))
        state_tokens = state_tokens + self.history_type_embed.weight[0]
        action_tokens = action_tokens + self.history_type_embed.weight[1]

        max_age = float(max((num_states - 1) * self.proprio_stride, num_actions, 1))
        state_age = torch.arange(
            -(num_states - 1) * self.proprio_stride,
            1,
            self.proprio_stride,
            device=device,
            dtype=torch.float32,
        )
        action_age = torch.arange(-num_actions, 0, device=device, dtype=torch.float32)
        state_age = state_age / max_age
        action_age = action_age / max_age
        state_tokens = state_tokens + self.age_mlp(state_age[:, None].to(dtype))[None]
        action_tokens = action_tokens + self.age_mlp(action_age[:, None].to(dtype))[None]

        history = torch.cat((state_tokens, action_tokens), dim=1)
        history_valid = torch.cat((proprio_mask, action_mask), dim=1)
        history = history.masked_fill(~history_valid[:, :, None], 0.0)
        encoded = self.temporal_encoder(history, src_key_padding_mask=~history_valid)
        queries = self.memory_queries.to(dtype).unsqueeze(0).expand(batch_size, -1, -1)
        memory, _ = self.memory_pool(
            queries,
            encoded,
            encoded,
            key_padding_mask=~history_valid,
            need_weights=False,
        )
        memory = self.memory_norm(memory)
        return state_tokens[:, -1], action_tokens, memory, action_age


class RelationAwareCrossAttention(nn.Module):
    """Cross-attention with per-head context-type, camera, age, and progress bias."""

    def __init__(self, *, model_dim: int, num_heads: int, num_views: int) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_views < 1:
            raise ValueError("num_views must be positive")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.num_views = num_views
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.context_type_bias = nn.Embedding(_NUM_CONTEXT_TYPES, num_heads)
        self.view_bias = nn.Embedding(num_views + 1, num_heads)
        self.age_scale = nn.Parameter(torch.zeros(num_heads))
        self.progress_scale = nn.Parameter(torch.zeros(num_heads))

    def init_weights(self) -> None:
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            std = 1.0 / math.sqrt(projection.in_features)
            nn.init.trunc_normal_(projection.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.context_type_bias.weight)
        nn.init.zeros_(self.view_bias.weight)
        nn.init.zeros_(self.age_scale)
        nn.init.zeros_(self.progress_scale)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        query_valid: torch.Tensor,
        context_valid: torch.Tensor,
        context_type: torch.Tensor,
        context_view: torch.Tensor,
        context_age: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = query.shape[:2]
        context_length = context.shape[1]
        q = self.q_proj(query).view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)

        relation = self.context_type_bias(context_type).permute(0, 2, 1)
        relation = relation + self.view_bias(context_view).permute(0, 2, 1)
        relation = relation + self.age_scale[None, :, None] * context_age[:, None, :].float()
        progress = torch.arange(query_length, device=query.device, dtype=torch.float32)
        progress = progress / max(query_length - 1, 1)
        relation = relation[:, :, None, :] + self.progress_scale[None, :, None, None] * progress[None, None, :, None]
        logits = logits + relation
        logits = logits.masked_fill(~context_valid[:, None, None, :], -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        attended = torch.matmul(weights.to(v.dtype), v)
        attended = attended.transpose(1, 2).reshape(batch_size, query_length, self.model_dim)
        attended = self.out_proj(attended).masked_fill(~query_valid[:, :, None], 0.0)

        entropy = -(weights.clamp_min(1.0e-12).log() * weights).sum(dim=-1)
        denominator = context_valid.sum(dim=-1).float().clamp_min(2).log()[:, None, None]
        normalized_entropy = entropy / denominator
        valid_queries = query_valid[:, None, :].expand(-1, self.num_heads, -1)
        entropy_mean = normalized_entropy.masked_fill(~valid_queries, 0.0).sum() / valid_queries.sum().clamp_min(1)
        valid_query_weights = valid_queries[:, :, :, None].to(weights.dtype)
        context_importance = (weights * valid_query_weights).sum(dim=(1, 2))
        context_importance = context_importance / valid_query_weights.sum(dim=(1, 2)).clamp_min(1)
        context_importance = context_importance.masked_fill(~context_valid, 0.0)
        return attended, entropy_mean, context_importance


class ActionExpertBlock(nn.Module):
    def __init__(self, *, model_dim: int, num_heads: int, num_views: int, ffn_multiplier: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(model_dim)
        self.self_attention = nn.MultiheadAttention(model_dim, num_heads, dropout=0.0, batch_first=True)
        self.cross_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.cross_attention = RelationAwareCrossAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_views=num_views,
        )
        self.ffn_norm = nn.LayerNorm(model_dim)
        ffn_dim = ffn_multiplier * model_dim
        self.ffn_in = nn.Linear(model_dim, 2 * ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, model_dim)

    def init_weights(self) -> None:
        for module in (self.self_norm, self.cross_norm, self.context_norm, self.ffn_norm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.self_attention.in_proj_weight)
        if self.self_attention.in_proj_bias is not None:
            nn.init.zeros_(self.self_attention.in_proj_bias)
        nn.init.xavier_uniform_(self.self_attention.out_proj.weight)
        if self.self_attention.out_proj.bias is not None:
            nn.init.zeros_(self.self_attention.out_proj.bias)
        self.cross_attention.init_weights()
        for module in (self.ffn_in, self.ffn_out):
            std = 1.0 / math.sqrt(module.in_features)
            nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        action: torch.Tensor,
        context: torch.Tensor,
        *,
        action_valid: torch.Tensor,
        context_valid: torch.Tensor,
        context_type: torch.Tensor,
        context_view: torch.Tensor,
        context_age: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.self_norm(action)
        residual, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~action_valid,
            need_weights=False,
        )
        action = action + residual.masked_fill(~action_valid[:, :, None], 0.0)
        residual, entropy, context_importance = self.cross_attention(
            self.cross_norm(action),
            self.context_norm(context),
            query_valid=action_valid,
            context_valid=context_valid,
            context_type=context_type,
            context_view=context_view,
            context_age=context_age,
        )
        action = action + residual
        left, right = self.ffn_in(self.ffn_norm(action)).chunk(2, dim=-1)
        residual = self.ffn_out(F.silu(left) * right)
        action = action + residual.masked_fill(~action_valid[:, :, None], 0.0)
        return action, entropy, context_importance


class StrongActionContextExpert(nn.Module):
    """Four-layer action query tower over multimodal context and learned memory."""

    def __init__(
        self,
        *,
        hidden_size: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_multiplier: int,
        state_dim: int,
        action_dim: int,
        proprio_history_length: int,
        proprio_history_stride: int,
        action_history_length: int,
        memory_layers: int,
        memory_tokens: int,
        num_views: int,
        view_token_width: int,
        initial_residual_gate: float,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_residual_gate < 1.0:
            raise ValueError("initial_residual_gate must be strictly between zero and one")
        if min(
            num_layers,
            ffn_multiplier,
            proprio_history_length,
            action_history_length,
            num_views,
            view_token_width,
        ) < 1:
            raise ValueError("Strong action expert layer, history, and view counts must be positive")
        if model_dim % num_heads:
            raise ValueError("Strong action expert model_dim must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.proprio_history_length = proprio_history_length
        self.action_history_length = action_history_length
        self.memory_tokens = memory_tokens
        self.num_views = num_views
        self.view_token_width = view_token_width
        self.initial_residual_gate = initial_residual_gate

        self.action_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.action_in = nn.Linear(hidden_size, model_dim, bias=False)
        self.context_in = nn.Linear(hidden_size, model_dim, bias=False)
        self.action_progress_mlp = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.context_type_embed = nn.Embedding(_NUM_CONTEXT_TYPES, model_dim)
        self.view_embed = nn.Embedding(num_views + 1, model_dim)
        self.control_memory = TemporalControlMemory(
            state_dim=state_dim,
            action_dim=action_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=memory_layers,
            num_memory_tokens=memory_tokens,
            proprio_stride=proprio_history_stride,
        )
        self.layers = nn.ModuleList(
            ActionExpertBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                num_views=num_views,
                ffn_multiplier=ffn_multiplier,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.action_out = nn.Linear(model_dim, hidden_size, bias=False)
        initial_logit = math.log(initial_residual_gate / (1.0 - initial_residual_gate))
        self.residual_gate_logit = nn.Parameter(torch.tensor([initial_logit]))

    def init_weights(self) -> None:
        for norm in (self.action_norm, self.context_norm, self.output_norm):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for projection in (self.action_in, self.context_in, self.action_progress_mlp[0], self.action_progress_mlp[2]):
            assert isinstance(projection, nn.Linear)
            std = 1.0 / math.sqrt(projection.in_features)
            nn.init.trunc_normal_(projection.weight, std=std, a=-3 * std, b=3 * std)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
        output_std = 1.0 / math.sqrt(self.action_out.in_features)
        nn.init.trunc_normal_(
            self.action_out.weight,
            std=output_std,
            a=-3 * output_std,
            b=3 * output_std,
        )
        nn.init.normal_(self.context_type_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.view_embed.weight, mean=0.0, std=0.02)
        self.control_memory.init_weights()
        for layer in self.layers:
            layer.init_weights()
        initial_logit = math.log(self.initial_residual_gate / (1.0 - self.initial_residual_gate))
        nn.init.constant_(self.residual_gate_logit, initial_logit)

    def _vision_view_ids(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.numel() == 0:
            return torch.empty(0, device=positions.device, dtype=torch.long)
        horizontal = positions[:, 2].long()
        lo = horizontal.amin()
        total_width = horizontal.amax() - lo + 1
        content_width = self.num_views * self.view_token_width
        torch._assert(
            total_width >= content_width,
            "Configured camera-token width exceeds the visual canvas",
        )
        # RoboCasa365's canonical VideoResize contract places valid camera
        # content at the top-left, so any full-canvas padding trails it.
        left_padding = torch.zeros_like(total_width)
        view = torch.div(
            horizontal - lo - left_padding,
            self.view_token_width,
            rounding_mode="floor",
        )
        return view.clamp_(0, self.num_views - 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        action_indexes: torch.Tensor,
        text_indexes: torch.Tensor,
        vision_indexes: torch.Tensor,
        position_ids: torch.Tensor,
        sample_lens: list[int],
        proprio_history: list[torch.Tensor] | None,
        proprio_history_mask: list[torch.Tensor] | None,
        action_history: list[torch.Tensor] | None,
        action_history_mask: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        control_fields = (proprio_history, proprio_history_mask, action_history, action_history_mask)
        if any(value is None for value in control_fields):
            raise ValueError("Strong action expert requires proprio/action history tensors and masks")
        assert proprio_history is not None
        assert proprio_history_mask is not None
        assert action_history is not None
        assert action_history_mask is not None
        num_samples = len(sample_lens)
        if not all(len(value) == num_samples for value in control_fields):
            raise ValueError("Strong action expert control fields must align with packed samples")
        if action_indexes.numel() == 0 or vision_indexes.numel() == 0:
            raise ValueError("Strong action expert requires action tokens and clean visual context")

        proprio = torch.stack(_strip_optional_collator_axis(proprio_history, expected_ndim=2))
        proprio_valid = torch.stack(
            _strip_optional_collator_axis(proprio_history_mask, expected_ndim=1)
        ).bool()
        history_action = torch.stack(_strip_optional_collator_axis(action_history, expected_ndim=2))
        history_action_valid = torch.stack(
            _strip_optional_collator_axis(action_history_mask, expected_ndim=1)
        ).bool()
        if proprio.shape[1:] != (self.proprio_history_length, self.state_dim):
            raise ValueError(
                "Strong action proprio contract mismatch: "
                f"expected [B,{self.proprio_history_length},{self.state_dim}], got {tuple(proprio.shape)}."
            )
        if history_action.shape[1:] != (self.action_history_length, self.action_dim):
            raise ValueError(
                "Strong action history contract mismatch: "
                f"expected [B,{self.action_history_length},{self.action_dim}], got {tuple(history_action.shape)}."
            )
        current_state, action_memory_input, memory_tokens, action_ages = self.control_memory(
            proprio,
            proprio_valid,
            history_action,
            history_action_valid,
        )

        action_sample_ids = _sample_ids(action_indexes, sample_lens)
        text_sample_ids = _sample_ids(text_indexes, sample_lens)
        vision_sample_ids = _sample_ids(vision_indexes, sample_lens)
        action_projected = self.action_in(self.action_norm(hidden_states[action_indexes]))
        projected_text = self.context_in(self.context_norm(hidden_states[text_indexes]))
        projected_vision = self.context_in(self.context_norm(hidden_states[vision_indexes]))

        action_groups: list[torch.Tensor] = []
        action_index_groups: list[torch.Tensor] = []
        context_groups: list[torch.Tensor] = []
        context_type_groups: list[torch.Tensor] = []
        context_view_groups: list[torch.Tensor] = []
        context_age_groups: list[torch.Tensor] = []
        context_valid_groups: list[torch.Tensor] = []
        action_lengths: list[int] = []
        nonvisual_view = self.num_views
        max_age = float(max(self.action_history_length, 1))

        for sample_id in range(num_samples):
            action_mask = action_sample_ids == sample_id
            text_mask = text_sample_ids == sample_id
            vision_mask = vision_sample_ids == sample_id
            sample_action = action_projected[action_mask]
            action_length = int(sample_action.shape[0])
            if action_length < 1:
                raise ValueError(f"Packed sample {sample_id} has no action tokens")
            progress = torch.arange(action_length, device=hidden_states.device, dtype=torch.float32)
            progress = progress / max(action_length - 1, 1)
            sample_action = sample_action + self.action_progress_mlp(progress[:, None].to(sample_action.dtype))
            action_groups.append(sample_action)
            action_index_groups.append(action_indexes[action_mask])
            action_lengths.append(action_length)

            sample_text = projected_text[text_mask]
            sample_vision_indexes = vision_indexes[vision_mask]
            sample_vision = projected_vision[vision_mask]
            vision_views = self._vision_view_ids(position_ids[:, sample_vision_indexes].T)
            text_types = torch.full(
                (sample_text.shape[0],), _TEXT_CONTEXT, device=hidden_states.device, dtype=torch.long
            )
            vision_types = torch.full(
                (sample_vision.shape[0],), _VISION_CONTEXT, device=hidden_states.device, dtype=torch.long
            )
            state_types = torch.full((1,), _PROPRIO_CONTEXT, device=hidden_states.device, dtype=torch.long)
            history_types = torch.full(
                (self.action_history_length,),
                _ACTION_HISTORY_CONTEXT,
                device=hidden_states.device,
                dtype=torch.long,
            )
            memory_types = torch.full(
                (self.memory_tokens,), _MEMORY_CONTEXT, device=hidden_states.device, dtype=torch.long
            )
            context_types = torch.cat((text_types, vision_types, state_types, history_types, memory_types))
            context_views = torch.cat(
                (
                    torch.full_like(text_types, nonvisual_view),
                    vision_views,
                    torch.full_like(state_types, nonvisual_view),
                    torch.full_like(history_types, nonvisual_view),
                    torch.full_like(memory_types, nonvisual_view),
                )
            )
            sample_context = torch.cat(
                (
                    sample_text,
                    sample_vision,
                    current_state[sample_id : sample_id + 1],
                    action_memory_input[sample_id],
                    memory_tokens[sample_id],
                ),
                dim=0,
            )
            sample_context = (
                sample_context
                + self.context_type_embed(context_types)
                + self.view_embed(context_views)
            )
            context_valid = torch.cat(
                (
                    torch.ones(
                        sample_text.shape[0] + sample_vision.shape[0] + 1,
                        device=hidden_states.device,
                        dtype=torch.bool,
                    ),
                    history_action_valid[sample_id],
                    torch.ones(self.memory_tokens, device=hidden_states.device, dtype=torch.bool),
                )
            )
            context_age = torch.cat(
                (
                    torch.zeros(sample_text.shape[0] + sample_vision.shape[0] + 1, device=hidden_states.device),
                    action_ages,
                    torch.full((self.memory_tokens,), -1.0 / max_age, device=hidden_states.device),
                )
            )
            context_groups.append(sample_context)
            context_type_groups.append(context_types)
            context_view_groups.append(context_views)
            context_age_groups.append(context_age)
            context_valid_groups.append(context_valid)

        action_batch = pad_sequence(action_groups, batch_first=True)
        context_batch = pad_sequence(context_groups, batch_first=True)
        context_types = pad_sequence(context_type_groups, batch_first=True)
        context_views = pad_sequence(context_view_groups, batch_first=True, padding_value=nonvisual_view)
        context_ages = pad_sequence(context_age_groups, batch_first=True)
        context_valid = pad_sequence(context_valid_groups, batch_first=True)
        action_valid = (
            torch.arange(action_batch.shape[1], device=hidden_states.device)[None]
            < torch.tensor(action_lengths, device=hidden_states.device)[:, None]
        )

        entropies = []
        for layer in self.layers:
            action_batch, entropy, _context_importance = layer(
                action_batch,
                context_batch,
                action_valid=action_valid,
                context_valid=context_valid,
                context_type=context_types,
                context_view=context_views,
                context_age=context_ages,
            )
            entropies.append(entropy)

        residuals = self.action_out(self.output_norm(action_batch))
        ordered_indexes = torch.cat(action_index_groups)
        ordered_residuals = torch.cat(
            [residuals[sample_id, :length] for sample_id, length in enumerate(action_lengths)],
            dim=0,
        )
        gate = self.residual_gate_logit.sigmoid().squeeze(0)
        updated = hidden_states.index_add(0, ordered_indexes, gate * ordered_residuals)
        valid_context_count = torch.stack([mask.sum() for mask in context_valid_groups]).float().mean()
        metrics = {
            "residual_gate": gate,
            "attention_entropy": torch.stack(entropies).mean(),
            "valid_context_tokens": valid_context_count,
            "memory_rms": memory_tokens.float().square().mean().sqrt(),
        }
        return updated, metrics
