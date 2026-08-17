# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DeMiAn-inspired action expert with explicit causal control memory."""

from __future__ import annotations

import math
from itertools import accumulate

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

_TEXT_CONTEXT = 0
_VISION_CONTEXT = 1
_PROPRIO_CONTEXT = 2
_ACTION_HISTORY_CONTEXT = 3
_MEMORY_CONTEXT = 4
_NUM_CONTEXT_TYPES = 5


def action_gradient_alignment_loss(
    action_loss: torch.Tensor,
    relevance_state: torch.Tensor,
    action_attention: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align action-to-slot attention with saliency of the state used by the action path."""
    zero = action_attention.float().sum() * 0.0
    if (
        not torch.is_grad_enabled()
        or not action_loss.requires_grad
        or not relevance_state.requires_grad
    ):
        return zero, zero.detach(), zero.detach()
    state_gradient = torch.autograd.grad(
        action_loss,
        relevance_state,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    if state_gradient is None:
        return zero, zero.detach(), zero.detach()

    saliency = state_gradient.detach().float().square().sum(dim=-1).sqrt()
    saliency_sum = saliency.sum(dim=-1, keepdim=True)
    valid = saliency_sum.squeeze(-1) > 1.0e-12
    target = saliency / saliency_sum.clamp_min(1.0e-12)
    attention = action_attention.float().clamp_min(1.0e-8)
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    per_sample_loss = -(target * attention.log()).sum(dim=-1)
    valid_weight = valid.float()
    valid_denominator = valid_weight.sum().clamp_min(1.0)
    loss = (per_sample_loss * valid_weight).sum() / valid_denominator
    entropy_denominator = math.log(max(relevance_state.shape[1], 2))
    target_entropy = -(target.clamp_min(1.0e-8).log() * target).sum(dim=-1)
    target_entropy = (target_entropy * valid_weight).sum() / valid_denominator / entropy_denominator
    return loss, valid.float().mean(), target_entropy


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


class VisualObjectMemory(nn.Module):
    """Pool sparse historical Wan latents into competitive persistent object slots."""

    def __init__(
        self,
        *,
        latent_channels: int,
        history_length: int,
        latent_height: int,
        latent_width: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        num_slots: int,
        num_views: int,
        intervention: str,
        contrastive_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if min(
            latent_channels,
            history_length,
            latent_height,
            latent_width,
            model_dim,
            num_heads,
            num_layers,
            num_slots,
            num_views,
        ) < 1:
            raise ValueError("Visual object memory dimensions must be positive")
        if model_dim % num_heads:
            raise ValueError("Visual object memory model_dim must be divisible by num_heads")
        if intervention not in {"none", "zero", "shuffle"}:
            raise ValueError(f"Unsupported visual memory intervention={intervention!r}")
        if contrastive_temperature <= 0.0:
            raise ValueError("Visual memory contrastive temperature must be positive")
        self.latent_channels = latent_channels
        self.history_length = history_length
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.num_slots = num_slots
        self.intervention = intervention
        self.contrastive_temperature = float(contrastive_temperature)

        self.latent_norm = nn.LayerNorm(latent_channels)
        self.latent_proj = nn.Linear(latent_channels, model_dim)
        self.history_embed = nn.Embedding(history_length, model_dim)
        self.spatial_embed = nn.Parameter(torch.empty(latent_height * latent_width, model_dim))
        self.view_embed = nn.Embedding(num_views, model_dim)
        self.object_queries = nn.Parameter(torch.empty(num_slots, model_dim))
        self.null_history_token = nn.Parameter(torch.empty(1, model_dim))
        self.slot_q = nn.Linear(model_dim, model_dim, bias=False)
        self.token_k = nn.Linear(model_dim, model_dim, bias=False)
        self.token_v = nn.Linear(model_dim, model_dim, bias=False)
        self.slot_out = nn.Linear(model_dim, model_dim, bias=False)
        self.temporal_predictor = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2 * model_dim),
            nn.SiLU(),
            nn.Linear(2 * model_dim, model_dim),
        )
        rows = 2 if num_slots >= 4 else 1
        columns = math.ceil(num_slots / rows)
        anchor_y = torch.linspace(-0.5, 0.5, rows)
        anchor_x = torch.linspace(-0.75, 0.75, columns)
        anchors = torch.stack(torch.meshgrid(anchor_y, anchor_x, indexing="ij"), dim=-1)
        self.register_buffer("slot_anchors", anchors.reshape(-1, 2)[:num_slots], persistent=True)
        self.anchor_bias_strength = 4.0
        relation_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_encoder = nn.TransformerEncoder(
            relation_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(model_dim)

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
        nn.init.normal_(self.spatial_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.object_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.null_history_token, mean=0.0, std=0.02)

    def forward(
        self,
        history_latent: torch.Tensor,
        history_mask: torch.Tensor,
        episode_ids: torch.Tensor | None = None,
        intervention_override: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        expected = (
            self.history_length,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        )
        if history_latent.ndim != 5 or tuple(history_latent.shape[1:]) != expected:
            raise ValueError(
                f"Visual history latent must be [B,{','.join(map(str, expected))}], "
                f"got {tuple(history_latent.shape)}."
            )
        if history_mask.shape != history_latent.shape[:2]:
            raise ValueError("Visual history mask must align with [B,R]")
        if episode_ids is not None and episode_ids.shape != history_latent.shape[:1]:
            raise ValueError("Visual memory episode IDs must align with the batch")
        intervention = self.intervention if intervention_override is None else intervention_override
        if intervention not in {"none", "zero", "shuffle"}:
            raise ValueError(f"Unsupported visual memory intervention override={intervention!r}")
        history_mask = history_mask.bool()
        if intervention == "zero":
            history_mask = torch.zeros_like(history_mask)
        elif intervention == "shuffle" and history_latent.shape[0] > 1:
            history_latent = torch.roll(history_latent, shifts=1, dims=0)
            history_mask = torch.roll(history_mask, shifts=1, dims=0)
            if episode_ids is not None:
                episode_ids = torch.roll(episode_ids, shifts=1, dims=0)
        slot_valid = history_mask.any(dim=1)

        batch_size = history_latent.shape[0]
        dtype = self.latent_proj.weight.dtype
        tokens = history_latent.permute(0, 1, 3, 4, 2).reshape(
            batch_size,
            self.history_length,
            self.latent_height * self.latent_width,
            self.latent_channels,
        )
        tokens = self.latent_proj(self.latent_norm(tokens.to(dtype)))
        tokens = tokens + self.history_embed.weight[None, :, None, :].to(dtype)
        tokens = tokens + self.spatial_embed[None, None].to(dtype)

        horizontal = torch.arange(self.latent_width, device=tokens.device)
        view_ids = torch.div(horizontal * self.view_embed.num_embeddings, self.latent_width, rounding_mode="floor")
        view_ids = view_ids.clamp_max(self.view_embed.num_embeddings - 1)
        view_ids = view_ids[None, :].expand(self.latent_height, -1).reshape(-1)
        tokens = tokens + self.view_embed(view_ids)[None, None].to(dtype)
        tokens = tokens.flatten(1, 2)
        token_valid = history_mask[:, :, None].expand(-1, -1, self.latent_height * self.latent_width).flatten(1)

        null_token = self.null_history_token.to(dtype).unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat((tokens.masked_fill(~token_valid[:, :, None], 0.0), null_token), dim=1)
        null_valid = ~slot_valid
        token_valid = torch.cat(
            (token_valid, null_valid[:, None]),
            dim=1,
        )
        queries = self.object_queries.to(dtype).unsqueeze(0).expand(batch_size, -1, -1)
        query = self.slot_q(queries).reshape(batch_size, self.num_slots, self.num_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        key = self.token_k(tokens).reshape(batch_size, tokens.shape[1], self.num_heads, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        value = self.token_v(tokens).reshape(batch_size, tokens.shape[1], self.num_heads, self.head_dim)
        value = value.permute(0, 2, 1, 3)
        assignment_logits = torch.einsum("bhnd,bhkd->bhnk", key, query) / math.sqrt(self.head_dim)

        y_coord = torch.linspace(-1.0, 1.0, self.latent_height, device=tokens.device)
        x_coord = torch.linspace(-1.0, 1.0, self.latent_width, device=tokens.device)
        spatial_coordinates = torch.stack(torch.meshgrid(y_coord, x_coord, indexing="ij"), dim=-1)
        spatial_coordinates = spatial_coordinates.reshape(-1, 2).repeat(self.history_length, 1)
        spatial_coordinates = torch.cat((spatial_coordinates, spatial_coordinates.new_zeros(1, 2)), dim=0)
        anchor_distance = (
            spatial_coordinates[:, None, :] - self.slot_anchors.to(spatial_coordinates.dtype)[None]
        ).square().sum(dim=-1)
        anchor_distance[-1] = 0.0
        assignment_logits = assignment_logits - self.anchor_bias_strength * anchor_distance[None, None]

        # Each token first chooses among slots. Slot-wise normalization keeps the
        # pooled feature scale stable when historical frames are missing.
        competitive_assignment = assignment_logits.float().softmax(dim=-1).to(dtype)
        competitive_assignment = competitive_assignment * token_valid[:, None, :, None]

        num_spatial_tokens = self.latent_height * self.latent_width
        num_history_tokens = self.history_length * num_spatial_tokens
        frame_assignment = competitive_assignment[:, :, :num_history_tokens].reshape(
            batch_size,
            self.num_heads,
            self.history_length,
            num_spatial_tokens,
            self.num_slots,
        )
        frame_value = value[:, :, :num_history_tokens].reshape(
            batch_size,
            self.num_heads,
            self.history_length,
            num_spatial_tokens,
            self.head_dim,
        )
        frame_weights = frame_assignment / frame_assignment.sum(dim=3, keepdim=True).clamp_min(1.0e-6)
        valid_frame = history_mask[:, None, :, None].expand(
            -1,
            self.num_heads,
            -1,
            self.num_slots,
        )
        valid_frame_weight = valid_frame.float()
        valid_frame_denominator = valid_frame_weight.sum().clamp_min(1.0)

        assignment_entropy = -(
            frame_assignment.float().clamp_min(1.0e-8).log() * frame_assignment.float()
        ).sum(dim=-1)
        assignment_entropy = assignment_entropy / math.log(max(self.num_slots, 2))
        valid_token = history_mask[:, None, :, None].expand(
            -1,
            self.num_heads,
            -1,
            num_spatial_tokens,
        )
        assignment_entropy_loss = (
            assignment_entropy * valid_token.float()
        ).sum() / valid_token.float().sum().clamp_min(1.0)

        slot_usage = frame_assignment.float().mean(dim=3)
        uniform_usage = 1.0 / self.num_slots
        assignment_balance_loss = (
            (slot_usage - uniform_usage).square().sum(dim=-1)
            * self.num_slots
            * history_mask[:, None].float()
        ).sum() / (
            history_mask.float().sum() * self.num_heads
        ).clamp_min(1.0)

        local_coordinates = spatial_coordinates[:num_spatial_tokens].float()
        slot_centroid = torch.einsum("bhrpk,pd->bhrkd", frame_weights.float(), local_coordinates)
        centered_coordinates = (
            local_coordinates[None, None, None, :, None, :]
            - slot_centroid[:, :, :, None, :, :]
        )
        spatial_variance = (
            frame_weights.float()[..., None] * centered_coordinates.square()
        ).sum(dim=(3, 5))
        spatial_compactness_loss = (
            spatial_variance * valid_frame_weight
        ).sum() / valid_frame_denominator

        frame_pooled = torch.einsum("bhrpk,bhrpd->bhrkd", frame_weights, frame_value)
        frame_pooled = frame_pooled.permute(0, 2, 4, 1, 3).reshape(
            batch_size,
            self.history_length,
            self.num_slots,
            self.model_dim,
        )
        frame_slots = self.output_norm(queries[:, None] + self.slot_out(frame_pooled))
        frame_positions = torch.arange(self.history_length, device=history_mask.device)[None].expand(
            batch_size,
            -1,
        )
        latest_frame = frame_positions.masked_fill(~history_mask, -1).amax(dim=1)
        previous_frame = frame_positions.masked_fill(
            ~(history_mask & (frame_positions < latest_frame[:, None])),
            -1,
        ).amax(dim=1)
        object_delta_valid = previous_frame >= 0
        batch_indexes = torch.arange(batch_size, device=history_mask.device)
        latest_slots = frame_slots[batch_indexes, latest_frame.clamp_min(0)]
        previous_slots = frame_slots[batch_indexes, previous_frame.clamp_min(0)]
        object_delta = (latest_slots - previous_slots).masked_fill(
            ~object_delta_valid[:, None, None],
            0.0,
        )
        delta_valid_weight = object_delta_valid[:, None, None].float()
        object_delta_mean_square = (
            (object_delta.float().square() * delta_valid_weight).sum()
            / (delta_valid_weight.sum() * self.num_slots * self.model_dim).clamp_min(1.0)
        )
        object_delta_rms = torch.where(
            object_delta_mean_square > 0.0,
            object_delta_mean_square.clamp_min(1.0e-12).sqrt(),
            torch.zeros_like(object_delta_mean_square),
        )
        temporal_prediction = self.temporal_predictor(frame_slots[:, :-1])
        temporal_target = frame_slots[:, 1:].detach()
        temporal_distance = 1.0 - (
            F.normalize(temporal_prediction.float(), dim=-1)
            * F.normalize(temporal_target.float(), dim=-1)
        ).sum(dim=-1)
        temporal_pair_valid = history_mask[:, :-1] & history_mask[:, 1:]
        temporal_pair_weight = temporal_pair_valid[:, :, None].float()
        temporal_pair_denominator = (temporal_pair_weight.sum() * self.num_slots).clamp_min(1.0)
        temporal_correspondence_loss = (
            temporal_distance * temporal_pair_weight
        ).sum() / temporal_pair_denominator
        temporal_pair_fraction = (
            temporal_pair_valid.float().mean()
            if temporal_pair_valid.numel()
            else temporal_pair_valid.new_zeros((), dtype=torch.float32)
        )

        normalized_prediction = F.normalize(temporal_prediction.float(), dim=-1)
        normalized_target = F.normalize(temporal_target.float(), dim=-1)
        object_contrastive_loss = temporal_prediction.float().sum() * 0.0
        positive_similarity = object_contrastive_loss.detach()
        object_negative_similarity = object_contrastive_loss.detach()
        if self.history_length > 1:
            object_similarity = torch.einsum(
                "brkd,brjd->brkj",
                normalized_prediction,
                normalized_target,
            )
            object_logits = object_similarity / self.contrastive_temperature
            object_labels = torch.arange(self.num_slots, device=object_logits.device)
            object_loss = F.cross_entropy(
                object_logits.reshape(-1, self.num_slots),
                object_labels.repeat(batch_size * (self.history_length - 1)),
                reduction="none",
            ).reshape(batch_size, self.history_length - 1, self.num_slots)
            object_contrastive_loss = (
                object_loss * temporal_pair_weight
            ).sum() / temporal_pair_denominator
            positive_similarity = (
                (normalized_prediction * normalized_target).sum(dim=-1) * temporal_pair_weight
            ).sum() / temporal_pair_denominator
            object_off_diagonal = ~torch.eye(
                self.num_slots,
                device=object_similarity.device,
                dtype=torch.bool,
            )[None, None]
            object_negative_weight = (
                object_off_diagonal
                & temporal_pair_valid[:, :, None, None]
            ).float()
            object_negative_similarity = (
                object_similarity * object_negative_weight
            ).sum() / object_negative_weight.sum().clamp_min(1.0)

        episode_contrastive_loss = temporal_prediction.float().sum() * 0.0
        episode_negative_similarity = episode_contrastive_loss.detach()
        episode_negative_fraction = episode_contrastive_loss.detach()
        if episode_ids is not None and dist.is_available() and dist.is_initialized():
            local_prototype = frame_slots.new_zeros((self.num_slots, self.model_dim))
            local_episode_id = episode_ids.new_full((), -1, dtype=torch.long)
            valid_samples = slot_valid & (episode_ids >= 0)
            if bool(valid_samples.any()):
                local_index = int(valid_samples.nonzero(as_tuple=False)[0, 0])
                valid_positions = history_mask[local_index].nonzero(as_tuple=False).flatten()
                local_prototype = frame_slots[local_index, int(valid_positions[-1])].detach()
                local_episode_id = episode_ids[local_index].long()
            gathered_prototypes = [torch.zeros_like(local_prototype) for _ in range(dist.get_world_size())]
            gathered_episode_ids = [torch.zeros_like(local_episode_id) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_prototypes, local_prototype)
            dist.all_gather(gathered_episode_ids, local_episode_id)
            prototype_bank = F.normalize(torch.stack(gathered_prototypes).float(), dim=-1)
            prototype_episode_ids = torch.stack(gathered_episode_ids)
            episode_losses: list[torch.Tensor] = []
            episode_negative_means: list[torch.Tensor] = []
            valid_pair_count = int(temporal_pair_valid.sum())
            for sample_id, pair_id in temporal_pair_valid.nonzero(as_tuple=False).tolist():
                negative_episodes = (
                    (prototype_episode_ids >= 0)
                    & (prototype_episode_ids != episode_ids[sample_id])
                )
                if not bool(negative_episodes.any()):
                    continue
                query_slots = normalized_prediction[sample_id, pair_id]
                positive_slots = normalized_target[sample_id, pair_id]
                negatives = prototype_bank[negative_episodes].reshape(-1, self.model_dim)
                positive_logits = (query_slots * positive_slots).sum(dim=-1, keepdim=True)
                negative_logits = torch.matmul(query_slots, negatives.T)
                logits = torch.cat((positive_logits, negative_logits), dim=-1)
                labels = torch.zeros(self.num_slots, device=logits.device, dtype=torch.long)
                episode_losses.append(
                    F.cross_entropy(logits / self.contrastive_temperature, labels)
                )
                episode_negative_means.append(negative_logits.mean())
            if episode_losses:
                episode_contrastive_loss = torch.stack(episode_losses).mean()
                episode_negative_similarity = torch.stack(episode_negative_means).mean()
                episode_negative_fraction = episode_negative_fraction.new_tensor(
                    len(episode_losses) / max(valid_pair_count, 1)
                )

        slot_weights = competitive_assignment / competitive_assignment.sum(dim=2, keepdim=True).clamp_min(1.0e-6)
        pooled = torch.einsum("bhnk,bhnd->bhkd", slot_weights, value)
        pooled = pooled.permute(0, 2, 1, 3).reshape(batch_size, self.num_slots, self.model_dim)
        slots = self.output_norm(self.relation_encoder(queries + self.slot_out(pooled)))

        normalized_slots = F.normalize(slots.float(), dim=-1)
        similarity = torch.matmul(normalized_slots, normalized_slots.transpose(-1, -2))
        off_diagonal = ~torch.eye(self.num_slots, device=slots.device, dtype=torch.bool)[None]
        off_diagonal = off_diagonal.expand(batch_size, -1, -1)
        per_sample_similarity = similarity.masked_select(off_diagonal).reshape(batch_size, -1)
        valid_weight = slot_valid.float()
        valid_denominator = valid_weight.sum().clamp_min(1.0)
        slot_cosine = (per_sample_similarity.mean(dim=1) * valid_weight).sum() / valid_denominator
        slot_abs_cosine = (per_sample_similarity.abs().mean(dim=1) * valid_weight).sum() / valid_denominator
        diversity_loss = (per_sample_similarity.square().mean(dim=1) * valid_weight).sum() / valid_denominator
        metrics = {
            "visual_memory_rms": slots.float().square().mean().sqrt(),
            "visual_memory_slot_cosine": slot_cosine,
            "visual_memory_slot_abs_cosine": slot_abs_cosine,
            "visual_memory_diversity_loss": diversity_loss,
            "visual_memory_assignment_entropy_loss": assignment_entropy_loss,
            "visual_memory_assignment_balance_loss": assignment_balance_loss,
            "visual_memory_spatial_compactness_loss": spatial_compactness_loss,
            "visual_memory_history_fraction": history_mask.float().mean(),
            "visual_memory_temporal_correspondence_loss": temporal_correspondence_loss,
            "visual_memory_temporal_pair_fraction": temporal_pair_fraction,
            "visual_memory_object_contrastive_loss": object_contrastive_loss,
            "visual_memory_episode_contrastive_loss": episode_contrastive_loss,
            "visual_memory_positive_similarity": positive_similarity,
            "visual_memory_object_negative_similarity": object_negative_similarity,
            "visual_memory_episode_negative_similarity": episode_negative_similarity,
            "visual_memory_episode_negative_fraction": episode_negative_fraction,
            "visual_memory_slots": slots,
            "visual_memory_object_delta": object_delta,
            "visual_memory_object_delta_valid": object_delta_valid,
            "visual_memory_object_delta_rms": object_delta_rms,
            "visual_memory_object_delta_active_fraction": object_delta_valid.float().mean(),
        }
        # Zero is represented by learned null slots, not by removing the branch.
        # This keeps the R4 conditioning path structurally active for matched
        # none/zero/shuffle interventions without introducing all-masked attention.
        return slots, torch.ones_like(slot_valid), metrics


class MemoryConditioningBlock(nn.Module):
    """Inject object memory into every action layer through fixed-scale attention and FiLM."""

    def __init__(self, *, model_dim: int, num_heads: int, residual_scale: float) -> None:
        super().__init__()
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("Memory conditioning residual_scale must be in (0, 1]")
        self.residual_scale = float(residual_scale)
        self.action_norm = nn.LayerNorm(model_dim)
        self.memory_norm = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.film = nn.Linear(model_dim, 2 * model_dim)

    def init_weights(self) -> None:
        for norm in (self.action_norm, self.memory_norm):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        nn.init.xavier_uniform_(self.cross_attention.in_proj_weight)
        if self.cross_attention.in_proj_bias is not None:
            nn.init.zeros_(self.cross_attention.in_proj_bias)
        nn.init.xavier_uniform_(self.cross_attention.out_proj.weight)
        if self.cross_attention.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attention.out_proj.bias)
        std = 1.0 / math.sqrt(self.film.in_features)
        nn.init.trunc_normal_(self.film.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.film.bias)

    def forward(
        self,
        action: torch.Tensor,
        memory: torch.Tensor,
        *,
        action_valid: torch.Tensor,
        memory_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if memory_valid is not None and memory_valid.shape != action.shape[:1]:
            raise ValueError("Memory validity must align with the action batch")
        normalized_action = self.action_norm(action)
        normalized_memory = self.memory_norm(memory)
        attended, attention = self.cross_attention(
            normalized_action,
            normalized_memory,
            normalized_memory,
            need_weights=True,
            average_attn_weights=False,
        )
        gamma, beta = self.film(attended).chunk(2, dim=-1)
        conditioned = attended + torch.tanh(gamma) * normalized_action + torch.tanh(beta)
        effective_valid = action_valid
        if memory_valid is not None:
            effective_valid = effective_valid & memory_valid[:, None]
        residual = (self.residual_scale * conditioned).masked_fill(~effective_valid[:, :, None], 0.0)
        valid_residual = residual.float().square().sum() / (
            effective_valid.sum().clamp_min(1) * residual.shape[-1]
        )
        mean_attention = attention.mean(dim=1)
        if memory_valid is not None:
            mean_attention = mean_attention.masked_fill(~memory_valid[:, None, None], 0.0)
        return action + residual, valid_residual.sqrt(), mean_attention


class MemoryDeltaMainResidual(nn.Module):
    """Make the final action residual depend directly on same-slot memory changes."""

    def __init__(
        self,
        *,
        model_dim: int,
        hidden_size: int,
        num_heads: int,
        gate_scale: float,
        residual_scale: float,
        component: str,
    ) -> None:
        super().__init__()
        if not 0.0 < gate_scale <= 1.0:
            raise ValueError("Memory delta gate_scale must be in (0, 1]")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("Memory delta residual_scale must be in (0, 1]")
        self.gate_scale = float(gate_scale)
        self.residual_scale = float(residual_scale)
        if component not in {"base", "gate", "residual", "full"}:
            raise ValueError("Memory delta component must be base, gate, residual, or full")
        self.component = component
        self.action_norm = nn.LayerNorm(model_dim)
        self.delta_norm = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.delta_out = nn.Linear(model_dim, hidden_size, bias=False)

    def init_weights(self) -> None:
        for norm in (self.action_norm, self.delta_norm):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        nn.init.xavier_uniform_(self.cross_attention.in_proj_weight)
        if self.cross_attention.in_proj_bias is not None:
            nn.init.zeros_(self.cross_attention.in_proj_bias)
        nn.init.xavier_uniform_(self.cross_attention.out_proj.weight)
        if self.cross_attention.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attention.out_proj.bias)
        std = 1.0 / math.sqrt(self.delta_out.in_features)
        nn.init.trunc_normal_(self.delta_out.weight, std=std, a=-3 * std, b=3 * std)

    def forward(
        self,
        action: torch.Tensor,
        base_residual: torch.Tensor,
        object_delta: torch.Tensor,
        object_delta_valid: torch.Tensor,
        *,
        action_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if object_delta.ndim != 3 or object_delta.shape[0] != action.shape[0]:
            raise ValueError("Object-local memory delta must align with the action batch")
        if base_residual.shape[:2] != action.shape[:2]:
            raise ValueError("Base residual must align with the padded action batch")
        if base_residual.shape[-1] != self.delta_out.out_features:
            raise ValueError("Base residual width must match the configured hidden size")
        if object_delta.shape[-1] != action.shape[-1]:
            raise ValueError("Object-local memory delta width must match the action model dimension")
        if object_delta_valid.shape != action.shape[:1]:
            raise ValueError("Object-local memory delta validity must align with the action batch")
        if action_valid.shape != action.shape[:2]:
            raise ValueError("Action validity must align with the padded action batch")

        normalized_delta = self.delta_norm(object_delta)
        attended, attention = self.cross_attention(
            self.action_norm(action),
            normalized_delta,
            normalized_delta,
            need_weights=True,
            average_attn_weights=False,
        )
        attention = attention.mean(dim=1)
        valid = object_delta_valid[:, None] & action_valid

        slot_squared_rms = object_delta.float().square().mean(dim=-1)
        slot_rms = torch.where(
            slot_squared_rms > 0.0,
            slot_squared_rms.clamp_min(1.0e-12).sqrt(),
            torch.zeros_like(slot_squared_rms),
        )
        local_rms = (attention.float() * slot_rms[:, None]).sum(dim=-1, keepdim=True)
        global_rms = slot_rms.mean(dim=-1, keepdim=True)[:, None]
        raw_gate_deviation = self.gate_scale * torch.tanh(0.5 * (local_rms + global_rms))
        raw_gate_deviation = raw_gate_deviation.masked_fill(~valid[:, :, None], 0.0)

        raw_directional_residual = self.residual_scale * self.delta_out(attended)
        raw_directional_residual = raw_directional_residual.masked_fill(~valid[:, :, None], 0.0)
        gate_deviation = (
            raw_gate_deviation
            if self.component in {"gate", "full"}
            else torch.zeros_like(raw_gate_deviation)
        )
        directional_residual = (
            raw_directional_residual
            if self.component in {"residual", "full"}
            else torch.zeros_like(raw_directional_residual)
        )
        residual = base_residual * (1.0 + gate_deviation.to(base_residual.dtype))
        residual = residual + directional_residual

        valid_weight = valid[:, :, None].float()
        valid_denominator = (valid_weight.sum() * base_residual.shape[-1]).clamp_min(1.0)
        raw_directional_mean_square = raw_directional_residual.float().square().sum() / valid_denominator
        raw_directional_rms = torch.where(
            raw_directional_mean_square > 0.0,
            raw_directional_mean_square.clamp_min(1.0e-12).sqrt(),
            torch.zeros_like(raw_directional_mean_square),
        )
        directional_mean_square = directional_residual.float().square().sum() / valid_denominator
        directional_rms = torch.where(
            directional_mean_square > 0.0,
            directional_mean_square.clamp_min(1.0e-12).sqrt(),
            torch.zeros_like(directional_mean_square),
        )
        applied_delta = (residual - base_residual).float() * valid_weight
        applied_delta_rms = (applied_delta.square().sum() / valid_denominator).sqrt()
        base_rms = ((base_residual.float() * valid_weight).square().sum() / valid_denominator).sqrt()
        delta_base_ratio = applied_delta_rms / base_rms.clamp_min(1.0e-12)
        delta_square_sum = applied_delta.square().sum()
        base_square_sum = (base_residual.float() * valid_weight).square().sum()
        delta_base_cosine = torch.where(
            (delta_square_sum > 0.0) & (base_square_sum > 0.0),
            (applied_delta * base_residual.float()).sum()
            / (delta_square_sum * base_square_sum).clamp_min(1.0e-24).sqrt(),
            torch.zeros_like(delta_square_sum),
        )
        gate_denominator = valid_weight.sum().clamp_min(1.0)
        metrics = {
            "visual_memory_delta_main_gate_mean": (
                1.0
                + (gate_deviation.float() * valid_weight).sum() / gate_denominator
            ),
            "visual_memory_delta_main_gate_deviation": (
                gate_deviation.float().abs() * valid_weight
            ).sum()
            / gate_denominator,
            "visual_memory_delta_main_residual_rms": directional_rms,
            "visual_memory_delta_main_raw_gate_deviation": (
                raw_gate_deviation.float().abs() * valid_weight
            ).sum()
            / gate_denominator,
            "visual_memory_delta_main_raw_residual_rms": raw_directional_rms,
            "visual_memory_delta_main_base_rms": base_rms,
            "visual_memory_delta_main_applied_delta_rms": applied_delta_rms,
            "visual_memory_delta_main_delta_base_ratio": delta_base_ratio,
            "visual_memory_delta_main_delta_base_cosine": delta_base_cosine,
            "visual_memory_delta_main_attention": (
                attention.float() * valid_weight
            ).sum(dim=1)
            / valid_weight.sum(dim=1).clamp_min(1.0),
        }
        return residual, metrics


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return attended, entropy_mean


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.self_norm(action)
        residual, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~action_valid,
            need_weights=False,
        )
        action = action + residual.masked_fill(~action_valid[:, :, None], 0.0)
        residual, entropy = self.cross_attention(
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
        return action, entropy


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
        visual_memory_enabled: bool = False,
        visual_history_length: int = 3,
        visual_latent_channels: int = 48,
        visual_latent_height: int = 15,
        visual_latent_width: int = 26,
        visual_memory_layers: int = 2,
        visual_memory_tokens: int = 8,
        visual_memory_initial_gate: float = 0.1,
        visual_memory_conditioning_scale: float = 0.25,
        visual_memory_delta_main_enabled: bool = False,
        visual_memory_delta_gate_scale: float = 0.25,
        visual_memory_delta_residual_scale: float = 0.25,
        visual_memory_delta_main_component: str = "full",
        visual_memory_exclusive_dynamic: bool = False,
        visual_memory_delta_conditioning_layers: int = 0,
        visual_memory_delta_conditioning_scale: float = 0.25,
        visual_memory_contrastive_temperature: float = 0.07,
        visual_memory_intervention: str = "none",
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
        self.visual_memory_enabled = bool(visual_memory_enabled)
        self.visual_history_length = int(visual_history_length)
        self.visual_memory_tokens = int(visual_memory_tokens)
        self.visual_memory_initial_gate = float(visual_memory_initial_gate)
        if self.visual_memory_enabled and not 0.0 < self.visual_memory_initial_gate < 1.0:
            raise ValueError("visual_memory_initial_gate must be strictly between zero and one")
        self.visual_memory_conditioning_scale = float(visual_memory_conditioning_scale)
        if self.visual_memory_enabled and not 0.0 < self.visual_memory_conditioning_scale <= 1.0:
            raise ValueError("visual_memory_conditioning_scale must be in (0, 1]")
        self.visual_memory_delta_main_enabled = bool(visual_memory_delta_main_enabled)
        if self.visual_memory_delta_main_enabled and not self.visual_memory_enabled:
            raise ValueError("visual_memory_delta_main_enabled requires visual_memory_enabled")
        self.visual_memory_delta_gate_scale = float(visual_memory_delta_gate_scale)
        self.visual_memory_delta_residual_scale = float(visual_memory_delta_residual_scale)
        self.visual_memory_delta_main_component = visual_memory_delta_main_component
        if self.visual_memory_delta_main_component not in {"base", "gate", "residual", "full"}:
            raise ValueError("visual_memory_delta_main_component must be base, gate, residual, or full")
        self.visual_memory_exclusive_dynamic = bool(visual_memory_exclusive_dynamic)
        if self.visual_memory_exclusive_dynamic and not (
            self.visual_memory_enabled and self.visual_memory_delta_main_enabled
        ):
            raise ValueError("Exclusive dynamic memory requires visual memory and the delta main residual")
        self.visual_memory_delta_conditioning_layers = int(visual_memory_delta_conditioning_layers)
        if not 0 <= self.visual_memory_delta_conditioning_layers <= num_layers:
            raise ValueError("Delta conditioning layers must be between zero and the action layer count")
        if self.visual_memory_delta_conditioning_layers and not self.visual_memory_exclusive_dynamic:
            raise ValueError("Delta conditioning layers require exclusive dynamic memory")
        self.visual_memory_delta_conditioning_scale = float(visual_memory_delta_conditioning_scale)
        if self.visual_memory_delta_conditioning_layers and not (
            0.0 < self.visual_memory_delta_conditioning_scale <= 1.0
        ):
            raise ValueError("Delta conditioning scale must be in (0, 1]")

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
        if self.visual_memory_enabled:
            self.visual_memory_context_embed = nn.Parameter(torch.empty(model_dim))
            visual_memory_initial_logit = math.log(
                self.visual_memory_initial_gate / (1.0 - self.visual_memory_initial_gate)
            )
            self.visual_memory_gate_logit = nn.Parameter(torch.tensor([visual_memory_initial_logit]))
            self.visual_object_memory = VisualObjectMemory(
                latent_channels=visual_latent_channels,
                history_length=visual_history_length,
                latent_height=visual_latent_height,
                latent_width=visual_latent_width,
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=visual_memory_layers,
                num_slots=visual_memory_tokens,
                num_views=num_views,
                intervention=visual_memory_intervention,
                contrastive_temperature=visual_memory_contrastive_temperature,
            )
            self.visual_memory_conditioners = nn.ModuleList(
                MemoryConditioningBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    residual_scale=self.visual_memory_conditioning_scale,
                )
                for _ in range(num_layers)
            )
            self.visual_memory_delta_conditioners = nn.ModuleList(
                MemoryConditioningBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    residual_scale=self.visual_memory_delta_conditioning_scale,
                )
                for _ in range(self.visual_memory_delta_conditioning_layers)
            )
            self.visual_memory_aux_norm = nn.LayerNorm(model_dim)
            self.visual_memory_aux_out = nn.Linear(model_dim, hidden_size, bias=False)
            if self.visual_memory_delta_main_enabled:
                self.visual_memory_delta_main = MemoryDeltaMainResidual(
                    model_dim=model_dim,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    gate_scale=self.visual_memory_delta_gate_scale,
                    residual_scale=self.visual_memory_delta_residual_scale,
                    component=self.visual_memory_delta_main_component,
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
        if self.visual_memory_enabled:
            nn.init.normal_(self.visual_memory_context_embed, mean=0.0, std=0.02)
            visual_memory_initial_logit = math.log(
                self.visual_memory_initial_gate / (1.0 - self.visual_memory_initial_gate)
            )
            nn.init.constant_(self.visual_memory_gate_logit, visual_memory_initial_logit)
            self.visual_object_memory.init_weights()
            for conditioner in self.visual_memory_conditioners:
                conditioner.init_weights()
            for conditioner in self.visual_memory_delta_conditioners:
                conditioner.init_weights()
            nn.init.ones_(self.visual_memory_aux_norm.weight)
            nn.init.zeros_(self.visual_memory_aux_norm.bias)
            aux_output_std = 1.0 / math.sqrt(self.visual_memory_aux_out.in_features)
            nn.init.trunc_normal_(
                self.visual_memory_aux_out.weight,
                std=aux_output_std,
                a=-3 * aux_output_std,
                b=3 * aux_output_std,
            )
            if self.visual_memory_delta_main_enabled:
                self.visual_memory_delta_main.init_weights()
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
        visual_history_latent: list[torch.Tensor] | None = None,
        visual_history_mask: list[torch.Tensor] | None = None,
        episode_ids: list[torch.Tensor] | None = None,
        action_seed_hidden: torch.Tensor | None = None,
        visual_memory_intervention_override: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
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
        if self.visual_memory_enabled:
            if action_seed_hidden is None:
                raise ValueError("R4 visual memory conditioning requires pre-transformer action seeds")
            if action_seed_hidden.shape != (action_indexes.numel(), self.hidden_size):
                raise ValueError(
                    "R4 action seeds must follow packed action order: "
                    f"expected {(action_indexes.numel(), self.hidden_size)}, got {tuple(action_seed_hidden.shape)}"
                )

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
        visual_memory_tokens: torch.Tensor | None = None
        visual_memory_valid: torch.Tensor | None = None
        visual_memory_metrics: dict[str, torch.Tensor] = {}
        if self.visual_memory_enabled:
            if visual_history_latent is None or visual_history_mask is None:
                raise ValueError("Visual object memory requires visual history latents and masks")
            if len(visual_history_latent) != num_samples or len(visual_history_mask) != num_samples:
                raise ValueError("Visual history fields must align with packed samples")
            visual_latent = torch.stack(
                _strip_optional_collator_axis(visual_history_latent, expected_ndim=4)
            )
            visual_valid = torch.stack(
                _strip_optional_collator_axis(visual_history_mask, expected_ndim=1)
            ).bool()
            visual_episode_ids = None
            if episode_ids is not None:
                if len(episode_ids) != num_samples:
                    raise ValueError("Visual memory episode IDs must align with packed samples")
                visual_episode_ids = torch.stack(
                    _strip_optional_collator_axis(episode_ids, expected_ndim=0)
                ).long()
            visual_memory_tokens, visual_memory_valid, visual_memory_metrics = self.visual_object_memory(
                visual_latent,
                visual_valid,
                visual_episode_ids,
                intervention_override=visual_memory_intervention_override,
            )

        action_sample_ids = _sample_ids(action_indexes, sample_lens)
        text_sample_ids = _sample_ids(text_indexes, sample_lens)
        vision_sample_ids = _sample_ids(vision_indexes, sample_lens)
        action_projected = self.action_in(self.action_norm(hidden_states[action_indexes]))
        action_seed_projected = (
            self.action_in(self.action_norm(action_seed_hidden))
            if action_seed_hidden is not None
            else None
        )
        projected_text = self.context_in(self.context_norm(hidden_states[text_indexes]))
        projected_vision = self.context_in(self.context_norm(hidden_states[vision_indexes]))

        action_groups: list[torch.Tensor] = []
        action_seed_groups: list[torch.Tensor] = []
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
            if action_seed_projected is not None:
                sample_action_seed = action_seed_projected[action_mask]
                sample_action_seed = sample_action_seed + self.action_progress_mlp(
                    progress[:, None].to(sample_action_seed.dtype)
                )
                action_seed_groups.append(sample_action_seed)
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
            include_absolute_visual_memory = (
                self.visual_memory_enabled and not self.visual_memory_exclusive_dynamic
            )
            visual_memory_types = torch.full(
                (self.visual_memory_tokens if include_absolute_visual_memory else 0,),
                _MEMORY_CONTEXT,
                device=hidden_states.device,
                dtype=torch.long,
            )
            context_types = torch.cat(
                (text_types, vision_types, state_types, history_types, memory_types, visual_memory_types)
            )
            context_views = torch.cat(
                (
                    torch.full_like(text_types, nonvisual_view),
                    vision_views,
                    torch.full_like(state_types, nonvisual_view),
                    torch.full_like(history_types, nonvisual_view),
                    torch.full_like(memory_types, nonvisual_view),
                    torch.full_like(visual_memory_types, nonvisual_view),
                )
            )
            sample_visual_memory = (
                visual_memory_tokens[sample_id]
                if visual_memory_tokens is not None and include_absolute_visual_memory
                else memory_tokens.new_empty((0, self.model_dim))
            )
            sample_context = torch.cat(
                (
                    sample_text,
                    sample_vision,
                    current_state[sample_id : sample_id + 1],
                    action_memory_input[sample_id],
                    memory_tokens[sample_id],
                    sample_visual_memory,
                ),
                dim=0,
            )
            sample_context = (
                sample_context
                + self.context_type_embed(context_types)
                + self.view_embed(context_views)
            )
            if sample_visual_memory.shape[0]:
                visual_gate = self.visual_memory_gate_logit.sigmoid()
                visual_start = sample_context.shape[0] - sample_visual_memory.shape[0]
                visual_scale = torch.ones(
                    sample_context.shape[0],
                    device=sample_context.device,
                    dtype=sample_context.dtype,
                )
                visual_scale[visual_start:] = visual_gate.to(sample_context.dtype)
                visual_identity = torch.zeros_like(sample_context)
                visual_identity[visual_start:] = self.visual_memory_context_embed.to(sample_context.dtype)
                sample_context = (sample_context + visual_identity) * visual_scale[:, None]
            sample_visual_valid = (
                visual_memory_valid[sample_id].expand(sample_visual_memory.shape[0])
                if visual_memory_valid is not None
                else torch.empty(0, device=hidden_states.device, dtype=torch.bool)
            )
            context_valid = torch.cat(
                (
                    torch.ones(
                        sample_text.shape[0] + sample_vision.shape[0] + 1,
                        device=hidden_states.device,
                        dtype=torch.bool,
                    ),
                    history_action_valid[sample_id],
                    torch.ones(
                        self.memory_tokens,
                        device=hidden_states.device,
                        dtype=torch.bool,
                    ),
                    sample_visual_valid,
                )
            )
            context_age = torch.cat(
                (
                    torch.zeros(sample_text.shape[0] + sample_vision.shape[0] + 1, device=hidden_states.device),
                    action_ages,
                    torch.full((self.memory_tokens,), -1.0 / max_age, device=hidden_states.device),
                    torch.full(
                        (sample_visual_memory.shape[0],),
                        -float(self.visual_history_length) / max(self.visual_history_length, 1),
                        device=hidden_states.device,
                    ),
                )
            )
            context_groups.append(sample_context)
            context_type_groups.append(context_types)
            context_view_groups.append(context_views)
            context_age_groups.append(context_age)
            context_valid_groups.append(context_valid)

        action_batch = pad_sequence(action_groups, batch_first=True)
        action_seed_batch = (
            pad_sequence(action_seed_groups, batch_first=True)
            if action_seed_groups
            else None
        )
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
        conditioned_residual_rms = []
        visual_memory_attentions = []
        auxiliary_conditioned_residual_rms = []
        delta_conditioned_residual_rms = []
        delta_memory_attentions = []
        delta_conditioning_start = len(self.layers) - self.visual_memory_delta_conditioning_layers
        for layer_id, layer in enumerate(self.layers):
            action_batch, entropy = layer(
                action_batch,
                context_batch,
                action_valid=action_valid,
                context_valid=context_valid,
                context_type=context_types,
                context_view=context_views,
                context_age=context_ages,
            )
            entropies.append(entropy)
            if self.visual_memory_enabled:
                assert visual_memory_tokens is not None
                assert action_seed_batch is not None
                conditioner = self.visual_memory_conditioners[layer_id]
                if not self.visual_memory_exclusive_dynamic:
                    action_batch, residual_rms, action_attention = conditioner(
                        action_batch,
                        visual_memory_tokens,
                        action_valid=action_valid,
                    )
                    conditioned_residual_rms.append(residual_rms)
                    visual_memory_attentions.append(action_attention)
                action_seed_batch, auxiliary_residual_rms, _ = conditioner(
                    action_seed_batch,
                    visual_memory_tokens,
                    action_valid=action_valid,
                )
                auxiliary_conditioned_residual_rms.append(auxiliary_residual_rms)
                if layer_id >= delta_conditioning_start:
                    delta_conditioner = self.visual_memory_delta_conditioners[
                        layer_id - delta_conditioning_start
                    ]
                    action_batch, delta_residual_rms, delta_attention = delta_conditioner(
                        action_batch,
                        visual_memory_metrics["visual_memory_object_delta"],
                        action_valid=action_valid,
                        memory_valid=visual_memory_metrics["visual_memory_object_delta_valid"],
                    )
                    delta_conditioned_residual_rms.append(delta_residual_rms)
                    delta_memory_attentions.append(delta_attention)

        residuals = self.action_out(self.output_norm(action_batch))
        delta_main_metrics: dict[str, torch.Tensor] = {}
        if self.visual_memory_delta_main_enabled:
            residuals, delta_main_metrics = self.visual_memory_delta_main(
                action_batch,
                residuals,
                visual_memory_metrics["visual_memory_object_delta"],
                visual_memory_metrics["visual_memory_object_delta_valid"],
                action_valid=action_valid,
            )
        ordered_indexes = torch.cat(action_index_groups)
        ordered_residuals = torch.cat(
            [residuals[sample_id, :length] for sample_id, length in enumerate(action_lengths)],
            dim=0,
        )
        gate = self.residual_gate_logit.sigmoid().squeeze(0)
        updated = hidden_states.index_add(0, ordered_indexes, gate * ordered_residuals)
        if self.visual_memory_exclusive_dynamic:
            # Keep the legacy absolute-memory parameters in the R5-compatible
            # FSDP/optimizer graph without allowing them to affect R6 outputs.
            legacy_layout_anchor = (
                self.visual_memory_context_embed.sum() + self.visual_memory_gate_logit.sum()
            ) * 0.0
            updated = updated + legacy_layout_anchor.to(updated.dtype)
        visual_memory_action_hidden: torch.Tensor | None = None
        if action_seed_batch is not None:
            auxiliary_hidden = self.visual_memory_aux_out(self.visual_memory_aux_norm(action_seed_batch))
            visual_memory_action_hidden = torch.cat(
                [auxiliary_hidden[sample_id, :length] for sample_id, length in enumerate(action_lengths)],
                dim=0,
            )
        valid_context_count = torch.stack([mask.sum() for mask in context_valid_groups]).float().mean()
        metrics = {
            "residual_gate": gate,
            "attention_entropy": torch.stack(entropies).mean(),
            "valid_context_tokens": valid_context_count,
            "memory_rms": memory_tokens.float().square().mean().sqrt(),
        }
        if self.visual_memory_enabled:
            metrics["visual_memory_gate"] = self.visual_memory_gate_logit.sigmoid()
            metrics["visual_memory_conditioning_scale"] = torch.tensor(
                self.visual_memory_conditioning_scale,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            metrics["visual_memory_conditioned_residual_rms"] = torch.stack(
                conditioned_residual_rms
            ).mean() if conditioned_residual_rms else torch.zeros((), device=hidden_states.device)
            metrics["visual_memory_aux_conditioned_residual_rms"] = torch.stack(
                auxiliary_conditioned_residual_rms
            ).mean()
            metrics["visual_memory_delta_conditioned_residual_rms"] = torch.stack(
                delta_conditioned_residual_rms
            ).mean() if delta_conditioned_residual_rms else torch.zeros((), device=hidden_states.device)
            metrics["visual_memory_main_context_tokens"] = torch.tensor(
                0 if self.visual_memory_exclusive_dynamic else self.visual_memory_tokens,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            metrics["visual_memory_delta_conditioning_layers"] = torch.tensor(
                self.visual_memory_delta_conditioning_layers,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            assert visual_memory_action_hidden is not None
            metrics["visual_memory_action_aux_hidden_rms"] = (
                visual_memory_action_hidden.float().square().mean().sqrt()
            )
            attention_weight = action_valid[None, :, :, None].float()
            if visual_memory_attentions:
                stacked_attention = torch.stack(visual_memory_attentions).float()
                metrics["visual_memory_action_attention"] = (
                    stacked_attention * attention_weight
                ).sum(dim=(0, 2)) / (
                    attention_weight.sum(dim=(0, 2)) * len(visual_memory_attentions)
                ).clamp_min(1.0)
            elif delta_memory_attentions:
                stacked_attention = torch.stack(delta_memory_attentions).float()
                metrics["visual_memory_action_attention"] = (
                    stacked_attention * attention_weight
                ).sum(dim=(0, 2)) / (
                    attention_weight.sum(dim=(0, 2)) * len(delta_memory_attentions)
                ).clamp_min(1.0)
            elif "visual_memory_delta_main_attention" in delta_main_metrics:
                metrics["visual_memory_action_attention"] = delta_main_metrics[
                    "visual_memory_delta_main_attention"
                ]
            else:
                metrics["visual_memory_action_attention"] = torch.zeros(
                    num_samples,
                    self.visual_memory_tokens,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            if delta_memory_attentions:
                stacked_delta_attention = torch.stack(delta_memory_attentions).float()
                metrics["visual_memory_delta_action_attention"] = (
                    stacked_delta_attention * attention_weight
                ).sum(dim=(0, 2)) / (
                    attention_weight.sum(dim=(0, 2)) * len(delta_memory_attentions)
                ).clamp_min(1.0)
            else:
                metrics["visual_memory_delta_action_attention"] = torch.zeros(
                    num_samples,
                    self.visual_memory_tokens,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
        metrics.update(visual_memory_metrics)
        metrics.update(delta_main_metrics)
        return updated, visual_memory_action_hidden, metrics
