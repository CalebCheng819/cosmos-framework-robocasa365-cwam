# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Online causal action-relevant state and state-only action expert."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from cosmos_framework.model.generator.mot.strong_action_expert import (
    _ACTION_HISTORY_CONTEXT,
    _MEMORY_CONTEXT,
    _PROPRIO_CONTEXT,
    _TEXT_CONTEXT,
    _VISION_CONTEXT,
    _NUM_CONTEXT_TYPES,
    ActionExpertBlock,
    TemporalControlMemory,
    _sample_ids,
    _strip_optional_collator_axis,
)


class CausalActionStateExpert(nn.Module):
    """Encode online state slots, predict their future, and decode actions from them.

    The action tower only cross-attends to ``state_slots``. Task, vision,
    proprioception, and executed-action history are consumed by the shared state
    encoder and cannot bypass the state representation into the action decoder.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        model_dim: int,
        num_heads: int,
        state_layers: int,
        state_slots: int,
        action_layers: int,
        dynamics_layers: int,
        ffn_multiplier: int,
        state_dim: int,
        history_action_dim: int,
        target_action_dim: int,
        proprio_history_length: int,
        proprio_history_stride: int,
        action_history_length: int,
        memory_layers: int,
        memory_tokens: int,
        num_views: int,
        view_token_width: int,
        v2_enabled: bool = False,
        object_slots: int = 12,
        geometry_slots: int = 8,
        dynamics_slots: int = 8,
        causal_memory_slots: int = 4,
        spatial_decoder_layers: int = 2,
        relative_geometry_max_patches: int = 32,
        vision_lora_rank: int = 16,
        history_replay_enabled: bool = True,
    ) -> None:
        super().__init__()
        positive = {
            "hidden_size": hidden_size,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "state_layers": state_layers,
            "state_slots": state_slots,
            "action_layers": action_layers,
            "dynamics_layers": dynamics_layers,
            "ffn_multiplier": ffn_multiplier,
            "state_dim": state_dim,
            "history_action_dim": history_action_dim,
            "target_action_dim": target_action_dim,
            "proprio_history_length": proprio_history_length,
            "proprio_history_stride": proprio_history_stride,
            "action_history_length": action_history_length,
            "memory_layers": memory_layers,
            "memory_tokens": memory_tokens,
            "num_views": num_views,
            "view_token_width": view_token_width,
        }
        invalid = {name: value for name, value in positive.items() if int(value) < 1}
        if invalid:
            raise ValueError(f"Causal action-state dimensions must be positive: {invalid}")
        if model_dim % num_heads:
            raise ValueError("Causal action-state model_dim must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.history_action_dim = history_action_dim
        self.target_action_dim = target_action_dim
        self.proprio_history_length = proprio_history_length
        self.action_history_length = action_history_length
        self.state_slots = state_slots
        self.memory_tokens = memory_tokens
        self.num_views = num_views
        self.view_token_width = view_token_width
        self.v2_enabled = bool(v2_enabled)
        self.object_slots = int(object_slots)
        self.geometry_slots = int(geometry_slots)
        self.dynamics_slots = int(dynamics_slots)
        self.causal_memory_slots = int(causal_memory_slots)
        self.relative_geometry_max_patches = int(relative_geometry_max_patches)
        self.history_replay_enabled = bool(history_replay_enabled)
        self._intervention_mode = "normal"
        self._intervention_topk = min(8, state_slots)
        self._external_state_slots: torch.Tensor | None = None

        self.context_norm = nn.LayerNorm(hidden_size)
        self.context_in = nn.Linear(hidden_size, model_dim, bias=False)
        self.context_type_embed = nn.Embedding(_NUM_CONTEXT_TYPES, model_dim)
        self.view_embed = nn.Embedding(num_views + 1, model_dim)
        self.state_queries = nn.Parameter(torch.empty(state_slots, model_dim))
        self.state_layers = nn.ModuleList(
            ActionExpertBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                num_views=num_views,
                ffn_multiplier=ffn_multiplier,
            )
            for _ in range(state_layers)
        )
        self.state_norm = nn.LayerNorm(model_dim)

        if self.v2_enabled:
            factorized_slots = (
                self.object_slots
                + self.geometry_slots
                + self.dynamics_slots
                + self.causal_memory_slots
            )
            if factorized_slots != state_slots:
                raise ValueError(
                    "CWAM-v2 factorized slot counts must sum to state_slots: "
                    f"got {factorized_slots} != {state_slots}"
                )
            if min(
                self.object_slots,
                self.geometry_slots,
                self.dynamics_slots,
                self.causal_memory_slots,
                int(spatial_decoder_layers),
                self.relative_geometry_max_patches,
                int(vision_lora_rank),
            ) < 1:
                raise ValueError("CWAM-v2 slot counts, decoder depth, relation budget, and LoRA rank must be positive")
            if self.causal_memory_slots > memory_tokens:
                raise ValueError("CWAM-v2 causal memory slots cannot exceed temporal control-memory tokens")
            slot_types = torch.cat(
                (
                    torch.full((self.object_slots,), 0, dtype=torch.long),
                    torch.full((self.geometry_slots,), 1, dtype=torch.long),
                    torch.full((self.dynamics_slots,), 2, dtype=torch.long),
                    torch.full((self.causal_memory_slots,), 3, dtype=torch.long),
                )
            )
            self.register_buffer("slot_type_ids", slot_types, persistent=False)
            self.slot_type_embed = nn.Embedding(4, model_dim)
            self.memory_slot_norm = nn.LayerNorm(model_dim)
            self.memory_slot_in = nn.Linear(model_dim, model_dim, bias=False)

            self.vision_lora_norm = nn.LayerNorm(model_dim)
            self.vision_lora_down = nn.Linear(model_dim, int(vision_lora_rank), bias=False)
            self.vision_lora_up = nn.Linear(int(vision_lora_rank), model_dim, bias=False)
            self.vision_lora_scale = 1.0 / float(vision_lora_rank)

            self.spatial_query_mlp = nn.Sequential(
                nn.Linear(9, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim),
            )
            self.spatial_decoder_layers = nn.ModuleList(
                ActionExpertBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    num_views=num_views,
                    ffn_multiplier=ffn_multiplier,
                )
                for _ in range(int(spatial_decoder_layers))
            )
            self.spatial_out_norm = nn.LayerNorm(model_dim)
            self.history_replay_head = nn.Linear(model_dim, history_action_dim)

        self.control_memory = TemporalControlMemory(
            state_dim=state_dim,
            action_dim=history_action_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=memory_layers,
            num_memory_tokens=memory_tokens,
            proprio_stride=proprio_history_stride,
        )

        self.action_norm = nn.LayerNorm(hidden_size)
        self.action_in = nn.Linear(hidden_size, model_dim, bias=False)
        self.action_progress_mlp = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.action_layers = nn.ModuleList(
            ActionExpertBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                num_views=num_views,
                ffn_multiplier=ffn_multiplier,
            )
            for _ in range(action_layers)
        )
        self.action_out_norm = nn.LayerNorm(model_dim)
        self.action_out = nn.Linear(model_dim, hidden_size, bias=False)

        self.dynamics_action_norm = nn.LayerNorm(target_action_dim)
        self.dynamics_action_in = nn.Linear(target_action_dim, model_dim)
        self.dynamics_layers = nn.ModuleList(
            ActionExpertBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                num_views=num_views,
                ffn_multiplier=ffn_multiplier,
            )
            for _ in range(dynamics_layers)
        )
        self.dynamics_norm = nn.LayerNorm(model_dim)
        self.dynamics_out = nn.Linear(model_dim, model_dim)

    def init_weights(self) -> None:
        for norm in (
            self.context_norm,
            self.state_norm,
            self.action_norm,
            self.action_out_norm,
            self.dynamics_action_norm,
            self.dynamics_norm,
        ):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for module in (
            self.context_in,
            self.action_in,
            self.action_progress_mlp[0],
            self.action_progress_mlp[2],
            self.action_out,
            self.dynamics_action_in,
            self.dynamics_out,
        ):
            assert isinstance(module, nn.Linear)
            std = 1.0 / math.sqrt(max(module.in_features, 1))
            nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.context_type_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.view_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.state_queries, mean=0.0, std=0.02)
        if self.v2_enabled:
            nn.init.normal_(self.slot_type_embed.weight, mean=0.0, std=0.02)
            for norm in (self.memory_slot_norm, self.vision_lora_norm, self.spatial_out_norm):
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)
            for module in (
                self.memory_slot_in,
                self.vision_lora_down,
                self.spatial_query_mlp[0],
                self.spatial_query_mlp[2],
                self.history_replay_head,
            ):
                assert isinstance(module, nn.Linear)
                std = 1.0 / math.sqrt(max(module.in_features, 1))
                nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            nn.init.zeros_(self.vision_lora_up.weight)
        self.control_memory.init_weights()
        spatial_layers = tuple(self.spatial_decoder_layers) if self.v2_enabled else ()
        for layer in (*self.state_layers, *self.action_layers, *self.dynamics_layers, *spatial_layers):
            layer.init_weights()

    def configure_intervention(
        self,
        mode: str,
        *,
        topk: int | None = None,
        external_state_slots: torch.Tensor | None = None,
    ) -> None:
        """Configure one evaluation-only C intervention for the next request."""
        allowed = {"normal", "zero", "shuffle", "cross_episode", "selected", "complement"}
        if mode not in allowed:
            raise ValueError(f"Unsupported C intervention {mode!r}; expected one of {sorted(allowed)}")
        if topk is not None:
            if type(topk) is not int or not 1 <= topk <= self.state_slots:
                raise ValueError(f"C intervention topk must be in [1,{self.state_slots}], got {topk!r}")
            self._intervention_topk = topk
        if mode in {"shuffle", "cross_episode"} and external_state_slots is None:
            raise ValueError(f"C intervention {mode!r} requires external state slots")
        if external_state_slots is not None:
            if external_state_slots.ndim not in (2, 3):
                raise ValueError("External C state must have shape [S,D] or [B,S,D]")
            if external_state_slots.shape[-2:] != (self.state_slots, self.model_dim):
                raise ValueError(
                    "External C state shape mismatch: "
                    f"expected [...,{self.state_slots},{self.model_dim}], got {tuple(external_state_slots.shape)}"
                )
            external_state_slots = external_state_slots.detach()
        self._intervention_mode = mode
        self._external_state_slots = external_state_slots

    def _apply_external_intervention(self, state: torch.Tensor) -> torch.Tensor:
        if self._intervention_mode == "zero":
            return torch.zeros_like(state)
        if self._intervention_mode not in {"shuffle", "cross_episode"}:
            return state
        assert self._external_state_slots is not None
        external = self._external_state_slots
        if external.ndim == 2:
            external = external.unsqueeze(0)
        if external.shape[0] == 1 and state.shape[0] != 1:
            external = external.expand(state.shape[0], -1, -1)
        if external.shape != state.shape:
            raise ValueError(
                f"External C state batch mismatch: expected {tuple(state.shape)}, got {tuple(external.shape)}"
            )
        return external.to(device=state.device, dtype=state.dtype)

    def _vision_view_ids(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.numel() == 0:
            return torch.empty(0, device=positions.device, dtype=torch.long)
        horizontal = positions[:, 2].long()
        lo = horizontal.amin()
        total_width = horizontal.amax() - lo + 1
        content_width = self.num_views * self.view_token_width
        torch._assert(total_width >= content_width, "Configured camera-token width exceeds the visual canvas")
        view = torch.div(horizontal - lo, self.view_token_width, rounding_mode="floor")
        return view.clamp_(0, self.num_views - 1)

    def _prepare_control(
        self,
        *,
        sample_lens: list[int],
        proprio_history: list[torch.Tensor] | None,
        proprio_history_mask: list[torch.Tensor] | None,
        action_history: list[torch.Tensor] | None,
        action_history_mask: list[torch.Tensor] | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        fields = (proprio_history, proprio_history_mask, action_history, action_history_mask)
        if any(value is None for value in fields):
            raise ValueError("Causal action state requires proprio/action history tensors and masks")
        assert proprio_history is not None
        assert proprio_history_mask is not None
        assert action_history is not None
        assert action_history_mask is not None
        if not all(len(value) == len(sample_lens) for value in fields):
            raise ValueError("Causal action-state control fields must align with packed samples")

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
                "Causal action-state proprio contract mismatch: "
                f"expected [B,{self.proprio_history_length},{self.state_dim}], got {tuple(proprio.shape)}"
            )
        if history_action.shape[1:] != (self.action_history_length, self.history_action_dim):
            raise ValueError(
                "Causal action-state action-history contract mismatch: "
                f"expected [B,{self.action_history_length},{self.history_action_dim}], "
                f"got {tuple(history_action.shape)}"
            )
        current_state, action_tokens, memory_tokens, action_ages = self.control_memory(
            proprio,
            proprio_valid,
            history_action,
            history_action_valid,
        )
        return (
            current_state,
            action_tokens,
            memory_tokens,
            action_ages,
            history_action_valid,
            proprio,
            proprio_valid,
            history_action,
        )

    def _encode_state_slots(
        self,
        *,
        text_groups: list[torch.Tensor],
        vision_groups: list[torch.Tensor],
        vision_view_groups: list[torch.Tensor],
        current_state: torch.Tensor,
        action_memory_input: torch.Tensor,
        memory_tokens: torch.Tensor,
        action_ages: torch.Tensor,
        history_action_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_samples = len(text_groups)
        if len(vision_groups) != num_samples or len(vision_view_groups) != num_samples:
            raise ValueError("State text/vision groups must use the same batch size")
        device = current_state.device
        nonvisual_view = self.num_views
        max_age = float(max(self.action_history_length, 1))
        contexts: list[torch.Tensor] = []
        context_types: list[torch.Tensor] = []
        context_views: list[torch.Tensor] = []
        context_ages: list[torch.Tensor] = []
        context_valids: list[torch.Tensor] = []

        for sample_id, (text, vision, vision_views) in enumerate(
            zip(text_groups, vision_groups, vision_view_groups, strict=True)
        ):
            if vision.shape[0] < 1:
                raise ValueError(f"Sample {sample_id} has no visual state tokens")
            text_types = torch.full((text.shape[0],), _TEXT_CONTEXT, device=device, dtype=torch.long)
            vision_types = torch.full((vision.shape[0],), _VISION_CONTEXT, device=device, dtype=torch.long)
            state_types = torch.full((1,), _PROPRIO_CONTEXT, device=device, dtype=torch.long)
            history_types = torch.full(
                (self.action_history_length,), _ACTION_HISTORY_CONTEXT, device=device, dtype=torch.long
            )
            memory_types = torch.full((self.memory_tokens,), _MEMORY_CONTEXT, device=device, dtype=torch.long)
            types = torch.cat((text_types, vision_types, state_types, history_types, memory_types))
            views = torch.cat(
                (
                    torch.full_like(text_types, nonvisual_view),
                    vision_views,
                    torch.full_like(state_types, nonvisual_view),
                    torch.full_like(history_types, nonvisual_view),
                    torch.full_like(memory_types, nonvisual_view),
                )
            )
            context = torch.cat(
                (
                    text,
                    vision,
                    current_state[sample_id : sample_id + 1],
                    action_memory_input[sample_id],
                    memory_tokens[sample_id],
                ),
                dim=0,
            )
            context = context + self.context_type_embed(types) + self.view_embed(views)
            valid = torch.cat(
                (
                    torch.ones(text.shape[0] + vision.shape[0] + 1, device=device, dtype=torch.bool),
                    history_action_valid[sample_id],
                    torch.ones(self.memory_tokens, device=device, dtype=torch.bool),
                )
            )
            ages = torch.cat(
                (
                    torch.zeros(text.shape[0] + vision.shape[0] + 1, device=device),
                    action_ages,
                    torch.full((self.memory_tokens,), -1.0 / max_age, device=device),
                )
            )
            contexts.append(context)
            context_types.append(types)
            context_views.append(views)
            context_ages.append(ages)
            context_valids.append(valid)

        context_batch = pad_sequence(contexts, batch_first=True)
        type_batch = pad_sequence(context_types, batch_first=True)
        view_batch = pad_sequence(context_views, batch_first=True, padding_value=nonvisual_view)
        age_batch = pad_sequence(context_ages, batch_first=True)
        valid_batch = pad_sequence(context_valids, batch_first=True)
        slots = self.state_queries.to(context_batch.dtype).unsqueeze(0).expand(num_samples, -1, -1)
        if self.v2_enabled:
            slots = slots + self.slot_type_embed(self.slot_type_ids).to(slots.dtype).unsqueeze(0)
            memory_seed = self.memory_slot_in(self.memory_slot_norm(memory_tokens[:, : self.causal_memory_slots]))
            slots = slots.clone()
            slots[:, -self.causal_memory_slots :] = (
                slots[:, -self.causal_memory_slots :] + memory_seed.to(slots.dtype)
            )
        slot_valid = torch.ones((num_samples, self.state_slots), device=device, dtype=torch.bool)
        entropies: list[torch.Tensor] = []
        for layer in self.state_layers:
            slots, entropy, _context_importance = layer(
                slots,
                context_batch,
                action_valid=slot_valid,
                context_valid=valid_batch,
                context_type=type_batch,
                context_view=view_batch,
                context_age=age_batch,
            )
            entropies.append(entropy)
        return self.state_norm(slots), torch.stack(entropies).mean()

    @staticmethod
    def _normalized_position_features(positions: torch.Tensor) -> torch.Tensor:
        coordinates = positions.float()
        lo = coordinates.amin(dim=0, keepdim=True)
        span = (coordinates.amax(dim=0, keepdim=True) - lo).clamp_min(1.0)
        normalized = 2.0 * (coordinates - lo) / span - 1.0
        return torch.cat(
            (normalized, torch.sin(math.pi * normalized), torch.cos(math.pi * normalized)),
            dim=-1,
        )

    def _decode_spatial_patches(
        self,
        *,
        state: torch.Tensor,
        vision_target_groups: list[torch.Tensor],
        vision_view_groups: list[torch.Tensor],
        vision_position_groups: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.v2_enabled:
            zero = state.sum() * 0.0
            return zero, zero, zero
        if not (
            len(vision_target_groups) == len(vision_view_groups) == len(vision_position_groups) == state.shape[0]
        ):
            raise ValueError("CWAM-v2 spatial targets, views, positions, and state batch must align")

        queries: list[torch.Tensor] = []
        lengths: list[int] = []
        for targets, views, positions in zip(
            vision_target_groups,
            vision_view_groups,
            vision_position_groups,
            strict=True,
        ):
            if targets.shape[0] != views.shape[0] or targets.shape[0] != positions.shape[0]:
                raise ValueError("CWAM-v2 patch targets must align with view and position metadata")
            coordinate_query = self.spatial_query_mlp(
                self._normalized_position_features(positions).to(dtype=state.dtype)
            )
            queries.append(coordinate_query + self.view_embed(views).to(coordinate_query.dtype))
            lengths.append(targets.shape[0])

        query_batch = pad_sequence(queries, batch_first=True)
        patch_valid = (
            torch.arange(query_batch.shape[1], device=query_batch.device)[None]
            < torch.tensor(lengths, device=query_batch.device)[:, None]
        )
        state_types = torch.full(state.shape[:2], _MEMORY_CONTEXT, device=state.device, dtype=torch.long)
        state_views = torch.full(state.shape[:2], self.num_views, device=state.device, dtype=torch.long)
        state_ages = torch.zeros(state.shape[:2], device=state.device)

        def decode_with_slots(state_valid: torch.Tensor) -> torch.Tensor:
            decoded = query_batch
            for layer in self.spatial_decoder_layers:
                decoded, _entropy, _importance = layer(
                    decoded,
                    state,
                    action_valid=patch_valid,
                    context_valid=state_valid,
                    context_type=state_types,
                    context_view=state_views,
                    context_age=state_ages,
                )
            return self.spatial_out_norm(decoded)

        spatial_state_valid = torch.zeros(state.shape[:2], device=state.device, dtype=torch.bool)
        spatial_state_valid[:, : self.object_slots + self.geometry_slots] = True
        geometry_state_valid = torch.zeros_like(spatial_state_valid)
        geometry_state_valid[:, self.object_slots : self.object_slots + self.geometry_slots] = True
        reconstructed = decode_with_slots(spatial_state_valid)
        geometry_reconstructed = decode_with_slots(geometry_state_valid)

        spatial_losses: list[torch.Tensor] = []
        relation_losses: list[torch.Tensor] = []
        reconstruction_cosines: list[torch.Tensor] = []
        for sample_id, (targets, length) in enumerate(zip(vision_target_groups, lengths, strict=True)):
            predicted = reconstructed[sample_id, :length].float()
            geometry_predicted = geometry_reconstructed[sample_id, :length].float()
            target = targets.detach().float()
            predicted_unit = F.normalize(predicted, dim=-1)
            geometry_predicted_unit = F.normalize(geometry_predicted, dim=-1)
            target_unit = F.normalize(target, dim=-1)
            cosine = (predicted_unit * target_unit).sum(dim=-1)
            reconstruction_cosines.append(cosine.mean())
            spatial_losses.append(1.0 - cosine.mean())

            relation_count = min(length, self.relative_geometry_max_patches)
            if relation_count < 2:
                relation_losses.append(predicted.sum() * 0.0)
                continue
            relation_indexes = torch.linspace(
                0,
                length - 1,
                steps=relation_count,
                device=predicted.device,
            ).round().long().unique()
            predicted_relation = (
                geometry_predicted_unit[relation_indexes] @ geometry_predicted_unit[relation_indexes].T
            )
            target_relation = target_unit[relation_indexes] @ target_unit[relation_indexes].T
            off_diagonal = ~torch.eye(
                relation_indexes.numel(),
                device=predicted.device,
                dtype=torch.bool,
            )
            relation_losses.append(
                F.mse_loss(predicted_relation[off_diagonal], target_relation[off_diagonal])
            )
        return (
            torch.stack(spatial_losses).mean(),
            torch.stack(relation_losses).mean(),
            torch.stack(reconstruction_cosines).mean(),
        )

    def _history_replay_loss(
        self,
        *,
        text_groups: list[torch.Tensor],
        vision_groups: list[torch.Tensor],
        vision_view_groups: list[torch.Tensor],
        proprio: torch.Tensor,
        proprio_valid: torch.Tensor,
        history_action: torch.Tensor,
        history_action_valid: torch.Tensor,
    ) -> torch.Tensor:
        if not self.v2_enabled or not self.history_replay_enabled:
            return vision_groups[0].sum() * 0.0
        replay_valid = history_action_valid.clone()
        replay_targets = torch.zeros(
            (history_action.shape[0], self.history_action_dim),
            device=history_action.device,
            dtype=history_action.dtype,
        )
        replay_samples = torch.zeros(history_action.shape[0], device=history_action.device, dtype=torch.bool)
        for sample_id in range(history_action.shape[0]):
            valid_indexes = torch.nonzero(replay_valid[sample_id], as_tuple=False).flatten()
            if valid_indexes.numel() == 0:
                continue
            held_out = valid_indexes[-1]
            replay_targets[sample_id] = history_action[sample_id, held_out].detach()
            replay_valid[sample_id, held_out] = False
            replay_samples[sample_id] = True
        if not bool(replay_samples.any().item()):
            return history_action.sum() * 0.0

        current_state, action_tokens, memory_tokens, action_ages = self.control_memory(
            proprio,
            proprio_valid,
            history_action,
            replay_valid,
        )
        replay_state, _ = self._encode_state_slots(
            text_groups=text_groups,
            vision_groups=vision_groups,
            vision_view_groups=vision_view_groups,
            current_state=current_state,
            action_memory_input=action_tokens,
            memory_tokens=memory_tokens,
            action_ages=action_ages,
            history_action_valid=replay_valid,
        )
        memory_summary = replay_state[:, -self.causal_memory_slots :].mean(dim=1)
        predicted = self.history_replay_head(memory_summary)
        return F.smooth_l1_loss(
            predicted[replay_samples].float(),
            replay_targets[replay_samples].float(),
        )

    def _decode_actions(
        self,
        *,
        action_seed_hidden: torch.Tensor,
        action_indexes: torch.Tensor,
        sample_lens: list[int],
        state_slots: torch.Tensor,
        state_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_sample_ids = _sample_ids(action_indexes, sample_lens)
        projected = self.action_in(self.action_norm(action_seed_hidden))
        groups: list[torch.Tensor] = []
        lengths: list[int] = []
        for sample_id in range(len(sample_lens)):
            group = projected[action_sample_ids == sample_id]
            if group.shape[0] < 1:
                raise ValueError(f"Packed sample {sample_id} has no action tokens")
            progress = torch.arange(group.shape[0], device=group.device, dtype=torch.float32)
            progress = progress / max(group.shape[0] - 1, 1)
            groups.append(group + self.action_progress_mlp(progress[:, None].to(group.dtype)))
            lengths.append(group.shape[0])
        action_batch = pad_sequence(groups, batch_first=True)
        action_valid = (
            torch.arange(action_batch.shape[1], device=action_batch.device)[None]
            < torch.tensor(lengths, device=action_batch.device)[:, None]
        )
        if state_valid is None:
            state_valid = torch.ones(state_slots.shape[:2], device=action_batch.device, dtype=torch.bool)
        elif state_valid.shape != state_slots.shape[:2]:
            raise ValueError("state_valid must align with the C slot batch")
        if not bool(torch.all(state_valid.any(dim=1)).item()):
            raise ValueError("Every action sample must retain at least one valid C slot")
        state_types = torch.full(state_slots.shape[:2], _MEMORY_CONTEXT, device=action_batch.device, dtype=torch.long)
        state_views = torch.full(state_slots.shape[:2], self.num_views, device=action_batch.device, dtype=torch.long)
        state_ages = torch.zeros(state_slots.shape[:2], device=action_batch.device)
        entropies: list[torch.Tensor] = []
        importances: list[torch.Tensor] = []
        for layer in self.action_layers:
            action_batch, entropy, importance = layer(
                action_batch,
                state_slots,
                action_valid=action_valid,
                context_valid=state_valid,
                context_type=state_types,
                context_view=state_views,
                context_age=state_ages,
            )
            entropies.append(entropy)
            importances.append(importance)
        hidden = self.action_out(self.action_out_norm(action_batch))
        ordered = torch.cat([hidden[i, :length] for i, length in enumerate(lengths)], dim=0)
        return ordered, torch.stack(entropies).mean(), torch.stack(importances).mean(dim=0)

    def _predict_future_state(
        self,
        current_state: torch.Tensor,
        clean_actions: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = _strip_optional_collator_axis(clean_actions, expected_ndim=2)
        if len(actions) != current_state.shape[0]:
            raise ValueError("Clean action chunks must align with state samples")
        for action in actions:
            if action.ndim != 2 or action.shape[-1] != self.target_action_dim:
                raise ValueError(
                    f"Clean action chunks must be [T,{self.target_action_dim}], got {tuple(action.shape)}"
                )
        parameter_dtype = self.dynamics_action_in.weight.dtype
        action_batch = pad_sequence(
            [
                self.dynamics_action_in(self.dynamics_action_norm(action.to(dtype=parameter_dtype)))
                for action in actions
            ],
            batch_first=True,
        )
        action_valid = (
            torch.arange(action_batch.shape[1], device=action_batch.device)[None]
            < torch.tensor([action.shape[0] for action in actions], device=action_batch.device)[:, None]
        )
        state_valid = torch.ones(current_state.shape[:2], device=current_state.device, dtype=torch.bool)
        action_types = torch.full(action_batch.shape[:2], _ACTION_HISTORY_CONTEXT, device=current_state.device, dtype=torch.long)
        action_views = torch.full(action_batch.shape[:2], self.num_views, device=current_state.device, dtype=torch.long)
        progress = torch.arange(action_batch.shape[1], device=current_state.device, dtype=torch.float32)
        action_ages = progress[None].expand(action_batch.shape[0], -1) / max(action_batch.shape[1] - 1, 1)
        predicted = current_state
        entropies: list[torch.Tensor] = []
        for layer in self.dynamics_layers:
            predicted, entropy, _context_importance = layer(
                predicted,
                action_batch,
                action_valid=state_valid,
                context_valid=action_valid,
                context_type=action_types,
                context_view=action_views,
                context_age=action_ages,
            )
            entropies.append(entropy)
        predicted = self.dynamics_out(self.dynamics_norm(predicted))
        return predicted, torch.stack(entropies).mean()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        action_seed_hidden: torch.Tensor,
        action_indexes: torch.Tensor,
        text_indexes: torch.Tensor,
        vision_indexes: torch.Tensor,
        position_ids: torch.Tensor,
        sample_lens: list[int],
        proprio_history: list[torch.Tensor] | None,
        proprio_history_mask: list[torch.Tensor] | None,
        action_history: list[torch.Tensor] | None,
        action_history_mask: list[torch.Tensor] | None,
        future_vision_groups: list[torch.Tensor] | None = None,
        future_view_groups: list[torch.Tensor] | None = None,
        clean_actions: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states must be [N,{self.hidden_size}], got {tuple(hidden_states.shape)}")
        if action_seed_hidden.shape != (action_indexes.numel(), self.hidden_size):
            raise ValueError("action_seed_hidden must be aligned with action_indexes")
        if vision_indexes.numel() < 1 or action_indexes.numel() < 1:
            raise ValueError("Causal action state requires current vision and action tokens")

        control = self._prepare_control(
            sample_lens=sample_lens,
            proprio_history=proprio_history,
            proprio_history_mask=proprio_history_mask,
            action_history=action_history,
            action_history_mask=action_history_mask,
        )
        (
            current_state,
            action_memory_input,
            memory_tokens,
            action_ages,
            history_action_valid,
            proprio,
            proprio_valid,
            history_action,
        ) = control
        text_sample_ids = _sample_ids(text_indexes, sample_lens)
        vision_sample_ids = _sample_ids(vision_indexes, sample_lens)
        text_projected = self.context_in(self.context_norm(hidden_states[text_indexes]))
        vision_base = self.context_in(self.context_norm(hidden_states[vision_indexes]))
        if self.v2_enabled:
            vision_projected = vision_base + self.vision_lora_up(
                F.silu(self.vision_lora_down(self.vision_lora_norm(vision_base)))
            ) * self.vision_lora_scale
            vision_lora_rms = (vision_projected - vision_base).float().square().mean().sqrt()
        else:
            vision_projected = vision_base
            vision_lora_rms = vision_base.sum() * 0.0
        text_groups = [text_projected[text_sample_ids == i] for i in range(len(sample_lens))]
        vision_groups = [vision_projected[vision_sample_ids == i] for i in range(len(sample_lens))]
        view_ids = self._vision_view_ids(position_ids[:, vision_indexes].T)
        view_groups = [view_ids[vision_sample_ids == i] for i in range(len(sample_lens))]
        vision_positions = position_ids[:, vision_indexes].T
        position_groups = [vision_positions[vision_sample_ids == i] for i in range(len(sample_lens))]
        vision_target_groups = [vision_base[vision_sample_ids == i].detach() for i in range(len(sample_lens))]

        state, state_entropy = self._encode_state_slots(
            text_groups=text_groups,
            vision_groups=vision_groups,
            vision_view_groups=view_groups,
            current_state=current_state,
            action_memory_input=action_memory_input,
            memory_tokens=memory_tokens,
            action_ages=action_ages,
            history_action_valid=history_action_valid,
        )
        action_state = self._apply_external_intervention(state)
        action_hidden, action_entropy, slot_importance = self._decode_actions(
            action_seed_hidden=action_seed_hidden,
            action_indexes=action_indexes,
            sample_lens=sample_lens,
            state_slots=action_state,
        )
        retained_fraction = state.new_tensor(1.0)
        retained_importance = state.new_tensor(1.0)
        if self._intervention_mode in {"selected", "complement"}:
            topk = min(self._intervention_topk, self.state_slots)
            selected = torch.zeros_like(slot_importance, dtype=torch.bool)
            selected.scatter_(1, slot_importance.topk(topk, dim=1).indices, True)
            state_valid = selected if self._intervention_mode == "selected" else ~selected
            if not torch.all(state_valid.any(dim=1)):
                raise ValueError("Complement intervention removed every C slot")
            action_hidden, action_entropy, _ = self._decode_actions(
                action_seed_hidden=action_seed_hidden,
                action_indexes=action_indexes,
                sample_lens=sample_lens,
                state_slots=action_state,
                state_valid=state_valid,
            )
            retained_fraction = state_valid.float().mean()
            retained_importance = (slot_importance * state_valid).sum(dim=1).mean()

        spatial_loss, relative_geometry_loss, spatial_cosine = self._decode_spatial_patches(
            state=state,
            vision_target_groups=vision_target_groups,
            vision_view_groups=view_groups,
            vision_position_groups=position_groups,
        )
        history_replay_loss = self._history_replay_loss(
            text_groups=text_groups,
            vision_groups=vision_groups,
            vision_view_groups=view_groups,
            proprio=proprio,
            proprio_valid=proprio_valid,
            history_action=history_action,
            history_action_valid=history_action_valid,
        )

        zero = state.sum() * 0.0
        dynamics_loss = zero
        dynamics_cosine = zero
        dynamics_entropy = zero
        if future_vision_groups is not None or future_view_groups is not None or clean_actions is not None:
            if future_vision_groups is None or future_view_groups is None or clean_actions is None:
                raise ValueError("Future vision, future views, and clean actions must be provided together")
            with torch.no_grad():
                projected_future = [self.context_in(self.context_norm(group)) for group in future_vision_groups]
                target_state, _ = self._encode_state_slots(
                    text_groups=[group.detach() for group in text_groups],
                    vision_groups=projected_future,
                    vision_view_groups=future_view_groups,
                    current_state=current_state.detach(),
                    action_memory_input=action_memory_input.detach(),
                    memory_tokens=memory_tokens.detach(),
                    action_ages=action_ages,
                    history_action_valid=history_action_valid,
                )
            predicted_state, dynamics_entropy = self._predict_future_state(state, clean_actions)
            if self.v2_enabled:
                dynamics_start = self.object_slots + self.geometry_slots
                dynamics_stop = dynamics_start + self.dynamics_slots
                predicted_state = predicted_state[:, dynamics_start:dynamics_stop]
                target_state = target_state[:, dynamics_start:dynamics_stop]
            predicted_unit = F.normalize(predicted_state.float(), dim=-1)
            target_unit = F.normalize(target_state.float().detach(), dim=-1)
            dynamics_cosine = (predicted_unit * target_unit).sum(dim=-1).mean()
            dynamics_loss = 1.0 - dynamics_cosine

        state_std = state.float().std(dim=(0, 1), unbiased=False).mean()
        variance_loss = F.relu(state.new_tensor(0.1) - state.float().std(dim=(0, 1), unbiased=False)).square().mean()
        metrics = {
            "dynamics_loss": dynamics_loss,
            "variance_loss": variance_loss,
            "dynamics_cosine": dynamics_cosine,
            "state_std": state_std,
            "state_rms": state.float().square().mean().sqrt(),
            "state_attention_entropy": state_entropy,
            "action_attention_entropy": action_entropy,
            "action_slot_retained_fraction": retained_fraction,
            "action_slot_retained_importance": retained_importance,
            "dynamics_attention_entropy": dynamics_entropy,
            "spatial_reconstruction_loss": spatial_loss,
            "spatial_reconstruction_cosine": spatial_cosine,
            "relative_geometry_loss": relative_geometry_loss,
            "history_replay_loss": history_replay_loss,
            "vision_lora_rms": vision_lora_rms,
            "causal_memory_rms": state[:, -self.causal_memory_slots :].float().square().mean().sqrt()
            if self.v2_enabled
            else zero,
        }
        return action_hidden, metrics
