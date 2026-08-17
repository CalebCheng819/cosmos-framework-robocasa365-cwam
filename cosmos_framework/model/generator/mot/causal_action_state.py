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
        self.control_memory.init_weights()
        for layer in (*self.state_layers, *self.action_layers, *self.dynamics_layers):
            layer.init_weights()

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return current_state, action_tokens, memory_tokens, action_ages, history_action_valid

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
        slot_valid = torch.ones((num_samples, self.state_slots), device=device, dtype=torch.bool)
        entropies: list[torch.Tensor] = []
        for layer in self.state_layers:
            slots, entropy = layer(
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

    def _decode_actions(
        self,
        *,
        action_seed_hidden: torch.Tensor,
        action_indexes: torch.Tensor,
        sample_lens: list[int],
        state_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        state_valid = torch.ones(state_slots.shape[:2], device=action_batch.device, dtype=torch.bool)
        state_types = torch.full(state_slots.shape[:2], _MEMORY_CONTEXT, device=action_batch.device, dtype=torch.long)
        state_views = torch.full(state_slots.shape[:2], self.num_views, device=action_batch.device, dtype=torch.long)
        state_ages = torch.zeros(state_slots.shape[:2], device=action_batch.device)
        entropies: list[torch.Tensor] = []
        for layer in self.action_layers:
            action_batch, entropy = layer(
                action_batch,
                state_slots,
                action_valid=action_valid,
                context_valid=state_valid,
                context_type=state_types,
                context_view=state_views,
                context_age=state_ages,
            )
            entropies.append(entropy)
        hidden = self.action_out(self.action_out_norm(action_batch))
        ordered = torch.cat([hidden[i, :length] for i, length in enumerate(lengths)], dim=0)
        return ordered, torch.stack(entropies).mean()

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
            predicted, entropy = layer(
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
        current_state, action_memory_input, memory_tokens, action_ages, history_action_valid = control
        text_sample_ids = _sample_ids(text_indexes, sample_lens)
        vision_sample_ids = _sample_ids(vision_indexes, sample_lens)
        text_projected = self.context_in(self.context_norm(hidden_states[text_indexes]))
        vision_projected = self.context_in(self.context_norm(hidden_states[vision_indexes]))
        text_groups = [text_projected[text_sample_ids == i] for i in range(len(sample_lens))]
        vision_groups = [vision_projected[vision_sample_ids == i] for i in range(len(sample_lens))]
        view_ids = self._vision_view_ids(position_ids[:, vision_indexes].T)
        view_groups = [view_ids[vision_sample_ids == i] for i in range(len(sample_lens))]

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
        action_hidden, action_entropy = self._decode_actions(
            action_seed_hidden=action_seed_hidden,
            action_indexes=action_indexes,
            sample_lens=sample_lens,
            state_slots=state,
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
            "dynamics_attention_entropy": dynamics_entropy,
        }
        return action_hidden, metrics
