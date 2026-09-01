# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""ImageWAM-compatible multi-robot conditions for Cosmos action tokens."""

from __future__ import annotations

import math

import torch
from torch import nn

from cosmos_framework.data.generator.sequence_packing import MultiRobotConditionData


def _sinusoidal_hub_seeds(num_hubs: int, hidden_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create hub-index seeds without introducing robot identity embeddings."""
    positions = torch.arange(num_hubs, device=device, dtype=torch.float32).unsqueeze(1)
    half = hidden_size // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    ).unsqueeze(0)
    seeds = torch.cat((torch.sin(positions * frequencies), torch.cos(positions * frequencies)), dim=1)
    if seeds.shape[1] < hidden_size:
        seeds = torch.nn.functional.pad(seeds, (0, hidden_size - seeds.shape[1]))
    return seeds[:, :hidden_size].to(dtype=dtype)


class DynamicHubTokenCommunication(nn.Module):
    """Permutation-equivariant agent communication through a dynamic hub set."""

    def __init__(self, hidden_size: int, num_heads: int, hub_ratio: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.hub_ratio = hub_ratio
        self.agent_norm = nn.LayerNorm(hidden_size)
        self.hub_seed_proj = nn.Linear(hidden_size, hidden_size)
        self.agent_to_hub = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.hub_self = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.hub_ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.hub_to_action = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        # FSDP2 ``fully_shard`` rejects zero-dimensional parameters.  Keep gates
        # as one-element vectors; broadcasting preserves the scalar semantics.
        self.output_gate = nn.Parameter(torch.zeros(1))

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        # action_tokens: [N,H,D]. Mean-over-horizon summaries contain no agent IDs.
        agent_summary = self.agent_norm(action_tokens.mean(dim=1)).unsqueeze(0)
        num_hubs = max(1, math.ceil(self.hub_ratio * action_tokens.shape[0]))
        seeds = _sinusoidal_hub_seeds(
            num_hubs, self.hidden_size, action_tokens.device, action_tokens.dtype
        ).unsqueeze(0)
        hubs = self.hub_seed_proj(seeds)
        hubs = hubs + self.agent_to_hub(hubs, agent_summary, agent_summary, need_weights=False)[0]
        hubs = hubs + self.hub_self(hubs, hubs, hubs, need_weights=False)[0]
        hubs = hubs + self.hub_ffn(hubs)
        flat = action_tokens.reshape(1, -1, self.hidden_size)
        communicated = self.hub_to_action(flat, hubs, hubs, need_weights=False)[0]
        return action_tokens + torch.tanh(self.output_gate) * communicated.reshape_as(action_tokens)


class SpatialGaussianCrossAttention(nn.Module):
    """Per-agent action-to-Gaussian spatial cross attention over a 7x10 grid."""

    def __init__(self, hidden_size: int, num_heads: int, gaussian_channels: int = 13):
        super().__init__()
        self.hidden_size = hidden_size
        stem_dim = max(64, hidden_size // 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(gaussian_channels, stem_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(stem_dim, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.row_position = nn.Parameter(torch.zeros(7, hidden_size))
        self.col_position = nn.Parameter(torch.zeros(10, hidden_size))
        self.action_norm = nn.LayerNorm(hidden_size)
        self.gaussian_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.output_gate = nn.Parameter(torch.tensor([0.1]))

    def forward(self, action_tokens: torch.Tensor, gaussian: torch.Tensor) -> torch.Tensor:
        if gaussian.ndim != 4 or tuple(gaussian.shape[1:]) != (13, 28, 40):
            raise ValueError(f"agent_gaussian must have shape [N,13,28,40], got {tuple(gaussian.shape)}")
        spatial = self.encoder(gaussian)
        if tuple(spatial.shape[-2:]) != (7, 10):
            raise RuntimeError(f"Gaussian encoder must produce 7x10 tokens, got {tuple(spatial.shape[-2:])}")
        spatial = spatial.permute(0, 2, 3, 1)
        spatial = spatial + self.row_position[None, :, None, :] + self.col_position[None, None, :, :]
        spatial = self.gaussian_norm(spatial.reshape(spatial.shape[0], 70, self.hidden_size))
        queried = self.cross_attention(self.action_norm(action_tokens), spatial, spatial, need_weights=False)[0]
        queried = queried + self.ffn(queried)
        return action_tokens + self.output_gate * queried


class MultiRobotActionConditioner(nn.Module):
    """Apply shared state, spatial Gaussian, and HubToken conditions sample-wise."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        state_dim: int = 18,
        gaussian_channels: int = 13,
        hub_ratio: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.state_gate = nn.Parameter(torch.zeros(1))
        self.gaussian_attention = SpatialGaussianCrossAttention(hidden_size, num_heads, gaussian_channels)
        self.hub_communication = DynamicHubTokenCommunication(hidden_size, num_heads, hub_ratio)

    def init_weights(self) -> None:
        """Initialize newly added modules after meta-device materialization."""
        for module in self.modules():
            if module is self:
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
            elif isinstance(module, nn.MultiheadAttention):
                module._reset_parameters()
        nn.init.normal_(self.gaussian_attention.row_position, std=0.02)
        nn.init.normal_(self.gaussian_attention.col_position, std=0.02)
        nn.init.zeros_(self.state_gate)
        nn.init.constant_(self.gaussian_attention.output_gate, 0.1)
        nn.init.zeros_(self.hub_communication.output_gate)

    def forward(
        self,
        packed_action_tokens: torch.Tensor,
        token_shapes: list[tuple[int, ...]],
        condition: MultiRobotConditionData,
    ) -> torch.Tensor:
        if len(token_shapes) != len(condition.agent_count):
            raise ValueError("Multi-robot condition count must match action payload count")
        lengths = [int(shape[0]) for shape in token_shapes]
        if sum(lengths) != packed_action_tokens.shape[0]:
            raise ValueError("Packed action token length does not match action token_shapes")
        gaussian_values = condition.agent_gaussian
        if gaussian_values is None or len(gaussian_values) != len(lengths):
            raise ValueError("Spatial Gaussian condition is required for every multi-robot action sample")

        outputs = []
        for index, tokens in enumerate(torch.split(packed_action_tokens, lengths, dim=0)):
            num_agents = int(condition.agent_count[index].item())
            horizon = int(condition.horizon_per_agent[index].item())
            if num_agents < 1 or num_agents > 4 or tokens.shape[0] != num_agents * horizon:
                raise ValueError(
                    f"Invalid agent-major action contract: N={num_agents}, H={horizon}, T={tokens.shape[0]}"
                )
            state = condition.agent_state[index]
            if tuple(state.shape) != (num_agents, self.state_dim):
                raise ValueError(
                    f"agent_state must have shape [{num_agents},{self.state_dim}], got {tuple(state.shape)}"
                )
            gaussian = gaussian_values[index]
            if gaussian.shape[0] != num_agents:
                raise ValueError("agent_gaussian first dimension must equal agent_count")

            per_agent = tokens.reshape(num_agents, horizon, self.hidden_size)
            state_embed = self.state_encoder(state.to(device=tokens.device, dtype=tokens.dtype))
            per_agent = per_agent + torch.tanh(self.state_gate) * state_embed[:, None, :]
            per_agent = self.gaussian_attention(
                per_agent, gaussian.to(device=tokens.device, dtype=tokens.dtype)
            )
            per_agent = self.hub_communication(per_agent)
            outputs.append(per_agent.reshape(-1, self.hidden_size))
        return torch.cat(outputs, dim=0)
