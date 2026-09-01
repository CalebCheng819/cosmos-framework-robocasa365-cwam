# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dependency-light codecs for the Cosmos3/RoboFactory HTTP bridge."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import torch


ACTION_HORIZON_PER_AGENT = 32


def decode_array(
    payload: Any, *, dtype: np.dtype, shape: tuple[int, ...], label: str
) -> torch.Tensor:
    if not isinstance(payload, str):
        raise ValueError(f"'{label}' must be a base64 string")
    raw = base64.b64decode(payload, validate=True)
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{label} contains {len(raw)} bytes, expected {expected_bytes} for {shape}"
        )
    array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    tensor = torch.from_numpy(array)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{label} contains non-finite values")
    return tensor


def chronological_views(
    raw_agent_major: torch.Tensor,
    normalizer: Any,
    *,
    agent_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = (agent_count * ACTION_HORIZON_PER_AGENT, 8)
    if tuple(raw_agent_major.shape) != expected:
        raise ValueError(
            f"Generated action must be {expected}, got {tuple(raw_agent_major.shape)}"
        )
    normalized_agent_major = normalizer.normalize_action(raw_agent_major)
    raw = raw_agent_major.reshape(agent_count, ACTION_HORIZON_PER_AGENT, 8)
    normalized = normalized_agent_major.reshape(
        agent_count, ACTION_HORIZON_PER_AGENT, 8
    )
    return (
        normalized.permute(1, 0, 2)
        .reshape(ACTION_HORIZON_PER_AGENT, agent_count * 8)
        .contiguous(),
        raw.permute(1, 0, 2)
        .reshape(ACTION_HORIZON_PER_AGENT, agent_count * 8)
        .contiguous(),
    )
