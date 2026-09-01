# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import base64

import numpy as np
import pytest
import torch

from cosmos_framework.scripts.action_policy_server_robofactory import RoboFactoryActionServerArgs
from cosmos_framework.scripts.robofactory_action_bridge import (
    ACTION_HORIZON_PER_AGENT,
    chronological_views,
    decode_array,
)


class _OffsetNormalizer:
    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action + 1000.0


def test_server_args_disable_compile_for_dynamic_multi_robot_metadata() -> None:
    args = RoboFactoryActionServerArgs.model_validate(
        {"checkpoint": {"checkpoint_path": "/checkpoint/model"}}
    )

    overrides = args.build_setup_overrides()

    assert overrides.guardrails is False
    assert overrides.use_torch_compile is False
    assert overrides.use_cuda_graphs is False


def test_decode_array_preserves_shape_and_values() -> None:
    source = np.arange(2 * 3, dtype=np.dtype("<f4")).reshape(2, 3)
    payload = base64.b64encode(source.tobytes()).decode("ascii")
    decoded = decode_array(payload, dtype=np.dtype("<f4"), shape=(2, 3), label="state")
    torch.testing.assert_close(decoded, torch.from_numpy(source))


def test_decode_array_rejects_wrong_size_and_non_finite() -> None:
    with pytest.raises(ValueError, match="expected"):
        decode_array("AA==", dtype=np.dtype("<f4"), shape=(2, 3), label="state")
    source = np.zeros((2, 3), dtype=np.dtype("<f4"))
    source[0, 0] = np.nan
    payload = base64.b64encode(source.tobytes()).decode("ascii")
    with pytest.raises(ValueError, match="non-finite"):
        decode_array(payload, dtype=np.dtype("<f4"), shape=(2, 3), label="state")


def test_chronological_views_transposes_agent_major_exactly_once() -> None:
    agent_count = 2
    source = torch.arange(
        agent_count * ACTION_HORIZON_PER_AGENT * 8, dtype=torch.float32
    ).reshape(agent_count * ACTION_HORIZON_PER_AGENT, 8)
    normalized, raw = chronological_views(
        source, _OffsetNormalizer(), agent_count=agent_count
    )
    expected_raw = source.reshape(agent_count, ACTION_HORIZON_PER_AGENT, 8)
    expected_raw = expected_raw.permute(1, 0, 2).reshape(ACTION_HORIZON_PER_AGENT, 16)
    torch.testing.assert_close(raw, expected_raw)
    torch.testing.assert_close(normalized, expected_raw + 1000.0)


def test_chronological_views_rejects_wrong_action_shape() -> None:
    with pytest.raises(ValueError, match="Generated action"):
        chronological_views(torch.zeros(32, 8), _OffsetNormalizer(), agent_count=2)
