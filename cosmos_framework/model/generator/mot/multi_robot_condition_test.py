# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace

import torch

from cosmos_framework.data.generator.sequence_packing import MultiRobotConditionData
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.mot.multi_robot_condition import MultiRobotActionConditioner


def _condition(state: torch.Tensor, gaussian: torch.Tensor) -> MultiRobotConditionData:
    return MultiRobotConditionData(
        agent_count=[torch.tensor(state.shape[0])],
        horizon_per_agent=[torch.tensor(4)],
        agent_state=[state],
        agent_gaussian=[gaussian],
    )


def test_multi_robot_conditioner_is_agent_permutation_equivariant() -> None:
    torch.manual_seed(7)
    module = MultiRobotActionConditioner(hidden_size=16, num_heads=4)
    module.eval()
    module.state_gate.data.fill_(0.4)
    module.gaussian_attention.output_gate.data.fill_(0.2)
    module.hub_communication.output_gate.data.fill_(0.3)
    tokens = torch.randn(3, 4, 16)
    state = torch.randn(3, 18)
    gaussian = torch.randn(3, 13, 28, 40)
    permutation = torch.tensor([2, 0, 1])

    output = module(tokens.reshape(-1, 16), [(12,)], _condition(state, gaussian)).reshape(3, 4, 16)
    permuted_output = module(
        tokens[permutation].reshape(-1, 16),
        [(12,)],
        _condition(state[permutation], gaussian[permutation]),
    ).reshape(3, 4, 16)
    torch.testing.assert_close(permuted_output, output[permutation], atol=1.0e-5, rtol=1.0e-5)


def test_multi_robot_conditioner_has_no_scalar_parameters_for_fsdp2() -> None:
    module = MultiRobotActionConditioner(hidden_size=16, num_heads=4)
    scalar_names = [name for name, parameter in module.named_parameters() if parameter.ndim == 0]
    assert scalar_names == []


def test_action_token_mask_excludes_invalid_robot_tokens_from_loss() -> None:
    pred = [torch.ones(2, 3)]
    target = [torch.zeros(2, 3)]
    condition_mask = [torch.zeros(2, 1)]
    rectified_flow = SimpleNamespace(
        train_time_weight=lambda timesteps, _kwargs: torch.ones_like(timesteps, dtype=torch.float32)
    )
    loss, per_instance = compute_flow_matching_loss(
        pred,
        target,
        condition_mask,
        timesteps=torch.ones(1, 1),
        has_valid_tokens=True,
        rectified_flow=rectified_flow,
        tensor_kwargs_fp32={},
        token_mask=[torch.tensor([1.0, 0.0])],
    )
    torch.testing.assert_close(loss, torch.tensor(0.5))
    torch.testing.assert_close(per_instance, torch.tensor([0.5]))
