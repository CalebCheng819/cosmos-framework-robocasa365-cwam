# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU regression tests for the explicit multi-robot inference contract."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.omni_mot_model import attach_multi_robot_action_contract


def _batch() -> dict[str, list[list[torch.Tensor]]]:
    return {
        "agent_count": [[torch.tensor(2)]],
        "action_horizon_per_agent": [[torch.tensor(32)]],
        "agent_state": [[torch.zeros(2, 18)]],
        "agent_gaussian": [[torch.zeros(2, 13, 28, 40)]],
        "action_token_mask": [[torch.ones(64)]],
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_attach_multi_robot_action_contract_accepts_service_batch_nesting() -> None:
    packed = SimpleNamespace(
        action=SimpleNamespace(token_shapes=[(64, 8)], token_mask=None),
        multi_robot_condition=None,
    )

    attach_multi_robot_action_contract(packed, _batch(), action_sample_indices=[0], batch_size=1)

    assert packed.multi_robot_condition is not None
    assert packed.multi_robot_condition.agent_count[0].item() == 2
    assert tuple(packed.multi_robot_condition.agent_state[0].shape) == (2, 18)
    assert tuple(packed.multi_robot_condition.agent_gaussian[0].shape) == (2, 13, 28, 40)
    assert tuple(packed.action.token_mask[0].shape) == (64,)


@pytest.mark.L0
@pytest.mark.CPU
def test_attach_multi_robot_action_contract_rejects_partial_condition() -> None:
    packed = SimpleNamespace(
        action=SimpleNamespace(token_shapes=[(64, 8)], token_mask=None),
        multi_robot_condition=None,
    )
    batch = _batch()
    del batch["agent_gaussian"]

    with pytest.raises(KeyError, match="agent_gaussian"):
        attach_multi_robot_action_contract(packed, batch, action_sample_indices=[0], batch_size=1)
