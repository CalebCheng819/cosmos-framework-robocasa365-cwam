# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for generic DCP warm-start state-dict transformations."""

import pytest
import torch

from cosmos_framework.checkpoint.dcp import _copy_reg_weights_to_ema, _state_dict_has_ema_tree


@pytest.mark.L0
@pytest.mark.CPU
def test_copy_regular_weights_to_ema_resets_every_matching_tensor() -> None:
    state_dict = {
        "net.layer.weight": torch.tensor([1.0, 2.0]),
        "net.layer.bias": torch.tensor([3.0]),
        "net_ema.layer.weight": torch.tensor([-1.0, -2.0]),
        "net_ema.layer.bias": torch.tensor([-3.0]),
        "other": torch.tensor([9.0]),
    }

    _copy_reg_weights_to_ema(state_dict)

    torch.testing.assert_close(state_dict["net_ema.layer.weight"], state_dict["net.layer.weight"])
    torch.testing.assert_close(state_dict["net_ema.layer.bias"], state_dict["net.layer.bias"])
    torch.testing.assert_close(state_dict["other"], torch.tensor([9.0]))


@pytest.mark.L0
@pytest.mark.CPU
def test_copy_regular_weights_to_ema_rejects_an_incomplete_ema_tree() -> None:
    with pytest.raises(AssertionError, match="net_ema.layer.weight"):
        _copy_reg_weights_to_ema({"net.layer.weight": torch.tensor([1.0])})


@pytest.mark.L0
@pytest.mark.CPU
def test_state_dict_has_ema_tree_distinguishes_ema_disabled_model() -> None:
    assert _state_dict_has_ema_tree(
        {
            "net.layer.weight": torch.tensor([1.0]),
            "net_ema.layer.weight": torch.tensor([1.0]),
        }
    )
    assert not _state_dict_has_ema_tree({"net.layer.weight": torch.tensor([1.0])})
