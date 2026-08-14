# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU unit tests for OmniMoT's shared train/validation objective wrapper."""

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel, optional_model_config_value


class _FlowMatchingHarness:
    def __init__(self) -> None:
        self._flow_matching_step = MagicMock(
            side_effect=lambda data_batch, iteration, update_train_stats, sampling_iteration=None: (
                {"draw": torch.rand(1), "data_batch": data_batch},
                torch.tensor(float(iteration if sampling_iteration is None else sampling_iteration)),
            )
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_optional_model_config_value_supports_published_configs_without_b4_fields() -> None:
    config = OmegaConf.create({"existing": 7})

    assert optional_model_config_value(config, "existing", 3) == 7
    assert optional_model_config_value(config, "action_relevance_dselect_adaptive_counts", (4, 8)) == (4, 8)


@pytest.mark.L0
@pytest.mark.CPU
def test_training_step_enables_training_stats_and_preserves_iteration() -> None:
    model = _FlowMatchingHarness()
    data_batch = {"sample": torch.tensor([1])}

    output, loss = OmniMoTModel.training_step(model, data_batch, iteration=17)

    model._flow_matching_step.assert_called_once_with(
        data_batch,
        iteration=17,
        update_train_stats=True,
    )
    assert output["data_batch"] is data_batch
    assert loss.item() == 17.0


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_step_uses_fixed_noise_without_advancing_global_rng() -> None:
    model = _FlowMatchingHarness()
    data_batch = {"sample": torch.tensor([1])}

    torch.manual_seed(1234)
    state_before_first = torch.random.get_rng_state().clone()
    first_output, first_loss = OmniMoTModel.validation_step(
        model,
        data_batch,
        iteration=500,
        validation_batch_index=3,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), state_before_first)

    torch.rand(7)
    state_before_second = torch.random.get_rng_state().clone()
    second_output, second_loss = OmniMoTModel.validation_step(
        model,
        data_batch,
        iteration=1000,
        validation_batch_index=3,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), state_before_second)

    torch.testing.assert_close(first_output["draw"], second_output["draw"])
    assert first_loss.item() == second_loss.item() == 3.0
    assert model._flow_matching_step.call_count == 2
    for call in model._flow_matching_step.call_args_list:
        assert call.kwargs["update_train_stats"] is False
        assert call.kwargs["sampling_iteration"] == 3

    third_output, third_loss = OmniMoTModel.validation_step(
        model,
        data_batch,
        iteration=1000,
        validation_batch_index=4,
    )
    assert not torch.equal(first_output["draw"], third_output["draw"])
    assert third_loss.item() == 4.0
