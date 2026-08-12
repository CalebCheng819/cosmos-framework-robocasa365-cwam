# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for OmniMoT denoise output filtering."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel, action_relevance_aux_warmup_fraction


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("iteration", "start_iter", "warmup_iters", "expected"),
    [
        (99, 100, 1000, 0.0),
        (100, 100, 1000, 0.001),
        (599, 100, 1000, 0.5),
        (1099, 100, 1000, 1.0),
        (1500, 100, 1000, 1.0),
        (100, 100, 0, 1.0),
    ],
)
def test_action_relevance_aux_warmup_fraction(
    iteration: int,
    start_iter: int,
    warmup_iters: int,
    expected: float,
) -> None:
    assert action_relevance_aux_warmup_fraction(
        iteration=iteration,
        start_iter=start_iter,
        warmup_iters=warmup_iters,
    ) == pytest.approx(expected)


@pytest.mark.L0
@pytest.mark.CPU
def test_action_relevance_aux_warmup_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        action_relevance_aux_warmup_fraction(iteration=-1, start_iter=0, warmup_iters=1000)


@pytest.mark.L0
@pytest.mark.CPU
def test_denoise_preserves_action_relevance_metrics() -> None:
    relevance_metrics = {
        "action_relevance_sparsity_loss": torch.tensor(0.1),
        "action_relevance_entropy": torch.tensor(0.2),
        "action_relevance_topk_mass": torch.tensor(0.8),
        "action_relevance_effective_support": torch.tensor(0.3),
        "action_relevance_residual_gate": torch.tensor(0.01),
        "action_relevance_selector_entropy": torch.tensor(0.4),
        "action_relevance_selector_overlap": torch.tensor(0.1),
        "action_relevance_selector_invalid_mass": torch.tensor(0.0),
        "action_relevance_selector_hard_fraction": torch.tensor(0.5),
    }
    network_output = {
        "preds_vision": [torch.zeros(1)],
        "preds_action": [torch.ones(1)],
        "preds_action_relevance_only": [torch.full((1,), 3.0)],
        "lbl_metadata_gen": torch.tensor(2.0),
        "last_hidden_state": torch.randn(2, 4),
        **relevance_metrics,
    }
    harness = SimpleNamespace(
        net=MagicMock(return_value=network_output),
        config=SimpleNamespace(action_gen=True, action_relevance_enabled=True, sound_gen=False),
    )
    packed_sequence = object()
    memory = object()

    output = OmniMoTModel.denoise(
        harness,
        data_batch_packed=packed_sequence,
        memory=memory,
    )

    assert output["preds_vision"] is network_output["preds_vision"]
    assert output["preds_action"] is network_output["preds_action"]
    assert output["preds_action_relevance_only"] is network_output["preds_action_relevance_only"]
    assert output["lbl_metadata_gen"] is network_output["lbl_metadata_gen"]
    for name, value in relevance_metrics.items():
        assert output[name] is value
    assert "last_hidden_state" not in output
    harness.net.assert_called_once_with(
        packed_seq=packed_sequence,
        memory=memory,
        video_temporal_causal=None,
    )
