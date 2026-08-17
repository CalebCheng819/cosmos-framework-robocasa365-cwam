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


@pytest.mark.L0
@pytest.mark.CPU
def test_denoise_accepts_causal_state_without_video_prediction() -> None:
    state_metrics = {
        "causal_action_state_dynamics_loss": torch.tensor(0.1),
        "causal_action_state_variance_loss": torch.tensor(0.2),
    }
    network_output = {
        "preds_action": [torch.ones(1)],
        "last_hidden_state": torch.randn(2, 4),
        **state_metrics,
    }
    harness = SimpleNamespace(
        net=MagicMock(return_value=network_output),
        config=SimpleNamespace(action_gen=True, action_relevance_enabled=False, sound_gen=False),
    )

    output = OmniMoTModel.denoise(harness, data_batch_packed=object())

    assert "preds_vision" not in output
    assert output["preds_action"] is network_output["preds_action"]
    for name, value in state_metrics.items():
        assert output[name] is value


@pytest.mark.L0
@pytest.mark.CPU
def test_causal_state_loss_skips_video_prediction() -> None:
    metric_names = (
        "causal_action_state_dynamics_loss",
        "causal_action_state_variance_loss",
        "causal_action_state_dynamics_cosine",
        "causal_action_state_state_std",
        "causal_action_state_state_rms",
        "causal_action_state_state_attention_entropy",
        "causal_action_state_action_attention_entropy",
        "causal_action_state_dynamics_attention_entropy",
    )
    out_net = {name: torch.tensor(1.0) for name in metric_names}
    config = SimpleNamespace(
        vision_gen=True,
        action_gen=False,
        action_relevance_enabled=False,
        strong_action_expert_enabled=False,
        causal_action_state_enabled=True,
        causal_action_state_dynamics_loss_weight=0.25,
        causal_action_state_dynamics_warmup_iters=1000,
        causal_action_state_variance_loss_weight=0.01,
        sound_gen=False,
        rectified_flow_training_config=SimpleNamespace(normalize_loss_by_active=False),
        lbl=SimpleNamespace(coeff_und=0.0, coeff_gen=0.0, method="uniform"),
    )
    harness = SimpleNamespace(
        config=config,
        tensor_kwargs_fp32={"device": torch.device("cpu"), "dtype": torch.float32},
        parallel_dims=None,
    )

    loss, losses = OmniMoTModel._compute_losses(
        harness,
        out_net=out_net,
        data_batch_packed=SimpleNamespace(),
        gen_data_noised=SimpleNamespace(),
        timesteps=torch.zeros(2, 1),
        is_image_batch=False,
        iteration=0,
    )

    assert loss.item() == pytest.approx(0.01025)
    assert losses["flow_matching_loss_vision"].item() == 0.0
    assert losses["flow_matching_loss_vision_per_instance"].shape == (2,)
