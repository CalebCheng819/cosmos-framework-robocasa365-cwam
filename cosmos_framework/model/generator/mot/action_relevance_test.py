# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

from cosmos_framework.data.generator.sequence_packing import ModalityData
from cosmos_framework.model.generator.mot.action_relevance import (
    ActionRelevanceAdapter,
    conditioned_vision_token_indexes,
)


def test_conditioned_vision_token_indexes_selects_only_clean_frames() -> None:
    vision = ModalityData(
        sequence_indexes=torch.arange(3, 15),
        token_shapes=[(2, 2, 2), (1, 2, 2)],
        condition_mask=[
            torch.tensor([[[1.0]], [[0.0]]]),
            torch.tensor([[[1.0]]]),
        ],
    )

    indexes = conditioned_vision_token_indexes(vision)

    torch.testing.assert_close(indexes, torch.tensor([3, 4, 5, 6, 11, 12, 13, 14]))


def test_action_relevance_updates_only_action_tokens_and_backpropagates() -> None:
    torch.manual_seed(7)
    adapter = ActionRelevanceAdapter(
        hidden_size=16,
        routing_dim=8,
        num_heads=2,
        topk=2,
        initial_residual_gate=0.1,
    )
    adapter.init_weights()
    assert adapter.residual_gate_logit.shape == (1,)
    assert all(parameter.ndim > 0 for parameter in adapter.parameters())
    hidden = torch.randn(12, 16, requires_grad=True)
    action_indexes = torch.tensor([4, 5, 10, 11])
    context_indexes = torch.tensor([0, 1, 2, 6, 7, 8])
    context_is_vision = torch.tensor([False, True, True, False, True, True])
    position_ids = torch.stack(
        (
            torch.arange(12),
            torch.arange(12) % 3,
            torch.arange(12) % 4,
        )
    )

    updated, metrics = adapter(
        hidden,
        action_indexes=action_indexes,
        context_indexes=context_indexes,
        context_is_vision=context_is_vision,
        position_ids=position_ids,
        sample_lens=[6, 6],
    )

    non_action = torch.tensor([0, 1, 2, 3, 6, 7, 8, 9])
    torch.testing.assert_close(updated[non_action], hidden[non_action])
    assert not torch.equal(updated[action_indexes], hidden[action_indexes])
    assert set(metrics) == {
        "sparsity_loss",
        "entropy",
        "topk_mass",
        "effective_support",
        "residual_gate",
    }
    assert all(bool(torch.isfinite(value)) for value in metrics.values())
    assert 0.0 <= metrics["topk_mass"].item() <= 1.0
    assert 0.0 <= metrics["effective_support"].item() <= 1.0
    torch.testing.assert_close(metrics["residual_gate"], torch.tensor(0.1))

    (updated[action_indexes].square().mean() + metrics["sparsity_loss"]).backward()
    for parameter in (
        adapter.query_proj.weight,
        adapter.key_proj.weight,
        adapter.value_proj.weight,
        adapter.output_proj.weight,
        adapter.relation_bias[2].weight,
        adapter.residual_gate_logit,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())


def test_action_relevance_supports_samples_without_actions() -> None:
    adapter = ActionRelevanceAdapter(
        hidden_size=8,
        routing_dim=8,
        num_heads=2,
        topk=1,
        initial_residual_gate=0.05,
    )
    adapter.init_weights()
    hidden = torch.randn(8, 8)

    updated, metrics = adapter(
        hidden,
        action_indexes=torch.tensor([6, 7]),
        context_indexes=torch.tensor([0, 1, 4, 5]),
        context_is_vision=torch.tensor([False, True, False, True]),
        position_ids=torch.stack((torch.arange(8), torch.zeros(8), torch.zeros(8))).long(),
        sample_lens=[4, 4],
    )

    torch.testing.assert_close(updated[:4], hidden[:4])
    assert bool(torch.isfinite(metrics["entropy"]))
