# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import ModalityData
from cosmos_framework.model.generator.mot.action_relevance import (
    ActionRelevanceAdapter,
    DSelectKSupportRouter,
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

    action_seed = hidden[action_indexes].detach().clone().requires_grad_(True)
    updated, auxiliary, metrics = adapter(
        hidden,
        action_indexes=action_indexes,
        context_indexes=context_indexes,
        context_is_vision=context_is_vision,
        position_ids=position_ids,
        sample_lens=[6, 6],
        action_seed_hidden=action_seed,
    )

    non_action = torch.tensor([0, 1, 2, 3, 6, 7, 8, 9])
    torch.testing.assert_close(updated[non_action], hidden[non_action])
    assert not torch.equal(updated[action_indexes], hidden[action_indexes])
    assert auxiliary.shape == action_seed.shape
    assert bool(torch.isfinite(auxiliary).all())
    assert not torch.equal(auxiliary, action_seed)
    assert set(metrics) == {
        "sparsity_loss",
        "entropy",
        "topk_mass",
        "effective_support",
        "residual_gate",
        "selector_entropy",
        "selector_overlap",
        "selector_invalid_mass",
        "selector_hard_fraction",
        "selector_expected_count",
        "selector_selected_count",
        "selector_count_entropy",
        "selector_budget_loss",
        "selector_utilization",
    }
    assert all(bool(torch.isfinite(value)) for value in metrics.values())
    assert 0.0 <= metrics["topk_mass"].item() <= 1.0
    assert 0.0 <= metrics["effective_support"].item() <= 1.0
    torch.testing.assert_close(metrics["residual_gate"], torch.tensor(0.1))

    (
        updated[action_indexes].square().mean()
        + auxiliary.square().mean()
        + metrics["sparsity_loss"]
    ).backward()
    for parameter in (
        adapter.query_proj.weight,
        adapter.key_proj.weight,
        adapter.value_proj.weight,
        adapter.output_proj.weight,
        adapter.auxiliary_norm.weight,
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

    updated, auxiliary, metrics = adapter(
        hidden,
        action_indexes=torch.tensor([6, 7]),
        context_indexes=torch.tensor([0, 1, 4, 5]),
        context_is_vision=torch.tensor([False, True, False, True]),
        position_ids=torch.stack((torch.arange(8), torch.zeros(8), torch.zeros(8))).long(),
        sample_lens=[4, 4],
    )

    torch.testing.assert_close(updated[:4], hidden[:4])
    assert auxiliary.shape == (2, 8)
    assert bool(torch.isfinite(metrics["entropy"]))


def test_action_relevance_auxiliary_query_does_not_use_post_transformer_action_states() -> None:
    torch.manual_seed(11)
    adapter = ActionRelevanceAdapter(
        hidden_size=16,
        routing_dim=8,
        num_heads=2,
        topk=2,
        initial_residual_gate=0.1,
    )
    adapter.init_weights()
    hidden = torch.randn(8, 16)
    action_indexes = torch.tensor([3, 7])
    context_indexes = torch.tensor([0, 1, 4, 5])
    action_seed = torch.randn(2, 16)
    position_ids = torch.stack((torch.arange(8), torch.zeros(8), torch.zeros(8))).long()

    _, auxiliary_before, metrics_before = adapter(
        hidden,
        action_indexes=action_indexes,
        context_indexes=context_indexes,
        context_is_vision=torch.tensor([False, True, False, True]),
        position_ids=position_ids,
        sample_lens=[4, 4],
        action_seed_hidden=action_seed,
    )
    changed_hidden = hidden.clone()
    changed_hidden[action_indexes] += 100.0
    _, auxiliary_after, metrics_after = adapter(
        changed_hidden,
        action_indexes=action_indexes,
        context_indexes=context_indexes,
        context_is_vision=torch.tensor([False, True, False, True]),
        position_ids=position_ids,
        sample_lens=[4, 4],
        action_seed_hidden=action_seed,
    )

    torch.testing.assert_close(auxiliary_before, auxiliary_after)
    for name in ("sparsity_loss", "entropy", "topk_mass", "effective_support"):
        torch.testing.assert_close(metrics_before[name], metrics_after[name])


def test_dselect_k_hard_codes_produce_at_most_k_support_positions() -> None:
    router = DSelectKSupportRouter(
        head_dim=2,
        num_selectors=2,
        capacity=8,
        gamma=1.0,
    )
    router.init_weights()
    with torch.no_grad():
        router.selector_logits.weight.zero_()
        router.selector_logits.bias.copy_(
            torch.tensor(
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                ]
            )
        )
        router.selector_mix_logits.weight.zero_()
        router.selector_mix_logits.bias.zero_()

    support, metrics = router(
        torch.zeros(1, 1, 1, 2),
        torch.zeros(1, 1, 8, 2),
        context_valid=torch.ones(1, 8, dtype=torch.bool),
        query_valid=torch.ones(1, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(support[0, 0, 0], torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]))
    assert int((support > 0.0).sum()) <= router.num_selectors
    torch.testing.assert_close(metrics["selector_entropy"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["selector_overlap"], torch.tensor(0.0), atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(metrics["selector_invalid_mass"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["selector_hard_fraction"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["selector_expected_count"], torch.tensor(2.0))
    torch.testing.assert_close(metrics["selector_selected_count"], torch.tensor(2.0))
    torch.testing.assert_close(metrics["selector_utilization"], torch.tensor(1.0))


def test_dselect_k_reports_mass_assigned_beyond_the_valid_context() -> None:
    router = DSelectKSupportRouter(
        head_dim=2,
        num_selectors=1,
        capacity=8,
        gamma=1.0,
    )
    router.init_weights()
    with torch.no_grad():
        router.selector_logits.weight.zero_()
        router.selector_logits.bias.fill_(1.0)  # Select code 111, outside a length-3 context.
        router.selector_mix_logits.weight.zero_()
        router.selector_mix_logits.bias.zero_()

    support, metrics = router(
        torch.zeros(1, 1, 1, 2),
        torch.zeros(1, 1, 3, 2),
        context_valid=torch.ones(1, 3, dtype=torch.bool),
        query_valid=torch.ones(1, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(support, torch.full_like(support, 1.0 / 3.0))
    torch.testing.assert_close(metrics["selector_invalid_mass"], torch.tensor(1.0))


def test_dselect_k_adapter_backpropagates_through_selector_parameters() -> None:
    torch.manual_seed(19)
    adapter = ActionRelevanceAdapter(
        hidden_size=16,
        routing_dim=8,
        num_heads=2,
        topk=2,
        initial_residual_gate=0.1,
        routing_mode="dselect_k",
        dselect_num_selectors=2,
        dselect_capacity=8,
    )
    adapter.init_weights()
    hidden = torch.randn(8, 16, requires_grad=True)
    action_indexes = torch.tensor([3, 7])
    action_seed = torch.randn(2, 16, requires_grad=True)

    updated, auxiliary, metrics = adapter(
        hidden,
        action_indexes=action_indexes,
        context_indexes=torch.tensor([0, 1, 2, 4, 5, 6]),
        context_is_vision=torch.tensor([False, True, True, False, True, True]),
        position_ids=torch.stack((torch.arange(8), torch.zeros(8), torch.zeros(8))).long(),
        sample_lens=[4, 4],
        action_seed_hidden=action_seed,
    )

    loss = updated[action_indexes].square().mean() + auxiliary.square().mean() + metrics["sparsity_loss"]
    loss.backward()
    assert adapter.dselect_router is not None
    for parameter in (
        adapter.dselect_router.selector_logits.weight,
        adapter.dselect_router.selector_logits.bias,
        adapter.dselect_router.selector_mix_logits.weight,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
    assert 0.0 <= metrics["selector_entropy"].item() <= 1.0
    assert 0.0 <= metrics["selector_overlap"].item() <= 1.0
    assert 0.0 <= metrics["selector_invalid_mass"].item() <= 1.0


def test_adaptive_dselect_k_uses_the_selected_prefix_budget() -> None:
    router = DSelectKSupportRouter(
        head_dim=2,
        num_selectors=4,
        capacity=8,
        gamma=1.0,
        adaptive_counts=(1, 2, 4),
        adaptive_target_selectors=2.0,
    )
    router.init_weights()
    assert router.selector_count_logits is not None
    with torch.no_grad():
        router.selector_logits.weight.zero_()
        router.selector_logits.bias.copy_(
            torch.tensor(
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                ]
            )
        )
        router.selector_mix_logits.weight.zero_()
        router.selector_mix_logits.bias.zero_()
        router.selector_count_logits.weight.zero_()
        router.selector_count_logits.bias.copy_(torch.tensor([-10.0, 10.0, -10.0]))

    router.eval()
    support, metrics = router(
        torch.zeros(1, 1, 1, 2),
        torch.zeros(1, 1, 8, 2),
        context_valid=torch.ones(1, 8, dtype=torch.bool),
        query_valid=torch.ones(1, 1, dtype=torch.bool),
    )

    assert int((support > 0.0).sum()) == 2
    torch.testing.assert_close(metrics["selector_selected_count"], torch.tensor(2.0))
    torch.testing.assert_close(metrics["selector_expected_count"], torch.tensor(2.0))
    torch.testing.assert_close(metrics["selector_utilization"], torch.tensor(0.5))


def test_adaptive_dselect_k_requires_multiple_candidate_counts() -> None:
    with pytest.raises(ValueError, match="at least two candidate counts"):
        DSelectKSupportRouter(
            head_dim=2,
            num_selectors=4,
            capacity=8,
            gamma=1.0,
            adaptive_counts=(4,),
            adaptive_target_selectors=4.0,
        )


def test_adaptive_dselect_k_backpropagates_through_the_count_predictor() -> None:
    torch.manual_seed(23)
    adapter = ActionRelevanceAdapter(
        hidden_size=16,
        routing_dim=8,
        num_heads=2,
        topk=2,
        initial_residual_gate=0.1,
        routing_mode="adaptive_dselect_k",
        dselect_num_selectors=2,
        dselect_capacity=8,
        dselect_adaptive_counts=(1, 2),
        dselect_adaptive_target_selectors=1.5,
    )
    adapter.init_weights()
    hidden = torch.randn(8, 16, requires_grad=True)
    action_indexes = torch.tensor([3, 7])

    updated, auxiliary, metrics = adapter(
        hidden,
        action_indexes=action_indexes,
        context_indexes=torch.tensor([0, 1, 2, 4, 5, 6]),
        context_is_vision=torch.tensor([False, True, True, False, True, True]),
        position_ids=torch.stack((torch.arange(8), torch.zeros(8), torch.zeros(8))).long(),
        sample_lens=[4, 4],
        action_seed_hidden=torch.randn(2, 16, requires_grad=True),
    )

    loss = updated[action_indexes].square().mean() + auxiliary.square().mean() + metrics["sparsity_loss"]
    loss.backward()
    assert adapter.dselect_router is not None
    assert adapter.dselect_router.selector_count_logits is not None
    for parameter in (
        adapter.dselect_router.selector_count_logits.weight,
        adapter.dselect_router.selector_count_logits.bias,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool((parameter.grad != 0.0).any())
    assert 1.0 <= metrics["selector_expected_count"].item() <= 2.0
    assert metrics["selector_selected_count"].item() in {1.0, 2.0}
