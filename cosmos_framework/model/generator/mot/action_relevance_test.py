# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import ModalityData
from cosmos_framework.model.generator.mot.action_relevance import (
    ACTION_RELEVANCE_INTERVENTIONS,
    ActionRelevanceAdapter,
    action_relevance_graph_capture_enabled,
    clear_action_relevance_graph_tensors,
    collect_action_relevance_evidence,
    collect_action_relevance_graph_tensors,
    configure_action_relevance_modules,
    conditioned_vision_token_indexes,
    estimate_action_relevance_jacobian_energy,
    set_action_relevance_fixed_routed_c,
)


def _adapter_inputs() -> tuple[ActionRelevanceAdapter, dict[str, object]]:
    torch.manual_seed(19)
    adapter = ActionRelevanceAdapter(
        hidden_size=16,
        routing_dim=8,
        num_heads=2,
        topk=2,
        initial_residual_gate=0.1,
    )
    adapter.init_weights()
    inputs: dict[str, object] = {
        "hidden_states": torch.randn(12, 16),
        "action_indexes": torch.tensor([4, 5, 10, 11]),
        "context_indexes": torch.tensor([0, 1, 2, 6, 7, 8]),
        "context_is_vision": torch.tensor([False, True, True, False, True, True]),
        "position_ids": torch.stack(
            (torch.arange(12), torch.arange(12) % 3, torch.arange(12) % 4)
        ),
        "sample_lens": [6, 6],
    }
    return adapter, inputs


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


def test_evidence_configuration_preserves_normal_output() -> None:
    adapter, inputs = _adapter_inputs()
    baseline, _ = adapter(**inputs)
    count = configure_action_relevance_modules(
        adapter,
        intervention_mode="normal",
        capture_evidence=True,
    )
    captured, _ = adapter(**inputs)

    assert count == 1
    torch.testing.assert_close(captured, baseline, rtol=0.0, atol=0.0)
    evidence = collect_action_relevance_evidence(adapter)
    assert len(evidence) == 1
    assert evidence[0]["intervention_mode"] == "normal"
    torch.testing.assert_close(evidence[0]["routed_c"], evidence[0]["routed_c_normal"])


def test_action_relevance_interventions_have_matched_support() -> None:
    adapter, inputs = _adapter_inputs()
    hidden = inputs["hidden_states"]
    action_indexes = inputs["action_indexes"]
    assert isinstance(hidden, torch.Tensor)
    assert isinstance(action_indexes, torch.Tensor)

    for mode in ACTION_RELEVANCE_INTERVENTIONS:
        adapter.configure_evidence(
            intervention_mode=mode,
            intervention_seed=13,
            capture_evidence=True,
        )
        donor = None
        if mode in {"shuffle", "cross_episode"}:
            donor = torch.full((2, 2, adapter.routing_dim), 0.125)
            set_action_relevance_fixed_routed_c(adapter, {"": donor})
        updated, metrics = adapter(**inputs)
        assert bool(torch.isfinite(updated).all())
        assert all(bool(torch.isfinite(value)) for value in metrics.values())
        evidence = adapter.last_evidence
        assert evidence is not None
        if mode in {"selected", "random_matched"}:
            support = (evidence["attention_used"] > 0).sum(dim=-1)
            assert int(support.max()) <= adapter.topk
        if mode == "zero":
            torch.testing.assert_close(
                updated[action_indexes], hidden[action_indexes], rtol=0.0, atol=0.0
            )
            assert int(torch.count_nonzero(evidence["routed_c"])) == 0
        if donor is not None:
            torch.testing.assert_close(evidence["routed_c"], donor)


def test_capture_graph_retains_routed_c_gradient() -> None:
    adapter, inputs = _adapter_inputs()
    hidden = inputs["hidden_states"]
    assert isinstance(hidden, torch.Tensor)
    inputs["hidden_states"] = hidden.requires_grad_()
    adapter.configure_evidence(capture_graph=True)

    updated, _ = adapter(**inputs)
    updated.square().mean().backward()

    assert adapter.last_routed_c is not None
    assert adapter.last_routed_c.grad is not None
    assert bool(torch.isfinite(adapter.last_routed_c.grad).all())


def test_capture_graph_works_with_frozen_adapter() -> None:
    adapter, inputs = _adapter_inputs()
    adapter.requires_grad_(False)
    adapter.configure_evidence(capture_graph=True)

    updated, _ = adapter(**inputs)
    graph_tensors = collect_action_relevance_graph_tensors(adapter)
    gradient = torch.autograd.grad(updated.square().mean(), graph_tensors[0][1])[0]

    assert action_relevance_graph_capture_enabled(adapter)
    assert len(graph_tensors) == 1
    assert graph_tensors[0][1].is_leaf
    assert bool(torch.isfinite(gradient).all())


def test_clear_graph_tensors_releases_routed_c_reference() -> None:
    adapter, inputs = _adapter_inputs()
    adapter.configure_evidence(capture_graph=True)

    adapter(**inputs)
    assert collect_action_relevance_graph_tensors(adapter)

    clear_action_relevance_graph_tensors(adapter)
    assert collect_action_relevance_graph_tensors(adapter) == []


@pytest.mark.parametrize("mode", ["shuffle", "cross_episode"])
def test_replay_intervention_requires_exact_fixed_donor(mode: str) -> None:
    adapter, inputs = _adapter_inputs()
    adapter.configure_evidence(intervention_mode=mode)
    inputs.update(
        hidden_states=torch.randn(6, 16),
        action_indexes=torch.tensor([4, 5]),
        context_indexes=torch.tensor([0, 1, 2]),
        context_is_vision=torch.tensor([False, True, True]),
        position_ids=torch.stack((torch.arange(6), torch.zeros(6), torch.zeros(6))).long(),
        sample_lens=[6],
    )

    try:
        adapter(**inputs)
    except ValueError as error:
        assert "requires an exact fixed routed-C donor" in str(error)
    else:
        raise AssertionError(f"{mode} must reject inference without an exact donor")


@pytest.mark.parametrize("mode", ["shuffle", "cross_episode"])
def test_replay_intervention_accepts_exact_fixed_single_sample_donor(mode: str) -> None:
    adapter, inputs = _adapter_inputs()
    inputs.update(
        hidden_states=torch.randn(6, 16),
        action_indexes=torch.tensor([4, 5]),
        context_indexes=torch.tensor([0, 1, 2]),
        context_is_vision=torch.tensor([False, True, True]),
        position_ids=torch.stack((torch.arange(6), torch.zeros(6), torch.zeros(6))).long(),
        sample_lens=[6],
    )
    adapter.configure_evidence(intervention_mode="normal", capture_evidence=True)
    adapter(**inputs)
    assert adapter.last_evidence is not None
    donor = torch.full_like(adapter.last_evidence["routed_c"], 0.25)

    adapter.configure_evidence(intervention_mode=mode, capture_evidence=True)
    assert set_action_relevance_fixed_routed_c(adapter, {"": donor}) == 1
    adapter(**inputs)

    assert adapter.last_evidence is not None
    torch.testing.assert_close(adapter.last_evidence["routed_c"], donor)


def test_fixed_routed_c_rejects_layer_and_shape_drift() -> None:
    adapter, inputs = _adapter_inputs()
    try:
        set_action_relevance_fixed_routed_c(adapter, {"wrong": torch.zeros(1)})
    except ValueError as error:
        assert "layer mismatch" in str(error)
    else:
        raise AssertionError("fixed donor must reject a mismatched layer name")

    adapter.configure_evidence(intervention_mode="cross_episode")
    set_action_relevance_fixed_routed_c(adapter, {"": torch.zeros(1, 1, 1)})
    try:
        adapter(**inputs)
    except ValueError as error:
        assert "shape mismatch" in str(error)
    else:
        raise AssertionError("fixed donor must reject a mismatched routed-C shape")


def test_jacobian_energy_matches_linear_action_map() -> None:
    routed = torch.randn(3, 5, 7, requires_grad=True)
    prediction = routed.sum(dim=1)

    energy = estimate_action_relevance_jacobian_energy(
        prediction,
        [("adapter", routed)],
        num_projections=4,
        seed=23,
    )["adapter"]

    assert energy.shape == (3, 7)
    expected = torch.full_like(energy, 1.0 / prediction.numel())
    torch.testing.assert_close(energy, expected)
