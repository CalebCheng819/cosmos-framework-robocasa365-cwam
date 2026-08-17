# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.strong_action_expert import (
    MemoryConditioningBlock,
    MemoryDeltaMainResidual,
    StrongActionContextExpert,
    action_gradient_alignment_loss,
)


def _expert(
    *,
    view_token_width: int = 2,
    visual_memory_enabled: bool = False,
    visual_memory_intervention: str = "none",
    visual_memory_delta_main_component: str = "full",
    visual_memory_exclusive_dynamic: bool = False,
    visual_memory_delta_conditioning_layers: int = 0,
) -> StrongActionContextExpert:
    module = StrongActionContextExpert(
        hidden_size=32,
        model_dim=16,
        num_heads=4,
        num_layers=2,
        ffn_multiplier=2,
        state_dim=6,
        action_dim=4,
        proprio_history_length=3,
        proprio_history_stride=2,
        action_history_length=5,
        memory_layers=1,
        memory_tokens=2,
        num_views=3,
        view_token_width=view_token_width,
        initial_residual_gate=0.1,
        visual_memory_enabled=visual_memory_enabled,
        visual_history_length=3,
        visual_latent_channels=8,
        visual_latent_height=3,
        visual_latent_width=6,
        visual_memory_layers=1,
        visual_memory_tokens=3,
        visual_memory_initial_gate=0.1,
        visual_memory_delta_main_enabled=visual_memory_enabled,
        visual_memory_delta_gate_scale=0.25,
        visual_memory_delta_residual_scale=0.25,
        visual_memory_delta_main_component=visual_memory_delta_main_component,
        visual_memory_exclusive_dynamic=visual_memory_exclusive_dynamic,
        visual_memory_delta_conditioning_layers=visual_memory_delta_conditioning_layers,
        visual_memory_delta_conditioning_scale=0.25,
        visual_memory_intervention=visual_memory_intervention,
    )
    module.init_weights()
    return module


def _inputs() -> dict:
    hidden_states = torch.randn(16, 32, requires_grad=True)
    position_ids = torch.zeros(3, 16, dtype=torch.long)
    position_ids[2, torch.tensor([1, 2, 9, 10])] = torch.tensor([0, 5, 0, 5])
    proprio = [torch.randn(3, 6, requires_grad=True) for _ in range(2)]
    action_history = [torch.randn(5, 4, requires_grad=True) for _ in range(2)]
    return {
        "hidden_states": hidden_states,
        "action_indexes": torch.tensor([3, 4, 5, 11, 12, 13]),
        "text_indexes": torch.tensor([0, 8]),
        "vision_indexes": torch.tensor([1, 2, 9, 10]),
        "position_ids": position_ids,
        "sample_lens": [8, 8],
        "proprio_history": proprio,
        "proprio_history_mask": [
            torch.tensor([False, True, True]),
            torch.tensor([True, True, True]),
        ],
        "action_history": action_history,
        "action_history_mask": [
            torch.tensor([False, False, True, True, True]),
            torch.tensor([True, True, True, True, True]),
        ],
    }


def test_strong_action_expert_updates_only_action_tokens_and_backpropagates_context() -> None:
    torch.manual_seed(7)
    expert = _expert()
    inputs = _inputs()
    original = inputs["hidden_states"].detach().clone()

    output, auxiliary, metrics = expert(**inputs)

    action_indexes = inputs["action_indexes"]
    untouched = torch.ones(output.shape[0], dtype=torch.bool)
    untouched[action_indexes] = False
    torch.testing.assert_close(output[untouched], original[untouched])
    assert not torch.equal(output[action_indexes], original[action_indexes])
    assert auxiliary is None
    assert metrics.keys() == {
        "residual_gate",
        "attention_entropy",
        "valid_context_tokens",
        "memory_rms",
    }
    torch.testing.assert_close(metrics["residual_gate"], torch.tensor(0.1), atol=1.0e-6, rtol=0.0)
    assert 0.0 <= metrics["attention_entropy"].item() <= 1.0

    output[action_indexes].square().mean().backward()
    hidden_grad = inputs["hidden_states"].grad
    assert hidden_grad is not None
    assert torch.count_nonzero(hidden_grad[inputs["text_indexes"]]) > 0
    assert torch.count_nonzero(hidden_grad[inputs["vision_indexes"]]) > 0
    assert all(value.grad is not None and torch.count_nonzero(value.grad) > 0 for value in inputs["proprio_history"])
    assert all(value.grad is not None and torch.count_nonzero(value.grad) > 0 for value in inputs["action_history"])


def test_strong_action_expert_rejects_missing_or_misaligned_control_context() -> None:
    expert = _expert()
    inputs = _inputs()
    inputs["action_history_mask"] = None
    with pytest.raises(ValueError, match="requires proprio/action history"):
        expert(**inputs)

    inputs = _inputs()
    inputs["proprio_history"] = inputs["proprio_history"][:1]
    with pytest.raises(ValueError, match="align with packed samples"):
        expert(**inputs)


def test_strong_action_expert_accepts_singleton_collator_axis() -> None:
    expert = _expert()
    inputs = _inputs()
    for field in (
        "proprio_history",
        "proprio_history_mask",
        "action_history",
        "action_history_mask",
    ):
        inputs[field] = [value.unsqueeze(0) for value in inputs[field]]

    output, auxiliary, metrics = expert(**inputs)

    assert output.shape == inputs["hidden_states"].shape
    assert auxiliary is None
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(value).all() for value in metrics.values())


def test_strong_action_expert_rejects_nonsingleton_collator_axis() -> None:
    expert = _expert()
    inputs = _inputs()
    inputs["proprio_history"] = [
        value.unsqueeze(0).expand(2, -1, -1) for value in inputs["proprio_history"]
    ]

    with pytest.raises(ValueError, match="proprio contract mismatch"):
        expert(**inputs)


def test_vision_view_ids_partition_horizontal_tokens_in_camera_order() -> None:
    expert = _expert()
    positions = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 0, 4],
            [0, 0, 5],
        ]
    )

    assert expert._vision_view_ids(positions).tolist() == [0, 0, 1, 1, 2, 2]


def test_vision_view_ids_assign_trailing_canvas_padding_to_last_view() -> None:
    expert = _expert()
    positions = torch.zeros(8, 3, dtype=torch.long)
    positions[:, 2] = torch.arange(8)

    assert expert._vision_view_ids(positions).tolist() == [0, 0, 1, 1, 2, 2, 2, 2]


def test_vision_view_ids_match_triple480_fullpad_production_grid() -> None:
    expert = _expert(view_token_width=8)
    positions = torch.zeros(26, 3, dtype=torch.long)
    positions[:, 2] = torch.arange(26)

    view_ids = expert._vision_view_ids(positions)

    assert view_ids.tolist() == [0] * 8 + [1] * 8 + [2] * 10


def test_visual_object_memory_is_checkpoint_additive_and_backpropagates() -> None:
    torch.manual_seed(11)
    baseline = _expert()
    visual = _expert(visual_memory_enabled=True)
    incompatible = visual.load_state_dict(baseline.state_dict(), strict=False)

    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith(("visual_memory_", "visual_object_memory."))
        for key in incompatible.missing_keys
    )

    inputs = _inputs()
    visual_latents = [torch.randn(3, 8, 3, 6, requires_grad=True) for _ in range(2)]
    inputs.update(
        visual_history_latent=visual_latents,
        visual_history_mask=[
            torch.tensor([False, True, True]),
            torch.tensor([True, True, True]),
        ],
        episode_ids=[torch.tensor(4), torch.tensor(5)],
        action_seed_hidden=torch.randn(6, 32, requires_grad=True),
    )
    output, auxiliary, metrics = visual(**inputs)

    assert torch.isfinite(output).all()
    assert auxiliary is not None
    assert auxiliary.shape == (6, 32)
    assert torch.isfinite(auxiliary).all()
    assert metrics["visual_memory_gate"].item() == pytest.approx(0.1)
    assert metrics["visual_memory_history_fraction"].item() == pytest.approx(5.0 / 6.0)
    assert 0.0 <= metrics["visual_memory_slot_abs_cosine"].item() <= 1.0
    assert 0.0 <= metrics["visual_memory_diversity_loss"].item() <= 1.0
    assert 0.0 <= metrics["visual_memory_assignment_entropy_loss"].item() <= 1.0
    assert metrics["visual_memory_assignment_balance_loss"].item() >= 0.0
    assert metrics["visual_memory_spatial_compactness_loss"].item() >= 0.0
    assert torch.isfinite(metrics["visual_memory_temporal_correspondence_loss"])
    assert torch.isfinite(metrics["visual_memory_object_contrastive_loss"])
    assert metrics["visual_memory_episode_contrastive_loss"].item() == 0.0
    assert metrics["visual_memory_temporal_pair_fraction"].item() == pytest.approx(0.75)
    assert metrics["visual_memory_episode_negative_fraction"].item() == 0.0
    assert metrics["visual_memory_conditioning_scale"].item() == pytest.approx(0.25)
    assert metrics["visual_memory_conditioned_residual_rms"].item() > 0.0
    assert metrics["visual_memory_aux_conditioned_residual_rms"].item() > 0.0
    assert metrics["visual_memory_delta_conditioned_residual_rms"].item() == 0.0
    assert metrics["visual_memory_main_context_tokens"].item() == 3.0
    assert metrics["visual_memory_delta_conditioning_layers"].item() == 0.0
    assert metrics["visual_memory_object_delta"].shape == (2, 3, 16)
    assert metrics["visual_memory_object_delta_valid"].tolist() == [True, True]
    assert metrics["visual_memory_object_delta_rms"].item() > 0.0
    assert metrics["visual_memory_object_delta_active_fraction"].item() == 1.0
    assert metrics["visual_memory_delta_main_gate_mean"].item() > 1.0
    assert metrics["visual_memory_delta_main_gate_deviation"].item() > 0.0
    assert metrics["visual_memory_delta_main_residual_rms"].item() > 0.0
    assert metrics["visual_memory_delta_main_applied_delta_rms"].item() > 0.0
    assert metrics["visual_memory_delta_main_delta_base_ratio"].item() > 0.0
    assert metrics["visual_memory_delta_main_attention"].shape == (2, 3)
    torch.testing.assert_close(
        metrics["visual_memory_delta_main_attention"].sum(dim=-1),
        torch.ones(2),
    )
    assert metrics["visual_memory_slots"].shape == (2, 3, 16)
    assert metrics["visual_memory_action_attention"].shape == (2, 3)
    torch.testing.assert_close(
        metrics["visual_memory_action_attention"].sum(dim=-1),
        torch.ones(2),
    )
    assert visual.visual_object_memory.slot_anchors.shape == (3, 2)
    (output[inputs["action_indexes"]].square().mean() + auxiliary.square().mean()).backward()
    assert visual.visual_memory_gate_logit.grad is not None
    assert torch.count_nonzero(visual.visual_memory_gate_logit.grad) > 0
    delta_out_grad = visual.visual_memory_delta_main.delta_out.weight.grad
    assert delta_out_grad is not None
    assert torch.isfinite(delta_out_grad).all()
    assert torch.count_nonzero(delta_out_grad) > 0
    assert all(value.grad is not None and torch.count_nonzero(value.grad) > 0 for value in visual_latents)


def test_memory_delta_components_decompose_full_residual_exactly() -> None:
    torch.manual_seed(19)
    modules = {
        component: MemoryDeltaMainResidual(
            model_dim=16,
            hidden_size=32,
            num_heads=4,
            gate_scale=0.25,
            residual_scale=0.25,
            component=component,
        )
        for component in ("base", "gate", "residual", "full")
    }
    modules["full"].init_weights()
    full_state = modules["full"].state_dict()
    for component, module in modules.items():
        if component != "full":
            module.load_state_dict(full_state)

    action = torch.randn(2, 5, 16)
    base = torch.randn(2, 5, 32)
    delta = torch.randn(2, 3, 16)
    delta_valid = torch.ones(2, dtype=torch.bool)
    action_valid = torch.ones(2, 5, dtype=torch.bool)
    outputs = {}
    metrics = {}
    for component, module in modules.items():
        outputs[component], metrics[component] = module(
            action,
            base,
            delta,
            delta_valid,
            action_valid=action_valid,
        )

    torch.testing.assert_close(outputs["base"], base)
    torch.testing.assert_close(
        outputs["full"],
        outputs["gate"] + outputs["residual"] - outputs["base"],
    )
    assert metrics["base"]["visual_memory_delta_main_applied_delta_rms"].item() == 0.0
    assert metrics["gate"]["visual_memory_delta_main_residual_rms"].item() == 0.0
    assert metrics["residual"]["visual_memory_delta_main_gate_deviation"].item() == 0.0
    assert metrics["full"]["visual_memory_delta_main_raw_gate_deviation"].item() > 0.0
    assert metrics["full"]["visual_memory_delta_main_raw_residual_rms"].item() > 0.0


def test_memory_conditioner_masks_invalid_delta_exactly() -> None:
    torch.manual_seed(23)
    conditioner = MemoryConditioningBlock(model_dim=16, num_heads=4, residual_scale=0.25)
    conditioner.init_weights()
    action = torch.randn(2, 5, 16)
    memory = torch.randn(2, 3, 16)
    action_valid = torch.ones(2, 5, dtype=torch.bool)

    updated, residual_rms, attention = conditioner(
        action,
        memory,
        action_valid=action_valid,
        memory_valid=torch.zeros(2, dtype=torch.bool),
    )

    torch.testing.assert_close(updated, action)
    assert residual_rms.item() == 0.0
    assert torch.count_nonzero(attention) == 0


def test_exclusive_dynamic_memory_removes_absolute_main_path() -> None:
    torch.manual_seed(29)
    r6a = _expert(
        visual_memory_enabled=True,
        visual_memory_exclusive_dynamic=True,
    )
    r6b = _expert(
        visual_memory_enabled=True,
        visual_memory_exclusive_dynamic=True,
        visual_memory_delta_conditioning_layers=2,
    )
    inputs = _inputs()
    inputs.update(
        visual_history_latent=[torch.randn(3, 8, 3, 6) for _ in range(2)],
        visual_history_mask=[torch.ones(3, dtype=torch.bool) for _ in range(2)],
        episode_ids=[torch.tensor(4), torch.tensor(5)],
        action_seed_hidden=torch.randn(6, 32),
    )

    output_a, _, metrics_a = r6a(**inputs)
    output_b, _, metrics_b = r6b(**inputs)

    (output_a.float().square().mean() + output_b.float().square().mean()).backward()

    assert r6a.visual_memory_context_embed.requires_grad
    assert r6a.visual_memory_gate_logit.requires_grad
    assert r6b.visual_memory_context_embed.requires_grad
    assert r6b.visual_memory_gate_logit.requires_grad
    assert r6a.visual_memory_context_embed.grad is None
    assert r6a.visual_memory_gate_logit.grad is None
    assert r6b.visual_memory_context_embed.grad is None
    assert r6b.visual_memory_gate_logit.grad is None
    assert metrics_a["visual_memory_main_context_tokens"].item() == 0.0
    assert metrics_a["visual_memory_conditioned_residual_rms"].item() == 0.0
    assert metrics_a["visual_memory_delta_conditioned_residual_rms"].item() == 0.0
    torch.testing.assert_close(
        metrics_a["visual_memory_action_attention"].sum(dim=-1),
        torch.ones(2),
    )
    assert metrics_b["visual_memory_main_context_tokens"].item() == 0.0
    assert metrics_b["visual_memory_conditioned_residual_rms"].item() == 0.0
    assert metrics_b["visual_memory_delta_conditioning_layers"].item() == 2.0
    assert metrics_b["visual_memory_delta_conditioned_residual_rms"].item() > 0.0
    assert torch.isfinite(output_b).all()


def test_visual_object_memory_uses_two_by_four_anchors_for_eight_slots() -> None:
    visual = StrongActionContextExpert(
        hidden_size=32,
        model_dim=16,
        num_heads=4,
        num_layers=1,
        ffn_multiplier=2,
        state_dim=6,
        action_dim=4,
        proprio_history_length=3,
        proprio_history_stride=2,
        action_history_length=5,
        memory_layers=1,
        memory_tokens=2,
        num_views=3,
        view_token_width=2,
        initial_residual_gate=0.1,
        visual_memory_enabled=True,
        visual_history_length=3,
        visual_latent_channels=8,
        visual_latent_height=3,
        visual_latent_width=6,
        visual_memory_layers=1,
        visual_memory_tokens=8,
        visual_memory_initial_gate=0.1,
    )

    anchors = visual.visual_object_memory.slot_anchors

    assert anchors.shape == (8, 2)
    assert torch.unique(anchors[:, 0]).numel() == 2
    assert torch.unique(anchors[:, 1]).numel() == 4


def test_visual_memory_zero_intervention_masks_history_from_action_context() -> None:
    torch.manual_seed(13)
    visual = _expert(visual_memory_enabled=True, visual_memory_intervention="zero")
    inputs_a = _inputs()
    inputs_b = dict(inputs_a)
    inputs_a.update(
        visual_history_latent=[torch.randn(3, 8, 3, 6) for _ in range(2)],
        visual_history_mask=[torch.ones(3, dtype=torch.bool) for _ in range(2)],
        action_seed_hidden=torch.randn(6, 32),
    )
    inputs_b.update(
        visual_history_latent=[torch.randn(3, 8, 3, 6) * 100 for _ in range(2)],
        visual_history_mask=[torch.ones(3, dtype=torch.bool) for _ in range(2)],
        action_seed_hidden=inputs_a["action_seed_hidden"],
    )

    output_a, auxiliary_a, metrics_a = visual(**inputs_a)
    output_b, auxiliary_b, metrics_b = visual(**inputs_b)

    torch.testing.assert_close(output_a, output_b)
    assert auxiliary_a is not None and auxiliary_b is not None
    torch.testing.assert_close(auxiliary_a, auxiliary_b)
    assert metrics_a["visual_memory_history_fraction"].item() == 0.0
    assert metrics_b["visual_memory_history_fraction"].item() == 0.0
    assert metrics_a["visual_memory_temporal_pair_fraction"].item() == 0.0
    assert metrics_b["visual_memory_temporal_pair_fraction"].item() == 0.0
    assert metrics_a["visual_memory_object_delta_rms"].item() == 0.0
    assert metrics_b["visual_memory_object_delta_rms"].item() == 0.0
    assert metrics_a["visual_memory_object_delta_active_fraction"].item() == 0.0
    assert metrics_b["visual_memory_object_delta_active_fraction"].item() == 0.0
    assert metrics_a["visual_memory_delta_main_gate_mean"].item() == 1.0
    assert metrics_b["visual_memory_delta_main_gate_mean"].item() == 1.0
    assert metrics_a["visual_memory_delta_main_gate_deviation"].item() == 0.0
    assert metrics_b["visual_memory_delta_main_gate_deviation"].item() == 0.0
    assert metrics_a["visual_memory_delta_main_residual_rms"].item() == 0.0
    assert metrics_b["visual_memory_delta_main_residual_rms"].item() == 0.0
    assert torch.count_nonzero(metrics_a["visual_memory_object_delta"]) == 0
    assert torch.count_nonzero(metrics_b["visual_memory_object_delta"]) == 0
    for metric_name in (
        "visual_memory_assignment_entropy_loss",
        "visual_memory_assignment_balance_loss",
        "visual_memory_spatial_compactness_loss",
        "visual_memory_temporal_correspondence_loss",
        "visual_memory_object_contrastive_loss",
        "visual_memory_episode_contrastive_loss",
    ):
        assert metrics_a[metric_name].item() == 0.0
        assert metrics_b[metric_name].item() == 0.0


def test_action_gradient_alignment_supervises_attention_without_second_order_gradients() -> None:
    torch.manual_seed(23)
    slots = torch.randn(2, 3, 4, requires_grad=True)
    saliency_scale = torch.tensor(
        [
            [[4.0], [1.0], [0.25]],
            [[0.5], [3.0], [1.0]],
        ]
    )
    action_loss = (slots * saliency_scale).square().mean()
    attention_logits = torch.zeros(2, 3, requires_grad=True)
    attention = attention_logits.softmax(dim=-1)

    alignment_loss, valid_fraction, target_entropy = action_gradient_alignment_loss(
        action_loss,
        slots,
        attention,
    )

    assert torch.isfinite(alignment_loss)
    assert valid_fraction.item() == 1.0
    assert 0.0 <= target_entropy.item() <= 1.0
    alignment_loss.backward()
    assert attention_logits.grad is not None
    assert torch.count_nonzero(attention_logits.grad) > 0
    assert slots.grad is None


def test_visual_memory_auxiliary_has_no_full_context_bypass() -> None:
    torch.manual_seed(17)
    visual = _expert(visual_memory_enabled=True)
    inputs_a = _inputs()
    inputs_a.update(
        visual_history_latent=[torch.randn(3, 8, 3, 6) for _ in range(2)],
        visual_history_mask=[torch.ones(3, dtype=torch.bool) for _ in range(2)],
        action_seed_hidden=torch.randn(6, 32),
    )
    inputs_b = dict(inputs_a)
    inputs_b["hidden_states"] = torch.randn_like(inputs_a["hidden_states"], requires_grad=True)

    output_a, auxiliary_a, _ = visual(**inputs_a)
    output_b, auxiliary_b, _ = visual(**inputs_b)

    assert not torch.equal(output_a, output_b)
    assert auxiliary_a is not None and auxiliary_b is not None
    torch.testing.assert_close(auxiliary_a, auxiliary_b)


def test_visual_memory_auxiliary_depends_on_matching_memory() -> None:
    torch.manual_seed(19)
    visual = _expert(visual_memory_enabled=True)
    action_seed = torch.randn(6, 32, requires_grad=True)
    inputs_a = _inputs()
    inputs_a.update(
        visual_history_latent=[torch.randn(3, 8, 3, 6, requires_grad=True) for _ in range(2)],
        visual_history_mask=[torch.ones(3, dtype=torch.bool) for _ in range(2)],
        action_seed_hidden=action_seed,
    )
    inputs_b = dict(inputs_a)
    inputs_b["visual_history_latent"] = [
        torch.randn(3, 8, 3, 6) + 4.0 for _ in range(2)
    ]

    _, auxiliary_a, _ = visual(**inputs_a)
    _, auxiliary_b, _ = visual(**inputs_b)

    assert auxiliary_a is not None and auxiliary_b is not None
    assert not torch.allclose(auxiliary_a, auxiliary_b)
    auxiliary_a.square().mean().backward()
    assert action_seed.grad is not None and torch.count_nonzero(action_seed.grad) > 0
    assert all(
        latent.grad is not None and torch.count_nonzero(latent.grad) > 0
        for latent in inputs_a["visual_history_latent"]
    )
