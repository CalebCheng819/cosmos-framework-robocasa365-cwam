# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.mot.strong_action_expert import StrongActionContextExpert


def _expert(*, view_token_width: int = 2) -> StrongActionContextExpert:
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

    output, metrics = expert(**inputs)

    action_indexes = inputs["action_indexes"]
    untouched = torch.ones(output.shape[0], dtype=torch.bool)
    untouched[action_indexes] = False
    torch.testing.assert_close(output[untouched], original[untouched])
    assert not torch.equal(output[action_indexes], original[action_indexes])
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

    output, metrics = expert(**inputs)

    assert output.shape == inputs["hidden_states"].shape
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(value) for value in metrics.values())


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
