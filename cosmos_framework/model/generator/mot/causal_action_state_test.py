# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace

import torch

from cosmos_framework.model.generator.mot.causal_action_state import CausalActionStateExpert
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork


def _expert() -> CausalActionStateExpert:
    module = CausalActionStateExpert(
        hidden_size=32,
        model_dim=16,
        num_heads=4,
        state_layers=2,
        state_slots=4,
        action_layers=2,
        dynamics_layers=1,
        ffn_multiplier=2,
        state_dim=6,
        history_action_dim=4,
        target_action_dim=8,
        proprio_history_length=3,
        proprio_history_stride=2,
        action_history_length=5,
        memory_layers=1,
        memory_tokens=2,
        num_views=2,
        view_token_width=2,
    )
    module.init_weights()
    return module


def _inputs() -> dict:
    hidden = torch.randn(16, 32, requires_grad=True)
    position_ids = torch.zeros(3, 16, dtype=torch.long)
    position_ids[2, torch.tensor([1, 2, 9, 10])] = torch.tensor([0, 3, 0, 3])
    return {
        "hidden_states": hidden,
        "action_seed_hidden": hidden[torch.tensor([3, 4, 5, 11, 12, 13])].clone(),
        "action_indexes": torch.tensor([3, 4, 5, 11, 12, 13]),
        "text_indexes": torch.tensor([0, 8]),
        "vision_indexes": torch.tensor([1, 2, 9, 10]),
        "position_ids": position_ids,
        "sample_lens": [8, 8],
        "proprio_history": [torch.randn(3, 6, requires_grad=True) for _ in range(2)],
        "proprio_history_mask": [
            torch.tensor([False, True, True]),
            torch.tensor([True, True, True]),
        ],
        "action_history": [torch.randn(5, 4, requires_grad=True) for _ in range(2)],
        "action_history_mask": [
            torch.tensor([False, False, True, True, True]),
            torch.tensor([True, True, True, True, True]),
        ],
        "future_vision_groups": [torch.randn(6, 32, requires_grad=True) for _ in range(2)],
        "future_view_groups": [torch.tensor([0, 0, 0, 1, 1, 1]) for _ in range(2)],
        "clean_actions": [torch.randn(3, 8) for _ in range(2)],
    }


def test_causal_state_decodes_actions_and_learns_controlled_dynamics() -> None:
    torch.manual_seed(11)
    expert = _expert()
    inputs = _inputs()

    action_hidden, metrics = expert(**inputs)

    assert action_hidden.shape == (6, 32)
    assert metrics.keys() == {
        "dynamics_loss",
        "variance_loss",
        "dynamics_cosine",
        "state_std",
        "state_rms",
        "state_attention_entropy",
        "action_attention_entropy",
        "dynamics_attention_entropy",
    }
    assert all(torch.isfinite(value) for value in metrics.values())

    (action_hidden.square().mean() + metrics["dynamics_loss"] + metrics["variance_loss"]).backward()
    hidden_grad = inputs["hidden_states"].grad
    assert hidden_grad is not None
    assert torch.count_nonzero(hidden_grad[inputs["text_indexes"]]) > 0
    assert torch.count_nonzero(hidden_grad[inputs["vision_indexes"]]) > 0
    assert all(value.grad is not None for value in inputs["proprio_history"])
    assert all(value.grad is not None for value in inputs["action_history"])
    assert all(value.grad is None for value in inputs["future_vision_groups"])


def test_causal_state_inference_needs_no_future_target() -> None:
    expert = _expert()
    inputs = _inputs()
    for key in ("future_vision_groups", "future_view_groups", "clean_actions"):
        inputs[key] = None

    action_hidden, metrics = expert(**inputs)

    assert action_hidden.shape == (6, 32)
    torch.testing.assert_close(metrics["dynamics_loss"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["dynamics_cosine"], torch.tensor(0.0))


def test_clean_future_vision_selects_future_frames_and_groups_samples() -> None:
    token_shapes = [(2, 1, 4), (2, 1, 4), (2, 1, 4)]
    clean_tokens = [torch.randn(1) for _ in token_shapes]
    embedded = torch.arange(24 * 3, dtype=torch.float32).reshape(24, 3)

    network = SimpleNamespace(
        config=SimpleNamespace(
            causal_action_state_num_views=2,
            causal_action_state_view_token_width=2,
        ),
        patchify_and_pack_latents=lambda tokens, shapes: (embedded, None),
        vae2llm=lambda patches: patches,
    )
    packed = SimpleNamespace(
        clean_vision_tokens=clean_tokens,
        sample_lens=[10, 10],
        num_vision_items_per_sample=[2, 1],
        vision=SimpleNamespace(
            token_shapes=token_shapes,
            condition_mask=[
                torch.tensor([[[1, 1, 1, 1]], [[0, 0, 0, 0]]]),
                torch.tensor([[[1, 1, 1, 1]], [[0, 0, 0, 0]]]),
                torch.tensor([[[1, 1, 1, 1]], [[0, 0, 0, 0]]]),
            ],
        ),
    )

    grouped = Cosmos3VFMNetwork._encode_clean_future_vision(network, packed, torch.float32)

    assert grouped is not None
    grouped_tokens, grouped_views = grouped
    torch.testing.assert_close(grouped_tokens[0], torch.cat((embedded[4:8], embedded[12:16])))
    torch.testing.assert_close(grouped_tokens[1], embedded[20:24])
    torch.testing.assert_close(grouped_views[0], torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]))
    torch.testing.assert_close(grouped_views[1], torch.tensor([0, 0, 1, 1]))
