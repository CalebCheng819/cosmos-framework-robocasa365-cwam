# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa365_nano import (
    action_policy_robocasa365_nano,
    action_policy_robocasa365_s1_nano,
)


def test_s1_enables_fresh_action_relevance_without_mutating_baseline() -> None:
    baseline = action_policy_robocasa365_nano
    s1 = action_policy_robocasa365_s1_nano

    assert baseline.model.config.get("action_relevance_enabled", False) is False
    assert s1.model.config.action_relevance_enabled is True
    assert s1.model.config.action_relevance_routing_dim == 512
    assert s1.model.config.action_relevance_num_heads == 8
    assert s1.model.config.action_relevance_topk == 32

    assert "action_relevance" not in baseline.optimizer.keys_to_select
    assert "action_relevance" in s1.optimizer.keys_to_select
    assert s1.optimizer.lr_multipliers.action_relevance == 5.0
    assert "action_relevance" in s1.checkpoint.keys_to_skip_loading
    assert s1.checkpoint.load_training_state is False
    assert s1.checkpoint.load_ema_to_reg is False
    assert "net_ema." in s1.checkpoint.keys_to_skip_loading
    assert "action2llm" in s1.checkpoint.keys_to_skip_loading
    assert "llm2action" in s1.checkpoint.keys_to_skip_loading


def test_s1_gradient_evidence_covers_router_and_action_heads() -> None:
    families = list(action_policy_robocasa365_s1_nano.trainer.callbacks.action_gradient_evidence.families)

    assert families == [
        "action2llm",
        "llm2action",
        "action_modality_embed",
        "action_relevance",
    ]
