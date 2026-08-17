# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from omegaconf import OmegaConf

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa365_nano import (
    action_policy_robocasa365_b2_nano,
    action_policy_robocasa365_b3_dselect_nano,
    action_policy_robocasa365_b4_adaptive_dselect_nano,
    action_policy_robocasa365_b5_visual_object_memory_nano,
    action_policy_robocasa365_nano,
    action_policy_robocasa365_s1_nano,
    action_policy_robocasa365_strong_nano,
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


def test_strong_recipe_isolated_three_view_control_contract() -> None:
    baseline = action_policy_robocasa365_nano
    strong = action_policy_robocasa365_strong_nano

    assert baseline.model.config.get("strong_action_expert_enabled", False) is False
    assert strong.model.config.strong_action_expert_enabled is True
    assert strong.model.config.action_relevance_enabled is False
    assert strong.model.config.strong_action_expert_num_layers == 4
    assert strong.model.config.strong_action_memory_tokens == 4
    assert strong.model.config.strong_action_view_token_width == 8
    assert strong.model.config.strong_action_input_resolution == "480"
    assert "strong_action_expert" not in baseline.optimizer.keys_to_select
    assert "strong_action_expert" in strong.optimizer.keys_to_select
    assert strong.optimizer.lr_multipliers.strong_action_expert == 5.0
    assert "strong_action_expert" in strong.checkpoint.keys_to_skip_loading
    assert strong.trainer.grad_accum_iter == 2
    assert strong.trainer.callbacks.production_batch_memory_gate.expected_samples_per_microbatch == 32

    train_loader = strong.dataloader_train
    dataset = train_loader.dataloader.datasets.robocasa365.dataset
    dataset_unresolved = OmegaConf.to_container(dataset, resolve=False)
    assert train_loader.max_samples_per_batch == 32
    assert dataset.camera_mode == "triple_view"
    assert dataset.resolution == "480"
    assert dataset.latent_cache_expected_stage == "full_padding"
    assert dataset_unresolved["control_cache_root"] == "${oc.env:ROBOCASA365_CONTROL_CACHE_ROOT}"
    assert dataset.proprio_history_length == 4
    assert dataset.proprio_history_stride == 12
    assert dataset.action_history_length == 16


def test_strong_gradient_evidence_covers_new_tower_and_action_heads() -> None:
    families = list(action_policy_robocasa365_strong_nano.trainer.callbacks.action_gradient_evidence.families)

    assert families == [
        "action2llm",
        "llm2action",
        "action_modality_embed",
        "strong_action_expert",
    ]


def test_b2_adds_relevance_auxiliary_objective_without_dropping_strong_context() -> None:
    b2 = action_policy_robocasa365_b2_nano

    assert b2.model.config.strong_action_expert_enabled is True
    assert b2.model.config.action_relevance_enabled is True
    assert b2.model.config.action_relevance_aux_loss_weight == 0.25
    assert b2.model.config.action_relevance_aux_start_iter == 0
    assert b2.model.config.action_relevance_aux_warmup_iters == 1000
    assert b2.model.config.action_relevance_sparsity_start_iter == 1000
    assert b2.dataloader_train.max_samples_per_batch == 32
    assert b2.trainer.grad_accum_iter == 2
    assert "strong_action_expert" in b2.optimizer.keys_to_select
    assert "action_relevance" in b2.optimizer.keys_to_select
    assert b2.optimizer.lr_multipliers.strong_action_expert == 5.0
    assert b2.optimizer.lr_multipliers.action_relevance == 5.0
    assert "strong_action_expert" in b2.checkpoint.keys_to_skip_loading
    assert "action_relevance" in b2.checkpoint.keys_to_skip_loading


def test_b2_gradient_evidence_covers_both_action_branches() -> None:
    families = list(action_policy_robocasa365_b2_nano.trainer.callbacks.action_gradient_evidence.families)

    assert families == [
        "action2llm",
        "llm2action",
        "action_modality_embed",
        "strong_action_expert",
        "action_relevance",
    ]


def test_b3_changes_only_the_b2_relevance_support_router() -> None:
    b2 = action_policy_robocasa365_b2_nano
    b3 = action_policy_robocasa365_b3_dselect_nano

    assert OmegaConf.select(
        b2.model.config,
        "action_relevance_routing_mode",
        default="soft_attention",
    ) == "soft_attention"
    assert b3.model.config.action_relevance_routing_mode == "dselect_k"
    assert b3.model.config.action_relevance_dselect_num_selectors == b2.model.config.action_relevance_topk
    assert b3.model.config.action_relevance_dselect_capacity == 8192
    assert b3.model.config.action_relevance_aux_loss_weight == b2.model.config.action_relevance_aux_loss_weight
    assert b3.model.config.strong_action_expert_enabled is True
    assert b3.dataloader_train.max_samples_per_batch == b2.dataloader_train.max_samples_per_batch
    assert b3.trainer.grad_accum_iter == b2.trainer.grad_accum_iter
    assert b3.optimizer == b2.optimizer
    assert b3.checkpoint == b2.checkpoint


def test_b4_changes_only_b3_selector_budget_control() -> None:
    b3 = action_policy_robocasa365_b3_dselect_nano
    b4 = action_policy_robocasa365_b4_adaptive_dselect_nano

    assert b3.model.config.action_relevance_routing_mode == "dselect_k"
    assert b4.model.config.action_relevance_routing_mode == "adaptive_dselect_k"
    assert list(b4.model.config.action_relevance_dselect_adaptive_counts) == [4, 8, 12, 16, 24, 32]
    assert b4.model.config.action_relevance_dselect_num_selectors == 32
    assert b4.model.config.action_relevance_dselect_adaptive_target_selectors == 24.0
    assert b4.model.config.action_relevance_dselect_adaptive_budget_weight == 1.0
    assert b4.model.config.action_relevance_dselect_adaptive_entropy_weight == 0.01
    assert b4.model.config.action_relevance_aux_loss_weight == b3.model.config.action_relevance_aux_loss_weight
    assert b4.model.config.strong_action_expert_enabled is True
    assert b4.dataloader_train.max_samples_per_batch == b3.dataloader_train.max_samples_per_batch
    assert b4.trainer.grad_accum_iter == b3.trainer.grad_accum_iter
    assert b4.optimizer == b3.optimizer
    assert b4.checkpoint == b3.checkpoint


def test_b5_adds_visual_object_memory_without_changing_b4_router() -> None:
    b4 = action_policy_robocasa365_b4_adaptive_dselect_nano
    b5 = action_policy_robocasa365_b5_visual_object_memory_nano

    assert b5.model.config.action_relevance_routing_mode == b4.model.config.action_relevance_routing_mode
    assert list(b5.model.config.action_relevance_dselect_adaptive_counts) == list(
        b4.model.config.action_relevance_dselect_adaptive_counts
    )
    assert b5.model.config.strong_action_visual_memory_enabled is True
    assert b5.model.config.strong_action_visual_history_length == 3
    assert b5.model.config.strong_action_visual_history_stride == 12
    assert b5.model.config.strong_action_visual_memory_tokens == 8
    assert b5.model.config.strong_action_visual_memory_layers == 2
    assert b5.model.config.strong_action_visual_memory_initial_gate == 0.1
    assert b5.model.config.strong_action_visual_memory_conditioning_scale == 0.25
    assert b5.model.config.strong_action_visual_memory_diversity_loss_weight == 0.05
    assert b5.model.config.strong_action_visual_memory_temporal_loss_weight == 0.05
    assert b5.model.config.strong_action_visual_memory_action_aux_loss_weight == 0.25
    assert b5.model.config.strong_action_visual_memory_action_aux_warmup_iters == 50
    assert b5.model.config.strong_action_visual_memory_intervention == "none"
    assert b5.dataloader_train.max_samples_per_batch == b4.dataloader_train.max_samples_per_batch
    assert b5.trainer.grad_accum_iter == b4.trainer.grad_accum_iter
    dataset = b5.dataloader_train.dataloader.datasets.robocasa365.dataset
    unresolved = OmegaConf.to_container(dataset, resolve=False)
    assert unresolved["visual_history_cache_root"] == "${oc.env:ROBOCASA365_VISUAL_HISTORY_CACHE_ROOT}"
    assert dataset.visual_history_length == 3
    assert dataset.visual_history_stride == 12
