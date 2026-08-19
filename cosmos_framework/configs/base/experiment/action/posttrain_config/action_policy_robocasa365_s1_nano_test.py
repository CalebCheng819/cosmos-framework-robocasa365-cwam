# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from omegaconf import OmegaConf

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa365_nano import (
    action_policy_robocasa365_b2_nano,
    action_policy_robocasa365_causal_state_nano,
    action_policy_robocasa365_cwam_v2_nano,
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


def test_causal_state_recipe_has_one_state_only_action_path() -> None:
    causal = action_policy_robocasa365_causal_state_nano

    assert causal.model.config.causal_action_state_enabled is True
    assert causal.model.config.strong_action_expert_enabled is False
    assert causal.model.config.action_relevance_enabled is False
    assert causal.model.config.causal_action_state_slots == 32
    assert causal.model.config.causal_action_state_encoder_layers == 4
    assert causal.model.config.causal_action_state_action_layers == 4
    assert causal.model.config.causal_action_state_dynamics_layers == 2
    assert causal.model.config.causal_action_state_dynamics_loss_weight == 0.25
    assert causal.dataloader_train.max_samples_per_batch == 32
    assert causal.trainer.grad_accum_iter == 2
    assert list(causal.trainer.callbacks.production_batch_memory_gate.allowed_world_sizes) == [6, 8]
    assert "causal_action_state" in causal.optimizer.keys_to_select
    assert "strong_action_expert" not in causal.optimizer.keys_to_select
    assert "moe_gen" not in causal.optimizer.keys_to_select
    assert "llm2vae" not in causal.optimizer.keys_to_select
    assert list(causal.optimizer.keys_to_select) == [
        "time_embedder",
        "vae2llm",
        "action2llm",
        "llm2action",
        "action_modality_embed",
        "causal_action_state",
    ]
    assert causal.optimizer.lr_multipliers.causal_action_state == 5.0
    assert "strong_action_expert" not in causal.optimizer.lr_multipliers
    assert "causal_action_state" in causal.checkpoint.keys_to_skip_loading
    assert "strong_action_expert" not in causal.checkpoint.keys_to_skip_loading


def test_causal_state_gradient_evidence_covers_state_and_action_heads() -> None:
    families = list(
        action_policy_robocasa365_causal_state_nano.trainer.callbacks.action_gradient_evidence.families
    )

    assert families == [
        "action2llm",
        "llm2action",
        "action_modality_embed",
        "causal_action_state",
    ]


def test_cwam_v2_enables_factorized_observation_only_objectives() -> None:
    v2 = action_policy_robocasa365_cwam_v2_nano

    assert v2.model.config.causal_action_state_enabled is True
    assert v2.model.config.causal_action_state_v2_enabled is True
    assert (
        v2.model.config.causal_action_state_object_slots
        + v2.model.config.causal_action_state_geometry_slots
        + v2.model.config.causal_action_state_dynamics_slots
        + v2.model.config.causal_action_state_causal_memory_slots
        == v2.model.config.causal_action_state_slots
    )
    assert v2.model.config.causal_action_state_spatial_loss_weight == 0.20
    assert v2.model.config.causal_action_state_relative_geometry_loss_weight == 0.10
    assert v2.model.config.causal_action_state_history_replay_loss_weight == 0.10
    assert list(v2.optimizer.keys_to_select) == list(
        action_policy_robocasa365_causal_state_nano.optimizer.keys_to_select
    )
    assert list(v2.checkpoint.keys_to_skip_loading) == [
        "causal_action_state.slot_type_embed",
        "causal_action_state.memory_slot_norm",
        "causal_action_state.memory_slot_in",
        "causal_action_state.vision_lora_norm",
        "causal_action_state.vision_lora_down",
        "causal_action_state.vision_lora_up",
        "causal_action_state.spatial_query_mlp",
        "causal_action_state.spatial_decoder_layers",
        "causal_action_state.spatial_out_norm",
        "causal_action_state.history_replay_head",
    ]
    assert "causal_action_state" not in v2.checkpoint.keys_to_skip_loading
    assert "action2llm" not in v2.checkpoint.keys_to_skip_loading
    assert "llm2action" not in v2.checkpoint.keys_to_skip_loading
    assert v2.checkpoint.load_ema_to_reg is True
    assert v2.checkpoint.load_training_state is False
    assert v2.checkpoint.strict_resume is False
