# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Nano action-policy SFT recipe for the CWAM RoboCasa365 view."""

import copy
import os

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.action_gradient_evidence import ActionGradientEvidence
from cosmos_framework.callbacks.production_batch_memory_gate import ProductionBatchMemoryGate
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_robocasa365_sft_dataset,
)
from cosmos_framework.data.generator.action.datasets.action_sft_visual_memory_dataset import (
    get_action_robocasa365_visual_memory_sft_dataset,
)
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.model.generator.tokenizers.precomputed_wan2pt2 import PrecomputedWan2pt2LatentTokenizer
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _latent_cache_settings(*, root_env: str, mode_env: str) -> tuple[str | None, str]:
    latent_cache_root = os.environ.get(root_env) or None
    latent_cache_mode = os.environ.get(
        mode_env,
        "required" if latent_cache_root else "disabled",
    )
    if latent_cache_mode not in {"disabled", "required"}:
        raise ValueError(f"{mode_env} must be disabled or required, got {latent_cache_mode!r}.")
    if latent_cache_mode == "required" and not latent_cache_root:
        raise ValueError(f"{root_env} is required when {mode_env}=required.")
    if latent_cache_mode == "disabled":
        latent_cache_root = None
    return latent_cache_root, latent_cache_mode


def _action_policy_robocasa365_nano_model_config(
    *,
    latent_cache_root_env: str = "ROBOCASA365_LATENT_CACHE_ROOT",
    latent_cache_mode_env: str = "ROBOCASA365_LATENT_CACHE_MODE",
    metadata_only_tokenizer_env: str = "CWAM_METADATA_ONLY_VIDEO_TOKENIZER",
) -> dict:
    """Return the Nano action-policy model settings used by this recipe."""
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)
    cfg["max_num_tokens_after_packing"] = 74000
    activation_checkpointing_mode = os.environ.get("CWAM_ACTIVATION_CHECKPOINTING_MODE", "selective")
    if activation_checkpointing_mode not in {"selective", "full", "none"}:
        raise ValueError(
            "CWAM_ACTIVATION_CHECKPOINTING_MODE must be one of selective/full/none, "
            f"got {activation_checkpointing_mode!r}."
        )
    cfg["activation_checkpointing"]["mode"] = activation_checkpointing_mode
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    latent_cache_root, latent_cache_mode = _latent_cache_settings(
        root_env=latent_cache_root_env,
        mode_env=latent_cache_mode_env,
    )
    metadata_only_raw = os.environ.get(metadata_only_tokenizer_env, "0")
    if metadata_only_raw not in {"0", "1"}:
        raise ValueError(f"{metadata_only_tokenizer_env} must be exactly 0 or 1.")
    metadata_only_tokenizer = metadata_only_raw == "1"
    if metadata_only_tokenizer and latent_cache_mode != "required":
        raise ValueError(
            f"{metadata_only_tokenizer_env}=1 requires "
            f"{latent_cache_mode_env}=required."
        )
    if metadata_only_tokenizer:
        # The strict cache reader validates the Wan weights/config/code hashes.
        # Keep only shape/time metadata in the training process so the unused
        # 2.8 GB Wan VAE does not occupy every GPU.
        cfg["tokenizer"]["_target_"] = (
            f"{PrecomputedWan2pt2LatentTokenizer.__module__}.{PrecomputedWan2pt2LatentTokenizer.__qualname__}"
        )
        cfg["tokenizer"]["latent_ch"] = 48
    # Preserve the public Cosmos3-Nano online-tokenizer contract in both
    # modes; the metadata-only implementation accepts this field without
    # loading weights.
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]
    # H-cluster GPU workers are offline. Require an explicitly staged local
    # Qwen processor instead of allowing a latent Hugging Face download.
    cfg["vlm_config"]["tokenizer"]["pretrained_model_name"] = "${oc.env:QWEN3_VL_TOKENIZER_PATH}"
    return cfg


def _robocasa365_packing_dataloader(
    *,
    split: str,
    cfg_dropout_rate: float,
    iterable_shuffle: bool,
    episode_shuffle_seed: int,
    num_workers: int,
    in_order: bool,
    restart_on_iter: bool = False,
    max_samples_per_batch: int = 128,
    camera_mode: str = "concat_view",
    resolution: str | None = None,
    latent_cache_root_env: str = "ROBOCASA365_LATENT_CACHE_ROOT",
    latent_cache_mode_env: str = "ROBOCASA365_LATENT_CACHE_MODE",
    latent_cache_expected_stage: str = "post_padding_crop",
    control_cache_root: str | None = None,
    proprio_history_length: int = 0,
    proprio_history_stride: int = 1,
    action_history_length: int = 0,
    visual_history_cache_root: str | None = None,
    visual_history_length: int = 3,
    visual_history_stride: int = 12,
):
    """Build a train/validation loader with one shared causal data contract."""
    latent_cache_root, latent_cache_mode = _latent_cache_settings(
        root_env=latent_cache_root_env,
        mode_env=latent_cache_mode_env,
    )
    fixed_index_trace_path = (
        (os.environ.get("ROBOCASA365_FIXED_INDEX_TRACE_PATH") or None) if split == "train" else None
    )
    dataset_factory = (
        get_action_robocasa365_visual_memory_sft_dataset
        if visual_history_cache_root is not None
        else get_action_robocasa365_sft_dataset
    )
    dataset_kwargs = dict(
        root="${oc.env:ROBOCASA365_ROOT}",
        source_root="${oc.env:ROBOCASA365_SOURCE_ROOT}",
        split=split,
        fps=20,
        chunk_length=16,
        image_size=256,
        mode="policy",
        camera_mode=camera_mode,
        caption_mode="policy",
        action_normalization="quantile_rot",
        embodiment_type="robocasa365",
        video_backend="lerobot",
        latent_cache_root=latent_cache_root,
        latent_cache_mode=latent_cache_mode,
        latent_cache_expected_stage=latent_cache_expected_stage,
        control_cache_root=control_cache_root,
        control_cache_verify_checksums=False,
        proprio_history_length=proprio_history_length,
        proprio_history_stride=proprio_history_stride,
        action_history_length=action_history_length,
        iterable_shuffle=iterable_shuffle,
        episode_shuffle_seed=episode_shuffle_seed,
        fixed_index_trace_path=fixed_index_trace_path,
        resolution=resolution,
        max_action_dim="${model.config.max_action_dim}",
        cfg_dropout_rate=cfg_dropout_rate,
        append_idle_frames=False,
        format_prompt_as_json=True,
        tokenizer_config="${model.config.vlm_config.tokenizer}",
    )
    if visual_history_cache_root is not None:
        dataset_kwargs.update(
            visual_history_cache_root=visual_history_cache_root,
            visual_history_length=visual_history_length,
            visual_history_stride=visual_history_stride,
        )
    return L(PackingDataLoader)(
        audio_sample_rate=48000,
        dataset_name=f"action_robocasa365_{split}",
        max_samples_per_batch=max_samples_per_batch,
        max_sequence_length=None,
        patch_spatial=2,
        sound_latent_fps=0,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        restart_on_iter=restart_on_iter,
        dataloader=L(RankPartitionedDataLoader)(
            batch_size=1,
            in_order=in_order,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            sampler=None,
            datasets=dict(
                robocasa365=dict(
                    ratio=1,
                    dataset=L(dataset_factory)(**dataset_kwargs),
                ),
            ),
        ),
    )


action_policy_robocasa365_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action_sft",
            name="action_policy_robocasa365_nano",
            wandb_mode="disabled",
        ),
        model=dict(config=_action_policy_robocasa365_nano_model_config()),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=5.0e-05,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,
            max_val_iter=8,
            run_validation=True,
            run_validation_on_start=True,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=500,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200,
                    log_memory_detail=True,
                    save_s3=False,
                    step_size=1,
                    upload_every_n_mul=5,
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                action_gradient_evidence=L(ActionGradientEvidence)(
                    families=["action2llm", "llm2action", "action_modality_embed"],
                    every_n=50,
                    fail_on_first_step=True,
                ),
                # This experiment is explicitly single-node 8xH200: the first
                # real production step either proves per-rank headroom and the
                # same process continues, or fails closed before a long run.
                production_batch_memory_gate=L(ProductionBatchMemoryGate)(
                    max_peak_reserved_fraction=0.94,
                    max_driver_used_fraction=0.94,
                    expected_samples_per_microbatch=128,
                    expected_grad_accum_iter=2,
                    expected_action_tokens_per_sample=16,
                    allowed_world_sizes=[8],
                    required=True,
                ),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # The public Nano checkpoint has no RoboCasa365-trained action
            # domain row; initialize all action heads cleanly for this baseline.
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=_robocasa365_packing_dataloader(
            split="train",
            cfg_dropout_rate=0.1,
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            num_workers=int(os.environ.get("ROBOCASA365_TRAIN_NUM_WORKERS", "4")),
            # Stable cross-worker ordering is required for reproducible traces
            # and is a prerequisite for any future exact cursor resume.
            in_order=True,
        ),
        dataloader_val=_robocasa365_packing_dataloader(
            split="val",
            cfg_dropout_rate=0.0,
            iterable_shuffle=True,
            episode_shuffle_seed=4242,
            num_workers=int(os.environ.get("ROBOCASA365_VAL_NUM_WORKERS", "2")),
            in_order=True,
            restart_on_iter=True,
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_nano",
    node=action_policy_robocasa365_nano,
)


def _action_policy_robocasa365_s1_nano_model_config() -> dict:
    """Enable the S1 soft action-relevance residual on the baseline policy."""
    cfg = _action_policy_robocasa365_nano_model_config()
    cfg.update(
        action_relevance_enabled=True,
        action_relevance_routing_dim=512,
        action_relevance_num_heads=8,
        action_relevance_topk=32,
        action_relevance_initial_gate=0.01,
        action_relevance_sparsity_weight=0.01,
        action_relevance_sparsity_start_iter=1000,
        action_relevance_sparsity_anneal_iters=4000,
    )
    return cfg


action_policy_robocasa365_s1_nano = copy.deepcopy(action_policy_robocasa365_nano)
action_policy_robocasa365_s1_nano.job.name = "action_policy_robocasa365_s1_nano"
action_policy_robocasa365_s1_nano.model.config = _action_policy_robocasa365_s1_nano_model_config()
action_policy_robocasa365_s1_nano.optimizer.keys_to_select.append("action_relevance")
action_policy_robocasa365_s1_nano.optimizer.lr_multipliers["action_relevance"] = 5.0
action_policy_robocasa365_s1_nano.trainer.callbacks.action_gradient_evidence.families.append("action_relevance")
action_policy_robocasa365_s1_nano.checkpoint.keys_to_skip_loading.append("action_relevance")

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_s1_nano",
    node=action_policy_robocasa365_s1_nano,
)


def _action_policy_robocasa365_strong_nano_model_config() -> dict:
    """Enable the three-view DeMiAn-inspired action tower and control memory."""
    cfg = _action_policy_robocasa365_nano_model_config(
        latent_cache_root_env="ROBOCASA365_STRONG_LATENT_CACHE_ROOT",
        latent_cache_mode_env="ROBOCASA365_STRONG_LATENT_CACHE_MODE",
        metadata_only_tokenizer_env="CWAM_STRONG_METADATA_ONLY_VIDEO_TOKENIZER",
    )
    cfg.update(
        action_relevance_enabled=False,
        strong_action_expert_enabled=True,
        strong_action_expert_model_dim=1024,
        strong_action_expert_num_heads=16,
        strong_action_expert_num_layers=4,
        strong_action_expert_ffn_multiplier=4,
        strong_action_state_dim=16,
        strong_action_history_action_dim=10,
        strong_action_proprio_history_length=4,
        strong_action_proprio_history_stride=12,
        strong_action_history_length=16,
        strong_action_memory_layers=2,
        strong_action_memory_tokens=4,
        strong_action_num_views=3,
        # 256 px/view / Wan spatial compression 16 / network patch 2.
        strong_action_view_token_width=8,
        strong_action_initial_gate=0.01,
        strong_action_input_resolution="480",
    )
    return cfg


def _strong_robocasa365_packing_dataloader(
    *,
    split: str,
    cfg_dropout_rate: float,
    iterable_shuffle: bool,
    episode_shuffle_seed: int,
    num_workers: int,
    in_order: bool,
    restart_on_iter: bool = False,
    visual_history_cache_root: str | None = None,
    visual_history_length: int = 3,
    visual_history_stride: int = 12,
):
    return _robocasa365_packing_dataloader(
        split=split,
        cfg_dropout_rate=cfg_dropout_rate,
        iterable_shuffle=iterable_shuffle,
        episode_shuffle_seed=episode_shuffle_seed,
        num_workers=num_workers,
        in_order=in_order,
        restart_on_iter=restart_on_iter,
        max_samples_per_batch=32,
        camera_mode="triple_view",
        resolution="480",
        latent_cache_root_env="ROBOCASA365_STRONG_LATENT_CACHE_ROOT",
        latent_cache_mode_env="ROBOCASA365_STRONG_LATENT_CACHE_MODE",
        latent_cache_expected_stage="full_padding",
        control_cache_root="${oc.env:ROBOCASA365_CONTROL_CACHE_ROOT}",
        proprio_history_length=4,
        proprio_history_stride=12,
        action_history_length=16,
        visual_history_cache_root=visual_history_cache_root,
        visual_history_length=visual_history_length,
        visual_history_stride=visual_history_stride,
    )


action_policy_robocasa365_strong_nano = copy.deepcopy(action_policy_robocasa365_nano)
action_policy_robocasa365_strong_nano.job.name = "action_policy_robocasa365_strong_nano"
action_policy_robocasa365_strong_nano.model.config = _action_policy_robocasa365_strong_nano_model_config()
action_policy_robocasa365_strong_nano.optimizer.keys_to_select.append("strong_action_expert")
action_policy_robocasa365_strong_nano.optimizer.lr_multipliers["strong_action_expert"] = 5.0
action_policy_robocasa365_strong_nano.trainer.grad_accum_iter = 2
action_policy_robocasa365_strong_nano.trainer.callbacks.action_gradient_evidence.families.append(
    "strong_action_expert"
)
strong_memory_gate = action_policy_robocasa365_strong_nano.trainer.callbacks.production_batch_memory_gate
strong_memory_gate.expected_samples_per_microbatch = 32
action_policy_robocasa365_strong_nano.checkpoint.keys_to_skip_loading.append("strong_action_expert")
action_policy_robocasa365_strong_nano.dataloader_train = _strong_robocasa365_packing_dataloader(
    split="train",
    cfg_dropout_rate=0.1,
    iterable_shuffle=True,
    episode_shuffle_seed=42,
    num_workers=int(os.environ.get("ROBOCASA365_TRAIN_NUM_WORKERS", "4")),
    in_order=True,
)
action_policy_robocasa365_strong_nano.dataloader_val = _strong_robocasa365_packing_dataloader(
    split="val",
    cfg_dropout_rate=0.0,
    iterable_shuffle=True,
    episode_shuffle_seed=4242,
    num_workers=int(os.environ.get("ROBOCASA365_VAL_NUM_WORKERS", "2")),
    in_order=True,
    restart_on_iter=True,
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_strong_nano",
    node=action_policy_robocasa365_strong_nano,
)


def _action_policy_robocasa365_b2_nano_model_config() -> dict:
    """Add action-relevance routing and its auxiliary action objective to Strong."""
    cfg = _action_policy_robocasa365_strong_nano_model_config()
    cfg.update(
        action_relevance_enabled=True,
        action_relevance_routing_dim=512,
        action_relevance_num_heads=8,
        action_relevance_topk=32,
        action_relevance_initial_gate=0.01,
        action_relevance_sparsity_weight=0.01,
        action_relevance_sparsity_start_iter=1000,
        action_relevance_sparsity_anneal_iters=4000,
        action_relevance_aux_loss_weight=0.25,
        action_relevance_aux_start_iter=0,
        action_relevance_aux_warmup_iters=1000,
    )
    return cfg


action_policy_robocasa365_b2_nano = copy.deepcopy(action_policy_robocasa365_strong_nano)
action_policy_robocasa365_b2_nano.job.name = "action_policy_robocasa365_b2_nano"
action_policy_robocasa365_b2_nano.model.config = _action_policy_robocasa365_b2_nano_model_config()
action_policy_robocasa365_b2_nano.optimizer.keys_to_select.append("action_relevance")
action_policy_robocasa365_b2_nano.optimizer.lr_multipliers["action_relevance"] = 5.0
action_policy_robocasa365_b2_nano.trainer.callbacks.action_gradient_evidence.families.append(
    "action_relevance"
)
action_policy_robocasa365_b2_nano.checkpoint.keys_to_skip_loading.append("action_relevance")

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_b2_nano",
    node=action_policy_robocasa365_b2_nano,
)


def _action_policy_robocasa365_b3_dselect_nano_model_config() -> dict:
    """Replace B2's top-k-mass proxy with matched-budget DSelect-k support."""
    cfg = _action_policy_robocasa365_b2_nano_model_config()
    cfg.update(
        action_relevance_routing_mode="dselect_k",
        action_relevance_dselect_num_selectors=32,
        action_relevance_dselect_capacity=8192,
        action_relevance_dselect_gamma=1.0,
        action_relevance_dselect_overlap_weight=0.1,
        action_relevance_dselect_invalid_weight=1.0,
    )
    return cfg


action_policy_robocasa365_b3_dselect_nano = copy.deepcopy(action_policy_robocasa365_b2_nano)
action_policy_robocasa365_b3_dselect_nano.job.name = "action_policy_robocasa365_b3_dselect_nano"
action_policy_robocasa365_b3_dselect_nano.model.config = (
    _action_policy_robocasa365_b3_dselect_nano_model_config()
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_b3_dselect_nano",
    node=action_policy_robocasa365_b3_dselect_nano,
)


def _action_policy_robocasa365_b4_adaptive_dselect_nano_model_config() -> dict:
    """Make B3's fixed selector budget adaptive while retaining its maximum capacity."""
    cfg = _action_policy_robocasa365_b3_dselect_nano_model_config()
    cfg.update(
        action_relevance_routing_mode="adaptive_dselect_k",
        action_relevance_dselect_adaptive_counts=[4, 8, 12, 16, 24, 32],
        action_relevance_dselect_adaptive_temperature=1.0,
        action_relevance_dselect_adaptive_target_selectors=24.0,
        action_relevance_dselect_adaptive_budget_weight=1.0,
        action_relevance_dselect_adaptive_entropy_weight=0.01,
    )
    return cfg


action_policy_robocasa365_b4_adaptive_dselect_nano = copy.deepcopy(action_policy_robocasa365_b3_dselect_nano)
action_policy_robocasa365_b4_adaptive_dselect_nano.job.name = "action_policy_robocasa365_b4_adaptive_dselect_nano"
action_policy_robocasa365_b4_adaptive_dselect_nano.model.config = (
    _action_policy_robocasa365_b4_adaptive_dselect_nano_model_config()
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_b4_adaptive_dselect_nano",
    node=action_policy_robocasa365_b4_adaptive_dselect_nano,
)


def _action_policy_robocasa365_b5_visual_object_memory_nano_model_config() -> dict:
    """Add persistent cross-replan visual object slots to B4."""

    cfg = _action_policy_robocasa365_b4_adaptive_dselect_nano_model_config()
    intervention = os.environ.get("CWAM_VISUAL_MEMORY_INTERVENTION", "none")
    if intervention not in {"none", "zero", "shuffle"}:
        raise ValueError("CWAM_VISUAL_MEMORY_INTERVENTION must be none, zero, or shuffle")
    delta_component = os.environ.get("CWAM_VISUAL_MEMORY_DELTA_COMPONENT", "full")
    if delta_component not in {"base", "gate", "residual", "full"}:
        raise ValueError("CWAM_VISUAL_MEMORY_DELTA_COMPONENT must be base, gate, residual, or full")
    exclusive_dynamic_raw = os.environ.get("CWAM_VISUAL_MEMORY_EXCLUSIVE_DYNAMIC", "0")
    if exclusive_dynamic_raw not in {"0", "1"}:
        raise ValueError("CWAM_VISUAL_MEMORY_EXCLUSIVE_DYNAMIC must be exactly 0 or 1")
    exclusive_dynamic = exclusive_dynamic_raw == "1"
    delta_conditioning_layers = int(os.environ.get("CWAM_VISUAL_MEMORY_DELTA_CONDITIONING_LAYERS", "0"))
    if delta_conditioning_layers not in {0, 4}:
        raise ValueError("CWAM_VISUAL_MEMORY_DELTA_CONDITIONING_LAYERS must be 0 or 4")
    if delta_conditioning_layers and not exclusive_dynamic:
        raise ValueError("Delta conditioning layers require exclusive dynamic memory")
    cfg.update(
        strong_action_visual_memory_enabled=True,
        strong_action_visual_history_length=3,
        strong_action_visual_history_stride=12,
        strong_action_visual_latent_channels=48,
        strong_action_visual_latent_height=15,
        strong_action_visual_latent_width=26,
        strong_action_visual_memory_layers=2,
        strong_action_visual_memory_tokens=8,
        strong_action_visual_memory_initial_gate=0.1,
        strong_action_visual_memory_conditioning_scale=0.25,
        strong_action_visual_memory_delta_main_enabled=True,
        strong_action_visual_memory_delta_gate_scale=0.25,
        strong_action_visual_memory_delta_residual_scale=0.25,
        strong_action_visual_memory_delta_main_component=delta_component,
        strong_action_visual_memory_exclusive_dynamic=exclusive_dynamic,
        strong_action_visual_memory_delta_conditioning_layers=delta_conditioning_layers,
        strong_action_visual_memory_delta_conditioning_scale=0.25,
        strong_action_visual_memory_diversity_loss_weight=0.20,
        strong_action_visual_memory_assignment_entropy_loss_weight=0.05,
        strong_action_visual_memory_assignment_balance_loss_weight=0.05,
        strong_action_visual_memory_spatial_compactness_loss_weight=0.05,
        strong_action_visual_memory_temporal_loss_weight=0.02,
        strong_action_visual_memory_object_contrastive_loss_weight=0.10,
        strong_action_visual_memory_episode_contrastive_loss_weight=0.10,
        strong_action_visual_memory_contrastive_temperature=0.07,
        strong_action_visual_memory_action_gradient_loss_weight=0.05,
        strong_action_visual_memory_action_aux_loss_weight=0.25,
        strong_action_visual_memory_action_aux_warmup_iters=50,
        strong_action_visual_memory_intervention=intervention,
    )
    return cfg


action_policy_robocasa365_b5_visual_object_memory_nano = copy.deepcopy(
    action_policy_robocasa365_b4_adaptive_dselect_nano
)
action_policy_robocasa365_b5_visual_object_memory_nano.job.name = (
    "action_policy_robocasa365_b5_visual_object_memory_nano"
)
action_policy_robocasa365_b5_visual_object_memory_nano.model.config = (
    _action_policy_robocasa365_b5_visual_object_memory_nano_model_config()
)
if action_policy_robocasa365_b5_visual_object_memory_nano.model.config.strong_action_visual_memory_exclusive_dynamic:
    action_policy_robocasa365_b5_visual_object_memory_nano.trainer.callbacks.action_gradient_evidence.excluded_parameter_patterns = [
        r"strong_action_expert\.visual_memory_context_embed$",
        r"strong_action_expert\.visual_memory_gate_logit$",
    ]
action_policy_robocasa365_b5_visual_object_memory_nano.dataloader_train = (
    _strong_robocasa365_packing_dataloader(
        split="train",
        cfg_dropout_rate=0.1,
        iterable_shuffle=True,
        episode_shuffle_seed=42,
        num_workers=int(os.environ.get("ROBOCASA365_TRAIN_NUM_WORKERS", "4")),
        in_order=True,
        visual_history_cache_root="${oc.env:ROBOCASA365_VISUAL_HISTORY_CACHE_ROOT}",
        visual_history_length=3,
        visual_history_stride=12,
    )
)
action_policy_robocasa365_b5_visual_object_memory_nano.dataloader_val = (
    _strong_robocasa365_packing_dataloader(
        split="val",
        cfg_dropout_rate=0.0,
        iterable_shuffle=True,
        episode_shuffle_seed=4242,
        num_workers=int(os.environ.get("ROBOCASA365_VAL_NUM_WORKERS", "2")),
        in_order=True,
        restart_on_iter=True,
        visual_history_cache_root="${oc.env:ROBOCASA365_VISUAL_HISTORY_CACHE_ROOT}",
        visual_history_length=3,
        visual_history_stride=12,
    )
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_b5_visual_object_memory_nano",
    node=action_policy_robocasa365_b5_visual_object_memory_nano,
)
