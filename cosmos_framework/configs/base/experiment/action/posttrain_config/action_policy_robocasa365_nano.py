# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Nano action-policy SFT recipe for the CWAM RoboCasa365 view."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.action_gradient_evidence import ActionGradientEvidence
from cosmos_framework.callbacks.production_batch_memory_gate import ProductionBatchMemoryGate
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_robocasa365_sft_dataset,
)
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _action_policy_robocasa365_nano_model_config() -> dict:
    """Return the Nano action-policy model settings used by this recipe."""
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    # Preserve the public Cosmos3-Nano tokenizer contract.
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
):
    """Build a train/validation loader with one shared causal data contract."""
    return L(PackingDataLoader)(
        audio_sample_rate=48000,
        dataset_name=f"action_robocasa365_{split}",
        max_samples_per_batch=128,
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
                    dataset=L(get_action_robocasa365_sft_dataset)(
                        root="${oc.env:ROBOCASA365_ROOT}",
                        source_root="${oc.env:ROBOCASA365_SOURCE_ROOT}",
                        split=split,
                        fps=20,
                        chunk_length=16,
                        image_size=256,
                        mode="policy",
                        camera_mode="concat_view",
                        caption_mode="policy",
                        action_normalization="quantile_rot",
                        embodiment_type="robocasa365",
                        video_backend="lerobot",
                        iterable_shuffle=iterable_shuffle,
                        episode_shuffle_seed=episode_shuffle_seed,
                        resolution=None,
                        max_action_dim="${model.config.max_action_dim}",
                        cfg_dropout_rate=cfg_dropout_rate,
                        # idle_frames is computed from the future action target.
                        # It is valid diagnostic metadata but must not condition
                        # a causal policy.
                        append_idle_frames=False,
                        format_prompt_as_json=True,
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                    ),
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
                # Preserve an acquired 6/8-H200 allocation: the first real
                # production step either proves per-rank headroom and the same
                # process continues, or fails closed before a long run.
                production_batch_memory_gate=L(ProductionBatchMemoryGate)(
                    max_peak_reserved_fraction=0.94,
                    max_driver_used_fraction=0.94,
                    expected_samples_per_microbatch=128,
                    expected_grad_accum_iter=2,
                    expected_action_tokens_per_sample=16,
                    allowed_world_sizes=[6, 8],
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
            num_workers=4,
            # Stable cross-worker ordering is required for reproducible traces
            # and is a prerequisite for any future exact cursor resume.
            in_order=True,
        ),
        dataloader_val=_robocasa365_packing_dataloader(
            split="val",
            cfg_dropout_rate=0.0,
            iterable_shuffle=True,
            episode_shuffle_seed=4242,
            num_workers=2,
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
