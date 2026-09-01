# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Nano recipe for the ImageWAM RoboFactory multi-robot contract.

Required environment variables:

* ``ROBOFACTORY_ROOT``: native table11 HDF5 tree.
* ``ROBOFACTORY_GAUSSIAN_CACHE_ROOT``: completed canonical Gaussian cache.
* ``ROBOFACTORY_ACTION_STATS_PATH``: ImageWAM-compatible global z-score JSON.
* ``QWEN3_VL_TOKENIZER_PATH``: local Qwen processor directory for offline H workers.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_robofactory_multi_robot_sft_dataset,
)
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _robofactory_multi_robot_model_config() -> dict:
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    # Match ImageWAM's lambda_video=0.5 and lambda_action=1.0 objective.
    cfg["rectified_flow_training_config"]["loss_scale"] = 0.5
    cfg["rectified_flow_training_config"]["image_loss_scale"] = 0.5
    cfg["rectified_flow_training_config"]["action_loss_weight"] = 1.0
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]
    cfg["vlm_config"]["tokenizer"]["pretrained_model_name"] = "${oc.env:QWEN3_VL_TOKENIZER_PATH}"
    cfg["multi_robot_conditioning_enabled"] = True
    cfg["multi_robot_state_dim"] = 18
    cfg["multi_robot_gaussian_channels"] = 13
    cfg["multi_robot_condition_num_heads"] = 8
    cfg["multi_robot_hub_ratio"] = 0.5
    return cfg


def _robofactory_loader(*, split: str, sample_stride: int, cfg_dropout_rate: float, num_workers: int):
    return L(PackingDataLoader)(
        audio_sample_rate=48000,
        dataset_name=f"action_robofactory_multi_robot_{split}",
        max_samples_per_batch=32,
        max_sequence_length=None,
        packing_group_key="agent_count",
        patch_spatial=2,
        sound_latent_fps=0,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        dataloader=L(RankPartitionedDataLoader)(
            batch_size=1,
            in_order=split != "train",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            sampler=None,
            datasets=dict(
                robofactory=dict(
                    ratio=1,
                    dataset=L(get_action_robofactory_multi_robot_sft_dataset)(
                        root="${oc.env:ROBOFACTORY_ROOT}",
                        gaussian_cache_root="${oc.env:ROBOFACTORY_GAUSSIAN_CACHE_ROOT}",
                        action_stats_path="${oc.env:ROBOFACTORY_ACTION_STATS_PATH}",
                        split=split,
                        val_ratio=0.1,
                        split_seed=42,
                        chunk_length=32,
                        image_stride=2,
                        sample_stride=sample_stride,
                        conditioning_fps=10.0,
                        accepted_agent_counts=[1, 2, 3, 4],
                        camera_path="obs/sensor_data/head_camera_global/rgb",
                        gaussian_preflight=True,
                        gaussian_source_path_strip_prefix="tasks",
                        permute_agents=True,
                        permutation_seed=42,
                        resolution=None,
                        max_action_dim="${model.config.max_action_dim}",
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                        cfg_dropout_rate=cfg_dropout_rate,
                        append_idle_frames=False,
                        format_prompt_as_json=True,
                        iterable_shuffle=split == "train",
                        episode_shuffle_seed=42,
                    ),
                ),
            ),
        ),
    )


action_policy_robofactory_multi_robot_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {"override /callbacks": ["basic", "optimization", "job_monitor"]},
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
            name="action_policy_robofactory_multi_robot_nano",
            wandb_mode="disabled",
        ),
        model=dict(config=_robofactory_multi_robot_model_config()),
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
                "multi_robot_conditioner",
            ],
            lr=5.0e-05,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
                "multi_robot_conditioner": 5.0,
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
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            # The formal run evaluates and checkpoints on the same 5k cadence.
            # Keep this in the Python recipe because the structured SFT TOML
            # schema intentionally does not expose validation_iter.
            validation_iter=5000,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # The selected warm-start is the released Nano policy-DROID DCP,
            # which already contains shape-compatible action2llm, llm2action,
            # and action_modality_embed parameters (32 domains, 64-D action,
            # 4096-D hidden).  Preserve those learned policy parameters.  Only
            # the RoboFactory multi-robot conditioner is new in this recipe;
            # EMA is reconstructed from the regular network by the loader.
            keys_to_skip_loading=[
                "net_ema.",
                "multi_robot_conditioner",
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
        # The canonical Table11 Gaussian cache was produced at source_stride=32.
        # Keep training windows on the same grid so every sampled episode/start
        # has a matching Gaussian stream segment.
        dataloader_train=_robofactory_loader(
            split="train", sample_stride=32, cfg_dropout_rate=0.1, num_workers=4
        ),
        dataloader_val=_robofactory_loader(
            split="val", sample_stride=32, cfg_dropout_rate=0.0, num_workers=2
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robofactory_multi_robot_nano",
    node=action_policy_robofactory_multi_robot_nano,
)
