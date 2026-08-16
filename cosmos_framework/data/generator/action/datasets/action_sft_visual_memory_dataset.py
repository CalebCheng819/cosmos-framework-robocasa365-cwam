# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action SFT factory for RoboCasa365 cross-replan visual memory."""

from __future__ import annotations

from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ActionFixedIndexTraceDataset,
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.generator.action.datasets.robocasa365_visual_memory_dataset import (
    RoboCasa365VisualMemoryDataset,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline


def get_action_robocasa365_visual_memory_sft_dataset(
    *,
    root: str,
    visual_history_cache_root: str,
    source_root: str | None = None,
    split: str = "train",
    fps: float = 20.0,
    chunk_length: int = 16,
    image_size: int = 256,
    mode: str = "policy",
    camera_mode: str = "triple_view",
    caption_mode: str = "policy",
    action_normalization: str | None = "quantile_rot",
    action_stats_path: str | None = None,
    embodiment_type: str = "robocasa365",
    video_backend: str = "lerobot",
    tolerance_s: float = 1e-4,
    decode_retries: int = 8,
    resample_on_decode_error: bool | None = None,
    validate_manifest: bool = True,
    latent_cache_root: str | None = None,
    latent_cache_mode: str = "required",
    latent_cache_expected_stage: str = "full_padding",
    resolution: str | int | None = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    format_prompt_as_json: bool = True,
    iterable_shuffle: bool = True,
    episode_shuffle_seed: int = 42,
    fixed_index_trace_path: str | None = None,
    control_cache_root: str | None = None,
    control_cache_verify_checksums: bool = False,
    proprio_history_length: int = 4,
    proprio_history_stride: int = 12,
    action_history_length: int = 16,
    visual_history_length: int = 3,
    visual_history_stride: int = 12,
) -> Dataset:
    """Build the B5 dataset while preserving the parent Wan cache contract."""

    dataset = RoboCasa365VisualMemoryDataset(
        root=root,
        source_root=source_root,
        split=split,
        fps=fps,
        chunk_length=chunk_length,
        image_size=image_size,
        mode=mode,
        camera_mode=camera_mode,
        caption_mode=caption_mode,
        action_normalization=action_normalization,
        action_stats_path=action_stats_path,
        embodiment_type=embodiment_type,
        video_backend=video_backend,
        tolerance_s=tolerance_s,
        decode_retries=decode_retries,
        resample_on_decode_error=resample_on_decode_error,
        validate_manifest=validate_manifest,
        latent_cache_root=latent_cache_root,
        latent_cache_mode=latent_cache_mode,
        latent_cache_expected_stage=latent_cache_expected_stage,
        control_cache_root=control_cache_root,
        control_cache_verify_checksums=control_cache_verify_checksums,
        proprio_history_length=proprio_history_length,
        proprio_history_stride=proprio_history_stride,
        action_history_length=action_history_length,
        transform_resolution=resolution,
        visual_history_cache_root=visual_history_cache_root,
        visual_history_length=visual_history_length,
        visual_history_stride=visual_history_stride,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(
        dataset,
        transform,
        resolution,
        action_normalizer=dataset.get_action_normalizer(),
    )
    if fixed_index_trace_path:
        if not iterable_shuffle:
            raise ValueError("fixed_index_trace_path requires iterable_shuffle=True.")
        return ActionFixedIndexTraceDataset(sft, fixed_index_trace_path)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft
