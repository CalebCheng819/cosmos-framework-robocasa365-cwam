# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import copy

import torch

from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.joint_dataloader import JointDataLoader, custom_collate_fn


def _sample(*, cached: bool) -> dict:
    sample = {
        "ai_caption": "Place the mug on the plate.",
        "action": torch.zeros(16, 10),
        "conditioning_fps": torch.tensor(20),
        "domain_id": torch.tensor(22),
        "idle_frames": torch.tensor(0),
        "mode": "policy",
        "viewpoint": "concat_view",
        "additional_view_description": (
            "The left half shows the left third-person agent view; "
            "the right half shows the wrist-mounted camera."
        ),
    }
    if cached:
        sample.update(
            {
                "video_latent": torch.zeros(48, 5, 10, 20, dtype=torch.bfloat16),
                "video_num_frames": torch.tensor(17),
                "image_size": torch.tensor([192.0, 320.0, 160.0, 320.0]),
            }
        )
    else:
        sample["video"] = torch.zeros(3, 17, 256, 512, dtype=torch.uint8)
    return sample


def test_cached_latent_keeps_prompt_and_sequence_plan_equal_to_online_path() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=64,
        append_idle_frames=False,
        format_prompt_as_json=True,
    )
    online = pipeline(copy.deepcopy(_sample(cached=False)), None)
    cached = pipeline(copy.deepcopy(_sample(cached=True)), None)

    assert cached["ai_caption"] == online["ai_caption"]
    assert cached["image_size"].tolist() == online["image_size"].tolist()
    assert cached["sequence_plan"] == online["sequence_plan"]
    assert cached["action_processing_record"] == online["action_processing_record"]
    torch.testing.assert_close(cached["action_raw"], online["action_raw"])
    torch.testing.assert_close(cached["action"], online["action"])
    assert "video" not in cached
    assert cached["video_latent"].shape == (48, 5, 10, 20)


def test_cached_latent_collation_and_token_count() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=64,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
    )
    transformed = pipeline(_sample(cached=True), None)
    online = pipeline(_sample(cached=False), None)
    batch = custom_collate_fn([transformed, copy.deepcopy(transformed)])
    assert isinstance(batch["video_latent"], list)
    assert len(batch["video_latent"]) == 2

    loader = object.__new__(JointDataLoader)
    loader.patch_spatial = 2
    loader.tokenizer_spatial_compression_factor = 16
    loader.tokenizer_temporal_compression_factor = 4
    loader.uniae_chunk_frames = None
    loader.sound_latent_fps = 0
    one_sample = {
        "video_latent": [transformed["video_latent"]],
        "action": [transformed["action"]],
    }
    online_sample = {
        "video": [online["video"]],
        "image_size": online["image_size"],
        "action": [online["action"]],
    }
    # 48 channels do not affect token count: 5 * ceil(10/2) * ceil(20/2)
    # vision tokens plus 16 action tokens.
    assert loader._compute_num_tokens_per_sample(one_sample) == 266
    assert loader._compute_num_tokens_per_sample(online_sample) == 266


def test_token_count_keeps_multi_item_image_sizes_aligned() -> None:
    loader = object.__new__(JointDataLoader)
    loader.patch_spatial = 2
    loader.tokenizer_spatial_compression_factor = 16
    loader.tokenizer_temporal_compression_factor = 4
    loader.uniae_chunk_frames = None
    loader.sound_latent_fps = 0
    sample = {
        "images": [
            torch.zeros(3, 192, 320),
            torch.zeros(3, 192, 320),
        ],
        "image_size": [
            torch.tensor([192.0, 320.0, 160.0, 320.0]),
            torch.tensor([192.0, 320.0, 160.0, 320.0]),
        ],
    }
    # Two post-crop image latents: 2 * ceil(10/2) * ceil(20/2).
    assert loader._compute_num_tokens_per_sample(sample) == 100
