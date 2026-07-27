# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.transforms import VideoResize
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader
from examples.validate_action_policy_robocasa365_nano_batch import (
    _tensor_bytes,
    _validate_packed_batch,
)


def _synthetic_loader() -> PackingDataLoader:
    loader = object.__new__(PackingDataLoader)
    loader.tokenizer_spatial_compression_factor = 16
    loader.tokenizer_temporal_compression_factor = 4
    loader.patch_spatial = 2
    loader.sound_latent_fps = 0
    loader.audio_sample_rate = 48000
    loader.uniae_chunk_frames = None
    loader.uniae_pad_frames = None
    return loader


def _synthetic_batch() -> dict:
    count = 2
    return {
        "__url__": ["robocasa365://synthetic/train"] * count,
        "__key__": [f"{index:09d}" for index in range(count)],
        "ai_caption": [
            json.dumps({"actions": [{"description": f"synthetic action {index}"}]}) for index in range(count)
        ],
        "mode": ["policy"] * count,
        "sequence_plan": [
            SimpleNamespace(
                has_text=True,
                has_vision=True,
                has_action=True,
                condition_frame_indexes_vision=[0],
                condition_frame_indexes_action=[],
            )
            for _ in range(count)
        ],
        "video": [[torch.zeros((3, 5, 16, 32), dtype=torch.uint8)] for _ in range(count)],
        "action": [[torch.ones((4, 8), dtype=torch.float32)] for _ in range(count)],
        "action_raw": [[torch.ones((4, 2), dtype=torch.float32)] for _ in range(count)],
        "text_token_ids": [
            [torch.arange(3, dtype=torch.int64)],
            [torch.arange(4, dtype=torch.int64)],
        ],
    }


def _validate(batch: dict) -> dict:
    return _validate_packed_batch(
        _synthetic_loader(),
        batch,
        expected_sample_count=2,
        expected_video_shape=(3, 5, 16, 32),
        expected_action_shape=(4, 8),
        expected_raw_action_shape=(4, 2),
    )


def test_resolution_none_maps_raw_concat_view_to_production_canvas() -> None:
    raw = torch.zeros((3, 17, 256, 512), dtype=torch.uint8)

    transformed = VideoResize()({"video": raw}, resolution=None)

    assert transformed["video"].shape == (3, 17, 192, 320)
    assert transformed["image_size"].tolist() == [192.0, 320.0, 160.0, 320.0]


def test_validate_packed_batch_recomputes_tokens_and_tensor_bytes() -> None:
    batch = _synthetic_batch()

    metrics = _validate(batch)

    # Per sample: text+EOS=(3/4)+1, vision=2 latent frames+2 markers,
    # action=4 tokens. Therefore the two samples contain 12 and 13 tokens.
    assert metrics["estimated_tokens"] == 25
    assert metrics["estimated_tokens_min_per_sample"] == 12
    assert metrics["estimated_tokens_max_per_sample"] == 13
    assert metrics["sample_count"] == 2
    assert metrics["unique_url_key_count"] == 2
    assert metrics["tensor_bytes"] == 15_736
    assert metrics["tensor_bytes"] == _tensor_bytes(batch)


def test_validate_packed_batch_rejects_duplicate_stable_identity() -> None:
    batch = _synthetic_batch()
    batch["__key__"][1] = batch["__key__"][0]

    with pytest.raises(AssertionError, match="duplicate stable"):
        _validate(batch)


def test_validate_packed_batch_rejects_nonfinite_action() -> None:
    batch = _synthetic_batch()
    batch["action"][1][0][0, 0] = torch.nan

    with pytest.raises(AssertionError, match="non-finite"):
        _validate(batch)
