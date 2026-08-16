# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.iter_speed import IterSpeed
from cosmos_framework.callbacks.production_batch_memory_gate import ProductionBatchMemoryGate
from cosmos_framework.data.generator.joint_dataloader import JointDataLoader, custom_collate_fn
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.data_and_condition import unwrap_and_densify
from cosmos_framework.utils.generator.data_utils import resolve_vision_batch_key


def _model_without_initialization() -> OmniMoTModel:
    model = object.__new__(OmniMoTModel)
    model.input_image_key = "images"
    model.input_video_key = "video"
    model.net = object()
    return model


def test_cached_latent_is_classified_as_video_and_train_stats_do_not_require_pixels() -> None:
    model = _model_without_initialization()
    batch = {"video_latent": [torch.zeros(48, 5, 10, 20)]}

    assert model.is_image_batch(batch) is False
    model._update_train_stats(batch)


def test_cached_latent_real_collate_split_and_pack_nesting_reaches_model_fast_path() -> None:
    latent_a = torch.zeros(48, 5, 10, 20, dtype=torch.bfloat16)
    latent_b = torch.ones(48, 5, 10, 20, dtype=torch.bfloat16)
    collated = custom_collate_fn(
        [
            {
                "video_latent": latent_a,
                "action": torch.zeros(16, 64),
                "domain_id": torch.tensor(22),
                "conditioning_fps": torch.tensor(20),
            },
            {
                "video_latent": latent_b,
                "action": torch.zeros(16, 64),
                "domain_id": torch.tensor(22),
                "conditioning_fps": torch.tensor(20),
            },
        ]
    )
    loader = object.__new__(JointDataLoader)
    loader.buffers = [deque()]
    loader.dataloaders = [iter([collated])]
    sample_a = loader._get_next_sample(0)
    sample_b = loader._get_next_sample(0)
    packed: dict = {}
    loader._update_output_batch(packed, sample_a)
    loader._update_output_batch(packed, sample_b)

    assert len(packed["video_latent"]) == 2
    assert len(packed["video_latent"][0]) == 1
    assert len(packed["video_latent"][1]) == 1
    assert packed["video_latent"][0][0] is latent_a
    assert packed["video_latent"][1][0] is latent_b
    dense = unwrap_and_densify(packed["video_latent"], {"device": "cpu", "dtype": torch.float32})
    assert dense is not None
    assert [tuple(value.shape) for value in dense] == [(48, 5, 10, 20), (48, 5, 10, 20)]
    assert all(value.dtype == torch.float32 for value in dense)

    model = _model_without_initialization()
    model.training = True
    model.tensor_kwargs = {"device": "cpu", "dtype": torch.bfloat16}
    model.tensor_kwargs_fp32 = {"device": "cpu", "dtype": torch.float32}
    model.tokenizer_vision_gen = SimpleNamespace(latent_ch=48)
    model.tokenizer_sound_gen = None
    model.log_enc_time_every_n = 100
    model.config = SimpleNamespace(
        diffusion_expert_config=SimpleNamespace(vision_temporal_position_mode="latent_index")
    )
    clean = model.get_data_and_condition(packed, iteration=1)
    assert clean.batch_size == 2
    assert clean.raw_state_vision is None
    assert [tuple(value.shape) for value in clean.x0_tokens_vision] == [
        (1, 48, 5, 10, 20),
        (1, 48, 5, 10, 20),
    ]
    assert all(value.dtype == torch.float32 for value in clean.x0_tokens_vision)


def test_online_visual_history_encoding_matches_sidecar_geometry_and_masks_padding() -> None:
    model = _model_without_initialization()
    model.tensor_kwargs_fp32 = {"device": "cpu", "dtype": torch.float32}
    model.config = SimpleNamespace(
        strong_action_visual_history_length=3,
        strong_action_visual_latent_channels=48,
        strong_action_visual_latent_height=15,
        strong_action_visual_latent_width=26,
    )
    latent = torch.arange(3 * 48 * 1 * 30 * 52, dtype=torch.float32).reshape(3, 48, 1, 30, 52)
    model.tokenizer_vision_gen = SimpleNamespace(encode=lambda _: latent)
    pixels = torch.randint(0, 256, (3, 3, 1, 480, 832), dtype=torch.uint8)
    mask = torch.tensor([False, True, True])

    encoded = model._encode_control_visual_history([pixels], [mask])

    assert len(encoded) == 1
    assert encoded[0].shape == (3, 48, 15, 26)
    assert encoded[0].dtype == torch.float32
    assert torch.count_nonzero(encoded[0][0]) == 0
    expected = torch.nn.functional.avg_pool2d(latent[1:, :, 0], kernel_size=2, stride=2)
    torch.testing.assert_close(encoded[0][1:], expected)


@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        ({"images": [torch.zeros(3, 16, 16)]}, ("images", True)),
        ({"video": [torch.zeros(3, 17, 16, 16)]}, ("video", False)),
        ({"video_latent": [torch.zeros(48, 5, 1, 1)]}, ("video_latent", False)),
    ],
)
def test_resolve_vision_batch_key(batch: dict, expected: tuple[str, bool]) -> None:
    assert resolve_vision_batch_key(batch) == expected


@pytest.mark.parametrize(
    "batch",
    [
        {},
        {"video": [], "video_latent": []},
        {"images": [], "video": []},
    ],
)
def test_resolve_vision_batch_key_rejects_missing_or_ambiguous_inputs(batch: dict) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        resolve_vision_batch_key(batch)


def test_required_memory_gate_counts_cached_latent_samples() -> None:
    gate = ProductionBatchMemoryGate(
        expected_samples_per_microbatch=2,
        expected_grad_accum_iter=1,
        allowed_world_sizes=[8],
    )
    gate._start_iteration = 10
    gate.on_training_step_batch_end(
        model=torch.nn.Identity(),
        data_batch={
            "video_latent": [torch.zeros(48, 5, 10, 20), torch.zeros(48, 5, 10, 20)],
            "__url__": ["dataset", "dataset"],
            "__key__": ["000", "001"],
        },
        output_batch={
            "batch_size": 2,
            "und_token_length": 1,
            "gen_token_length": 1,
            "vision_token_length": 500,
            "action_token_length": 32,
        },
        loss=torch.tensor(0.0),
        iteration=10,
    )
    assert gate._microbatches[0]["input_sample_count"] == 2


def test_iter_speed_accepts_cached_latent_video_batch() -> None:
    callback = IterSpeed(every_n=1, hit_thres=0, save_s3=False, barrier_after_run=False)
    callback.time = 1.0
    callback.every_n_impl(
        trainer=SimpleNamespace(),
        model=_model_without_initialization(),
        data_batch={"video_latent": [torch.zeros(48, 5, 10, 20)]},
        output_batch={},
        loss=torch.tensor(0.0),
        iteration=1,
    )
