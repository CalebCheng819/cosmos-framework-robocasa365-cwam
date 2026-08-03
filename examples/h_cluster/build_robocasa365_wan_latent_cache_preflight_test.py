# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the Wan latent-cache builder's adaptive batch preflight."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pytest
import torch

from examples.h_cluster import build_robocasa365_wan_latent_cache as builder


_REQUESTED_BATCH_SIZE = 4
_LATENT_SHAPE = (48, 5, 1, 1)


class _FakeTokenizer:
    """Return correctly shaped latents with injectable per-call behavior."""

    def __init__(self, behavior: Callable[[int, int, torch.Tensor], str] | None = None) -> None:
        self._behavior = behavior or (lambda _batch_size, _call_index, _video: "equal")
        self.calls: list[int] = []

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        batch_size = int(video.shape[0])
        call_index = len(self.calls)
        self.calls.append(batch_size)
        behavior = self._behavior(batch_size, call_index, video)
        if behavior == "cuda_oom":
            raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")

        latent = torch.full(
            (batch_size, *_LATENT_SHAPE),
            0.25,
            device=video.device,
            dtype=torch.float32,
        )
        if behavior == "different":
            latent[0, 0, 0, 0, 0] += 1.0
        return latent


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        requested_vae_batch_size=_REQUESTED_BATCH_SIZE,
        expected_image_size=(16, 16),
        expected_latent_shape=_LATENT_SHAPE,
        latent_stage=builder.CACHE_STAGE,
        splits=("train",),
        vae_batch_size=None,
    )


def _run_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: _FakeTokenizer,
) -> tuple[argparse.Namespace, list[dict[str, object]]]:
    args = _args()
    resized_videos = torch.arange(
        _REQUESTED_BATCH_SIZE * 3 * 17 * 16 * 16,
        dtype=torch.int64,
    ).remainder_(256).to(torch.uint8).reshape(_REQUESTED_BATCH_SIZE, 3, 17, 16, 16)
    base_receipt = {
        "rank": 0,
        "split": "train",
        "sample_indices": list(range(_REQUESTED_BATCH_SIZE)),
    }
    monkeypatch.setattr(
        builder,
        "_prepare_real_backend_preflight",
        lambda *_args, **_kwargs: (tokenizer, resized_videos, base_receipt),
    )

    returned_tokenizer, receipts = builder._distributed_real_backend_preflight(
        args,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        source_root=Path("/not/read"),
    )
    assert returned_tokenizer is tokenizer
    return args, receipts


def test_preflight_keeps_requested_batch_only_when_n_and_every_intermediate_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer()

    args, receipts = _run_preflight(monkeypatch, tokenizer)

    assert args.vae_batch_size == _REQUESTED_BATCH_SIZE
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["effective_vae_batch_size"] == _REQUESTED_BATCH_SIZE
    assert receipt["requested_batch_gate"] == {
        "batch_size": _REQUESTED_BATCH_SIZE,
        "status": "passed",
        "batched_vs_singleton_torch_equal": True,
    }
    assert receipt["intermediate_batch_gates"] == [
        {
            "batch_size": 2,
            "status": "passed",
            "batched_vs_singleton_torch_equal": True,
        },
        {
            "batch_size": 3,
            "status": "passed",
            "batched_vs_singleton_torch_equal": True,
        },
    ]
    assert receipt["fallback_batch_gate"] is None
    assert tokenizer.calls == [1, 1, 1, 1, 4, 2, 3]


def test_preflight_falls_back_to_one_when_an_intermediate_batch_is_not_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer(
        lambda batch_size, _call_index, _video: "different" if batch_size == 3 else "equal"
    )

    args, receipts = _run_preflight(monkeypatch, tokenizer)

    assert args.vae_batch_size == 1
    receipt = receipts[0]
    assert receipt["effective_vae_batch_size"] == 1
    assert receipt["requested_batch_gate"]["status"] == "passed"
    assert receipt["intermediate_batch_gates"][-1] == {
        "batch_size": 3,
        "status": "batch_not_bitwise_equal",
        "batched_vs_singleton_torch_equal": False,
    }
    assert receipt["fallback_batch_gate"] == {
        "batch_size": 1,
        "status": "passed",
        "batched_vs_singleton_torch_equal": True,
    }


def test_preflight_falls_back_to_one_after_requested_batch_cuda_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer(
        lambda batch_size, _call_index, _video: (
            "cuda_oom" if batch_size == _REQUESTED_BATCH_SIZE else "equal"
        )
    )

    args, receipts = _run_preflight(monkeypatch, tokenizer)

    assert args.vae_batch_size == 1
    receipt = receipts[0]
    assert receipt["effective_vae_batch_size"] == 1
    assert receipt["requested_batch_gate"] == {
        "batch_size": _REQUESTED_BATCH_SIZE,
        "status": "cuda_oom",
        "batched_vs_singleton_torch_equal": False,
    }
    assert receipt["intermediate_batch_gates"] == []
    assert receipt["fallback_batch_gate"] == {
        "batch_size": 1,
        "status": "passed",
        "batched_vs_singleton_torch_equal": True,
    }


def test_preflight_treats_nondeterministic_batch_one_fallback_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def behavior(batch_size: int, call_index: int, _video: torch.Tensor) -> str:
        if batch_size == _REQUESTED_BATCH_SIZE:
            return "cuda_oom"
        # After four requested-batch singleton references and the requested
        # batch OOM, the fallback's reference/candidate calls are indices 5/6.
        # Perturb only the candidate so the batch-size-1 equality gate fails.
        if batch_size == 1 and call_index == _REQUESTED_BATCH_SIZE + 2:
            return "different"
        return "equal"

    tokenizer = _FakeTokenizer(behavior)

    with pytest.raises(RuntimeError, match="Wan singleton preflight was not bitwise deterministic"):
        _run_preflight(monkeypatch, tokenizer)


def test_full_padding_stage_keeps_the_complete_480_canvas() -> None:
    args = argparse.Namespace(
        latent_stage=builder.CACHE_STAGE_FULL_PADDING,
        chunk_length=16,
    )
    image_size = (480, 832, 256, 768)
    latent = torch.arange(48 * 5 * 30 * 52, dtype=torch.float32).reshape(48, 5, 30, 52)

    assert builder._derive_latent_shape(args, image_size) == (48, 5, 30, 52)
    selected = builder._select_latent_stage(
        latent,
        image_size=torch.tensor(image_size),
        latent_stage=builder.CACHE_STAGE_FULL_PADDING,
    )
    assert selected.shape == (48, 5, 30, 52)
    assert torch.equal(selected, latent)


def test_post_padding_crop_stage_remains_backward_compatible() -> None:
    args = argparse.Namespace(latent_stage=builder.CACHE_STAGE, chunk_length=16)
    image_size = (480, 832, 256, 768)
    latent = torch.arange(48 * 5 * 30 * 52, dtype=torch.float32).reshape(48, 5, 30, 52)

    assert builder._derive_latent_shape(args, image_size) == (48, 5, 16, 48)
    selected = builder._select_latent_stage(
        latent,
        image_size=torch.tensor(image_size),
        latent_stage=builder.CACHE_STAGE,
    )
    assert selected.shape == (48, 5, 16, 48)
    assert torch.equal(selected, latent[:, :, :16, :48])


def _visual_identity(*, timestamp: float = 16.05, video_path: str = "left.mp4") -> dict[str, object]:
    return {
        "source_revision": "revision-v1",
        "source_prefix": "pretrain/atomic/task/day",
        "source_episode_index": 1,
        "source_frame_index": 87,
        "camera_mode": "agentview_left",
        "camera_fields": ["robot0_agentview_left"],
        "camera_video_paths": [video_path],
        "camera_effective_start_timestamps": [timestamp],
        "fps": 20.0,
        "num_frames": 17,
    }


class _FakeIdentityDataset:
    def __init__(self, identities: list[dict[str, object]]) -> None:
        self.identities = identities

    def __len__(self) -> int:
        return len(self.identities)

    def get_visual_window_identity(self, index: int) -> dict[str, object]:
        return self.identities[index]


def test_full_identity_audit_counts_canonical_overlaps_once(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _visual_identity()
    dataset = _FakeIdentityDataset([identity, dict(identity)])
    monkeypatch.setattr(builder, "_make_dataset", lambda *_args, **_kwargs: dataset)
    args = argparse.Namespace(splits=("train",))

    audit = builder._audit_visual_identity_plan(args, source_root=Path("/not/read"))

    assert audit["status"] == "passed"
    assert audit["logical_samples"] == 2
    assert audit["unique_records"] == 1
    assert audit["duplicate_samples"] == 1
    assert audit["physical_anchors"] == 1
    assert audit["splits"]["train"]["logical_samples"] == 2
    assert audit["splits"]["train"]["unique_records"] == 1
    assert audit["splits"]["train"]["duplicate_samples"] == 1
    assert len(audit["splits"]["train"]["ordered_visual_identity_sha256"]) == 64


def test_full_identity_audit_still_rejects_real_locator_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _FakeIdentityDataset([_visual_identity(), _visual_identity(timestamp=16.10)])
    monkeypatch.setattr(builder, "_make_dataset", lambda *_args, **_kwargs: dataset)
    args = argparse.Namespace(splits=("train",))

    with pytest.raises(RuntimeError, match="disagree on the effective video locator"):
        builder._audit_visual_identity_plan(args, source_root=Path("/not/read"))
