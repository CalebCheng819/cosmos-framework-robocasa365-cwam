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
