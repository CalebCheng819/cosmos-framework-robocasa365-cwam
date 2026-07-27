# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import itertools

import torch
from torch.utils.data import DataLoader, Dataset

from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn


class _IndexedVideoDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        return {
            "video": torch.full((3, 1, 16, 16), index, dtype=torch.uint8),
            "text_token_ids": torch.tensor([index], dtype=torch.long),
        }


def _sample_ids(batches: list[dict]) -> list[int]:
    return [int(batch["video"][0][0][0, 0, 0, 0]) for batch in batches]


def test_packing_loader_restart_on_iter_replays_fixed_validation_prefix() -> None:
    inner = DataLoader(
        _IndexedVideoDataset(),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )
    loader = PackingDataLoader(
        dataloader=inner,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_samples_per_batch=1,
        restart_on_iter=True,
    )

    first = list(itertools.islice(iter(loader), 3))
    second = list(itertools.islice(iter(loader), 3))

    assert _sample_ids(first) == [0, 1, 2]
    assert _sample_ids(second) == [0, 1, 2]
    assert loader.global_id == 3
