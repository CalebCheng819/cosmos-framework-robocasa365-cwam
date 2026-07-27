# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace

from torch.utils.data import Dataset

import cosmos_framework.data.generator.action.datasets.action_sft_dataset as action_sft_module
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset


class _BlockedDataset(Dataset):
    def __init__(self) -> None:
        self.blocks = [(0, 3), (3, 2), (5, 4), (9, 1), (10, 2), (12, 3), (15, 2), (17, 3)]

    def __len__(self) -> int:
        return 20

    def __getitem__(self, index: int) -> int:
        return index

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return list(self.blocks)


def test_rank_worker_shards_keep_permanent_disjoint_block_ownership(monkeypatch) -> None:
    base = _BlockedDataset()
    world_size = 2
    num_workers = 2
    first_epoch_indices: dict[int, list[int]] = {}

    for rank in range(world_size):
        for worker in range(num_workers):
            monkeypatch.setattr(
                action_sft_module,
                "get_worker_info",
                lambda worker=worker: SimpleNamespace(id=worker, num_workers=num_workers),
            )
            dataset = ActionIterableShuffleDataset(base, seed=17)
            dataset.shard_world_size = world_size
            dataset.shard_rank = rank
            global_shard = rank * num_workers + worker
            owned_blocks = list(range(global_shard, len(base.blocks), world_size * num_workers))
            epoch_size = sum(base.blocks[block][1] for block in owned_blocks)
            iterator = iter(dataset)
            first_epoch_indices[global_shard] = [next(iterator) for _ in range(epoch_size)]

            # Entering the next local epoch must not migrate into another
            # rank-worker's physical episode blocks.
            owned_indices = {
                index
                for block in owned_blocks
                for index in range(base.blocks[block][0], base.blocks[block][0] + base.blocks[block][1])
            }
            assert {next(iterator) for _ in range(epoch_size)} == owned_indices

    all_seen: set[int] = set()
    for global_shard, indices in first_epoch_indices.items():
        current = set(indices)
        assert len(current) == len(indices)
        assert all_seen.isdisjoint(current), f"shard {global_shard} overlaps an earlier shard"
        all_seen.update(current)
    assert all_seen == set(range(len(base)))
