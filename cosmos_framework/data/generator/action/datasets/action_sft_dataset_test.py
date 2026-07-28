# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import itertools
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import cosmos_framework.data.generator.action.datasets.action_sft_dataset as action_sft_module
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ActionFixedIndexTraceDataset,
    ActionIterableShuffleDataset,
    build_action_fixed_index_trace,
    write_action_fixed_index_trace,
)


class _BlockedDataset(Dataset):
    def __init__(self) -> None:
        self.blocks = [(0, 3), (3, 2), (5, 4), (9, 1), (10, 2), (12, 3), (15, 2), (17, 3)]

    def __len__(self) -> int:
        return 20

    def __getitem__(self, index: int) -> int:
        return index

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return list(self.blocks)


class _RandomBlockedDataset(_BlockedDataset):
    def __getitem__(self, index: int) -> dict:
        return {
            "__key__": f"{index:09d}",
            "python_random": random.random(),
            "numpy_random": float(np.random.random()),
            "torch_random": float(torch.rand(()).item()),
        }


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


def test_fixed_trace_reproduces_baseline_worker_interleave(monkeypatch) -> None:
    base = _BlockedDataset()
    world_size = 2
    baseline_workers = 2
    items_per_rank = 24
    trace = build_action_fixed_index_trace(
        base,
        world_size=world_size,
        baseline_num_workers=baseline_workers,
        seed=17,
        indices_per_rank=items_per_rank,
    )

    for rank in range(world_size):
        worker_iterators = []
        for worker in range(baseline_workers):
            monkeypatch.setattr(
                action_sft_module,
                "get_worker_info",
                lambda worker=worker: SimpleNamespace(id=worker, num_workers=baseline_workers),
            )
            iterable = ActionIterableShuffleDataset(base, seed=17)
            iterable.shard_world_size = world_size
            iterable.shard_rank = rank
            iterator = iter(iterable)
            # ``__iter__`` resolves get_worker_info lazily on first next().
            # Prime it while this worker's monkeypatch is active.
            first = next(iterator)
            worker_iterators.append(itertools.chain([first], iterator))
        observed = [
            next(worker_iterators[position % baseline_workers]) for position in range(items_per_rank)
        ]
        assert observed == trace[rank].tolist()


def test_fixed_trace_is_identical_across_worker_counts_and_determinizes_random_transforms(
    monkeypatch,
    tmp_path,
) -> None:
    base = _RandomBlockedDataset()
    trace_path = tmp_path / "fixed-index-trace.npy"
    write_action_fixed_index_trace(
        base,
        trace_path,
        world_size=2,
        baseline_num_workers=2,
        seed=17,
        transform_seed=29,
        indices_per_rank=144,
        batch_size_per_rank=4,
        grad_accum_iter=1,
        warmup_steps=1,
        measure_steps=1,
        prefetch_factor=4,
        in_order=True,
    )

    observed_by_workers: dict[int, list[dict]] = {}
    for num_workers in (1, 2, 4):
        dataset = ActionFixedIndexTraceDataset(base, trace_path)
        dataset.shard_world_size = 2
        dataset.shard_rank = 1
        worker_outputs: list[iter] = []
        for worker in range(num_workers):
            monkeypatch.setattr(
                action_sft_module,
                "get_worker_info",
                lambda worker=worker, num_workers=num_workers: SimpleNamespace(
                    id=worker,
                    num_workers=num_workers,
                ),
            )
            iterator = iter(dataset)
            first = next(iterator)
            worker_outputs.append(itertools.chain([first], iterator))
        observed_by_workers[num_workers] = [
            next(worker_outputs[position % num_workers]) for position in range(len(dataset))
        ]

    reference = observed_by_workers[1]
    for num_workers, observed in observed_by_workers.items():
        assert observed == reference, f"worker count {num_workers} changed the logical transformed trace"
    assert [sample["__trace_position__"] for sample in reference] == list(range(len(reference)))
    assert all(sample["__trace_rank__"] == 1 for sample in reference)


def test_fixed_trace_real_dataloader_order_is_identical_for_0_2_4_8_workers(tmp_path) -> None:
    base = _RandomBlockedDataset()
    trace_path = tmp_path / "fixed-index-trace.npy"
    write_action_fixed_index_trace(
        base,
        trace_path,
        world_size=1,
        baseline_num_workers=2,
        seed=17,
        transform_seed=29,
        indices_per_rank=144,
        batch_size_per_rank=4,
        grad_accum_iter=1,
        warmup_steps=1,
        measure_steps=1,
        prefetch_factor=4,
        in_order=True,
    )

    observed_by_workers: dict[int, list[tuple]] = {}
    for num_workers in (0, 2, 4, 8):
        dataset = ActionFixedIndexTraceDataset(base, trace_path)
        dataset.shard_world_size = 1
        dataset.shard_rank = 0
        kwargs = {
            "batch_size": 1,
            "num_workers": num_workers,
            "in_order": True,
        }
        if num_workers:
            kwargs["prefetch_factor"] = 4
        loader = DataLoader(dataset, **kwargs)
        observed_by_workers[num_workers] = [
            (
                batch["__key__"][0],
                int(batch["__trace_position__"].item()),
                float(batch["python_random"].item()),
                float(batch["numpy_random"].item()),
                float(batch["torch_random"].item()),
            )
            for batch in loader
        ]

    reference = observed_by_workers[0]
    for num_workers, observed in observed_by_workers.items():
        assert observed == reference, f"real DataLoader worker count {num_workers} changed trace order/content"
