# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.robofactory_gaussian_cache import CANONICAL_CHANNELS
from cosmos_framework.data.generator.action.datasets.robofactory_multi_robot_dataset import (
    RoboFactoryMultiRobotDataset,
    load_imagewam_zscore_normalizer,
)


def _write_hdf5(root: Path, *, agent_count: int = 2) -> Path:
    path = root / "tasks/PlaceFood-rf/demo.h5"
    path.parent.mkdir(parents=True)
    transitions = 48
    with h5py.File(path, "w") as handle:
        trajectory = handle.create_group("trajectory_0")
        actions = trajectory.create_group("actions")
        agents = trajectory.create_group("obs/agent")
        for agent_index in range(agent_count):
            action = np.zeros((transitions, 8), dtype=np.float32)
            action[:] = agent_index * 10_000 + np.arange(transitions)[:, None] * 100 + np.arange(8)
            actions.create_dataset(f"panda-{agent_index}", data=action)
            agent = agents.create_group(f"panda-{agent_index}")
            qpos = np.full((transitions + 1, 9), agent_index + 1, dtype=np.float32)
            qvel = np.full((transitions + 1, 9), agent_index + 101, dtype=np.float32)
            agent.create_dataset("qpos", data=qpos)
            agent.create_dataset("qvel", data=qvel)
        frames = np.empty((transitions + 1, 8, 12, 3), dtype=np.uint8)
        for frame_index in range(transitions + 1):
            frames[frame_index].fill(frame_index)
        trajectory.create_dataset("obs/sensor_data/head_camera_global/rgb", data=frames)
    return path


def _write_gaussian_cache(root: Path, *, agent_count: int = 2) -> Path:
    root.mkdir(parents=True)
    frames = []
    streams = []
    for agent_index in range(agent_count):
        offset = len(frames)
        for sample_index in range(2):
            frames.append(np.full((13, 28, 40), agent_index * 10 + sample_index, dtype="<f2"))
        streams.append(
            {
                "source_path": "PlaceFood-rf/demo.h5",
                "trajectory": "trajectory_0",
                "agent_name": f"panda-{agent_index}",
                "segments": [
                    {
                        "shard": "shard-00000",
                        "offset": offset,
                        "count": 2,
                        "source_start": 0,
                        "source_stride": 16,
                    }
                ],
            }
        )
    array = np.stack(frames).astype("<f2", copy=False)
    shard_path = root / "shard-00000.bin"
    array.tofile(shard_path)
    manifest = {
        "schema": {
            "name": "fastwam.canonical-gaussian-cache",
            "version": 1,
            "channels": list(CANONICAL_CHANNELS),
            "dtype": "float16",
            "byte_order": "little",
            "covariance_order": "row-major-ij",
            "height": 28,
            "width": 40,
        },
        "shards": [
            {
                "id": "shard-00000",
                "path": shard_path.name,
                "frames": int(array.shape[0]),
                "bytes": int(array.nbytes),
            }
        ],
        "streams": streams,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return root


def test_dataset_emits_agent_major_imagewam_contract(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_hdf5(data_root)
    cache_root = _write_gaussian_cache(tmp_path / "gaussian")
    dataset = RoboFactoryMultiRobotDataset(
        data_root,
        split="all",
        val_ratio=0.0,
        sample_stride=16,
        gaussian_cache_root=cache_root,
        permute_agents=False,
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["video"].shape == (3, 17, 8, 12)
    assert sample["video"][0, :, 0, 0].tolist() == list(range(0, 33, 2))
    assert sample["action"].shape == (64, 8)
    assert sample["action"][:32, 0].tolist() == [float(index * 100) for index in range(32)]
    assert sample["action"][32:, 0].tolist() == [float(10_000 + index * 100) for index in range(32)]
    assert sample["agent_state"].shape == (2, 18)
    assert sample["agent_state"][0].tolist() == [1.0] * 9 + [101.0] * 9
    assert sample["agent_gaussian"].shape == (2, 13, 28, 40)
    assert sample["agent_gaussian"][:, 0, 0, 0].tolist() == [0.0, 10.0]
    assert sample["agent_count"].item() == 2
    assert sample["action_horizon_per_agent"].item() == 32
    assert bool(sample["action_token_mask"].all())


def test_permutation_is_synchronized_across_action_state_and_gaussian(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_hdf5(data_root)
    cache_root = _write_gaussian_cache(tmp_path / "gaussian")
    dataset = RoboFactoryMultiRobotDataset(
        data_root,
        split="all",
        val_ratio=0.0,
        sample_stride=16,
        gaussian_cache_root=cache_root,
        permute_agents=True,
        permutation_seed=3,
    )

    sample = dataset[0]
    action_identity = (sample["action"][:, 0].reshape(2, 32)[:, 0] // 10_000).to(torch.long)
    state_identity = sample["agent_state"][:, 0].to(torch.long) - 1
    gaussian_identity = (sample["agent_gaussian"][:, 0, 0, 0] // 10).to(torch.long)
    assert torch.equal(action_identity, state_identity)
    assert torch.equal(action_identity, gaussian_identity)


def test_gaussian_cache_requires_completion_marker(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_hdf5(data_root)
    cache_root = _write_gaussian_cache(tmp_path / "gaussian")
    (cache_root / "COMPLETE").unlink()
    with pytest.raises(FileNotFoundError, match="COMPLETE"):
        RoboFactoryMultiRobotDataset(
            data_root,
            split="all",
            val_ratio=0.0,
            gaussian_cache_root=cache_root,
        )


def test_imagewam_global_zscore_is_loaded_without_clipping(tmp_path: Path) -> None:
    stats = {
        "action": {
            "default": {
                "global_mean": list(range(8)),
                "global_std": [2.0] * 8,
            }
        }
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    normalizer = load_imagewam_zscore_normalizer(path)
    raw = torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
    normalized = normalizer.normalize_action(raw)
    assert torch.equal(normalized, torch.ones_like(normalized))
    assert torch.equal(normalizer.denormalize_action(normalized), raw)
