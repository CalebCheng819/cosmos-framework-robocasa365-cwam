# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.robocasa365_control_cache import (
    RoboCasa365ControlCache,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _cache(tmp_path: Path) -> tuple[RoboCasa365ControlCache, str]:
    root = tmp_path / "control"
    arrays = root / "arrays"
    arrays.mkdir(parents=True)
    states = np.arange(12 * 16, dtype=np.float32).reshape(12, 16)
    actions = np.arange(12 * 10, dtype=np.float32).reshape(12, 10)
    action_valid = np.ones(12, dtype=np.bool_)
    action_valid[6] = False
    actions[6] = 0.0
    episode_offsets = np.asarray([[0, 5], [5, 7]], dtype=np.int64)
    values = {
        "states": ("arrays/states.npy", states),
        "actions_rot6d10": ("arrays/actions.npy", actions),
        "action_valid": ("arrays/action_valid.npy", action_valid),
        "episode_offsets": ("arrays/episode_offsets.npy", episode_offsets),
    }
    descriptors = {}
    for name, (relative_path, value) in values.items():
        path = root / relative_path
        np.save(path, value, allow_pickle=False)
        descriptors[name] = {
            "path": relative_path,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    dataset_sha = "a" * 64
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "dataset_manifest_sha256": dataset_sha,
            "arrays": descriptors,
        },
    )
    _write_json(
        root / "_SUCCESS.json",
        {
            "status": "complete",
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    )
    return RoboCasa365ControlCache(root, expected_dataset_manifest_sha256=dataset_sha), dataset_sha


def test_history_is_episode_local_right_aligned_and_future_free(tmp_path: Path) -> None:
    cache, _ = _cache(tmp_path)

    history = cache.history(
        episode_index=1,
        frame_index=3,
        proprio_history_length=4,
        proprio_history_stride=2,
        action_history_length=4,
    )

    assert history.proprio_history_mask.tolist() == [False, False, True, True]
    assert history.action_history_mask.tolist() == [False, True, False, True]
    torch.testing.assert_close(history.proprio_history[2], torch.from_numpy(cache.states[6]))
    torch.testing.assert_close(history.proprio_history[-1], torch.from_numpy(cache.states[8]))
    torch.testing.assert_close(history.action_history[-1], torch.from_numpy(cache.actions[7]))
    assert torch.count_nonzero(history.proprio_history[:2]) == 0
    assert torch.count_nonzero(history.action_history[0]) == 0
    assert torch.count_nonzero(history.action_history[2]) == 0
    assert not torch.any(history.action_history == torch.from_numpy(cache.actions[8]))


def test_cache_rejects_dataset_manifest_drift(tmp_path: Path) -> None:
    cache, _ = _cache(tmp_path)

    with pytest.raises(RuntimeError, match="dataset manifest mismatch"):
        RoboCasa365ControlCache(cache.root, expected_dataset_manifest_sha256="b" * 64)
