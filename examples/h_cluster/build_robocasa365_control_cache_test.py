# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_HELPER = Path(__file__).with_name("build_robocasa365_control_cache.py")
_SPEC = importlib.util.spec_from_file_location("build_robocasa365_control_cache", _HELPER)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_action_projection_uses_dataset_global_episode_index(tmp_path: Path) -> None:
    root = tmp_path / "view"
    action_path = root / "data/actions/actions_rot6d_f32.npy"
    action_path.parent.mkdir(parents=True)
    expected = np.arange(30, dtype=np.float32).reshape(3, 10)
    np.save(action_path, expected, allow_pickle=False)
    (root / "dataset_manifest.json").write_text(
        json.dumps({"dense_actions": {"rot6d10": "data/actions/actions_rot6d_f32.npy"}})
    )
    split_path = root / "splits/train.parquet"
    split_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_dataset_episode_index": 1,
                    "source_episode_index": 999,
                    "source_episode_length": 3,
                    "source_frame_offset": 0,
                    "source_global_frame_start": 2,
                    "action_offset": 0,
                    "action_length": 3,
                }
            ]
        ),
        split_path,
    )

    actions, valid, overlap = _MODULE._project_dense_actions(
        root,
        splits=("train",),
        episode_offsets=np.asarray([[0, 2], [2, 3]], dtype=np.int64),
        total_frames=5,
    )

    assert overlap == 0
    assert valid.tolist() == [False, False, True, True, True]
    np.testing.assert_array_equal(actions[2:], expected)
