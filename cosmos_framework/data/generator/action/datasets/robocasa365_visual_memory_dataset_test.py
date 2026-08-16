# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from cosmos_framework.data.generator.action.datasets.robocasa365_visual_memory_dataset import (
    RoboCasa365VisualMemoryDataset,
    _PhysicalInterval,
)


def _mapping_only_dataset() -> RoboCasa365VisualMemoryDataset:
    dataset = object.__new__(RoboCasa365VisualMemoryDataset)
    dataset._physical_intervals = {
        4: (
            _PhysicalInterval(source_start=0, source_stop=8, flat_start=10),
            _PhysicalInterval(source_start=8, source_stop=20, flat_start=100),
        ),
        5: (_PhysicalInterval(source_start=0, source_stop=20, flat_start=500),),
    }
    dataset._physical_interval_starts = {4: (0, 8), 5: (0,)}
    dataset._physical_prefix_max_stop = {4: (8, 20), 5: (20,)}
    return dataset


def test_physical_frame_mapping_crosses_segment_boundaries_within_episode() -> None:
    dataset = _mapping_only_dataset()

    assert dataset._flat_index_for_physical_frame(4, 3) == 13
    assert dataset._flat_index_for_physical_frame(4, 8) == 100
    assert dataset._flat_index_for_physical_frame(4, 15) == 107


def test_physical_frame_mapping_never_crosses_episode_or_missing_frame() -> None:
    dataset = _mapping_only_dataset()

    assert dataset._flat_index_for_physical_frame(4, -1) is None
    assert dataset._flat_index_for_physical_frame(4, 20) is None
    assert dataset._flat_index_for_physical_frame(999, 3) is None
    assert dataset._flat_index_for_physical_frame(5, 3) == 503
