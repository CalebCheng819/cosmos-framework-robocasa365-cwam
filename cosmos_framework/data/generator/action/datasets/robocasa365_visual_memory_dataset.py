# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboCasa365 segment view augmented with cross-replan visual memory."""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.data.generator.action.datasets.robocasa365_segment_dataset import RoboCasa365SegmentDataset
from cosmos_framework.data.generator.action.datasets.robocasa365_visual_history_cache import (
    RoboCasa365VisualHistoryCache,
)


@dataclass(frozen=True, slots=True)
class _PhysicalInterval:
    source_start: int
    source_stop: int
    flat_start: int


class RoboCasa365VisualMemoryDataset(RoboCasa365SegmentDataset):
    """Add fixed-lag historical latents without changing current-frame preprocessing."""

    def __init__(
        self,
        *args: Any,
        visual_history_cache_root: str,
        visual_history_length: int = 3,
        visual_history_stride: int = 12,
        **kwargs: Any,
    ) -> None:
        if visual_history_length < 1 or visual_history_stride < 1:
            raise ValueError("Visual history length and stride must be positive")
        super().__init__(*args, **kwargs)
        if self._latent_cache is None:
            raise ValueError("Visual memory dataset requires latent_cache_mode='required'")
        self._visual_history_length = int(visual_history_length)
        self._visual_history_stride = int(visual_history_stride)
        self._visual_history_cache = RoboCasa365VisualHistoryCache(
            visual_history_cache_root,
            expected_parent_cache_id=self._latent_cache.cache_id,
            expected_num_records=self._latent_cache.num_unique_records,
        )

        intervals: dict[int, list[_PhysicalInterval]] = defaultdict(list)
        previous_flat = 0
        for segment, cumulative in zip(self._segments, self._valid_cum.tolist()):
            if "source_dataset_episode_index" not in segment or "source_frame_offset" not in segment:
                raise KeyError("Visual memory requires physical source episode/frame indexes")
            count = int(cumulative) - previous_flat
            start = int(segment["source_frame_offset"])
            intervals[int(segment["source_dataset_episode_index"])].append(
                _PhysicalInterval(start, start + count, previous_flat)
            )
            previous_flat = int(cumulative)
        self._physical_intervals: dict[int, tuple[_PhysicalInterval, ...]] = {}
        self._physical_interval_starts: dict[int, tuple[int, ...]] = {}
        self._physical_prefix_max_stop: dict[int, tuple[int, ...]] = {}
        for episode, episode_intervals in intervals.items():
            ordered = tuple(sorted(episode_intervals, key=lambda item: (item.source_start, item.flat_start)))
            prefix: list[int] = []
            max_stop = -1
            for interval in ordered:
                max_stop = max(max_stop, interval.source_stop)
                prefix.append(max_stop)
            self._physical_intervals[episode] = ordered
            self._physical_interval_starts[episode] = tuple(item.source_start for item in ordered)
            self._physical_prefix_max_stop[episode] = tuple(prefix)

    def _flat_index_for_physical_frame(self, episode: int, frame: int) -> int | None:
        intervals = self._physical_intervals.get(episode)
        if intervals is None:
            return None
        starts = self._physical_interval_starts[episode]
        prefix_max_stop = self._physical_prefix_max_stop[episode]
        index = bisect.bisect_right(starts, frame) - 1
        while index >= 0 and prefix_max_stop[index] > frame:
            interval = intervals[index]
            if interval.source_start <= frame < interval.source_stop:
                return interval.flat_start + frame - interval.source_start
            index -= 1
        return None

    def _build_item(self, idx: int) -> dict[str, Any]:
        result = super()._build_item(idx)
        segment, window_offset = self.resolve_window(idx)
        episode = int(segment["source_dataset_episode_index"])
        current_frame = int(segment["source_frame_offset"]) + int(window_offset)
        values: list[torch.Tensor] = []
        mask = torch.zeros(self._visual_history_length, dtype=torch.bool)
        zero = torch.zeros(self._visual_history_cache.shape, dtype=torch.bfloat16)
        lags = range(
            self._visual_history_length * self._visual_history_stride,
            0,
            -self._visual_history_stride,
        )
        for position, lag in enumerate(lags):
            history_index = self._flat_index_for_physical_frame(episode, current_frame - lag)
            if history_index is None:
                values.append(zero.clone())
                continue
            ordinal = self._latent_cache.ordinal_for_sample(history_index)
            values.append(self._visual_history_cache.read_ordinal(ordinal))
            mask[position] = True
        result["control_visual_history_latent"] = torch.stack(values)
        result["control_visual_history_mask"] = mask
        return result
