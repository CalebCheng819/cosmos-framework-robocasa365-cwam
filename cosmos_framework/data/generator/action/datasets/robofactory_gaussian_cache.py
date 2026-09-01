# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read-only access to the canonical per-agent RoboFactory Gaussian cache.

The cache contract is shared with ImageWAM/FastWAM.  Identity is pinned by a
versioned directory and manifest; this reader deliberately does not calculate
content digests while loading training samples.
"""

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


CANONICAL_CHANNELS = (
    "mean_x",
    "mean_y",
    "mean_z",
    "cov_xx",
    "cov_xy",
    "cov_xz",
    "cov_yx",
    "cov_yy",
    "cov_yz",
    "cov_zx",
    "cov_zy",
    "cov_zz",
    "opacity",
)


def normalize_source_path(value: str) -> str:
    """Return a safe normalized relative POSIX path."""
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"source_path must be normalized and relative, got {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class _Segment:
    shard: str
    offset: int
    count: int
    source_start: int
    source_stride: int

    def locate(self, timestep: int) -> int | None:
        delta = int(timestep) - self.source_start
        if delta < 0 or delta % self.source_stride:
            return None
        index = delta // self.source_stride
        return None if index >= self.count else self.offset + index


class _Stream:
    def __init__(self, record: Mapping[str, Any]) -> None:
        self.segments = tuple(
            _Segment(
                shard=str(segment["shard"]),
                offset=int(segment["offset"]),
                count=int(segment["count"]),
                source_start=int(segment["source_start"]),
                source_stride=int(segment["source_stride"]),
            )
            for segment in record["segments"]
        )
        if not self.segments or any(
            segment.count <= 0 or segment.source_stride <= 0 for segment in self.segments
        ):
            raise ValueError("Gaussian stream segments must have positive count and stride")
        if any(segment.offset < 0 for segment in self.segments):
            raise ValueError("Gaussian stream segment offsets must be non-negative")
        self.starts = tuple(segment.source_start for segment in self.segments)
        if self.starts != tuple(sorted(self.starts)):
            raise ValueError("Gaussian stream segments must be ordered by source_start")
        for previous, current in zip(self.segments, self.segments[1:]):
            previous_end = previous.source_start + (previous.count - 1) * previous.source_stride
            if current.source_start <= previous_end:
                raise ValueError("Gaussian stream segments must not overlap in source time")

    def locate(self, timestep: int) -> tuple[str, int] | None:
        index = bisect.bisect_right(self.starts, int(timestep)) - 1
        if index < 0:
            return None
        offset = self.segments[index].locate(int(timestep))
        return None if offset is None else (self.segments[index].shard, offset)


class GaussianCache:
    """Pickle-safe random access to canonical ``[13,28,40]`` float16 frames."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        max_open_shards: int = 32,
        source_path_strip_prefix: str | None = None,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        if not self.cache_root.is_dir():
            raise FileNotFoundError(f"Gaussian cache directory does not exist: {self.cache_root}")
        complete = self.cache_root / "COMPLETE"
        manifest_path = self.cache_root / "manifest.json"
        if not complete.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Gaussian cache requires COMPLETE and manifest.json: {self.cache_root}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        schema = self.manifest.get("schema", {})
        if schema.get("name") != "fastwam.canonical-gaussian-cache" or int(schema.get("version", -1)) != 1:
            raise ValueError(f"Unsupported Gaussian cache schema: {schema}")
        if tuple(schema.get("channels", ())) != CANONICAL_CHANNELS:
            raise ValueError("Gaussian cache channel order is not canonical")
        if schema.get("dtype") != "float16" or schema.get("byte_order") != "little":
            raise ValueError("Gaussian cache must use little-endian float16 frames")
        if schema.get("covariance_order") != "row-major-ij":
            raise ValueError("Gaussian covariance channels must use row-major (i,j) order")
        self.frame_shape = (13, int(schema["height"]), int(schema["width"]))
        if self.frame_shape != (13, 28, 40):
            raise ValueError(f"RoboFactory Gaussian frames must be [13,28,40], got {self.frame_shape}")

        self.max_open_shards = int(max_open_shards)
        if self.max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        self.source_path_strip_prefix = (
            None if source_path_strip_prefix is None else normalize_source_path(source_path_strip_prefix)
        )
        self._shards = {str(item["id"]): dict(item) for item in self.manifest.get("shards", [])}
        if not self._shards:
            raise ValueError("Gaussian cache manifest contains no shards")
        for shard_id, record in self._shards.items():
            path = self._safe_file(record["path"])
            expected = int(record["bytes"])
            if path.stat().st_size != expected:
                raise ValueError(f"Gaussian shard size mismatch for {shard_id}: {path}")
            if expected != int(record["frames"]) * int(np.prod(self.frame_shape)) * 2:
                raise ValueError(f"Gaussian shard frame/byte declaration mismatch for {shard_id}")

        self._streams: dict[tuple[str, str, str], _Stream] = {}
        for record in self.manifest.get("streams", []):
            key = (
                normalize_source_path(record["source_path"]),
                str(record["trajectory"]),
                str(record["agent_name"]),
            )
            if key in self._streams:
                raise ValueError(f"Duplicate Gaussian stream: {key}")
            stream = _Stream(record)
            for segment in stream.segments:
                shard = self._shards.get(segment.shard)
                if shard is None:
                    raise ValueError(f"Gaussian stream references unknown shard: {segment.shard}")
                if segment.offset + segment.count > int(shard["frames"]):
                    raise ValueError(
                        f"Gaussian stream segment exceeds shard {segment.shard}: "
                        f"offset={segment.offset} count={segment.count} frames={shard['frames']}"
                    )
            self._streams[key] = stream
        if not self._streams:
            raise ValueError("Gaussian cache manifest contains no streams")
        self._arrays: OrderedDict[str, np.memmap] = OrderedDict()

    def _safe_file(self, relative_path: str) -> Path:
        relative = normalize_source_path(relative_path)
        path = self.cache_root / relative
        resolved = path.resolve(strict=True)
        if resolved != path or not path.is_file():
            raise RuntimeError(f"Gaussian cache file must be a regular non-symlink: {path}")
        return path

    def _locate(self, source_path: str, trajectory: str, timestep: int, agent_name: str):
        normalized_source_path = normalize_source_path(source_path)
        suffix = (str(trajectory), str(agent_name))
        stream = self._streams.get((normalized_source_path, *suffix))
        if stream is None and self.source_path_strip_prefix is not None:
            prefix = f"{self.source_path_strip_prefix}/"
            if normalized_source_path.startswith(prefix):
                stream = self._streams.get((normalized_source_path[len(prefix) :], *suffix))
        return None if stream is None else stream.locate(int(timestep))

    def preflight(self, keys: Iterable[tuple[str, str, int, str]]) -> int:
        checked = 0
        missing: list[tuple[str, str, int, str]] = []
        missing_count = 0
        for source_path, trajectory, timestep, agent_name in keys:
            checked += 1
            if self._locate(source_path, trajectory, timestep, agent_name) is None:
                missing_count += 1
                if len(missing) < 16:
                    missing.append((source_path, trajectory, int(timestep), agent_name))
        if missing_count:
            raise KeyError(f"Gaussian cache missing {missing_count}/{checked} frames; sample={missing}")
        return checked

    def _array(self, shard_id: str) -> np.memmap:
        array = self._arrays.pop(shard_id, None)
        if array is not None:
            self._arrays[shard_id] = array
            return array
        while len(self._arrays) >= self.max_open_shards:
            _, old = self._arrays.popitem(last=False)
            mmap = getattr(old, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()
        record = self._shards[shard_id]
        array = np.memmap(
            self.cache_root / normalize_source_path(record["path"]),
            mode="r",
            dtype=np.dtype("<f2"),
            shape=(int(record["frames"]), *self.frame_shape),
            order="C",
        )
        self._arrays[shard_id] = array
        return array

    def get_agents(
        self,
        source_path: str,
        trajectory: str,
        timestep: int,
        agent_names: Sequence[str],
    ) -> torch.Tensor:
        frames = []
        for agent_name in agent_names:
            location = self._locate(source_path, trajectory, timestep, agent_name)
            if location is None:
                raise KeyError(f"Missing Gaussian frame: {source_path}:{trajectory}:{timestep}:{agent_name}")
            shard_id, offset = location
            frame = np.array(self._array(shard_id)[offset], dtype=np.float16, copy=True, order="C")
            frames.append(torch.from_numpy(frame))
        return torch.stack(frames, dim=0)

    def close(self) -> None:
        arrays = getattr(self, "_arrays", None)
        while arrays:
            _, array = arrays.popitem(last=False)
            mmap = getattr(array, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = OrderedDict()
        return state

    def __del__(self):
        self.close()
