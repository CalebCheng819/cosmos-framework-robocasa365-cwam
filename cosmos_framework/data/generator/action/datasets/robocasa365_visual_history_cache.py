# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compact fixed-record sidecar for historical RoboCasa365 Wan latents."""

from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class VisualHistoryShard:
    shard_id: int
    first_ordinal: int
    num_records: int
    path: Path
    size_bytes: int

    @property
    def stop_ordinal(self) -> int:
        return self.first_ordinal + self.num_records


class RoboCasa365VisualHistoryCache:
    """Read pooled first-frame latents by the parent cache's unique ordinal."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_parent_cache_id: str,
        expected_num_records: int,
        expected_shape: tuple[int, int, int] = (48, 15, 26),
    ) -> None:
        self.root = Path(root)
        success_path = self.root / "_SUCCESS.json"
        manifest_path = self.root / "manifest.json"
        if not success_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Visual history cache is not published under {self.root}.")
        success = json.loads(success_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if success.get("status") != "complete" or int(manifest.get("schema_version", -1)) != 1:
            raise ValueError("Visual history cache requires a complete schema-v1 publication")
        if manifest.get("parent_cache_id") != expected_parent_cache_id:
            raise RuntimeError(
                "Visual history parent cache mismatch: "
                f"sidecar={manifest.get('parent_cache_id')!r}, active={expected_parent_cache_id!r}."
            )
        if int(manifest.get("num_records", -1)) != int(expected_num_records):
            raise RuntimeError("Visual history record count does not match the active parent cache")
        if manifest.get("storage_dtype") != "bfloat16":
            raise ValueError("Visual history storage_dtype must be bfloat16")
        shape = manifest.get("shape")
        if shape != list(expected_shape):
            raise RuntimeError(f"Visual history shape mismatch: sidecar={shape!r}, expected={expected_shape!r}.")
        if manifest.get("pooling") != {"source_temporal_index": 0, "spatial": "avg_pool2d_2x2"}:
            raise RuntimeError("Visual history cache pooling contract mismatch")

        self.shape = expected_shape
        self.numel = expected_shape[0] * expected_shape[1] * expected_shape[2]
        self.record_nbytes = self.numel * 2
        if int(manifest.get("record_nbytes", -1)) != self.record_nbytes:
            raise ValueError("Visual history record_nbytes does not match shape and dtype")
        raw_shards = manifest.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError("Visual history manifest.shards must be non-empty")
        shards: list[VisualHistoryShard] = []
        expected_first = 0
        for raw in raw_shards:
            if not isinstance(raw, dict):
                raise ValueError("Visual history shard entries must be objects")
            shard_id = int(raw.get("shard_id", -1))
            first = int(raw.get("first_ordinal", -1))
            count = int(raw.get("num_records", -1))
            size = int(raw.get("size_bytes", -1))
            relative_path = raw.get("path")
            if shard_id < 0 or first != expected_first or count < 1 or not isinstance(relative_path, str):
                raise ValueError(f"Invalid visual history shard metadata: {raw!r}")
            path = (self.root / relative_path).resolve()
            if self.root.resolve() not in path.parents:
                raise ValueError(f"Visual history shard escapes cache root: {relative_path!r}")
            if size != count * self.record_nbytes or not path.is_file() or path.stat().st_size != size:
                raise RuntimeError(f"Visual history shard size mismatch: {path}")
            shards.append(VisualHistoryShard(shard_id, first, count, path, size))
            expected_first += count
        if expected_first != expected_num_records:
            raise RuntimeError(
                f"Visual history shards cover {expected_first} records, expected {expected_num_records}."
            )
        self.shards = shards
        self._shard_first_ordinals = [shard.first_ordinal for shard in shards]
        self._fds: dict[Path, int] = {}
        self._fd_pid = os.getpid()

    def read_ordinal(self, ordinal: int) -> torch.Tensor:
        ordinal = int(ordinal)
        if ordinal < 0 or ordinal >= self.shards[-1].stop_ordinal:
            raise IndexError(f"Visual history ordinal {ordinal} is out of range")
        shard_index = bisect.bisect_right(self._shard_first_ordinals, ordinal) - 1
        shard = self.shards[shard_index]
        offset = (ordinal - shard.first_ordinal) * self.record_nbytes
        payload = os.pread(self._get_fd(shard.path), self.record_nbytes, offset)
        if len(payload) != self.record_nbytes:
            raise RuntimeError(f"Short visual history read at ordinal={ordinal}")
        value = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(self.shape).clone()
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Visual history record contains non-finite values at ordinal={ordinal}")
        return value.contiguous()

    def _get_fd(self, path: Path) -> int:
        pid = os.getpid()
        if pid != self._fd_pid:
            self.close()
            self._fd_pid = pid
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def close(self) -> None:
        for fd in getattr(self, "_fds", {}).values():
            os.close(fd)
        if hasattr(self, "_fds"):
            self._fds.clear()

    def __del__(self) -> None:
        self.close()
