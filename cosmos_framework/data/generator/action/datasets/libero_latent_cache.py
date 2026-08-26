# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict fixed-record Wan latent cache for one LIBERO suite.

The cache is indexed by the exact map-style sample index exposed by
``LIBEROLeRobotDataset``.  Shards contain aligned BF16 records and are staged
from object storage onto a worker-local POSIX filesystem before training.
"""

from __future__ import annotations

import bisect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LiberoLatentLayout:
    shape: tuple[int, int, int, int]
    record_nbytes: int
    record_stride_bytes: int
    video_num_frames: int
    image_size: tuple[int, int, int, int]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"LIBERO latent cache metadata is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"LIBERO latent cache metadata must be an object: {path}")
    return value


class LiberoLatentCache:
    """Read and validate a complete local LIBERO latent cache."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_dataset_name: str,
        expected_suite: str,
        expected_split: str,
        expected_val_ratio: float,
        expected_seed: int,
        expected_chunk_length: int,
        expected_camera_mode: str,
        expected_dataset_image_size: int,
        expected_num_samples: int,
        expected_image_size: tuple[int, int, int, int],
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        success = _read_json(self.root / "_SUCCESS.json")
        if success.get("schema_version") != CACHE_SCHEMA_VERSION or success.get("status") != "complete":
            raise ValueError(f"LIBERO latent cache is not complete: {self.root}")
        if success.get("manifest_path") != "manifest.json":
            raise ValueError("LIBERO latent cache _SUCCESS.json must reference manifest.json.")

        self.manifest = _read_json(self.root / "manifest.json")
        if self.manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported LIBERO latent cache schema version.")
        dataset = self.manifest.get("dataset")
        if not isinstance(dataset, dict):
            raise ValueError("LIBERO latent cache manifest.dataset must be an object.")
        expected_dataset = {
            "name": expected_dataset_name,
            "suite": expected_suite,
            "split": expected_split,
            "val_ratio": float(expected_val_ratio),
            "seed": int(expected_seed),
            "chunk_length": int(expected_chunk_length),
            "camera_mode": expected_camera_mode,
            "image_size": int(expected_dataset_image_size),
            "num_samples": int(expected_num_samples),
        }
        for key, expected in expected_dataset.items():
            actual = dataset.get(key)
            if isinstance(expected, float):
                matches = isinstance(actual, (int, float)) and abs(float(actual) - expected) < 1e-12
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    f"LIBERO latent cache dataset contract mismatch for {key}: "
                    f"cache={actual!r}, active={expected!r}."
                )

        latent = self.manifest.get("latent")
        if not isinstance(latent, dict):
            raise ValueError("LIBERO latent cache manifest.latent must be an object.")
        raw_shape = latent.get("shape")
        raw_image_size = latent.get("image_size")
        if latent.get("storage_dtype") != "bfloat16":
            raise ValueError("LIBERO latent cache storage_dtype must be bfloat16.")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 4
            or any(type(v) is not int or v <= 0 for v in raw_shape)
        ):
            raise ValueError("LIBERO latent cache latent.shape must contain four positive integers.")
        if (
            not isinstance(raw_image_size, list)
            or len(raw_image_size) != 4
            or any(type(v) is not int or v <= 0 for v in raw_image_size)
        ):
            raise ValueError("LIBERO latent cache latent.image_size must contain four positive integers.")
        shape = tuple(raw_shape)
        image_size = tuple(raw_image_size)
        record_nbytes = int(latent.get("record_nbytes", -1))
        stride = int(latent.get("record_stride_bytes", -1))
        expected_nbytes = math.prod(shape) * torch.tensor([], dtype=torch.bfloat16).element_size()
        if record_nbytes != expected_nbytes or stride < record_nbytes:
            raise ValueError(
                f"LIBERO latent cache record layout is invalid: bytes={record_nbytes}, "
                f"stride={stride}, expected_bytes={expected_nbytes}."
            )
        if image_size != expected_image_size:
            raise ValueError(
                f"LIBERO latent cache resize contract mismatch: cache={image_size}, active={expected_image_size}."
            )
        if int(latent.get("video_num_frames", -1)) != expected_chunk_length + 1:
            raise ValueError("LIBERO latent cache video_num_frames does not match chunk_length + 1.")
        self.layout = LiberoLatentLayout(
            shape=shape,  # type: ignore[arg-type]
            record_nbytes=record_nbytes,
            record_stride_bytes=stride,
            video_num_frames=int(latent["video_num_frames"]),
            image_size=image_size,  # type: ignore[arg-type]
        )

        shards = self.manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("LIBERO latent cache manifest.shards must be a non-empty list.")
        self._shards: list[dict[str, Any]] = []
        self._starts: list[int] = []
        next_index = 0
        for expected_id, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise ValueError("LIBERO latent cache shard entries must be objects.")
            shard_id = shard.get("shard_id")
            first_index = shard.get("first_index")
            num_records = shard.get("num_records")
            size_bytes = shard.get("size_bytes")
            relative = shard.get("path")
            if (
                shard_id != expected_id
                or first_index != next_index
                or type(num_records) is not int
                or num_records <= 0
                or type(size_bytes) is not int
                or size_bytes != num_records * stride
                or not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise ValueError(f"Invalid LIBERO latent cache shard entry: {shard!r}")
            shard_path = self.root / relative
            if not shard_path.is_file() or shard_path.stat().st_size != size_bytes:
                raise ValueError(f"LIBERO latent cache shard is missing or truncated: {shard_path}")
            self._starts.append(next_index)
            self._shards.append(shard)
            next_index += num_records
        if next_index != expected_num_samples:
            raise ValueError(
                f"LIBERO latent cache shard coverage mismatch: cache={next_index}, active={expected_num_samples}."
            )
        self._fds: dict[int, int] = {}
        self._pid = os.getpid()

    @property
    def video_num_frames(self) -> int:
        return self.layout.video_num_frames

    @property
    def image_size(self) -> tuple[int, int, int, int]:
        return self.layout.image_size

    def _fd(self, shard_id: int) -> int:
        if os.getpid() != self._pid:
            self._fds = {}
            self._pid = os.getpid()
        fd = self._fds.get(shard_id)
        if fd is None:
            fd = os.open(self.root / self._shards[shard_id]["path"], os.O_RDONLY)
            self._fds[shard_id] = fd
        return fd

    def read_sample(self, index: int) -> torch.Tensor:
        index = int(index)
        if index < 0:
            index += int(self.manifest["dataset"]["num_samples"])
        total = int(self.manifest["dataset"]["num_samples"])
        if not 0 <= index < total:
            raise IndexError(f"LIBERO latent cache index out of range: {index} for {total} samples.")
        shard_id = bisect.bisect_right(self._starts, index) - 1
        shard = self._shards[shard_id]
        local_index = index - int(shard["first_index"])
        offset = local_index * self.layout.record_stride_bytes
        payload = os.pread(self._fd(shard_id), self.layout.record_nbytes, offset)
        if len(payload) != self.layout.record_nbytes:
            raise OSError(
                f"Short read from LIBERO latent cache shard={shard_id}, index={index}: "
                f"got={len(payload)}, expected={self.layout.record_nbytes}."
            )
        return torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(self.layout.shape)

    def close(self) -> None:
        fds = getattr(self, "_fds", {})
        for fd in fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        fds.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_fds"] = {}
        state["_pid"] = os.getpid()
        return state

    def __del__(self) -> None:
        self.close()
