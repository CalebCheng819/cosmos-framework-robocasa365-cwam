# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Immutable episode-indexed proprioception and action-history cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


CONTROL_CACHE_SCHEMA_VERSION = 1
CONTROL_CACHE_STATE_DIM = 16
CONTROL_CACHE_ACTION_DIM = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must contain lowercase hexadecimal characters") from error
    if value != value.lower():
        raise ValueError(f"{label} must use lowercase hexadecimal characters")
    return value


@dataclass(frozen=True)
class ControlHistory:
    """One deployable, future-free control context window."""

    proprio_history: torch.Tensor
    proprio_history_mask: torch.Tensor
    action_history: torch.Tensor
    action_history_mask: torch.Tensor


class RoboCasa365ControlCache:
    """Load state16 and rot6d10 actions in original episode frame order.

    The arrays are intentionally read into normal process memory. GPFS does not
    support the mmap access pattern used by ``numpy.load``, and loading before
    DataLoader workers fork gives copy-on-write sharing within each rank.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        expected_dataset_manifest_sha256: str,
        verify_checksums: bool = False,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        success_path = self.root / "_SUCCESS.json"
        if not manifest_path.is_file() or not success_path.is_file():
            raise FileNotFoundError(
                f"RoboCasa365 control cache requires manifest.json and _SUCCESS.json under {self.root}."
            )

        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        success = json.loads(success_path.read_text())
        if success.get("status") != "complete":
            raise RuntimeError(f"RoboCasa365 control cache is not complete: {success.get('status')!r}")
        if success.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError(
                "RoboCasa365 control cache manifest checksum mismatch: "
                f"success={success.get('manifest_sha256')!r}, actual={manifest_sha256!r}."
            )
        if int(manifest.get("schema_version", -1)) != CONTROL_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported control cache schema_version={manifest.get('schema_version')!r}; "
                f"expected {CONTROL_CACHE_SCHEMA_VERSION}."
            )
        active_dataset_sha = _require_sha256(
            expected_dataset_manifest_sha256,
            "expected_dataset_manifest_sha256",
        )
        cached_dataset_sha = _require_sha256(
            manifest.get("dataset_manifest_sha256"),
            "manifest.dataset_manifest_sha256",
        )
        if active_dataset_sha != cached_dataset_sha:
            raise RuntimeError(
                "RoboCasa365 control cache dataset manifest mismatch: "
                f"cache={cached_dataset_sha}, active={active_dataset_sha}."
            )

        arrays = manifest.get("arrays")
        if not isinstance(arrays, dict):
            raise ValueError("RoboCasa365 control cache manifest.arrays must be an object")
        self.states = self._load_array(
            arrays,
            "states",
            expected_dtype=np.dtype(np.float32),
            expected_tail=(CONTROL_CACHE_STATE_DIM,),
            verify_checksum=verify_checksums,
        )
        self.actions = self._load_array(
            arrays,
            "actions_rot6d10",
            expected_dtype=np.dtype(np.float32),
            expected_tail=(CONTROL_CACHE_ACTION_DIM,),
            verify_checksum=verify_checksums,
        )
        self.action_valid = self._load_array(
            arrays,
            "action_valid",
            expected_dtype=np.dtype(np.bool_),
            expected_tail=(),
            verify_checksum=verify_checksums,
        )
        self.episode_offsets = self._load_array(
            arrays,
            "episode_offsets",
            expected_dtype=np.dtype(np.int64),
            expected_tail=(2,),
            verify_checksum=verify_checksums,
        )
        self._validate_layout()
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256

    def _load_array(
        self,
        arrays: dict[str, Any],
        name: str,
        *,
        expected_dtype: np.dtype[Any],
        expected_tail: tuple[int, ...],
        verify_checksum: bool,
    ) -> np.ndarray:
        descriptor = arrays.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"Control cache arrays.{name} must be an object")
        relative_path = descriptor.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Control cache arrays.{name}.path must be non-empty")
        path = self.root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Control cache array is missing: {path}")
        if int(descriptor.get("size_bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"Control cache array size mismatch for {name}: {path}")
        expected_sha = _require_sha256(descriptor.get("sha256"), f"arrays.{name}.sha256")
        if verify_checksum and sha256_file(path) != expected_sha:
            raise RuntimeError(f"Control cache array checksum mismatch for {name}: {path}")

        value = np.load(path, allow_pickle=False)
        expected_shape = descriptor.get("shape")
        if not isinstance(expected_shape, list) or tuple(expected_shape) != value.shape:
            raise RuntimeError(
                f"Control cache shape mismatch for {name}: manifest={expected_shape!r}, actual={value.shape}."
            )
        if value.dtype != expected_dtype:
            raise ValueError(f"Control cache {name} must have dtype {expected_dtype}, got {value.dtype}")
        if value.ndim != len(expected_tail) + 1 or tuple(value.shape[1:]) != expected_tail:
            raise ValueError(
                f"Control cache {name} must have shape [N,{','.join(map(str, expected_tail))}], "
                f"got {value.shape}."
            )
        return value

    def _validate_layout(self) -> None:
        num_frames = int(self.states.shape[0])
        if self.actions.shape[0] != num_frames or self.action_valid.shape[0] != num_frames:
            raise ValueError("Control cache state/action arrays must have the same leading frame count")
        if not np.isfinite(self.states).all():
            raise ValueError("Control cache states contain non-finite values")
        if not np.isfinite(self.actions).all():
            raise ValueError("Control cache actions contain non-finite values")
        if self.episode_offsets.shape[0] < 1:
            raise ValueError("Control cache must contain at least one episode")
        starts = self.episode_offsets[:, 0]
        lengths = self.episode_offsets[:, 1]
        if np.any(starts < 0) or np.any(lengths <= 0):
            raise ValueError("Control cache episode offsets must use non-negative starts and positive lengths")
        if np.any(starts[1:] != starts[:-1] + lengths[:-1]):
            raise ValueError("Control cache episodes must form one contiguous frame partition")
        if int(starts[0]) != 0 or int(starts[-1] + lengths[-1]) != num_frames:
            raise ValueError("Control cache episode offsets do not cover every source frame exactly once")

    @property
    def num_episodes(self) -> int:
        return int(self.episode_offsets.shape[0])

    def history(
        self,
        *,
        episode_index: int,
        frame_index: int,
        proprio_history_length: int,
        proprio_history_stride: int,
        action_history_length: int,
    ) -> ControlHistory:
        if not 0 <= int(episode_index) < self.num_episodes:
            raise IndexError(f"Control cache episode_index={episode_index} is out of range")
        if proprio_history_length < 1 or proprio_history_stride < 1 or action_history_length < 1:
            raise ValueError("Control history lengths and stride must be positive")
        episode_start, episode_length = (int(value) for value in self.episode_offsets[int(episode_index)])
        if not 0 <= int(frame_index) < episode_length:
            raise IndexError(
                f"Control cache frame_index={frame_index} is outside episode {episode_index} length={episode_length}"
            )

        state_offsets = np.arange(
            -(proprio_history_length - 1) * proprio_history_stride,
            1,
            proprio_history_stride,
            dtype=np.int64,
        )
        state_frames = int(frame_index) + state_offsets
        state_valid = (state_frames >= 0) & (state_frames < episode_length)
        state_values = np.zeros((proprio_history_length, CONTROL_CACHE_STATE_DIM), dtype=np.float32)
        state_values[state_valid] = self.states[episode_start + state_frames[state_valid]]

        action_frames = int(frame_index) + np.arange(-action_history_length, 0, dtype=np.int64)
        action_in_episode = (action_frames >= 0) & (action_frames < episode_length)
        action_valid = np.zeros(action_history_length, dtype=np.bool_)
        action_values = np.zeros((action_history_length, CONTROL_CACHE_ACTION_DIM), dtype=np.float32)
        if action_in_episode.any():
            positions = np.flatnonzero(action_in_episode)
            global_indexes = episode_start + action_frames[action_in_episode]
            cache_valid = self.action_valid[global_indexes]
            valid_positions = positions[cache_valid]
            action_valid[valid_positions] = True
            action_values[valid_positions] = self.actions[global_indexes[cache_valid]]

        if not bool(state_valid[-1]):
            raise RuntimeError("Current proprioception must always be valid")
        return ControlHistory(
            proprio_history=torch.from_numpy(state_values),
            proprio_history_mask=torch.from_numpy(state_valid),
            action_history=torch.from_numpy(action_values),
            action_history_mask=torch.from_numpy(action_valid),
        )
