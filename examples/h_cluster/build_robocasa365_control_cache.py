# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build an episode-indexed RoboCasa365 state/action-history cache.

The cache copies 16-D proprioception from the pinned LeRobot parquet rows and
projects the converted view's already-audited rot6d10 actions back onto original
episode frame indexes. Overlapping DeMiAn segments must agree numerically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_framework.data.generator.action.datasets.robocasa365_control_cache import (
    CONTROL_CACHE_ACTION_DIM,
    CONTROL_CACHE_SCHEMA_VERSION,
    CONTROL_CACHE_STATE_DIM,
    sha256_file,
)


_STATE_KEY = "observation.state"
_STATE_SEMANTICS = (
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
    "base_quat_x",
    "base_quat_y",
    "base_quat_z",
    "base_quat_w",
    "eef_pos_rel_x",
    "eef_pos_rel_y",
    "eef_pos_rel_z",
    "eef_quat_rel_x",
    "eef_quat_rel_y",
    "eef_quat_rel_z",
    "eef_quat_rel_w",
    "gripper_qpos_0",
    "gripper_qpos_1",
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _fixed_width_numpy(column: pa.ChunkedArray, *, width: int, label: str) -> np.ndarray:
    combined = column.combine_chunks()
    if isinstance(combined, pa.FixedSizeListArray):
        if combined.type.list_size != width:
            raise ValueError(f"{label} must have width {width}, got {combined.type.list_size}")
        values = combined.values.to_numpy(zero_copy_only=False)
        return np.asarray(values).reshape(len(combined), width)
    value = np.asarray(combined.to_pylist())
    if value.shape != (len(combined), width):
        raise ValueError(f"{label} must have shape [N,{width}], got {value.shape}")
    return value


def _load_source_states(source_root: Path, *, total_frames: int, total_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    parquet_paths = sorted(source_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No LeRobot parquet files found under {source_root / 'data'}")

    states = np.zeros((total_frames, CONTROL_CACHE_STATE_DIM), dtype=np.float32)
    seen = np.zeros(total_frames, dtype=np.bool_)
    episode_min_index = np.full(total_episodes, total_frames, dtype=np.int64)
    episode_max_index = np.full(total_episodes, -1, dtype=np.int64)
    episode_min_frame = np.full(total_episodes, total_frames, dtype=np.int64)
    episode_max_frame = np.full(total_episodes, -1, dtype=np.int64)
    episode_counts = np.zeros(total_episodes, dtype=np.int64)

    required = {"index", "episode_index", "frame_index", _STATE_KEY}
    for path in parquet_paths:
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = required - schema_names
        if missing:
            raise KeyError(f"Raw RoboCasa365 parquet {path} is missing columns {sorted(missing)}")
        table = pq.read_table(path, columns=sorted(required))
        indexes = np.asarray(table["index"].combine_chunks().to_numpy(), dtype=np.int64)
        episodes = np.asarray(table["episode_index"].combine_chunks().to_numpy(), dtype=np.int64)
        frames = np.asarray(table["frame_index"].combine_chunks().to_numpy(), dtype=np.int64)
        state_values = _fixed_width_numpy(table[_STATE_KEY], width=CONTROL_CACHE_STATE_DIM, label=_STATE_KEY)
        state_values = np.asarray(state_values, dtype=np.float32)

        if np.any(indexes < 0) or np.any(indexes >= total_frames):
            raise ValueError(f"Global frame index outside [0,{total_frames}) in {path}")
        if np.any(episodes < 0) or np.any(episodes >= total_episodes):
            raise ValueError(f"Episode index outside [0,{total_episodes}) in {path}")
        if np.any(frames < 0) or not np.isfinite(state_values).all():
            raise ValueError(f"Invalid frame index or non-finite state in {path}")
        if seen[indexes].any():
            raise ValueError(f"Duplicate global frame indexes in source parquet stream: {path}")

        states[indexes] = state_values
        seen[indexes] = True
        np.minimum.at(episode_min_index, episodes, indexes)
        np.maximum.at(episode_max_index, episodes, indexes)
        np.minimum.at(episode_min_frame, episodes, frames)
        np.maximum.at(episode_max_frame, episodes, frames)
        np.add.at(episode_counts, episodes, 1)

    if not seen.all():
        missing = np.flatnonzero(~seen)
        raise ValueError(f"Raw state scan missed {missing.size} global frames; first={missing[:10].tolist()}")
    if np.any(episode_counts <= 0):
        raise ValueError("Every episode index must contain at least one frame")
    if np.any(episode_min_frame != 0) or np.any(episode_max_frame + 1 != episode_counts):
        raise ValueError("Every source episode must use contiguous frame_index values starting at zero")
    if np.any(episode_max_index - episode_min_index + 1 != episode_counts):
        raise ValueError("Every source episode must occupy one contiguous global index interval")
    if np.any(episode_min_index[1:] != episode_max_index[:-1] + 1):
        raise ValueError("Source episodes must be contiguous and ordered by episode_index")

    episode_offsets = np.stack((episode_min_index, episode_counts), axis=1).astype(np.int64, copy=False)
    return states, episode_offsets


def _project_dense_actions(
    dataset_root: Path,
    *,
    splits: tuple[str, ...],
    episode_offsets: np.ndarray,
    total_frames: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    dataset_manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())
    relative_action_path = dataset_manifest.get("dense_actions", {}).get("rot6d10")
    if not isinstance(relative_action_path, str) or not relative_action_path:
        raise KeyError("Converted dataset manifest is missing dense_actions.rot6d10")
    dense_actions = np.load(dataset_root / relative_action_path, allow_pickle=False)
    if (
        dense_actions.dtype != np.float32
        or dense_actions.ndim != 2
        or dense_actions.shape[1] != CONTROL_CACHE_ACTION_DIM
    ):
        raise ValueError(f"Converted actions must be float32 [N,{CONTROL_CACHE_ACTION_DIM}], got {dense_actions.shape}")

    actions = np.zeros((total_frames, CONTROL_CACHE_ACTION_DIM), dtype=np.float32)
    valid = np.zeros(total_frames, dtype=np.bool_)
    overlap_rows = 0
    for split in splits:
        split_path = dataset_root / "splits" / f"{split}.parquet"
        if not split_path.is_file():
            raise FileNotFoundError(f"Converted split manifest is missing: {split_path}")
        columns = ("source_episode_index", "source_frame_offset", "action_offset", "action_length")
        table = pq.read_table(split_path, columns=list(columns))
        for row in table.to_pylist():
            episode_index = int(row["source_episode_index"])
            source_frame = int(row["source_frame_offset"])
            action_offset = int(row["action_offset"])
            action_length = int(row["action_length"])
            if not 0 <= episode_index < episode_offsets.shape[0]:
                raise ValueError(f"Invalid source_episode_index={episode_index} in {split_path}")
            episode_start, episode_length = (int(value) for value in episode_offsets[episode_index])
            if source_frame < 0 or source_frame + action_length > episode_length:
                raise ValueError(
                    f"Segment action interval exceeds source episode: episode={episode_index}, "
                    f"frame={source_frame}, length={action_length}, episode_length={episode_length}."
                )
            source_slice = dense_actions[action_offset : action_offset + action_length]
            if source_slice.shape != (action_length, CONTROL_CACHE_ACTION_DIM):
                raise ValueError(f"Dense action interval exceeds converted action cache in {split_path}")
            global_indexes = episode_start + source_frame + np.arange(action_length, dtype=np.int64)
            overlap = valid[global_indexes]
            if overlap.any():
                previous = actions[global_indexes[overlap]]
                candidate = source_slice[overlap]
                if not np.allclose(previous, candidate, rtol=1e-5, atol=1e-6):
                    raise ValueError(
                        f"Overlapping DeMiAn segments disagree on rot6d10 actions in split={split}, "
                        f"episode={episode_index}."
                    )
                overlap_rows += int(overlap.sum())
            actions[global_indexes[~overlap]] = source_slice[~overlap]
            valid[global_indexes] = True

    return actions, valid, overlap_rows


def _array_descriptor(root: Path, relative_path: str, value: np.ndarray) -> dict[str, Any]:
    path = root / relative_path
    return {
        "path": relative_path,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_cache(args: argparse.Namespace) -> Path:
    dataset_manifest_path = args.dataset_root / "dataset_manifest.json"
    source_info_path = args.source_root / "meta" / "info.json"
    if not dataset_manifest_path.is_file() or not source_info_path.is_file():
        raise FileNotFoundError("Dataset root and source root do not contain their required manifests")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to replace existing control cache: {args.output_root}")

    source_info = json.loads(source_info_path.read_text())
    total_frames = int(source_info["total_frames"])
    total_episodes = int(source_info["total_episodes"])
    if total_frames < 1 or total_episodes < 1:
        raise ValueError("Raw source info must contain positive total_frames and total_episodes")

    states, episode_offsets = _load_source_states(
        args.source_root,
        total_frames=total_frames,
        total_episodes=total_episodes,
    )
    actions, action_valid, overlap_rows = _project_dense_actions(
        args.dataset_root,
        splits=args.splits,
        episode_offsets=episode_offsets,
        total_frames=total_frames,
    )

    stage = args.output_root.with_name(f".{args.output_root.name}.partial-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        arrays = {
            "states": ("arrays/states_f32.npy", states),
            "actions_rot6d10": ("arrays/actions_rot6d10_f32.npy", actions),
            "action_valid": ("arrays/action_valid_bool.npy", action_valid),
            "episode_offsets": ("arrays/episode_offsets_i64.npy", episode_offsets),
        }
        for relative_path, value in arrays.values():
            _write_npy(stage / relative_path, value)

        manifest = {
            "schema_version": CONTROL_CACHE_SCHEMA_VERSION,
            "created_at_unix_ns": time.time_ns(),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "source": {
                "root": str(args.source_root),
                "info_sha256": sha256_file(source_info_path),
                "total_frames": total_frames,
                "total_episodes": total_episodes,
                "state_key": _STATE_KEY,
                "state_semantics": list(_STATE_SEMANTICS),
            },
            "action_projection": {
                "splits": list(args.splits),
                "valid_frames": int(action_valid.sum()),
                "coverage": float(action_valid.mean()),
                "validated_overlap_rows": overlap_rows,
                "layout": "[dpos3,rot6d_columns6,gripper1]",
            },
            "arrays": {
                name: _array_descriptor(stage, relative_path, value)
                for name, (relative_path, value) in arrays.items()
            },
        }
        manifest_bytes = _json_bytes(manifest)
        _write_bytes(stage / "manifest.json", manifest_bytes)
        _write_bytes(
            stage / "_SUCCESS.json",
            _json_bytes(
                {
                    "status": "complete",
                    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                }
            ),
        )
        _fsync_dir(stage / "arrays")
        _fsync_dir(stage)
        os.rename(stage, args.output_root)
        _fsync_dir(args.output_root.parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return args.output_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.resolve(strict=True)
    args.source_root = args.source_root.resolve(strict=True)
    args.output_root = args.output_root.resolve(strict=False)
    args.splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    if not args.splits or len(set(args.splits)) != len(args.splits):
        parser.error("--splits must contain one or more unique names")
    if args.output_root in {args.dataset_root, args.source_root}:
        parser.error("--output-root must differ from dataset and source roots")
    return args


def main() -> None:
    output = build_cache(_parse_args())
    print(output)


if __name__ == "__main__":
    main()
