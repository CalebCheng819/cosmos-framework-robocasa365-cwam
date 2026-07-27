# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboCasa365 DeMiAn-segment dataset for Cosmos action training.

The CWAM conversion keeps the original RoboCasa365 H.264 videos in place and
materializes only compact segment manifests, dense actions, captions, splits,
and normalization statistics.  This loader maps a flat training index to one
of those segment-local windows:

* 17 video frames at 20 Hz;
* 16 raw frame-wise-relative actions in
  ``[dpos(3), rot6d-columns(6), gripper(1)]`` format; and
* the canonical RoboCasa policy instruction; or, only for explicitly named
  oracle diagnostics, that instruction augmented with a temporally aligned
  DeMiAn current-action caption.

The rich current-action annotation can describe actions after the policy
conditioning frame.  It is therefore non-causal and must never be used as an
input to a deployable action policy or a policy-training baseline.

Video timestamps in the converted manifest are already absolute within the
shared source MP4.  The only valid rule is therefore
``camera_from_timestamp + (window_offset + frame_index) / fps``.  In
particular, callers must not add the source episode timestamp a second time.

GPFS does not support the mmap access pattern used by ``numpy.load``.  The
dense action cache is deliberately loaded into ordinary process memory before
DataLoader workers fork, allowing copy-on-write sharing without mmap.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_processing import (
    ActionNormalizationMethod,
    ActionNormalizer,
    load_action_stats,
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.action_spec import Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.pose_utils import compute_idle_frames
from cosmos_framework.utils import log

CameraMode = Literal[
    "concat_view",
    "agent_wrist",
    "agentview_left",
    "agentview_right",
    "wrist",
]
CaptionMode = Literal["policy", "oracle_policy_rich"]
VideoBackend = Literal["lerobot", "opencv"]

_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")
_VALID_MODES = {*_MODE_CHOICES, "joint"}
_VALID_SPLITS = {"train", "val", "test"}
_CAMERA_FIELDS = {
    "agentview_left": "robot0_agentview_left",
    "agentview_right": "robot0_agentview_right",
    "wrist": "robot0_eye_in_hand",
}
_CAMERAS_BY_MODE = {
    "concat_view": ("agentview_left", "wrist"),
    "agent_wrist": ("agentview_left", "wrist"),
    "agentview_left": ("agentview_left",),
    "agentview_right": ("agentview_right",),
    "wrist": ("wrist",),
}
_VIEWPOINT_BY_MODE = {
    "concat_view": "concat_view",
    "agent_wrist": "concat_view",
    "agentview_left": "third_person_view",
    "agentview_right": "third_person_view",
    "wrist": "wrist_view",
}
_ACTION_SPEC = build_action_spec(Pos(), Rot("rot6d"), Gripper())
_STAT_KEYS = {"mean", "std", "min", "max", "q01", "q99"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RoboCasa365SegmentDataset(Dataset):
    """Map-style view of fixed-length windows from the CWAM RoboCasa365 view."""

    def __init__(
        self,
        root: str,
        source_root: str | None = None,
        *,
        split: str = "train",
        fps: float = 20.0,
        chunk_length: int = 16,
        mode: str = "policy",
        camera_mode: CameraMode = "concat_view",
        image_size: int = 256,
        caption_mode: CaptionMode = "policy",
        action_normalization: str | None = "quantile_rot",
        action_stats_path: str | None = None,
        embodiment_type: str = "robocasa365",
        video_backend: VideoBackend = "lerobot",
        tolerance_s: float = 1e-4,
        decode_retries: int = 8,
        resample_on_decode_error: bool | None = None,
        validate_manifest: bool = True,
    ) -> None:
        super().__init__()
        split = split.lower().strip()
        mode = mode.lower().strip()
        camera_mode = cast(CameraMode, camera_mode.lower().strip())
        caption_mode = cast(CaptionMode, caption_mode.lower().strip())
        video_backend = cast(VideoBackend, video_backend.lower().strip())

        if split not in _VALID_SPLITS:
            raise ValueError(f"Unsupported split={split!r}; use one of {sorted(_VALID_SPLITS)}.")
        if mode not in _VALID_MODES:
            raise ValueError(f"Unsupported mode={mode!r}; use one of {sorted(_VALID_MODES)}.")
        if camera_mode not in _CAMERAS_BY_MODE:
            raise ValueError(f"Unsupported camera_mode={camera_mode!r}; use one of {sorted(_CAMERAS_BY_MODE)}.")
        if caption_mode not in {"policy", "oracle_policy_rich"}:
            raise ValueError("caption_mode must be 'policy' or the explicitly non-causal 'oracle_policy_rich'.")
        if video_backend not in {"lerobot", "opencv"}:
            raise ValueError("video_backend must be 'lerobot' or 'opencv'.")
        if float(fps) <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")
        if int(chunk_length) <= 0 or int(chunk_length) % 4 != 0:
            raise ValueError(f"chunk_length must be positive and divisible by 4, got {chunk_length}.")
        if int(image_size) <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}.")
        if int(decode_retries) < 1:
            raise ValueError(f"decode_retries must be at least 1, got {decode_retries}.")

        self._root = Path(root)
        self._split = split
        self._fps = float(fps)
        self._chunk_length = int(chunk_length)
        self._mode = mode
        self._camera_mode = camera_mode
        self._camera_names = _CAMERAS_BY_MODE[camera_mode]
        self._viewpoint = _VIEWPOINT_BY_MODE[camera_mode]
        self._image_size = int(image_size)
        self._caption_mode = caption_mode
        self._action_normalization = action_normalization
        self._domain_name = embodiment_type
        self._domain_id = get_domain_id(embodiment_type)
        self._video_backend = video_backend
        self._tolerance_s = float(tolerance_s)
        self._decode_retries = int(decode_retries)
        # Fail fast by default. Random replacement would allow the same
        # physical episode to cross rank/worker shards and make data errors
        # invisible. Explicit opt-in remains available for exploratory runs.
        self._resample_on_decode_error = False if resample_on_decode_error is None else bool(resample_on_decode_error)

        manifest_path = self._root / "dataset_manifest.json"
        success_path = self._root / "_SUCCESS.json"
        if not manifest_path.is_file() or not success_path.is_file():
            raise FileNotFoundError(
                f"RoboCasa365 view must contain dataset_manifest.json and _SUCCESS.json under {self._root}."
            )
        self._manifest = json.loads(manifest_path.read_text())
        self._dataset_uri = f"robocasa365://{self._manifest.get('dataset_id', 'unknown')}/{split}"
        success = json.loads(success_path.read_text())
        if success.get("status") != "complete":
            raise RuntimeError(f"RoboCasa365 conversion is not complete: status={success.get('status')!r}.")
        if int(self._manifest.get("schema_version", -1)) != 1:
            raise ValueError(
                f"Unsupported RoboCasa365 manifest schema_version={self._manifest.get('schema_version')!r}; "
                "this loader supports version 1."
            )
        if success.get("dataset_id") not in (None, self._manifest.get("dataset_id")):
            raise RuntimeError(
                f"RoboCasa365 dataset_id mismatch: manifest={self._manifest.get('dataset_id')!r}, "
                f"success={success.get('dataset_id')!r}."
            )
        if validate_manifest:
            expected_sha = success.get("dataset_manifest_sha256")
            actual_sha = _sha256(manifest_path)
            if not expected_sha or actual_sha != expected_sha:
                raise RuntimeError(
                    "RoboCasa365 dataset_manifest.json checksum mismatch: "
                    f"expected={expected_sha!r}, actual={actual_sha!r}."
                )

        contract = self._manifest.get("contracts", {})
        expected_actions = int(contract.get("window_action_steps", self._chunk_length))
        expected_frames = int(contract.get("window_video_frames", self._chunk_length + 1))
        manifest_fps = float(self._manifest.get("inputs", {}).get("fps", self._fps))
        if expected_actions != self._chunk_length or expected_frames != self._chunk_length + 1:
            raise ValueError(
                "Requested chunk_length does not match the converted view: "
                f"requested={self._chunk_length}, manifest_actions={expected_actions}, "
                f"manifest_frames={expected_frames}."
            )
        if not np.isclose(manifest_fps, self._fps):
            raise ValueError(f"Requested fps={self._fps} does not match manifest fps={manifest_fps}.")

        source_from_manifest = self._manifest.get("inputs", {}).get("source_root")
        resolved_source = source_root or source_from_manifest
        if not resolved_source:
            raise ValueError("source_root is required when it is absent from dataset_manifest.json.")
        self._source_root = Path(resolved_source)
        if not self._source_root.is_dir():
            raise FileNotFoundError(f"RoboCasa365 source_root not found: {self._source_root}.")

        split_path = self._root / "splits" / f"{split}.parquet"
        if not split_path.is_file():
            raise FileNotFoundError(f"RoboCasa365 split manifest not found: {split_path}.")
        rows = pq.read_table(split_path).to_pylist()
        rows = [row for row in rows if bool(row["eligible_for_chunk"]) and int(row["valid_window_count"]) > 0]
        # Keep every physical source episode on one contiguous shuffle block.
        # This prevents overlapping DeMiAn segments from the same episode being
        # assigned to different ranks/workers and improves shared-MP4 seek locality.
        rows.sort(
            key=lambda row: (
                str(row["split_group"]),
                int(row["source_frame_offset"]),
                int(row["demian_episode_index"]),
            )
        )
        if not rows:
            raise RuntimeError(f"RoboCasa365 split {split!r} contains no eligible windows.")
        self._segments: list[dict[str, Any]] = rows
        valid_counts = np.asarray([int(row["valid_window_count"]) for row in rows], dtype=np.int64)
        self._valid_cum = np.cumsum(valid_counts, dtype=np.int64)
        self._shuffle_blocks = self._build_shuffle_blocks(valid_counts)

        dense_rel = self._manifest.get("dense_actions", {}).get("rot6d10")
        if not dense_rel:
            raise KeyError("dataset_manifest.json is missing dense_actions.rot6d10.")
        action_path = self._root / dense_rel
        # Do not add mmap_mode here: mmap is unsupported on the H-cluster GPFS.
        self._actions = np.load(action_path, allow_pickle=False)
        if self._actions.dtype != np.float32 or self._actions.ndim != 2 or self._actions.shape[1] != 10:
            raise ValueError(
                f"Expected dense rot6d actions with shape [N,10] float32, got {self._actions.shape} "
                f"{self._actions.dtype} from {action_path}."
            )
        expected_action_rows = self._manifest.get("counts", {}).get("action_rows")
        if expected_action_rows is not None and int(self._actions.shape[0]) != int(expected_action_rows):
            raise ValueError(
                f"Dense action row count mismatch: manifest={expected_action_rows}, array={self._actions.shape[0]}."
            )
        if not np.isfinite(self._actions).all():
            bad_rows = np.flatnonzero(~np.isfinite(self._actions).all(axis=1))
            raise ValueError(
                f"Dense action cache contains non-finite values in {bad_rows.size} rows; "
                f"first rows={bad_rows[:10].tolist()}."
            )
        self._validate_segment_offsets()

        if action_stats_path:
            stats_path = Path(action_stats_path)
            if not stats_path.is_absolute():
                stats_path = self._root / stats_path
        else:
            stats_path = self._root / self._manifest.get(
                "normalization_stats",
                "stats/robocasa365_native_frame_wise_relative_rot6d.json",
            )
        self._action_normalizer = self._resolve_action_normalizer(action_normalization, stats_path)
        self._captions = self._load_rich_captions() if caption_mode == "oracle_policy_rich" else {}

        log.info(
            f"Loaded RoboCasa365 segments root={self._root} source_root={self._source_root} "
            f"split={split!r} mode={mode!r} camera_mode={camera_mode!r} "
            f"segments={len(self._segments)} physical_groups={len(self._shuffle_blocks)} "
            f"windows={len(self)} caption_mode={caption_mode!r}"
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        value = value.lower().strip()
        if value not in _VALID_MODES:
            raise ValueError(f"Unsupported mode={value!r}; use one of {sorted(_VALID_MODES)}.")
        self._mode = value

    @property
    def domain_name(self) -> str:
        return self._domain_name

    @property
    def domain_id(self) -> int:
        return self._domain_id

    @property
    def viewpoint(self) -> str:
        return self._viewpoint

    @property
    def action_dim(self) -> int:
        return 10

    @property
    def action_names(self) -> list[str]:
        return _ACTION_SPEC.names

    def get_action_normalizer(self, _sample: dict[str, Any] | None = None) -> ActionNormalizer | None:
        """Return the transform-time normalizer; dataset actions stay raw."""
        return self._action_normalizer

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Return physical-episode blocks covering every flat index exactly once."""
        return list(self._shuffle_blocks)

    def __len__(self) -> int:
        return int(self._valid_cum[-1])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        n = len(self)
        idx = int(idx)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"RoboCasa365 sample index {idx} out of range for length {n}.")

        attempts = self._decode_retries if self._resample_on_decode_error else 1
        last_error: Exception | None = None
        requested_idx = idx
        for attempt in range(attempts):
            try:
                return self._build_item(idx)
            except Exception as error:  # noqa: BLE001 - decode failures are backend-specific
                last_error = error
                if attempt + 1 < attempts:
                    log.warning(
                        f"RoboCasa365: sample idx={idx} failed to load ({type(error).__name__}: {error}); resampling"
                    )
                    idx = random.randint(0, n - 1)
        raise RuntimeError(
            f"RoboCasa365: failed to load requested idx={requested_idx} after {attempts} attempt(s); "
            f"last error: {last_error}"
        ) from last_error

    def _build_item(self, idx: int) -> dict[str, Any]:
        segment_index = int(np.searchsorted(self._valid_cum, idx, side="right"))
        previous = int(self._valid_cum[segment_index - 1]) if segment_index else 0
        window_offset = idx - previous
        segment = self._segments[segment_index]

        action_start = int(segment["action_offset"]) + window_offset
        action_stop = action_start + self._chunk_length
        action_np = np.ascontiguousarray(self._actions[action_start:action_stop])
        if action_np.shape != (self._chunk_length, self.action_dim):
            raise RuntimeError(
                f"Invalid action slice for segment={segment['demian_episode_index']}: "
                f"start={action_start}, stop={action_stop}, shape={action_np.shape}."
            )
        action = torch.from_numpy(action_np).float()
        video = self._load_video(segment, window_offset)
        caption = self._build_caption(segment, window_offset)
        mode = random.choice(_MODE_CHOICES) if self._mode == "joint" else self._mode
        idle_frames = compute_idle_frames(
            action,
            _ACTION_SPEC,
            eps_t=5e-3 / self._fps,
            eps_r=np.deg2rad(1.5) / self._fps,
            eps_g=1e-2,
            joint_threshold=5e-3 / self._fps,
            min_streak=3,
        )

        result: dict[str, Any] = {
            "__url__": self._dataset_uri,
            "__key__": f"{idx:09d}",
            "ai_caption": caption,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(int(self._fps), dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
        }
        if len(self._camera_names) == 2:
            result["additional_view_description"] = (
                "The left half shows the left third-person agent view; the right half shows the wrist-mounted camera."
            )
        return result

    def _build_shuffle_blocks(self, valid_counts: np.ndarray) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        block_start = 0
        block_length = 0
        current_group: str | None = None
        for row, count in zip(self._segments, valid_counts.tolist()):
            group = str(row["split_group"])
            if current_group is not None and group != current_group:
                blocks.append((block_start, block_length))
                block_start += block_length
                block_length = 0
            current_group = group
            block_length += int(count)
        if block_length:
            blocks.append((block_start, block_length))
        return blocks

    def _validate_segment_offsets(self) -> None:
        action_rows = int(self._actions.shape[0])
        for row in self._segments:
            offset = int(row["action_offset"])
            length = int(row["action_length"])
            windows = int(row["valid_window_count"])
            if offset < 0 or length < self._chunk_length + 1:
                raise ValueError(
                    f"Invalid action interval for segment={row['demian_episode_index']}: "
                    f"offset={offset}, length={length}."
                )
            expected_windows = length - self._chunk_length
            if windows != expected_windows:
                raise ValueError(
                    f"valid_window_count mismatch for segment={row['demian_episode_index']}: "
                    f"manifest={windows}, expected={expected_windows}."
                )
            if offset + length > action_rows:
                raise ValueError(
                    f"Action interval exceeds dense cache for segment={row['demian_episode_index']}: "
                    f"stop={offset + length}, rows={action_rows}."
                )

    @staticmethod
    def _resolve_action_normalizer(
        method: str | None,
        stats_path: Path,
    ) -> ActionNormalizer | None:
        if method is None:
            return None
        if method not in {
            "quantile",
            "quantile_rot",
            "quantile_rot_scale_floor",
            "asinh_rot",
            "piecewise_asinh_rot",
            "meanstd",
            "minmax",
        }:
            raise ValueError(f"Unsupported action_normalization={method!r}.")
        raw_stats = load_action_stats(str(stats_path), stats_key="global_raw")
        if set(raw_stats) != _STAT_KEYS:
            raise ValueError(
                f"Expected stats keys {sorted(_STAT_KEYS)} in global_raw at {stats_path}, got {sorted(raw_stats)}."
            )
        stats = {key: torch.from_numpy(value).float() for key, value in raw_stats.items()}
        for key, value in stats.items():
            if value.shape != (10,):
                raise ValueError(f"RoboCasa365 stat {key!r} must have shape (10,), got {tuple(value.shape)}.")
            if not torch.isfinite(value).all():
                raise ValueError(f"RoboCasa365 stat {key!r} contains non-finite values.")
        return resolve_action_normalization(cast(ActionNormalizationMethod, method), stats)

    def _load_rich_captions(self) -> dict[int, tuple[np.ndarray, tuple[str, ...]]]:
        ranges_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._segments:
            if bool(row["rich_caption_enabled"]):
                ranges_by_path[str(row["caption_shard_path"])].append(row)

        captions: dict[int, tuple[np.ndarray, tuple[str, ...]]] = {}
        for relative_path, segments in sorted(ranges_by_path.items()):
            path = self._root / relative_path
            table = pq.read_table(path, columns=["demian_episode_index", "frame_index", "current_action"])
            for segment in segments:
                start = int(segment["caption_row_offset"])
                stop = int(segment["caption_row_stop"])
                if start < 0 or stop < start or stop > table.num_rows:
                    raise ValueError(
                        f"Invalid caption interval [{start}, {stop}) in {relative_path} "
                        f"for segment={segment['demian_episode_index']}."
                    )
                block = table.slice(start, stop - start)
                episode_index = int(segment["demian_episode_index"])
                block_episodes = block["demian_episode_index"].to_numpy(zero_copy_only=False)
                if block.num_rows and not np.all(block_episodes == episode_index):
                    raise ValueError(
                        f"Caption interval in {relative_path} crosses episodes for segment={episode_index}."
                    )
                frames = np.asarray(block["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int32)
                texts = tuple(str(value or "").strip() for value in block["current_action"].to_pylist())
                if frames.size and np.any(np.diff(frames) < 0):
                    raise ValueError(f"Caption frames are not sorted for segment={episode_index}.")
                captions[episode_index] = (frames, texts)
        return captions

    def _build_caption(self, segment: dict[str, Any], window_offset: int) -> str:
        policy_task = str(segment["policy_task_text"]).strip()
        if not policy_task:
            raise ValueError(f"Empty policy_task_text for segment={segment['demian_episode_index']}.")
        if self._caption_mode != "oracle_policy_rich" or not bool(segment["rich_caption_enabled"]):
            return policy_task

        rich = self._captions.get(int(segment["demian_episode_index"]))
        if not rich or rich[0].size == 0:
            return policy_task
        frames, texts = rich
        target_frame = window_offset + self._chunk_length // 2
        caption_index = bisect.bisect_right(frames, target_frame) - 1
        if caption_index < 0 or not texts[caption_index]:
            return policy_task
        return f"{policy_task}\nCurrent action: {texts[caption_index]}"

    def _load_video(self, segment: dict[str, Any], window_offset: int) -> torch.Tensor:
        frames_by_camera: list[torch.Tensor] = []
        frame_offsets = np.arange(self._chunk_length + 1, dtype=np.float64) + float(window_offset)
        for camera_name in self._camera_names:
            field = _CAMERA_FIELDS[camera_name]
            relative_path = str(segment[f"{field}_video_path"])
            base_timestamp = float(segment[f"{field}_from_timestamp"])
            timestamps = (base_timestamp + frame_offsets / self._fps).tolist()
            frames = self._decode_video(self._source_root / relative_path, timestamps)
            frames_by_camera.append(self._resize_video(frames))

        video_tchw = torch.cat(frames_by_camera, dim=-1) if len(frames_by_camera) == 2 else frames_by_camera[0]
        if video_tchw.shape[0] != self._chunk_length + 1 or video_tchw.shape[1] != 3:
            raise ValueError(f"Decoded video must have shape [T,3,H,W], got {tuple(video_tchw.shape)}.")
        if video_tchw.dtype == torch.uint8:
            return video_tchw.permute(1, 0, 2, 3).contiguous()
        if not torch.is_floating_point(video_tchw):
            raise TypeError(f"Decoded video must be floating point or uint8, got {video_tchw.dtype}.")
        video_min = float(video_tchw.amin().item())
        video_max = float(video_tchw.amax().item())
        if video_min < 0.0 or video_max > 1.0:
            raise ValueError(f"Decoded floating-point video must be in [0,1], got [{video_min}, {video_max}].")
        return (video_tchw * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3).contiguous()

    def _decode_video(self, path: Path, timestamps: list[float]) -> torch.Tensor:
        if self._video_backend == "lerobot":
            from lerobot.datasets.video_utils import decode_video_frames

            return decode_video_frames(path, timestamps, self._tolerance_s)

        # OpenCV is intended for CPU data validation and environments where the
        # optional LeRobot decoder is unavailable.  Production training should
        # use the LeRobot backend for its timestamp tolerance checks.
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}.")
        decoded: list[torch.Tensor] = []
        try:
            for timestamp in timestamps:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError(f"OpenCV failed to decode {path} at {timestamp:.6f}s.")
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                decoded.append(torch.from_numpy(frame_rgb).permute(2, 0, 1))
        finally:
            capture.release()
        return torch.stack(decoded, dim=0)

    def _resize_video(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"Decoder must return [T,3,H,W], got {tuple(frames.shape)}.")
        if frames.shape[-2:] == (self._image_size, self._image_size):
            return frames
        source_dtype = frames.dtype
        resized = F.interpolate(
            frames.float(),
            size=(self._image_size, self._image_size),
            mode="bilinear",
            align_corners=False,
        )
        if source_dtype == torch.uint8:
            return resized.round().clamp(0.0, 255.0).to(torch.uint8)
        return resized.to(source_dtype)
