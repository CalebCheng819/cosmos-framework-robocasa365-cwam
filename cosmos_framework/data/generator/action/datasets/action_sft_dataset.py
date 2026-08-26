# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Map-style action datasets composed with ``ActionTransformPipeline``.

The base ``DROIDLeRobotDataset.__getitem__`` returns the raw sample
(``video``/``action``/``ai_caption``/``viewpoint``/``mode``/``domain_id``/
``idle_frames``). The model expects each sample to be passed through
``ActionTransformPipeline`` (spatial resize/pad, text tokenization, action
padding to ``max_action_dim``, and ``sequence_plan`` construction). This thin
wrapper composes the two so the experiment can hand a single map-style dataset
to ``RankPartitionedDataLoader`` (mirroring how the vision recipe uses
``get_sft_dataset``).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from cosmos_framework.data.generator.action.action_processing import ActionNormalizer
from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import (
    DROIDLeRobotDataset,
)
from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset
from cosmos_framework.data.generator.action.datasets.robocasa365_segment_dataset import RoboCasa365SegmentDataset
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline


class ActionSFTDataset(Dataset):
    """Wraps a map-style action dataset and applies ``ActionTransformPipeline`` per sample."""

    def __init__(
        self,
        dataset: Dataset,
        transform: ActionTransformPipeline,
        resolution: str | int | None,
        action_normalizer: ActionNormalizer | None = None,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._transform = transform
        self._resolution = resolution
        self._action_normalizer = action_normalizer

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._transform(
            self._dataset[idx],
            self._resolution,
            action_normalizer=self._action_normalizer,
        )

    def get_shuffle_blocks(self):
        """Delegate flat-index shuffle blocks to the inner action dataset."""
        return self._dataset.get_shuffle_blocks()


class ActionIterableShuffleDataset(IterableDataset):
    """Streaming view of a map-style ``ActionSFTDataset``.

    Each ``(rank, worker)`` owns a permanent DISJOINT subset of episodes (sharded
    over ``shard_world_size * num_workers``), shuffles only its owned episode
    ORDER, and streams the windows WITHIN each episode sequentially. Permanent
    ownership prevents a faster worker entering its next local epoch from
    overlapping a physical episode still being consumed by a slower worker.
    The local order is re-shuffled each epoch and streams indefinitely (the
    trainer stops at ``max_iter``).

    ``shard_world_size`` / ``shard_rank`` are set by ``RankPartitionedDataLoader``.
    """

    def __init__(self, dataset: "ActionSFTDataset", seed: int = 42):
        super().__init__()
        self._dataset = dataset
        self._seed = int(seed)
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:  # informational only; iteration is infinite
        return len(self._dataset)

    def __iter__(self):
        blocks = self._dataset.get_shuffle_blocks()
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1
        global_shard = int(self.shard_rank) * nw + wid
        total_shards = max(1, int(self.shard_world_size) * nw)
        owned_blocks = list(range(global_shard, len(blocks), total_shards))
        if not owned_blocks:
            raise RuntimeError(
                "ActionIterableShuffleDataset has fewer physical episode blocks "
                f"({len(blocks)}) than rank-worker shards ({total_shards}); "
                f"shard {global_shard} would never yield a sample."
            )
        epoch = 0
        while True:
            g = torch.Generator()
            g.manual_seed(self._seed + epoch * total_shards + global_shard)
            order = torch.randperm(len(owned_blocks), generator=g).tolist()
            for local_index in order:
                b = owned_blocks[local_index]
                start, length = blocks[b]
                for idx in range(start, start + length):
                    yield self._dataset[idx]
            epoch += 1


FIXED_INDEX_TRACE_SCHEMA_VERSION = 1
FIXED_INDEX_TRACE_ALGORITHM = "baseline-worker-interleave-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shuffle_blocks_sha256(blocks: Sequence[tuple[int, int]]) -> str:
    payload = json.dumps(
        [[int(start), int(length)] for start, length in blocks],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def fixed_index_trace_manifest_path(trace_path: str | Path) -> Path:
    """Return the required sidecar path for one ``.npy`` fixed trace."""

    path = Path(trace_path)
    if path.suffix != ".npy":
        raise ValueError(f"Fixed action index trace must use a .npy suffix, got {path}.")
    return path.with_suffix(".manifest.json")


def _baseline_worker_index_stream(
    blocks: Sequence[tuple[int, int]],
    *,
    global_shard: int,
    total_shards: int,
    seed: int,
) -> Iterator[int]:
    """Reproduce one permanent worker shard from ActionIterableShuffleDataset."""

    owned_blocks = list(range(global_shard, len(blocks), total_shards))
    if not owned_blocks:
        raise RuntimeError(
            "Cannot build a fixed trace: fewer physical episode blocks "
            f"({len(blocks)}) than baseline rank-worker shards ({total_shards}); "
            f"shard {global_shard} would never yield a sample."
        )
    epoch = 0
    while True:
        generator = torch.Generator()
        generator.manual_seed(int(seed) + epoch * total_shards + global_shard)
        order = torch.randperm(len(owned_blocks), generator=generator).tolist()
        for local_index in order:
            start, length = blocks[owned_blocks[local_index]]
            yield from range(int(start), int(start) + int(length))
        epoch += 1


def build_action_fixed_index_trace(
    dataset: Dataset,
    *,
    world_size: int,
    baseline_num_workers: int,
    seed: int,
    indices_per_rank: int,
) -> np.ndarray:
    """Materialize the exact baseline-worker-interleaved index stream.

    The baseline iterable dataset permanently assigns episode blocks to each
    ``(rank, worker)``. PyTorch's ordered DataLoader then consumes one item from
    each worker in round-robin task order. This function records that logical
    order so 4/8/12/16-worker probes can all replay the same per-rank samples.
    """

    world_size = int(world_size)
    baseline_num_workers = int(baseline_num_workers)
    indices_per_rank = int(indices_per_rank)
    if world_size <= 0 or baseline_num_workers <= 0 or indices_per_rank <= 0:
        raise ValueError("world_size, baseline_num_workers, and indices_per_rank must be positive.")
    if indices_per_rank % baseline_num_workers:
        raise ValueError("indices_per_rank must be divisible by baseline_num_workers.")
    blocks = [(int(start), int(length)) for start, length in dataset.get_shuffle_blocks()]
    if not blocks or any(start < 0 or length <= 0 for start, length in blocks):
        raise ValueError("Fixed trace requires non-empty positive shuffle blocks.")
    dataset_length = len(dataset)
    cursor = 0
    for start, length in blocks:
        if start != cursor:
            raise ValueError("Shuffle blocks must cover [0, len(dataset)) exactly once in order.")
        cursor += length
    if cursor != dataset_length:
        raise ValueError("Shuffle blocks must cover [0, len(dataset)) exactly once in order.")

    trace = np.empty((world_size, indices_per_rank), dtype=np.int64)
    total_shards = world_size * baseline_num_workers
    for rank in range(world_size):
        streams = [
            _baseline_worker_index_stream(
                blocks,
                global_shard=rank * baseline_num_workers + worker,
                total_shards=total_shards,
                seed=seed,
            )
            for worker in range(baseline_num_workers)
        ]
        for position in range(indices_per_rank):
            trace[rank, position] = next(streams[position % baseline_num_workers])
    return trace


def write_action_fixed_index_trace(
    dataset: Dataset,
    trace_path: str | Path,
    *,
    world_size: int = 8,
    baseline_num_workers: int = 4,
    seed: int = 42,
    transform_seed: int = 42,
    indices_per_rank: int = 15_552,
    batch_size_per_rank: int = 128,
    grad_accum_iter: int = 2,
    warmup_steps: int = 10,
    measure_steps: int = 50,
    prefetch_factor: int = 4,
    in_order: bool = True,
    dataset_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an immutable trace array and checksum-linked completion sidecar."""

    path = Path(trace_path)
    manifest_path = fixed_index_trace_manifest_path(path)
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to replace an existing fixed trace: {path} / {manifest_path}.")
    if not in_order:
        raise ValueError("Fixed trace replay requires DataLoader in_order=True.")
    if indices_per_rank < (warmup_steps + measure_steps) * grad_accum_iter * batch_size_per_rank + 128:
        raise ValueError("indices_per_rank is too short for all probe batches plus the trainer's terminal fetch.")

    trace = build_action_fixed_index_trace(
        dataset,
        world_size=world_size,
        baseline_num_workers=baseline_num_workers,
        seed=seed,
        indices_per_rank=indices_per_rank,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_array = path.with_name(f".{path.name}.incomplete-{os.getpid()}-{time.time_ns()}")
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.incomplete-{os.getpid()}-{time.time_ns()}"
    )
    array_published = False
    try:
        with temporary_array.open("xb") as handle:
            np.save(handle, trace, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        array_sha256 = _sha256_file(temporary_array)
        blocks = dataset.get_shuffle_blocks()
        manifest = {
            "schema_version": FIXED_INDEX_TRACE_SCHEMA_VERSION,
            "status": "complete",
            "algorithm": FIXED_INDEX_TRACE_ALGORITHM,
            "array_path": path.name,
            "array_sha256": array_sha256,
            "dtype": "int64",
            "shape": [int(value) for value in trace.shape],
            "dataset_length": len(dataset),
            "shuffle_blocks_sha256": _shuffle_blocks_sha256(blocks),
            "shuffle_seed": int(seed),
            "transform_seed": int(transform_seed),
            "source_world_size": int(world_size),
            "baseline_num_workers": int(baseline_num_workers),
            "indices_per_rank": int(indices_per_rank),
            "batch_size_per_rank": int(batch_size_per_rank),
            "grad_accum_iter": int(grad_accum_iter),
            "warmup_steps": int(warmup_steps),
            "measure_steps": int(measure_steps),
            "total_optimizer_steps": int(warmup_steps + measure_steps),
            "prefetch_factor": int(prefetch_factor),
            "in_order": True,
            "dataset_contract": dict(dataset_contract or {}),
        }
        with temporary_manifest.open("x") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_array, path)
        array_published = True
        os.replace(temporary_manifest, manifest_path)
        return manifest
    except BaseException:
        for temporary in (temporary_array, temporary_manifest):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        if array_published and not manifest_path.exists():
            path.unlink(missing_ok=True)
        raise


class ActionFixedIndexTraceDataset(IterableDataset):
    """Finite, checksum-bound per-rank trace replay independent of worker count."""

    def __init__(self, dataset: ActionSFTDataset, trace_path: str | Path):
        super().__init__()
        self._dataset = dataset
        self._trace_path = Path(trace_path)
        manifest_path = fixed_index_trace_manifest_path(self._trace_path)
        if not self._trace_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Fixed action trace requires both {self._trace_path} and {manifest_path}."
            )
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema_version") != FIXED_INDEX_TRACE_SCHEMA_VERSION
            or manifest.get("status") != "complete"
            or manifest.get("algorithm") != FIXED_INDEX_TRACE_ALGORITHM
            or manifest.get("array_path") != self._trace_path.name
            or manifest.get("dtype") != "int64"
            or manifest.get("in_order") is not True
        ):
            raise RuntimeError(f"Fixed action trace manifest has an unsupported contract: {manifest_path}.")
        actual_sha = _sha256_file(self._trace_path)
        if actual_sha != manifest.get("array_sha256"):
            raise RuntimeError(
                f"Fixed action trace checksum mismatch: expected={manifest.get('array_sha256')!r}, "
                f"actual={actual_sha}."
            )
        trace = np.load(self._trace_path, allow_pickle=False)
        if trace.dtype != np.int64 or trace.ndim != 2 or list(trace.shape) != manifest.get("shape"):
            raise RuntimeError(
                f"Fixed action trace array contract mismatch: shape={trace.shape}, dtype={trace.dtype}."
            )
        if int(manifest.get("source_world_size", -1)) != trace.shape[0]:
            raise RuntimeError("Fixed action trace source_world_size does not match its row count.")
        if int(manifest.get("indices_per_rank", -1)) != trace.shape[1]:
            raise RuntimeError("Fixed action trace indices_per_rank does not match its column count.")
        if int(manifest.get("dataset_length", -1)) != len(dataset):
            raise RuntimeError("Fixed action trace dataset_length does not match the active dataset.")
        if manifest.get("shuffle_blocks_sha256") != _shuffle_blocks_sha256(dataset.get_shuffle_blocks()):
            raise RuntimeError("Fixed action trace shuffle-block contract does not match the active dataset.")
        if trace.size == 0 or int(trace.min()) < 0 or int(trace.max()) >= len(dataset):
            raise RuntimeError("Fixed action trace contains an out-of-range dataset index.")
        self._trace = np.ascontiguousarray(trace)
        self._manifest = manifest
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:
        return int(self._trace.shape[1])

    @property
    def trace_manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    def __iter__(self):
        if int(self.shard_world_size) != self._trace.shape[0]:
            raise RuntimeError(
                f"Fixed action trace requires world_size={self._trace.shape[0]}, "
                f"got {self.shard_world_size}."
            )
        rank = int(self.shard_rank)
        if rank < 0 or rank >= self._trace.shape[0]:
            raise RuntimeError(f"Fixed action trace rank {rank} is out of range.")
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        transform_seed = int(self._manifest["transform_seed"])
        for position in range(worker_id, self._trace.shape[1], num_workers):
            index = int(self._trace[rank, position])
            digest = hashlib.sha256(f"{transform_seed}:{rank}:{position}".encode()).digest()
            sample_seed = int.from_bytes(digest[:8], "little", signed=False)
            random.seed(sample_seed)
            np.random.seed(sample_seed % (2**32))
            torch.manual_seed(sample_seed % (2**63))
            sample = self._dataset[index]
            if not isinstance(sample, dict):
                raise TypeError("ActionFixedIndexTraceDataset requires dictionary samples.")
            sample["__trace_position__"] = position
            sample["__trace_rank__"] = rank
            yield sample


def get_action_droid_sft_dataset(
    *,
    root: str,
    fps: float = 15.0,
    chunk_length: int = 32,
    action_space: str = "joint_pos",
    mode: str = "policy",
    use_state: bool = True,
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    use_image_augmentation: bool = False,
    use_filter_dict: bool = False,
    filter_dict_path: str | None = None,
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
    use_success_only: bool = True,
) -> Dataset:
    """Build the DROID action SFT dataset: ``action_space='joint_pos'`` (8D) +
    ``use_state`` (raw/un-normalized), concat_view, chunk_length 32.

    Reads ``root`` (a merged/versioned DROID LeRobot root) as a single flat
    dataset; ``use_success_only=True`` filters to the ``success/`` split."""
    shard_kwargs = dict(
        fps=fps,
        chunk_length=chunk_length,
        viewpoint=viewpoint,
        action_space=action_space,
        mode=mode,
        use_state=use_state,
        action_normalization=action_normalization,
        use_image_augmentation=use_image_augmentation,  # i4: bundles random-crop+resize+ColorJitter
        use_filter_dict=use_filter_dict,
        filter_dict_path=filter_dict_path,
        use_success_only=use_success_only,
    )
    dataset: Dataset = DROIDLeRobotDataset(root=root, **shard_kwargs)
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


def get_action_libero_sft_dataset(
    *,
    root: str,
    fps: float = 20.0,
    chunk_length: int = 16,
    image_size: int = 256,
    mode: str = "policy",
    camera_mode: str = "concat_view",
    action_space: str = "frame_wise_relative",
    rotation_space: str = "6d",
    pose_coordinate_frame: str = "native",
    action_normalization: str | None = "quantile_rot",
    action_stats_path: str | None = None,
    split: str = "train",
    val_ratio: float = 0.01,
    seed: int = 0,
    resolution: str | int | None = None,
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = True,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
    latent_cache_root: str | None = None,
    latent_cache_mode: str = "disabled",
) -> Dataset:
    """Build the LIBERO action-policy SFT dataset (GA reproduction defaults).

    Feeds ``LIBEROLeRobotDataset`` (frame-wise-relative rot6d actions,
    ``quantile_rot``-normalized, concat_view third-person + wrist at 256x256 each
    → 256x512) through ``ActionTransformPipeline``. ``root`` is a LOCAL LeRobot dir
    (read parquet + video directly); pre-sync the HF dataset once, e.g.
    ``hf download lerobot/libero_10 --repo-type dataset --local-dir <root>``. Point
    ``root`` at libero_10 alone. The
    dataset is FPS-agnostic (decodes at real frame timestamps); ``fps`` is metadata
    for ``conditioning_fps`` / prompt duration.
    """
    dataset = LIBEROLeRobotDataset(
        root=root,
        image_size=image_size,
        chunk_length=chunk_length,
        fps=fps,
        mode=mode,
        split=split,
        val_ratio=val_ratio,
        seed=seed,
        camera_mode=camera_mode,
        action_space=action_space,
        rotation_space=rotation_space,
        pose_coordinate_frame=pose_coordinate_frame,
        action_normalization=action_normalization,
        action_stats_path=action_stats_path,
        latent_cache_root=latent_cache_root,
        latent_cache_mode=latent_cache_mode,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


def get_action_robocasa365_sft_dataset(
    *,
    root: str,
    source_root: str | None = None,
    split: str = "train",
    fps: float = 20.0,
    chunk_length: int = 16,
    image_size: int = 256,
    mode: str = "policy",
    camera_mode: str = "concat_view",
    caption_mode: str = "policy",
    action_normalization: str | None = "quantile_rot",
    action_stats_path: str | None = None,
    embodiment_type: str = "robocasa365",
    video_backend: str = "lerobot",
    tolerance_s: float = 1e-4,
    decode_retries: int = 8,
    resample_on_decode_error: bool | None = None,
    validate_manifest: bool = True,
    latent_cache_root: str | None = None,
    latent_cache_mode: str = "disabled",
    resolution: str | int | None = None,
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = True,
    format_prompt_as_json: bool = True,
    iterable_shuffle: bool = True,
    episode_shuffle_seed: int = 42,
    fixed_index_trace_path: str | None = None,
) -> Dataset:
    """Build the CWAM RoboCasa365 segment dataset for Cosmos action SFT.

    ``root`` points at the compact converted view containing split manifests,
    dense rot6d actions, captions, and statistics. ``source_root`` points at
    the original RoboCasa365 LeRobot tree whose shared MP4s are referenced by
    the view; when omitted, the conversion-time path from
    ``dataset_manifest.json`` is used.

    The inner dataset intentionally returns raw 10D actions. Normalization is
    passed to :class:`ActionTransformPipeline`, which preserves ``action_raw``
    for evaluation, normalizes the real channels once, and pads the model input
    to ``max_action_dim``. ``caption_mode='policy'`` is the causal policy
    contract. ``oracle_policy_rich`` is reserved for explicitly labelled
    non-causal diagnostics and must not be used for policy training.
    """
    dataset = RoboCasa365SegmentDataset(
        root=root,
        source_root=source_root,
        split=split,
        fps=fps,
        chunk_length=chunk_length,
        image_size=image_size,
        mode=mode,
        camera_mode=camera_mode,
        caption_mode=caption_mode,
        action_normalization=action_normalization,
        action_stats_path=action_stats_path,
        embodiment_type=embodiment_type,
        video_backend=video_backend,
        tolerance_s=tolerance_s,
        decode_retries=decode_retries,
        resample_on_decode_error=resample_on_decode_error,
        validate_manifest=validate_manifest,
        latent_cache_root=latent_cache_root,
        latent_cache_mode=latent_cache_mode,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(
        dataset,
        transform,
        resolution,
        action_normalizer=dataset.get_action_normalizer(),
    )
    if fixed_index_trace_path:
        if not iterable_shuffle:
            raise ValueError("fixed_index_trace_path requires iterable_shuffle=True.")
        return ActionFixedIndexTraceDataset(sft, fixed_index_trace_path)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft
