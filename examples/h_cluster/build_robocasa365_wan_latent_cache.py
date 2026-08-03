# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build the strict RoboCasa365 post-crop Wan latent cache.

The builder is designed for one ``torchrun`` process per GPU. Rank ranges are
contiguous in physical-window order. CPU workers merge overlapping windows
from one physical episode into one decoder request, while the GPU encodes a
configurable batch. Shards and receipts are resumable under a deterministic
``.incomplete-*`` directory; ``_SUCCESS.json`` is written last and the complete
tree is atomically renamed into place.

When ``--nvme-source-root`` is supplied, one local rank per host copies and
content-verifies the small raw RoboCasa tree before the global build starts.
The rjob submission itself must also request host NVMe with the H-platform
``--store-host-nvme`` flag.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import socket
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from cosmos_framework.data.generator.action.datasets.robocasa365_latent_cache import (
    CACHE_SCHEMA_VERSION,
    CACHE_STAGE,
    IDENTITY_DIGEST_ALGORITHM,
    OFFICIAL_WAN22_VAE_SHA256,
    WAN_CACHE_VAE_BEHAVIOR_CONTRACT,
    active_preprocessing_code_sha256,
    canonical_json_bytes,
    sha256_file,
    validate_published_cache_files,
)
from cosmos_framework.data.generator.action.datasets.robocasa365_segment_dataset import (
    RoboCasa365SegmentDataset,
)
from cosmos_framework.data.generator.action.transforms import VideoResize

_CAMERA_FIELDS_BY_MODE: dict[str, tuple[str, ...]] = {
    "concat_view": ("robot0_agentview_left", "robot0_eye_in_hand"),
    "agent_wrist": ("robot0_agentview_left", "robot0_eye_in_hand"),
    "triple_view": ("robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"),
    "agentview_left": ("robot0_agentview_left",),
    "agentview_right": ("robot0_agentview_right",),
    "wrist": ("robot0_eye_in_hand",),
}
_NVME_RECEIPT = "_CWAM_SOURCE_STAGE_RECEIPT.json"


@dataclass(frozen=True)
class WorkBatch:
    split: str
    sample_indices: tuple[int, ...]
    ordinals: tuple[int, ...]


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_json_atomic(path: Path, value: object) -> None:
    _write_bytes_atomic(path, _json_bytes(value))


def _write_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        pq.write_table(table, temporary, compression="zstd", row_group_size=65_536)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _parse_int_tuple(value: str, *, length: int, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be {length} comma-separated integers.") from error
    if len(parsed) != length or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{label} must contain {length} positive integers.")
    return parsed


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _tree_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative == _NVME_RECEIPT or path.is_dir():
            continue
        if path.is_symlink():
            raise RuntimeError(f"Source staging refuses symlinks: {path}.")
        if not path.is_file():
            raise RuntimeError(f"Source staging found a non-regular entry: {path}.")
        file_sha = sha256_file(path)
        size = path.stat().st_size
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "little"))
        digest.update(bytes.fromhex(file_sha))
        file_count += 1
        total_bytes += size
    return {
        "algorithm": "sha256-relative-path-size-content-v1",
        "tree_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _stage_source_to_nvme(source_root: Path, destination: Path) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    destination = destination.resolve(strict=False)
    source_inventory = _tree_inventory(source_root)
    receipt_path = destination / _NVME_RECEIPT
    if destination.exists():
        if not destination.is_dir() or not receipt_path.is_file():
            raise RuntimeError(
                f"Refusing to overwrite an unverified existing NVMe source destination: {destination}."
            )
        receipt = json.loads(receipt_path.read_text())
        staged_inventory = _tree_inventory(destination)
        if (
            receipt.get("status") != "complete"
            or receipt.get("source_inventory") != source_inventory
            or receipt.get("staged_inventory") != staged_inventory
            or staged_inventory != source_inventory
        ):
            raise RuntimeError(f"Existing NVMe source stage failed receipt/content validation: {destination}.")
        return staged_inventory

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.incomplete-{socket.gethostname()}-{os.getpid()}-{time.time_ns()}"
    )
    shutil.copytree(source_root, temporary, symlinks=False, copy_function=shutil.copy2)
    staged_inventory = _tree_inventory(temporary)
    if staged_inventory != source_inventory:
        raise RuntimeError(
            f"NVMe source stage content mismatch: source={source_inventory!r}, staged={staged_inventory!r}."
        )
    _write_json_atomic(
        temporary / _NVME_RECEIPT,
        {
            "schema_version": 1,
            "status": "complete",
            "source_root": str(source_root),
            "source_inventory": source_inventory,
            "staged_inventory": staged_inventory,
        },
    )
    os.rename(temporary, destination)
    _fsync_dir(destination.parent)
    return staged_inventory


def _distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Wan latent cache construction requires CUDA.")
    if world_size > 1 and not dist.is_initialized():
        # This builder only exchanges small control/evidence objects; VAE work
        # is independent per GPU. Use a CPU Gloo control plane with a timeout
        # longer than the slowest rank's full shard range, avoiding NCCL's
        # default collective watchdog during rank-zero planning/publication or
        # uneven video-decode work.
        dist.init_process_group(backend="gloo", timeout=timedelta(hours=48))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _all_rank_errors(local_error: str | None, *, world_size: int) -> None:
    if world_size == 1:
        if local_error is not None:
            raise RuntimeError(local_error)
        return
    errors: list[str | None] = [None] * world_size
    dist.all_gather_object(errors, local_error)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failures:
        raise RuntimeError("Distributed prerequisite failed:\n" + "\n".join(failures))


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _make_dataset(args: argparse.Namespace, split: str, source_root: Path) -> RoboCasa365SegmentDataset:
    return RoboCasa365SegmentDataset(
        root=str(args.dataset_root),
        source_root=str(source_root),
        split=split,
        fps=args.fps,
        chunk_length=args.chunk_length,
        mode="policy",
        camera_mode=args.camera_mode,
        image_size=args.dataset_image_size,
        caption_mode="policy",
        action_normalization=None,
        video_backend=args.video_backend,
        decode_retries=1,
        resample_on_decode_error=False,
        validate_manifest=True,
        latent_cache_root=None,
        latent_cache_mode="disabled",
        load_actions=False,
        transform_resolution=args.resolution,
    )


def _derive_image_size(args: argparse.Namespace) -> tuple[int, int, int, int]:
    num_cameras = len(_CAMERA_FIELDS_BY_MODE[args.camera_mode])
    probe = {
        "video": torch.empty(
            (3, 1, args.dataset_image_size, args.dataset_image_size * num_cameras),
            dtype=torch.uint8,
        )
    }
    transformed = VideoResize()(probe, args.resolution)
    return tuple(int(value) for value in transformed["image_size"].tolist())  # type: ignore[return-value]


def _build_request(
    args: argparse.Namespace,
    *,
    source_root: Path,
    source_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_manifest_path = args.dataset_root / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    dataset_id = dataset_manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("RoboCasa365 dataset manifest must contain a non-empty dataset_id.")
    source_revision = dataset_manifest.get("inputs", {}).get("robocasa365_revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("RoboCasa365 dataset manifest must bind inputs.robocasa365_revision.")

    expected_image_size = _derive_image_size(args)
    if expected_image_size != args.expected_image_size:
        raise RuntimeError(
            f"Active VideoResize image_size={expected_image_size} does not match "
            f"--expected-image-size={args.expected_image_size}."
        )
    expected_shape = (
        int(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["latent_channels"]),
        (args.chunk_length + 1 - 1)
        // int(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["temporal_compression_factor"])
        + 1,
        max(
            expected_image_size[2]
            // int(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["spatial_compression_factor"]),
            1,
        ),
        max(
            expected_image_size[3]
            // int(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["spatial_compression_factor"]),
            1,
        ),
    )
    if expected_shape != args.expected_latent_shape:
        raise RuntimeError(
            f"Derived post-crop latent shape={expected_shape} does not match "
            f"--expected-latent-shape={args.expected_latent_shape}."
        )
    if args.chunk_length + 1 not in WAN_CACHE_VAE_BEHAVIOR_CONTRACT["encode_exact_durations"]:
        raise ValueError("Requested frame count is absent from the active Wan encode_exact_durations contract.")

    vae_path = args.wan_vae.resolve(strict=True)
    if not vae_path.is_file():
        raise FileNotFoundError(f"Wan VAE weights must be one regular file: {vae_path}.")
    vae_sha256 = sha256_file(vae_path)
    if vae_sha256 != OFFICIAL_WAN22_VAE_SHA256:
        raise RuntimeError(
            "Wan VAE weights do not match the audited official checksum: "
            f"actual={vae_sha256}, expected={OFFICIAL_WAN22_VAE_SHA256}."
        )
    code_sha = active_preprocessing_code_sha256()
    code_sha["cache_builder"] = sha256_file(Path(__file__))
    split_sha = {
        split: sha256_file(args.dataset_root / "splits" / f"{split}.parquet") for split in args.splits
    }
    preprocessing = {
        "schema_version": 1,
        "camera_mode": args.camera_mode,
        "ordered_camera_fields": list(_CAMERA_FIELDS_BY_MODE[args.camera_mode]),
        "fps": args.fps,
        "dataset_image_size": args.dataset_image_size,
        "video_resize": {
            "resolution": args.resolution,
            "resolution_selection": (
                "auto_get_vision_data_resolution" if args.resolution is None else "explicit_resolution_tier"
            ),
            "keep_aspect_ratio": True,
            "pad_keys": ["video"],
            "resize_interpolation": "bicubic",
            "antialias": True,
            "padding_origin": "top_left",
            "padding_mode": "reflect_else_edge",
            "image_size": list(expected_image_size),
        },
        "post_encode_crop": {
            "stage": CACHE_STAGE,
            "spatial_rounding": "floor",
            "minimum_latent_size": 1,
        },
        # Only active training-path hashes are copied into the published
        # preprocessing contract. The builder hash remains in provenance.
        "code_sha256": active_preprocessing_code_sha256(),
    }
    return {
        "schema_version": 1,
        "dataset": {
            "dataset_id": dataset_id,
            "manifest_sha256": sha256_file(dataset_manifest_path),
            "source_revision": source_revision,
            "split_manifest_sha256": split_sha,
        },
        "source": {
            "active_root": str(source_root),
            "inventory": dict(source_inventory),
        },
        "preprocessing": preprocessing,
        "vae": {
            "weights_path": str(vae_path),
            "weights_sha256": vae_sha256,
            "weights_size_bytes": vae_path.stat().st_size,
            "behavior_relevant_config": WAN_CACHE_VAE_BEHAVIOR_CONTRACT,
        },
        "latent": {
            "shape": list(expected_shape),
            "storage_dtype": args.storage_dtype,
            "logical_dtype": "float32",
            "video_num_frames": args.chunk_length + 1,
            "image_size": list(expected_image_size),
            "record_alignment_bytes": args.record_alignment_bytes,
            "records_per_shard": args.records_per_shard,
        },
        "build": {
            "splits": list(args.splits),
            "requested_vae_batch_size": args.requested_vae_batch_size,
            "vae_batch_size": args.vae_batch_size,
            "equivalence_samples_per_rank": args.equivalence_samples_per_rank,
            "real_backend_preflight": args.real_backend_preflight_receipts,
            "code_sha256": code_sha,
            "versions": {
                "torch": torch.__version__,
                "torchvision": _package_version("torchvision"),
                "lerobot": _package_version("lerobot"),
                "torchcodec": _package_version("torchcodec"),
            },
        },
    }


def _physical_anchor(identity: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(
        {
            key: identity[key]
            for key in (
                "source_revision",
                "source_prefix",
                "source_episode_index",
                "source_frame_index",
                "camera_mode",
                "camera_fields",
                "fps",
                "num_frames",
            )
        }
    )


def _physical_locator(identity: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(
        {
            "camera_video_paths": identity["camera_video_paths"],
            "camera_effective_start_timestamps": identity["camera_effective_start_timestamps"],
        }
    )


def _enumerate_and_write_plan(
    args: argparse.Namespace,
    *,
    source_root: Path,
    stage_root: Path,
    request: dict[str, Any],
    request_sha256: str,
    cache_id: str,
) -> dict[str, Any]:
    identity_to_ordinal: dict[bytes, int] = {}
    identity_payload_by_digest: dict[bytes, bytes] = {}
    locator_by_anchor: dict[bytes, bytes] = {}
    records: list[dict[str, Any]] = []
    mappings: dict[str, np.ndarray] = {}
    split_info: dict[str, dict[str, Any]] = {}

    for split in args.splits:
        dataset = _make_dataset(args, split, source_root)
        mapping = np.empty(len(dataset), dtype=np.uint32)
        ordered_digest = hashlib.sha256()
        for sample_index in range(len(dataset)):
            identity = dataset.get_visual_window_identity(sample_index)
            payload = canonical_json_bytes(identity)
            digest = hashlib.sha256(payload).digest()
            previous_payload = identity_payload_by_digest.get(digest)
            if previous_payload is not None and previous_payload != payload:
                raise RuntimeError("SHA-256 collision while enumerating visual identities.")
            identity_payload_by_digest.setdefault(digest, payload)

            anchor = _physical_anchor(identity)
            locator = _physical_locator(identity)
            previous_locator = locator_by_anchor.setdefault(anchor, locator)
            if previous_locator != locator:
                raise RuntimeError(
                    "Overlapping RoboCasa365 segments disagree on the effective video locator for "
                    f"split={split}, sample_index={sample_index}, identity={identity!r}."
                )

            ordinal = identity_to_ordinal.get(digest)
            if ordinal is None:
                ordinal = len(records)
                if ordinal >= np.iinfo(np.uint32).max:
                    raise OverflowError("RoboCasa365 latent cache exceeded uint32 ordinal capacity.")
                identity_to_ordinal[digest] = ordinal
                records.append(
                    {
                        "ordinal": ordinal,
                        "identity_sha256": digest.hex(),
                        "split": split,
                        "sample_index": sample_index,
                        "source_prefix": str(identity["source_prefix"]),
                        "source_episode_index": int(identity["source_episode_index"]),
                        "source_frame_index": int(identity["source_frame_index"]),
                    }
                )
            mapping[sample_index] = ordinal
            ordered_digest.update(len(payload).to_bytes(8, "little"))
            ordered_digest.update(payload)

        mapping_relative = f"mappings/{split}.npy"
        mapping_path = stage_root / mapping_relative
        _write_npy_atomic(mapping_path, mapping)
        mappings[split] = mapping
        split_info[split] = {
            "logical_samples": len(dataset),
            "unique_records": int(np.unique(mapping).shape[0]),
            "mapping_path": mapping_relative,
            "mapping_sha256": sha256_file(mapping_path),
            "source_split_manifest_sha256": request["dataset"]["split_manifest_sha256"][split],
            "ordered_visual_identity_sha256": ordered_digest.hexdigest(),
        }

    identity_relative = "identity/ordinal-sha256.bin"
    _write_bytes_atomic(
        stage_root / identity_relative,
        b"".join(bytes.fromhex(record["identity_sha256"]) for record in records),
    )
    index_relative = "index/records.parquet"
    index_table = pa.Table.from_pylist(records)
    _write_parquet_atomic(stage_root / index_relative, index_table)
    total_records = len(records)
    total_shards = math.ceil(total_records / args.records_per_shard)
    plan = {
        "schema_version": 1,
        "status": "planned",
        "cache_id": cache_id,
        "request_sha256": request_sha256,
        "request": request,
        "splits": split_info,
        "logical_samples": sum(info["logical_samples"] for info in split_info.values()),
        "unique_records": total_records,
        "records_per_shard": args.records_per_shard,
        "total_shards": total_shards,
        "ordinal_identity_path": identity_relative,
        "ordinal_identity_size_bytes": (stage_root / identity_relative).stat().st_size,
        "ordinal_identity_sha256": sha256_file(stage_root / identity_relative),
        "records_index_path": index_relative,
        "records_index_size_bytes": (stage_root / index_relative).stat().st_size,
        "records_index_sha256": sha256_file(stage_root / index_relative),
    }
    _write_json_atomic(stage_root / "build_plan.json", plan)
    return plan


def _load_and_validate_plan(
    stage_root: Path,
    *,
    request: dict[str, Any],
    request_sha256: str,
    cache_id: str,
) -> dict[str, Any]:
    plan_path = stage_root / "build_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Existing cache stage has no complete build_plan.json: {stage_root}.")
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("schema_version") != 1
        or plan.get("status") != "planned"
        or plan.get("cache_id") != cache_id
        or plan.get("request_sha256") != request_sha256
        or plan.get("request") != request
    ):
        raise RuntimeError("Existing latent-cache stage does not match the current immutable build request.")
    for split, split_info in plan["splits"].items():
        mapping_path = stage_root / split_info["mapping_path"]
        if sha256_file(mapping_path) != split_info["mapping_sha256"]:
            raise RuntimeError(f"Staged mapping checksum mismatch for split={split!r}.")
    for path_key, sha_key, size_key in (
        ("ordinal_identity_path", "ordinal_identity_sha256", "ordinal_identity_size_bytes"),
        ("records_index_path", "records_index_sha256", "records_index_size_bytes"),
    ):
        path = stage_root / plan[path_key]
        if path.stat().st_size != plan[size_key] or sha256_file(path) != plan[sha_key]:
            raise RuntimeError(f"Staged build-plan artifact failed validation: {path}.")
    return plan


def _shard_relative_path(shape: tuple[int, ...], shard_id: int) -> str:
    shape_name = "x".join(str(value) for value in shape)
    return f"shards/shape-{shape_name}/shard-{shard_id:06d}.bin"


def _receipt_path(stage_root: Path, shard_id: int) -> Path:
    return stage_root / "receipts" / f"shard-{shard_id:06d}.json"


def _validate_shard_receipt(
    stage_root: Path,
    *,
    plan_sha256: str,
    cache_id: str,
    shard_id: int,
    first_ordinal: int,
    num_records: int,
    stride: int,
    verify_checksum: bool,
) -> dict[str, Any] | None:
    receipt_path = _receipt_path(stage_root, shard_id)
    if not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("schema_version") != 1
            or receipt.get("status") != "complete"
            or receipt.get("cache_id") != cache_id
            or receipt.get("plan_sha256") != plan_sha256
            or type(receipt.get("shard_id")) is not int
            or receipt["shard_id"] != shard_id
            or receipt.get("first_ordinal") != first_ordinal
            or receipt.get("num_records") != num_records
            or receipt.get("size_bytes") != num_records * stride
        ):
            return None
        shard_path = stage_root / receipt["path"]
        if (
            not shard_path.is_file()
            or shard_path.stat().st_size != receipt["size_bytes"]
            or (verify_checksum and sha256_file(shard_path) != receipt["sha256"])
        ):
            return None
        return receipt
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


class _VideoWindowBatchDataset(Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        source_root: Path,
        work_batches: list[WorkBatch],
    ) -> None:
        self.work_batches = work_batches
        self.datasets = {
            split: _make_dataset(args, split, source_root) for split in sorted({batch.split for batch in work_batches})
        }
        self.resize = VideoResize()
        self.expected_image_size = args.expected_image_size

    def __len__(self) -> int:
        return len(self.work_batches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = self.work_batches[index]
        videos = self.datasets[batch.split].load_video_windows(list(batch.sample_indices))
        transformed_videos: list[torch.Tensor] = []
        image_sizes: list[torch.Tensor] = []
        for video in videos:
            transformed = self.resize({"video": video}, None)
            if tuple(int(value) for value in transformed["image_size"].tolist()) != self.expected_image_size:
                raise RuntimeError(
                    f"VideoResize drifted from expected image_size={self.expected_image_size}: "
                    f"got {transformed['image_size'].tolist()}."
                )
            transformed_videos.append(transformed["video"])
            image_sizes.append(transformed["image_size"])
        return {
            "ordinals": torch.tensor(batch.ordinals, dtype=torch.long),
            "video": torch.stack(transformed_videos, dim=0),
            "image_size": torch.stack(image_sizes, dim=0),
        }


def _build_work_batches(
    stage_root: Path,
    plan: Mapping[str, Any],
    *,
    assigned_shards: range,
    completed_shards: set[int],
    vae_batch_size: int,
) -> list[WorkBatch]:
    records_per_shard = int(plan["records_per_shard"])
    first_ordinal = assigned_shards.start * records_per_shard
    stop_ordinal = min(assigned_shards.stop * records_per_shard, int(plan["unique_records"]))
    if first_ordinal >= stop_ordinal:
        return []
    table = pq.read_table(
        stage_root / plan["records_index_path"],
        columns=["ordinal", "split", "sample_index", "source_prefix", "source_episode_index"],
    ).slice(first_ordinal, stop_ordinal - first_ordinal)
    rows = table.to_pylist()
    batches: list[WorkBatch] = []
    current: list[dict[str, Any]] = []
    current_key: tuple[Any, ...] | None = None
    for row in rows:
        ordinal = int(row["ordinal"])
        shard_id = ordinal // records_per_shard
        if shard_id in completed_shards:
            if current:
                batches.append(
                    WorkBatch(
                        split=str(current[0]["split"]),
                        sample_indices=tuple(int(item["sample_index"]) for item in current),
                        ordinals=tuple(int(item["ordinal"]) for item in current),
                    )
                )
                current = []
                current_key = None
            continue
        key = (
            shard_id,
            str(row["split"]),
            str(row["source_prefix"]),
            int(row["source_episode_index"]),
        )
        if current and (
            key != current_key
            or len(current) >= vae_batch_size
            or ordinal != int(current[-1]["ordinal"]) + 1
        ):
            batches.append(
                WorkBatch(
                    split=str(current[0]["split"]),
                    sample_indices=tuple(int(item["sample_index"]) for item in current),
                    ordinals=tuple(int(item["ordinal"]) for item in current),
                )
            )
            current = []
        current_key = key
        current.append(row)
    if current:
        batches.append(
            WorkBatch(
                split=str(current[0]["split"]),
                sample_indices=tuple(int(item["sample_index"]) for item in current),
                ordinals=tuple(int(item["ordinal"]) for item in current),
            )
        )
    return batches


class _FixedShardWriter:
    def __init__(
        self,
        stage_root: Path,
        *,
        cache_id: str,
        plan_sha256: str,
        shard_id: int,
        first_ordinal: int,
        num_records: int,
        shape: tuple[int, ...],
        storage_dtype: str,
        record_nbytes: int,
        stride: int,
    ) -> None:
        self.stage_root = stage_root
        self.cache_id = cache_id
        self.plan_sha256 = plan_sha256
        self.shard_id = shard_id
        self.first_ordinal = first_ordinal
        self.num_records = num_records
        self.shape = shape
        self.storage_dtype = storage_dtype
        self.record_nbytes = record_nbytes
        self.stride = stride
        self.relative_path = _shard_relative_path(shape, shard_id)
        self.final_path = stage_root / self.relative_path
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.final_path.with_suffix(".bin.partial")
        try:
            self.partial_path.unlink()
        except FileNotFoundError:
            pass
        self.handle = self.partial_path.open("xb")
        self.written = 0
        self.singleton_equivalence_samples = 0

    def write(self, ordinal: int, latent: torch.Tensor, *, singleton_equivalent: bool) -> None:
        expected_ordinal = self.first_ordinal + self.written
        if ordinal != expected_ordinal:
            raise RuntimeError(f"Shard {self.shard_id} expected ordinal {expected_ordinal}, got {ordinal}.")
        logical = latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(logical.shape) != self.shape or not torch.isfinite(logical).all():
            raise RuntimeError(
                f"Invalid logical latent for ordinal={ordinal}: shape={tuple(logical.shape)}, dtype={logical.dtype}."
            )
        dtype = torch.bfloat16 if self.storage_dtype == "bfloat16" else torch.float32
        stored = logical.to(dtype=dtype).contiguous()
        if not torch.equal(logical, stored.float()):
            raise RuntimeError(
                f"BF16/FP32 storage round trip is not bitwise equal to online float32 latent at ordinal={ordinal}."
            )
        payload = stored.view(torch.uint8).numpy().tobytes()
        if len(payload) != self.record_nbytes:
            raise RuntimeError(
                f"Record byte count drift at ordinal={ordinal}: got={len(payload)}, expected={self.record_nbytes}."
            )
        self.handle.write(payload)
        self.handle.write(b"\0" * (self.stride - self.record_nbytes))
        self.written += 1
        self.singleton_equivalence_samples += int(singleton_equivalent)

    def finalize(self) -> dict[str, Any]:
        if self.written != self.num_records:
            raise RuntimeError(
                f"Shard {self.shard_id} is incomplete: written={self.written}, expected={self.num_records}."
            )
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        expected_size = self.num_records * self.stride
        if self.partial_path.stat().st_size != expected_size:
            raise RuntimeError(f"Shard {self.shard_id} partial file has the wrong size.")
        os.replace(self.partial_path, self.final_path)
        _fsync_dir(self.final_path.parent)
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "cache_id": self.cache_id,
            "plan_sha256": self.plan_sha256,
            "shard_id": self.shard_id,
            "path": self.relative_path,
            "first_ordinal": self.first_ordinal,
            "num_records": self.num_records,
            "size_bytes": expected_size,
            "sha256": sha256_file(self.final_path),
            "storage_roundtrip_samples": self.num_records,
            "singleton_equivalence_samples": self.singleton_equivalence_samples,
        }
        _write_json_atomic(_receipt_path(self.stage_root, self.shard_id), receipt)
        return receipt

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()


def _load_tokenizer(args: argparse.Namespace):
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

    return Wan2pt2VAEInterface(
        bucket_name="",
        object_store_credential_path_pretrained="",
        vae_path=str(args.wan_vae),
        chunk_duration=int(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["chunk_duration"]),
        keep_decoder_cache=bool(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["keep_decoder_cache"]),
        use_streaming_encode=bool(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["use_streaming_encode"]),
        encode_chunk_frames=dict(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["encode_chunk_frames"]),
        encode_exact_durations=list(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["encode_exact_durations"]),
        spatial_compression_factor=int(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["spatial_compression_factor"]),
        temporal_compression_factor=int(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["temporal_compression_factor"]),
        causal=bool(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["causal"]),
    )


def _preflight_indices(dataset: RoboCasa365SegmentDataset, count: int) -> list[int]:
    """Choose one physical-episode run large enough for a real decode gate."""

    for start, length in dataset.get_shuffle_blocks():
        if int(length) >= count:
            indices = list(range(int(start), int(start) + count))
            physical_keys = {
                (
                    str(segment["source_prefix"]),
                    int(segment["source_episode_index"]),
                )
                for segment, _ in (dataset.resolve_window(index) for index in indices)
            }
            if len(physical_keys) == 1:
                return indices
    raise RuntimeError(
        f"Could not find one physical RoboCasa365 episode with at least {count} preflight windows."
    )


def _prepare_real_backend_preflight(
    args: argparse.Namespace,
    *,
    rank: int,
    source_root: Path,
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    """Decode real MP4 windows and prove union-loader pixel identity."""

    split = args.splits[rank % len(args.splits)]
    dataset = _make_dataset(args, split, source_root)
    indices = _preflight_indices(dataset, max(1, int(args.requested_vae_batch_size)))
    union_videos = dataset.load_video_windows(indices)
    if union_videos.dtype != torch.uint8 or union_videos.ndim != 5:
        raise RuntimeError(
            f"Real backend preflight expected uint8 [B,C,T,H,W], got "
            f"{union_videos.shape} {union_videos.dtype}."
        )

    # Compare every reconstructed union window against the exact ordinary
    # online decode path. This catches timestamp-union/reassembly drift at any
    # offset in the tested physical episode before a 100+ GiB cache write.
    checked_positions = list(range(len(indices)))
    resize = VideoResize()
    for position in checked_positions:
        segment, window_offset = dataset.resolve_window(indices[position])
        singleton_pixels = dataset._load_video(segment, window_offset)
        if not torch.equal(singleton_pixels, union_videos[position]):
            raise RuntimeError(
                "Real backend union decode is not bitwise equal to the ordinary online decode path: "
                f"split={split}, sample_index={indices[position]}."
            )
        singleton_resized = resize({"video": singleton_pixels}, None)
        union_resized = resize({"video": union_videos[position]}, None)
        if (
            not torch.equal(singleton_resized["video"], union_resized["video"])
            or not torch.equal(singleton_resized["image_size"], union_resized["image_size"])
        ):
            raise RuntimeError(
                "Real backend VideoResize result is not bitwise equal after union reconstruction: "
                f"split={split}, sample_index={indices[position]}."
            )

    transformed = [resize({"video": video}, None) for video in union_videos]
    resized_videos = torch.stack([item["video"] for item in transformed], dim=0)
    image_sizes = torch.stack([item["image_size"] for item in transformed], dim=0)
    if any(
        tuple(int(value) for value in image_size.tolist()) != args.expected_image_size
        for image_size in image_sizes
    ):
        raise RuntimeError("Real backend preflight VideoResize image_size drifted from the build contract.")
    tokenizer = _load_tokenizer(args)
    return tokenizer, resized_videos, {
        "rank": rank,
        "split": split,
        "sample_indices": indices,
        "ordinary_vs_union_checked_positions": checked_positions,
        "ordinary_vs_union_pixel_torch_equal": True,
        "ordinary_vs_union_resize_torch_equal": True,
    }


def _is_cuda_oom(error: BaseException) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or "cuda out of memory" in message


def _preflight_batch_equivalence(
    args: argparse.Namespace,
    *,
    tokenizer: Any,
    resized_videos: torch.Tensor,
    device: torch.device,
    batch_size: int,
    singleton_reference: torch.Tensor | None = None,
) -> tuple[bool, dict[str, Any], torch.Tensor]:
    """Compare active batched encoding with online singleton encoding."""

    videos_u8 = resized_videos[:batch_size]
    video = videos_u8.to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)
    try:
        with torch.inference_mode():
            if singleton_reference is None:
                singleton = torch.cat(
                    [tokenizer.encode(video[index : index + 1]).contiguous().float() for index in range(batch_size)],
                    dim=0,
                )
            else:
                if singleton_reference.shape[0] < batch_size:
                    raise RuntimeError("Wan preflight singleton reference is shorter than the requested batch.")
                singleton = singleton_reference[:batch_size]
            batched = tokenizer.encode(video).contiguous().float()
        spatial = int(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["spatial_compression_factor"])
        expected_pre_crop_shape = (
            batch_size,
            *args.expected_latent_shape[:2],
            max(args.expected_image_size[0] // spatial, 1),
            max(args.expected_image_size[1] // spatial, 1),
        )
        if tuple(batched.shape) != expected_pre_crop_shape:
            raise RuntimeError(
                "Wan preflight produced an unexpected pre-crop shape: "
                f"got={tuple(batched.shape)}, expected={expected_pre_crop_shape}."
            )
        if not torch.isfinite(batched).all() or not torch.isfinite(singleton).all():
            raise RuntimeError("Wan preflight produced non-finite latent values.")
        equal = torch.equal(batched, singleton)
        return (
            equal,
            {
                "batch_size": batch_size,
                "status": "passed" if equal else "batch_not_bitwise_equal",
                "batched_vs_singleton_torch_equal": equal,
            },
            singleton,
        )
    except BaseException as error:  # noqa: BLE001 - classify only an exact CUDA OOM as a safe batch fallback
        if batch_size > 1 and _is_cuda_oom(error):
            torch.cuda.empty_cache()
            empty_reference = (
                singleton_reference
                if singleton_reference is not None
                else torch.empty((0,), device=device, dtype=torch.float32)
            )
            return (
                False,
                {
                    "batch_size": batch_size,
                    "status": "cuda_oom",
                    "batched_vs_singleton_torch_equal": False,
                },
                empty_reference,
            )
        raise


def _distributed_real_backend_preflight(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    source_root: Path,
) -> tuple[Any, list[dict[str, Any]]]:
    """Run per-rank real decode/encode gates and choose one global batch size."""

    tokenizer: Any | None = None
    resized_videos: torch.Tensor | None = None
    base_receipt: dict[str, Any] | None = None
    local_error: str | None = None
    try:
        tokenizer, resized_videos, base_receipt = _prepare_real_backend_preflight(
            args,
            rank=rank,
            source_root=source_root,
        )
    except BaseException as error:  # noqa: BLE001 - coordinate every preflight failure
        local_error = f"{type(error).__name__}: {error}"
    _all_rank_errors(local_error, world_size=world_size)
    assert tokenizer is not None and resized_videos is not None and base_receipt is not None

    requested = int(args.requested_vae_batch_size)
    local_error = None
    requested_result: dict[str, Any] | None = None
    singleton_reference: torch.Tensor | None = None
    try:
        requested_equal, requested_result, singleton_reference = _preflight_batch_equivalence(
            args,
            tokenizer=tokenizer,
            resized_videos=resized_videos,
            device=device,
            batch_size=requested,
        )
    except BaseException as error:  # noqa: BLE001
        requested_equal = False
        local_error = f"{type(error).__name__}: {error}"
    _all_rank_errors(local_error, world_size=world_size)
    assert requested_result is not None

    requested_results: list[dict[str, Any] | None] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(requested_results, requested_result)
    else:
        requested_results[0] = requested_result
    use_requested = all(
        result is not None and result.get("batched_vs_singleton_torch_equal") is True
        for result in requested_results
    )
    intermediate_results: list[dict[str, Any]] = []
    if use_requested and requested > 2:
        assert singleton_reference is not None
        local_error = None
        try:
            for batch_size in range(2, requested):
                equal, result, _ = _preflight_batch_equivalence(
                    args,
                    tokenizer=tokenizer,
                    resized_videos=resized_videos,
                    device=device,
                    batch_size=batch_size,
                    singleton_reference=singleton_reference,
                )
                intermediate_results.append(result)
                if not equal:
                    break
        except BaseException as error:  # noqa: BLE001
            local_error = f"{type(error).__name__}: {error}"
        _all_rank_errors(local_error, world_size=world_size)
        all_intermediate_results: list[list[dict[str, Any]] | None] = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(all_intermediate_results, intermediate_results)
        else:
            all_intermediate_results[0] = intermediate_results
        use_requested = all(
            rank_results is not None
            and len(rank_results) == requested - 2
            and all(result.get("batched_vs_singleton_torch_equal") is True for result in rank_results)
            for rank_results in all_intermediate_results
        )
    effective = requested if use_requested else 1

    fallback_result: dict[str, Any] | None = None
    if effective == 1 and requested != 1:
        local_error = None
        try:
            fallback_reference = (
                singleton_reference
                if (
                    singleton_reference is not None
                    and singleton_reference.ndim > 0
                    and singleton_reference.shape[0] >= 1
                )
                else None
            )
            fallback_equal, fallback_result, _ = _preflight_batch_equivalence(
                args,
                tokenizer=tokenizer,
                resized_videos=resized_videos,
                device=device,
                batch_size=1,
                singleton_reference=fallback_reference,
            )
            if not fallback_equal:
                raise RuntimeError("Wan singleton preflight was not bitwise deterministic.")
        except BaseException as error:  # noqa: BLE001
            local_error = f"{type(error).__name__}: {error}"
        _all_rank_errors(local_error, world_size=world_size)
        assert fallback_result is not None
    elif not use_requested:
        raise RuntimeError("Wan batch-size-1 preflight failed; no exact fallback is available.")

    local_receipt = {
        **base_receipt,
        "requested_vae_batch_size": requested,
        "effective_vae_batch_size": effective,
        "requested_batch_gate": requested_result,
        "intermediate_batch_gates": intermediate_results,
        "fallback_batch_gate": fallback_result,
    }
    receipts: list[dict[str, Any] | None] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(receipts, local_receipt)
    else:
        receipts[0] = local_receipt
    args.vae_batch_size = effective
    return tokenizer, [receipt for receipt in receipts if receipt is not None]


def _build_rank_shards(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    source_root: Path,
    stage_root: Path,
    plan: dict[str, Any],
    tokenizer: Any,
) -> None:
    plan_sha256 = sha256_file(stage_root / "build_plan.json")
    total_shards = int(plan["total_shards"])
    shard_start = total_shards * rank // world_size
    shard_stop = total_shards * (rank + 1) // world_size
    assigned_shards = range(shard_start, shard_stop)
    shape = tuple(int(value) for value in args.expected_latent_shape)
    dtype_bytes = 2 if args.storage_dtype == "bfloat16" else 4
    record_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype_bytes
    stride = math.ceil(record_nbytes / args.record_alignment_bytes) * args.record_alignment_bytes

    completed: set[int] = set()
    for shard_id in assigned_shards:
        first = shard_id * args.records_per_shard
        count = min(args.records_per_shard, int(plan["unique_records"]) - first)
        receipt = _validate_shard_receipt(
            stage_root,
            plan_sha256=plan_sha256,
            cache_id=plan["cache_id"],
            shard_id=shard_id,
            first_ordinal=first,
            num_records=count,
            stride=stride,
            verify_checksum=True,
        )
        if receipt is not None:
            completed.add(shard_id)

    work_batches = _build_work_batches(
        stage_root,
        plan,
        assigned_shards=assigned_shards,
        completed_shards=completed,
        vae_batch_size=args.vae_batch_size,
    )
    if not work_batches:
        print(f"[rank {rank}] all assigned shards already complete", flush=True)
        return
    batch_dataset = _VideoWindowBatchDataset(args, source_root, work_batches)
    loader_kwargs: dict[str, Any] = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": args.loader_workers,
        "pin_memory": args.pin_memory,
    }
    if args.loader_workers > 0:
        loader_kwargs.update(
            {
                # The Wan tokenizer is initialized on CUDA before the loader is
                # first iterated. Never fork a process after CUDA
                # initialization; fresh spawned workers only decode/resize on
                # CPU and cannot inherit a poisoned CUDA runtime.
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
                "prefetch_factor": args.loader_prefetch_factor,
            }
        )
    loader = DataLoader(batch_dataset, **loader_kwargs)
    current_writer: _FixedShardWriter | None = None
    encoded_records = 0
    singleton_checks = 0
    assigned_records = sum(len(batch.ordinals) for batch in work_batches)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for batch in loader:
                ordinals = [int(value) for value in batch["ordinals"].tolist()]
                video_u8 = batch["video"]
                image_size = batch["image_size"]
                if video_u8.dtype != torch.uint8 or video_u8.ndim != 5:
                    raise RuntimeError(f"Builder expected uint8 [B,C,T,H,W], got {video_u8.shape} {video_u8.dtype}.")
                video = video_u8.to(device=device, dtype=torch.float32, non_blocking=True).div_(127.5).sub_(1.0)
                batched = tokenizer.encode(video).contiguous().float()
                if not torch.isfinite(batched).all():
                    raise RuntimeError(f"Wan encoder produced non-finite values for ordinals={ordinals!r}.")
                cropped: list[torch.Tensor] = []
                singleton_equal: list[bool] = []
                for batch_index, ordinal in enumerate(ordinals):
                    orig_h = int(image_size[batch_index, 2].item())
                    orig_w = int(image_size[batch_index, 3].item())
                    spatial = int(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["spatial_compression_factor"])
                    latent = batched[
                        batch_index,
                        :,
                        :,
                        : max(orig_h // spatial, 1),
                        : max(orig_w // spatial, 1),
                    ].contiguous()
                    if tuple(latent.shape) != shape:
                        raise RuntimeError(
                            f"Post-crop Wan latent shape mismatch at ordinal={ordinal}: "
                            f"got={tuple(latent.shape)}, expected={shape}."
                        )
                    should_check = singleton_checks < args.equivalence_samples_per_rank
                    is_equal = False
                    if should_check:
                        if video.shape[0] == 1:
                            singleton = batched
                        else:
                            singleton = tokenizer.encode(video[batch_index : batch_index + 1]).contiguous().float()
                        singleton = singleton[
                            0,
                            :,
                            :,
                            : max(orig_h // spatial, 1),
                            : max(orig_w // spatial, 1),
                        ].contiguous()
                        if not torch.equal(latent, singleton):
                            raise RuntimeError(
                                "Batched Wan encode is not bitwise equal to the active online singleton path "
                                f"at ordinal={ordinal}; reduce --vae-batch-size or investigate kernels."
                            )
                        is_equal = True
                        singleton_checks += 1
                    cropped.append(latent)
                    singleton_equal.append(is_equal)

                for ordinal, latent, is_equal in zip(ordinals, cropped, singleton_equal, strict=True):
                    shard_id = ordinal // args.records_per_shard
                    if current_writer is None or current_writer.shard_id != shard_id:
                        if current_writer is not None:
                            current_writer.finalize()
                        first = shard_id * args.records_per_shard
                        count = min(args.records_per_shard, int(plan["unique_records"]) - first)
                        current_writer = _FixedShardWriter(
                            stage_root,
                            cache_id=plan["cache_id"],
                            plan_sha256=plan_sha256,
                            shard_id=shard_id,
                            first_ordinal=first,
                            num_records=count,
                            shape=shape,
                            storage_dtype=args.storage_dtype,
                            record_nbytes=record_nbytes,
                            stride=stride,
                        )
                    current_writer.write(ordinal, latent, singleton_equivalent=is_equal)
                    encoded_records += 1
                if encoded_records % args.log_every_records < len(ordinals):
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = encoded_records / elapsed
                    eta = (assigned_records - encoded_records) / max(rate, 1e-9)
                    print(
                        f"[rank {rank}] encoded={encoded_records}/{assigned_records} "
                        f"rate={rate:.2f} records/s eta={eta / 3600:.2f}h",
                        flush=True,
                    )
        if current_writer is not None:
            current_writer.finalize()
    except BaseException:
        if current_writer is not None:
            current_writer.abort()
        raise


def _verify_mapping_full_coverage(
    args: argparse.Namespace,
    *,
    source_root: Path,
    stage_root: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    identity_bytes = (stage_root / plan["ordinal_identity_path"]).read_bytes()
    if len(identity_bytes) != int(plan["unique_records"]) * 32:
        raise RuntimeError("Ordinal identity inventory size changed before full-coverage validation.")
    seen: dict[bytes, int] = {}
    locator_by_anchor: dict[bytes, bytes] = {}
    next_ordinal = 0
    total_logical = 0
    for split in args.splits:
        dataset = _make_dataset(args, split, source_root)
        split_info = plan["splits"][split]
        mapping = np.load(stage_root / split_info["mapping_path"], allow_pickle=False)
        ordered_digest = hashlib.sha256()
        for sample_index in range(len(dataset)):
            identity = dataset.get_visual_window_identity(sample_index)
            payload = canonical_json_bytes(identity)
            digest = hashlib.sha256(payload).digest()
            ordinal = int(mapping[sample_index])
            inventory_digest = identity_bytes[ordinal * 32 : (ordinal + 1) * 32]
            if inventory_digest != digest:
                raise RuntimeError(
                    f"Mapping identity mismatch for split={split}, sample_index={sample_index}, ordinal={ordinal}."
                )
            previous_ordinal = seen.get(digest)
            if previous_ordinal is None:
                if ordinal != next_ordinal:
                    raise RuntimeError(
                        "Mapping violates canonical first-occurrence ordinal assignment: "
                        f"split={split}, sample={sample_index}, got={ordinal}, expected={next_ordinal}."
                    )
                seen[digest] = ordinal
                next_ordinal += 1
            elif ordinal != previous_ordinal:
                raise RuntimeError(
                    f"One canonical identity maps to multiple ordinals: {previous_ordinal} and {ordinal}."
                )
            anchor = _physical_anchor(identity)
            locator = _physical_locator(identity)
            previous_locator = locator_by_anchor.setdefault(anchor, locator)
            if previous_locator != locator:
                raise RuntimeError("Overlapping segments disagree on effective path/timestamp during prepublish.")
            ordered_digest.update(len(payload).to_bytes(8, "little"))
            ordered_digest.update(payload)
        if ordered_digest.hexdigest() != split_info["ordered_visual_identity_sha256"]:
            raise RuntimeError(f"Ordered visual identity digest changed for split={split!r}.")
        total_logical += len(dataset)
    if next_ordinal != int(plan["unique_records"]) or total_logical != int(plan["logical_samples"]):
        raise RuntimeError(
            f"Mapping full-coverage counts mismatch: logical={total_logical}, unique={next_ordinal}, plan={plan!r}."
        )
    return {
        "status": "passed",
        "full_coverage": True,
        "algorithm": "canonical-first-occurrence-v1",
        "identity_digest_algorithm": IDENTITY_DIGEST_ALGORITHM,
        "logical_samples": total_logical,
        "unique_records": next_ordinal,
        "ordinal_identity_path": plan["ordinal_identity_path"],
        "ordinal_identity_record_bytes": 32,
        "ordinal_identity_size_bytes": plan["ordinal_identity_size_bytes"],
        "ordinal_identity_sha256": plan["ordinal_identity_sha256"],
    }


def _finalize_and_publish(
    args: argparse.Namespace,
    *,
    source_root: Path,
    stage_root: Path,
    output_root: Path,
    plan: dict[str, Any],
) -> None:
    shape = tuple(int(value) for value in args.expected_latent_shape)
    dtype_bytes = 2 if args.storage_dtype == "bfloat16" else 4
    record_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype_bytes
    stride = math.ceil(record_nbytes / args.record_alignment_bytes) * args.record_alignment_bytes
    plan_sha256 = sha256_file(stage_root / "build_plan.json")
    shard_entries: list[dict[str, Any]] = []
    storage_checks = 0
    singleton_checks = 0
    for shard_id in range(int(plan["total_shards"])):
        first = shard_id * args.records_per_shard
        count = min(args.records_per_shard, int(plan["unique_records"]) - first)
        receipt = _validate_shard_receipt(
            stage_root,
            plan_sha256=plan_sha256,
            cache_id=plan["cache_id"],
            shard_id=shard_id,
            first_ordinal=first,
            num_records=count,
            stride=stride,
            # The shared validator performs one complete checksum pass after
            # manifest/_SUCCESS materialization. Avoid reading 100+ GiB twice.
            verify_checksum=False,
        )
        if receipt is None:
            raise RuntimeError(f"Missing or invalid completed shard receipt for shard_id={shard_id}.")
        shard_entries.append(
            {
                key: receipt[key]
                for key in ("shard_id", "path", "first_ordinal", "num_records", "size_bytes", "sha256")
            }
        )
        storage_checks += int(receipt["storage_roundtrip_samples"])
        singleton_checks += int(receipt["singleton_equivalence_samples"])
    if storage_checks != int(plan["unique_records"]) or singleton_checks < 1:
        raise RuntimeError(
            f"Equivalence coverage is insufficient: storage={storage_checks}, singleton={singleton_checks}."
        )
    transient_files = [
        path.relative_to(stage_root).as_posix()
        for path in stage_root.rglob("*")
        if path.is_file() and (".partial" in path.name or ".tmp-" in path.name)
    ]
    if transient_files:
        raise RuntimeError(f"Refusing to publish a cache with transient files: {transient_files[:20]!r}.")

    mapping_gate = _verify_mapping_full_coverage(
        args,
        source_root=source_root,
        stage_root=stage_root,
        plan=plan,
    )
    request = plan["request"]
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_id": plan["cache_id"],
        "dataset": {
            "dataset_id": request["dataset"]["dataset_id"],
            "manifest_sha256": request["dataset"]["manifest_sha256"],
        },
        "preprocessing": request["preprocessing"],
        "vae": {
            "weights_sha256": request["vae"]["weights_sha256"],
            "weights_size_bytes": request["vae"]["weights_size_bytes"],
            "behavior_relevant_config": request["vae"]["behavior_relevant_config"],
        },
        "latent": {
            "shape": list(shape),
            "storage_dtype": args.storage_dtype,
            "logical_dtype": "float32",
            "stage": CACHE_STAGE,
            "record_nbytes": record_nbytes,
            "record_stride_bytes": stride,
            "byte_order": "little",
            "video_num_frames": args.chunk_length + 1,
            "image_size": list(args.expected_image_size),
        },
        "splits": plan["splits"],
        "mapping_gate": mapping_gate,
        "shards": shard_entries,
        "equivalence": {
            "status": "passed",
            "torch_equal": True,
            "num_samples": singleton_checks,
            "online_reference": "active_singleton_Wan2pt2VAEInterface_encode_then_post_padding_crop",
            "candidate": f"batch_encode_size_{args.vae_batch_size}_then_{args.storage_dtype}_roundtrip",
            "storage_roundtrip_samples": storage_checks,
        },
        "provenance": {
            "request_sha256": plan["request_sha256"],
            "build_plan_path": "build_plan.json",
            "build_plan_sha256": plan_sha256,
            "records_index_path": plan["records_index_path"],
            "records_index_sha256": plan["records_index_sha256"],
            "source_inventory": request["source"]["inventory"],
        },
    }
    _write_json_atomic(stage_root / "manifest.json", manifest)
    manifest_sha = sha256_file(stage_root / "manifest.json")
    _write_json_atomic(
        stage_root / "_SUCCESS.json",
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": "complete",
            "cache_id": plan["cache_id"],
            "manifest_path": "manifest.json",
            "manifest_sha256": manifest_sha,
        },
    )
    validated = validate_published_cache_files(stage_root, verify_shard_checksums=True)
    if validated["cache_id"] != plan["cache_id"]:
        raise RuntimeError("Final shared validator returned the wrong cache_id.")
    if output_root.exists():
        raise FileExistsError(f"Refusing to replace an existing cache output during atomic publish: {output_root}.")
    os.rename(stage_root, output_root)
    _fsync_dir(output_root.parent)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wan-vae", type=Path, required=True)
    parser.add_argument("--nvme-source-root", type=Path)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--camera-mode", choices=sorted(_CAMERA_FIELDS_BY_MODE), default="concat_view")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--chunk-length", type=int, default=16)
    parser.add_argument("--dataset-image-size", type=int, default=256)
    parser.add_argument("--resolution", choices=("256", "480", "704", "720", "768"))
    parser.add_argument("--video-backend", choices=("lerobot", "opencv"), default="lerobot")
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--loader-prefetch-factor", type=int, default=2)
    parser.add_argument("--vae-batch-size", type=int, default=8)
    parser.add_argument("--equivalence-samples-per-rank", type=int, default=8)
    parser.add_argument("--records-per-shard", type=int, default=4096)
    parser.add_argument("--record-alignment-bytes", type=int, default=4096)
    parser.add_argument("--storage-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--expected-latent-shape", default="48,5,10,20")
    parser.add_argument("--expected-image-size", default="192,320,160,320")
    parser.add_argument("--log-every-records", type=int, default=1024)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.resolve(strict=True)
    args.source_root = args.source_root.resolve(strict=True)
    args.wan_vae = args.wan_vae.resolve(strict=True)
    args.output_root = args.output_root.resolve(strict=False)
    if args.nvme_source_root is not None:
        args.nvme_source_root = args.nvme_source_root.resolve(strict=False)
    args.splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    if not args.splits or len(set(args.splits)) != len(args.splits):
        parser.error("--splits must contain one or more unique comma-separated split names.")
    args.expected_latent_shape = _parse_int_tuple(
        args.expected_latent_shape,
        length=4,
        label="--expected-latent-shape",
    )
    args.expected_image_size = _parse_int_tuple(
        args.expected_image_size,
        length=4,
        label="--expected-image-size",
    )
    for name in (
        "loader_workers",
        "loader_prefetch_factor",
        "vae_batch_size",
        "equivalence_samples_per_rank",
        "records_per_shard",
        "record_alignment_bytes",
        "log_every_records",
    ):
        if int(getattr(args, name)) < (0 if name == "loader_workers" else 1):
            parser.error(f"--{name.replace('_', '-')} has an invalid value.")
    if args.output_root == args.dataset_root or args.output_root == args.source_root:
        parser.error("--output-root must not equal the dataset or raw source root.")
    return args


def main() -> None:
    args = _parse_args()
    args.requested_vae_batch_size = int(args.vae_batch_size)
    args.real_backend_preflight_receipts = []
    rank, world_size, local_rank, device = _distributed_context()
    active_source_root = args.source_root
    local_error: str | None = None
    local_inventory: dict[str, Any] | None = None
    try:
        if args.nvme_source_root is not None:
            if local_rank == 0:
                local_inventory = _stage_source_to_nvme(args.source_root, args.nvme_source_root)
            active_source_root = args.nvme_source_root
        elif rank == 0:
            local_inventory = _tree_inventory(args.source_root)
    except BaseException as error:  # noqa: BLE001 - propagate every rank's staging failure together
        local_error = f"{type(error).__name__}: {error}"
    _all_rank_errors(local_error, world_size=world_size)
    _barrier(world_size)

    tokenizer, preflight_receipts = _distributed_real_backend_preflight(
        args,
        rank=rank,
        world_size=world_size,
        device=device,
        source_root=active_source_root,
    )
    args.real_backend_preflight_receipts = preflight_receipts

    if rank == 0:
        try:
            if args.nvme_source_root is not None:
                local_inventory = _tree_inventory(active_source_root)
            assert local_inventory is not None
            request = _build_request(args, source_root=active_source_root, source_inventory=local_inventory)
            request_sha256 = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
            cache_id = f"robocasa365-wan-{request_sha256[:24]}"
            stage_root = args.output_root.parent / f".{args.output_root.name}.incomplete-{cache_id}"
            if args.output_root.exists():
                published = validate_published_cache_files(args.output_root, verify_shard_checksums=True)
                if published["cache_id"] != cache_id:
                    raise RuntimeError(
                        f"Existing output cache_id={published['cache_id']!r} does not match requested {cache_id!r}."
                    )
                coordination = {
                    "error": None,
                    "already_published": True,
                    "cache_id": cache_id,
                    "output_root": str(args.output_root),
                }
            else:
                if stage_root.exists():
                    plan_path = stage_root / "build_plan.json"
                    if plan_path.is_file():
                        plan = _load_and_validate_plan(
                            stage_root,
                            request=request,
                            request_sha256=request_sha256,
                            cache_id=cache_id,
                        )
                    else:
                        request_path = stage_root / "build_request.json"
                        if not request_path.is_file() or json.loads(request_path.read_text()) != request:
                            raise RuntimeError(
                                f"Existing incomplete stage has no matching build request and cannot resume: "
                                f"{stage_root}."
                            )
                        plan = _enumerate_and_write_plan(
                            args,
                            source_root=active_source_root,
                            stage_root=stage_root,
                            request=request,
                            request_sha256=request_sha256,
                            cache_id=cache_id,
                        )
                else:
                    stage_root.mkdir(parents=False, exist_ok=False)
                    _write_json_atomic(stage_root / "build_request.json", request)
                    plan = _enumerate_and_write_plan(
                        args,
                        source_root=active_source_root,
                        stage_root=stage_root,
                        request=request,
                        request_sha256=request_sha256,
                        cache_id=cache_id,
                    )
                coordination = {
                    "error": None,
                    "already_published": False,
                    "stage_root": str(stage_root),
                    "request_sha256": request_sha256,
                    "cache_id": cache_id,
                    "plan_sha256": sha256_file(stage_root / "build_plan.json"),
                }
        except BaseException as error:  # noqa: BLE001 - broadcast rank-zero planning failures
            coordination = {
                "error": f"{type(error).__name__}: {error}",
                "already_published": False,
            }
    else:
        coordination = None
        plan = None
    if world_size > 1:
        payload: list[dict[str, Any] | None] = [coordination]
        dist.broadcast_object_list(payload, src=0)
        coordination = payload[0]
    assert coordination is not None
    if coordination.get("error"):
        raise RuntimeError(f"Rank-zero latent-cache planning failed: {coordination['error']}")
    if coordination["already_published"]:
        if rank == 0:
            print(f"Cache already published and fully verified: {coordination['output_root']}", flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return
    stage_root = Path(coordination["stage_root"])
    local_error = None
    try:
        plan = json.loads((stage_root / "build_plan.json").read_text())
        if sha256_file(stage_root / "build_plan.json") != coordination["plan_sha256"]:
            raise RuntimeError("Build plan changed after distributed coordination.")
    except BaseException as error:  # noqa: BLE001
        local_error = f"{type(error).__name__}: {error}"
    _all_rank_errors(local_error, world_size=world_size)
    assert plan is not None

    local_error = None
    try:
        _build_rank_shards(
            args,
            rank=rank,
            world_size=world_size,
            device=device,
            source_root=active_source_root,
            stage_root=stage_root,
            plan=plan,
            tokenizer=tokenizer,
        )
    except BaseException as error:  # noqa: BLE001 - coordinate decoder/encoder/shard failures
        local_error = f"{type(error).__name__}: {error}"
    _all_rank_errors(local_error, world_size=world_size)

    if rank == 0:
        try:
            _finalize_and_publish(
                args,
                source_root=active_source_root,
                stage_root=stage_root,
                output_root=args.output_root,
                plan=plan,
            )
            finalization = {"error": None, "output_root": str(args.output_root)}
        except BaseException as error:  # noqa: BLE001 - broadcast rank-zero publication failures
            finalization = {"error": f"{type(error).__name__}: {error}"}
    else:
        finalization = None
    if world_size > 1:
        payload = [finalization]
        dist.broadcast_object_list(payload, src=0)
        finalization = payload[0]
    assert finalization is not None
    if finalization.get("error"):
        raise RuntimeError(f"Rank-zero latent-cache publication failed: {finalization['error']}")
    if rank == 0:
        print(f"Published strict Wan latent cache: {finalization['output_root']}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
