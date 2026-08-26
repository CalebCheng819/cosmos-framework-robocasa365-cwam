# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build a resumable, strict LIBERO Wan VAE latent cache directly to OSS."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset
from cosmos_framework.data.generator.action.datasets.robocasa365_latent_cache import WAN_CACHE_VAE_BEHAVIOR_CONTRACT
from cosmos_framework.data.generator.action.transforms import VideoResize

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LATENT_SHAPE = (48, 5, 10, 20)
IMAGE_SIZE = (192, 320, 160, 320)
STORAGE_DTYPE = torch.bfloat16


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)


def _run_rclone(args: argparse.Namespace, *command: str, capture: bool = False) -> str:
    result = subprocess.run(
        [str(args.rclone), *command],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def _remote_json(args: argparse.Namespace, remote: str) -> dict[str, Any] | None:
    result = subprocess.run(
        [str(args.rclone), "cat", remote],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _remote_size(args: argparse.Namespace, remote: str) -> int | None:
    result = subprocess.run(
        [str(args.rclone), "lsjson", "--stat", remote],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return int(value["Size"]) if isinstance(value, dict) and type(value.get("Size")) is int else None


def _remote(suite_remote: str, relative: str) -> str:
    return f"{suite_remote.rstrip('/')}/{relative.lstrip('/')}"


def _make_dataset(args: argparse.Namespace, suite: str) -> LIBEROLeRobotDataset:
    return LIBEROLeRobotDataset(
        root=str(args.dataset_root / f"{suite}{args.suite_suffix}"),
        fps=20.0,
        chunk_length=args.chunk_length,
        mode="policy",
        camera_mode="concat_view",
        image_size=args.dataset_image_size,
        action_space="frame_wise_relative",
        rotation_space="6d",
        pose_coordinate_frame="native",
        action_normalization="quantile_rot",
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        latent_cache_mode="disabled",
    )


def _load_tokenizer(args: argparse.Namespace):
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

    contract = WAN_CACHE_VAE_BEHAVIOR_CONTRACT
    return Wan2pt2VAEInterface(
        bucket_name="",
        object_store_credential_path_pretrained="",
        vae_path=str(args.wan_vae),
        chunk_duration=int(contract["chunk_duration"]),
        keep_decoder_cache=bool(contract["keep_decoder_cache"]),
        use_streaming_encode=bool(contract["use_streaming_encode"]),
        encode_chunk_frames=dict(contract["encode_chunk_frames"]),
        encode_exact_durations=list(contract["encode_exact_durations"]),
        spatial_compression_factor=int(contract["spatial_compression_factor"]),
        temporal_compression_factor=int(contract["temporal_compression_factor"]),
        causal=bool(contract["causal"]),
    )


def _resize(videos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    transform = VideoResize()
    transformed = [transform({"video": video}, None) for video in videos]
    resized = torch.stack([item["video"] for item in transformed], dim=0)
    image_sizes = torch.stack([item["image_size"] for item in transformed], dim=0)
    observed = {tuple(int(value) for value in item.tolist()) for item in image_sizes}
    if observed != {IMAGE_SIZE}:
        raise RuntimeError(f"VideoResize contract drifted: observed={observed}, expected={IMAGE_SIZE}.")
    return resized, image_sizes


def _encode(
    tokenizer: Any,
    videos_u8: torch.Tensor,
    image_sizes: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    video = videos_u8.to(device=device, dtype=torch.float32, non_blocking=True).div_(127.5).sub_(1.0)
    with torch.inference_mode():
        encoded = tokenizer.encode(video).contiguous().float()
    if not torch.isfinite(encoded).all():
        raise RuntimeError("Wan VAE returned non-finite latent values.")
    spatial = int(WAN_CACHE_VAE_BEHAVIOR_CONTRACT["spatial_compression_factor"])
    output: list[torch.Tensor] = []
    for index in range(encoded.shape[0]):
        orig_h = int(image_sizes[index, 2].item())
        orig_w = int(image_sizes[index, 3].item())
        latent = encoded[index, :, :, : max(orig_h // spatial, 1), : max(orig_w // spatial, 1)].contiguous()
        if tuple(latent.shape) != LATENT_SHAPE:
            raise RuntimeError(
                f"Post-crop latent shape mismatch: got={tuple(latent.shape)}, expected={LATENT_SHAPE}."
            )
        output.append(latent)
    return output


def _preflight(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    tokenizer: Any,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    local: list[dict[str, Any]] = []
    count = max(1, args.vae_batch_size)
    for suite in SUITES[rank::world_size]:
        dataset = _make_dataset(args, suite)
        indices: list[int] | None = None
        for start, length in dataset.get_shuffle_blocks():
            if length >= count:
                indices = list(range(start, start + count))
                break
        if indices is None:
            raise RuntimeError(f"No {suite} episode contains {count} cache preflight windows.")
        union = dataset.load_video_windows(indices)
        for position, index in enumerate(indices):
            exact = dataset.load_video_exact(index)
            if not torch.equal(union[position], exact):
                raise RuntimeError(f"Union decode differs from online decode for suite={suite}, index={index}.")
        resized, image_sizes = _resize(union)
        singleton = torch.stack(
            [
                _encode(tokenizer, resized[index : index + 1], image_sizes[index : index + 1], device)[0]
                for index in range(count)
            ],
            dim=0,
        )
        batched = torch.stack(_encode(tokenizer, resized, image_sizes, device), dim=0)
        local.append(
            {
                "rank": rank,
                "suite": suite,
                "sample_indices": indices,
                "union_vs_online_pixel_equal": True,
                "requested_vae_batch_size": args.vae_batch_size,
                "batched_vs_singleton_equal": bool(torch.equal(batched, singleton)),
            }
        )
    rank_receipts: list[list[dict[str, Any]] | None] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(rank_receipts, local)
    else:
        rank_receipts[0] = local
    complete = [receipt for receipts in rank_receipts if receipts is not None for receipt in receipts]
    observed_suites = {str(receipt["suite"]) for receipt in complete}
    if observed_suites != set(SUITES):
        raise RuntimeError(f"Cache preflight suite coverage drifted: observed={sorted(observed_suites)}.")
    effective = args.vae_batch_size if all(receipt["batched_vs_singleton_equal"] for receipt in complete) else 1
    if effective == 1 and args.vae_batch_size != 1:
        print("Wan batch encode was not bitwise equal to singleton; using batch_size=1.", flush=True)
    return effective, complete


def _receipt_valid(
    args: argparse.Namespace,
    *,
    suite_remote: str,
    suite: str,
    shard_id: int,
    first_index: int,
    num_records: int,
    size_bytes: int,
) -> bool:
    relative = f"receipts/shard-{shard_id:06d}.json"
    receipt = _remote_json(args, _remote(suite_remote, relative))
    if receipt is None:
        return False
    expected = {
        "schema_version": 1,
        "status": "complete",
        "suite": suite,
        "shard_id": shard_id,
        "first_index": first_index,
        "num_records": num_records,
        "size_bytes": size_bytes,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        return False
    path = receipt.get("path")
    return isinstance(path, str) and _remote_size(args, _remote(suite_remote, path)) == size_bytes


def _build_shard(
    args: argparse.Namespace,
    *,
    rank: int,
    suite: str,
    dataset: LIBEROLeRobotDataset,
    suite_remote: str,
    shard_id: int,
    tokenizer: Any,
    device: torch.device,
    vae_batch_size: int,
    record_nbytes: int,
    stride: int,
) -> dict[str, Any]:
    first = shard_id * args.records_per_shard
    count = min(args.records_per_shard, len(dataset) - first)
    size_bytes = count * stride
    if _receipt_valid(
        args,
        suite_remote=suite_remote,
        suite=suite,
        shard_id=shard_id,
        first_index=first,
        num_records=count,
        size_bytes=size_bytes,
    ):
        return {"status": "resumed", "records": count}

    rank_root = args.staging_root / f"rank-{rank:02d}" / suite
    rank_root.mkdir(parents=True, exist_ok=True)
    partial = rank_root / f"shard-{shard_id:06d}.bin.partial"
    final = partial.with_suffix("")
    partial.unlink(missing_ok=True)
    final.unlink(missing_ok=True)
    singleton_checks = 0
    with partial.open("xb") as handle:
        for batch_first in range(first, first + count, vae_batch_size):
            indices = list(range(batch_first, min(batch_first + vae_batch_size, first + count)))
            videos = dataset.load_video_windows(indices)
            resized, image_sizes = _resize(videos)
            latents = _encode(tokenizer, resized, image_sizes, device)
            if singleton_checks < args.equivalence_samples_per_rank and len(indices) > 1:
                singleton = _encode(tokenizer, resized[:1], image_sizes[:1], device)[0]
                if not torch.equal(latents[0], singleton):
                    raise RuntimeError(
                        f"Batched Wan output drifted after preflight for suite={suite}, index={indices[0]}."
                    )
                singleton_checks += 1
            for latent in latents:
                stored = latent.to(device="cpu", dtype=STORAGE_DTYPE).contiguous()
                if not torch.equal(latent.cpu(), stored.float()):
                    raise RuntimeError("Wan latent is not exactly representable by the BF16 cache contract.")
                payload = stored.view(torch.uint8).numpy().tobytes()
                if len(payload) != record_nbytes:
                    raise RuntimeError("Wan latent record byte count drifted.")
                handle.write(payload)
                handle.write(b"\0" * (stride - record_nbytes))
        handle.flush()
        os.fsync(handle.fileno())
    if partial.stat().st_size != size_bytes:
        raise RuntimeError(f"Local shard size mismatch: {partial}")
    os.replace(partial, final)
    shard_relative = f"shards/shard-{shard_id:06d}.bin"
    _run_rclone(args, "copyto", str(final), _remote(suite_remote, shard_relative), "--s3-no-check-bucket")
    remote_size = _remote_size(args, _remote(suite_remote, shard_relative))
    if remote_size != size_bytes:
        raise RuntimeError(
            f"OSS shard size mismatch suite={suite}, shard={shard_id}: got={remote_size}, expected={size_bytes}."
        )
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "suite": suite,
        "shard_id": shard_id,
        "path": shard_relative,
        "first_index": first,
        "num_records": count,
        "size_bytes": size_bytes,
        "singleton_equivalence_samples": singleton_checks,
    }
    receipt_path = rank_root / f"receipt-{shard_id:06d}.json"
    _write_json(receipt_path, receipt)
    _run_rclone(
        args,
        "copyto",
        str(receipt_path),
        _remote(suite_remote, f"receipts/shard-{shard_id:06d}.json"),
        "--s3-no-check-bucket",
    )
    final.unlink()
    receipt_path.unlink()
    return {"status": "built", "records": count}


def _finalize_suite(
    args: argparse.Namespace,
    *,
    suite: str,
    dataset: LIBEROLeRobotDataset,
    suite_remote: str,
    record_nbytes: int,
    stride: int,
    preflight: list[dict[str, Any]],
    effective_vae_batch_size: int,
) -> None:
    total_shards = math.ceil(len(dataset) / args.records_per_shard)
    shards: list[dict[str, Any]] = []
    singleton_checks = 0
    for shard_id in range(total_shards):
        first = shard_id * args.records_per_shard
        count = min(args.records_per_shard, len(dataset) - first)
        size_bytes = count * stride
        receipt = _remote_json(args, _remote(suite_remote, f"receipts/shard-{shard_id:06d}.json"))
        if receipt is None or not _receipt_valid(
            args,
            suite_remote=suite_remote,
            suite=suite,
            shard_id=shard_id,
            first_index=first,
            num_records=count,
            size_bytes=size_bytes,
        ):
            raise RuntimeError(f"Cannot finalize {suite}: shard {shard_id} is missing or invalid.")
        singleton_checks += int(receipt.get("singleton_equivalence_samples", 0))
        shards.append(
            {
                key: receipt[key]
                for key in ("shard_id", "path", "first_index", "num_records", "size_bytes")
            }
        )
    if sum(int(shard["num_records"]) for shard in shards) != len(dataset):
        raise RuntimeError(f"Cannot finalize {suite}: shard coverage is incomplete.")
    manifest = {
        "schema_version": 1,
        "provenance": {
            "experiment_id": args.experiment_id,
            "source_commit": args.source_commit,
        },
        "dataset": {
            "name": dataset._root.name,
            "suite": suite,
            "split": "train",
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "chunk_length": args.chunk_length,
            "camera_mode": "concat_view",
            "image_size": args.dataset_image_size,
            "fps": dataset._fps,
            "num_samples": len(dataset),
        },
        "latent": {
            "shape": list(LATENT_SHAPE),
            "storage_dtype": "bfloat16",
            "record_nbytes": record_nbytes,
            "record_stride_bytes": stride,
            "video_num_frames": args.chunk_length + 1,
            "image_size": list(IMAGE_SIZE),
            "stage": "post_video_resize_post_padding_crop",
        },
        "vae": {
            "family": "Wan2.2_VAE",
            "behavior_relevant_config": WAN_CACHE_VAE_BEHAVIOR_CONTRACT,
            "effective_batch_size": effective_vae_batch_size,
        },
        "equivalence": {
            "status": "passed",
            "union_decode_vs_online_pixel_equal": True,
            "batched_vs_singleton_equal": all(
                bool(receipt["batched_vs_singleton_equal"]) for receipt in preflight
            ),
            "preflight": preflight,
            "runtime_singleton_checks": singleton_checks,
        },
        "shards": shards,
    }
    metadata_root = args.staging_root / "finalize" / suite
    manifest_path = metadata_root / "manifest.json"
    success_path = metadata_root / "_SUCCESS.json"
    _write_json(manifest_path, manifest)
    _write_json(
        success_path,
        {
            "schema_version": 1,
            "status": "complete",
            "manifest_path": "manifest.json",
            "num_samples": len(dataset),
            "num_shards": total_shards,
        },
    )
    _run_rclone(args, "copyto", str(manifest_path), _remote(suite_remote, "manifest.json"), "--s3-no-check-bucket")
    _run_rclone(args, "copyto", str(success_path), _remote(suite_remote, "_SUCCESS.json"), "--s3-no-check-bucket")
    print(
        f"CACHE_SUITE_COMPLETE suite={suite} samples={len(dataset)} shards={total_shards} "
        f"remote={suite_remote}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--suite-suffix", default="_no_noops_lerobot")
    parser.add_argument("--wan-vae", type=Path, required=True)
    parser.add_argument("--output-remote", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--rclone", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--records-per-shard", type=int, default=1024)
    parser.add_argument("--record-alignment-bytes", type=int, default=4096)
    parser.add_argument("--vae-batch-size", type=int, default=8)
    parser.add_argument("--equivalence-samples-per-rank", type=int, default=4)
    parser.add_argument("--chunk-length", type=int, default=16)
    parser.add_argument("--dataset-image-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.output_remote.startswith("eailab-hdd2:eailab/chengjuntao/"):
        parser.error("--output-remote must stay under eailab-hdd2:eailab/chengjuntao/.")
    if args.records_per_shard <= 0 or args.vae_batch_size <= 0:
        parser.error("records-per-shard and vae-batch-size must be positive.")
    if not args.experiment_id.strip() or not args.source_commit.strip():
        parser.error("experiment-id and source-commit must be non-empty.")
    return args


def main() -> None:
    args = _parse_args()
    dist.init_process_group("nccl", timeout=timedelta(hours=24))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.staging_root.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(args)
    effective_batch, preflight = _preflight(
        args,
        rank=rank,
        world_size=world_size,
        tokenizer=tokenizer,
        device=device,
    )
    if rank == 0:
        print(
            f"CACHE_PREFLIGHT_COMPLETE world_size={world_size} requested_batch={args.vae_batch_size} "
            f"effective_batch={effective_batch}",
            flush=True,
        )
    record_nbytes = int(np.prod(LATENT_SHAPE, dtype=np.int64)) * torch.tensor([], dtype=STORAGE_DTYPE).element_size()
    stride = math.ceil(record_nbytes / args.record_alignment_bytes) * args.record_alignment_bytes
    started = time.monotonic()
    rank_records = 0
    for suite in SUITES:
        dataset = _make_dataset(args, suite)
        suite_remote = _remote(args.output_remote, suite)
        total_shards = math.ceil(len(dataset) / args.records_per_shard)
        for shard_id in range(rank, total_shards, world_size):
            result = _build_shard(
                args,
                rank=rank,
                suite=suite,
                dataset=dataset,
                suite_remote=suite_remote,
                shard_id=shard_id,
                tokenizer=tokenizer,
                device=device,
                vae_batch_size=effective_batch,
                record_nbytes=record_nbytes,
                stride=stride,
            )
            rank_records += int(result["records"])
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"[rank {rank}] suite={suite} shard={shard_id + 1}/{total_shards} "
                f"status={result['status']} rank_records={rank_records} "
                f"avg_rate={rank_records / elapsed:.2f} records/s",
                flush=True,
            )
        dist.barrier()
        if rank == 0:
            _finalize_suite(
                args,
                suite=suite,
                dataset=dataset,
                suite_remote=suite_remote,
                record_nbytes=record_nbytes,
                stride=stride,
                preflight=preflight,
                effective_vae_batch_size=effective_batch,
            )
        dist.barrier()
    if rank == 0:
        print(
            f"CACHE_BUILD_COMPLETE total_seconds={time.monotonic() - started:.1f} remote={args.output_remote}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
