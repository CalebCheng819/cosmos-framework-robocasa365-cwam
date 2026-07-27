# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Single-GPU runtime gate for the RoboCasa365 Cosmos3-Nano policy recipe.

This gate deliberately does not construct the transformer or an optimizer.
It verifies that the pinned CUDA environment can decode a real converted
sample and run the production Wan2.2 VAE encode/decode path on that sample.
The multi-GPU FSDP load/forward/backward/save/resume gate remains separate.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

# ``cosmos_framework.utils.flags`` is fixed at import time. This executable is
# a training-runtime CUDA gate, so inherited values must not silently change
# its semantics.
os.environ["COSMOS_DEVICE"] = "cuda"
os.environ["COSMOS_INTERNAL"] = "0"
os.environ["COSMOS_TRAINING"] = "1"

import torch

from cosmos_framework.data.generator.action.datasets.robocasa365_segment_dataset import (
    RoboCasa365SegmentDataset,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
    Wan2pt2VAEInterface,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wan-vae", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-index", type=int)
    return parser.parse_args()


def _cuda_snapshot() -> dict[str, Any]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this gate must run in a GPU allocation.")
    if not args.wan_vae.is_file() or args.wan_vae.stat().st_size <= 0:
        raise FileNotFoundError(args.wan_vae)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    dataset = RoboCasa365SegmentDataset(
        root=str(args.dataset_root),
        source_root=str(args.source_root),
        split=args.split,
        caption_mode="policy",
        video_backend="lerobot",
        resample_on_decode_error=False,
    )
    sample_index = len(dataset) // 2 if args.sample_index is None else args.sample_index
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample index {sample_index} is outside [0, {len(dataset)}).")

    decode_started = time.perf_counter()
    sample = dataset[sample_index]
    decode_seconds = time.perf_counter() - decode_started
    source_video_u8 = sample["video"]
    if tuple(source_video_u8.shape) != (3, 17, 256, 512) or source_video_u8.dtype != torch.uint8:
        raise AssertionError(
            f"Unexpected decoded video: shape={tuple(source_video_u8.shape)}, dtype={source_video_u8.dtype}"
        )
    # Match the recipe's auto-resolution spatial path before invoking the VAE.
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=64,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
        format_prompt_as_json=False,
    )
    transformed = pipeline(sample, None, action_normalizer=dataset.get_action_normalizer())
    video_u8 = transformed["video"]
    if tuple(video_u8.shape) != (3, 17, 192, 320) or video_u8.dtype != torch.uint8:
        raise AssertionError(f"Unexpected production video: shape={tuple(video_u8.shape)}, dtype={video_u8.dtype}")

    video = video_u8.unsqueeze(0).to(device=device, dtype=torch.float32)
    video = video.div_(127.5).sub_(1.0)
    if not torch.isfinite(video).all() or video.min() < -1.0 or video.max() > 1.0:
        raise AssertionError("Normalized VAE input must be finite and in [-1, 1].")

    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    tokenizer = Wan2pt2VAEInterface(
        bucket_name="",
        object_store_credential_path_pretrained="",
        vae_path=str(args.wan_vae),
        encode_chunk_frames={"256": 68, "480": 24, "720": 8, "768": 8},
        encode_exact_durations=[17],
        spatial_compression_factor=16,
        temporal_compression_factor=4,
        causal=True,
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    encode_started = time.perf_counter()
    with torch.inference_mode():
        latent = tokenizer.encode(video)
    torch.cuda.synchronize(device)
    encode_seconds = time.perf_counter() - encode_started
    expected_latent_shape = (1, 48, 5, 12, 20)
    if tuple(latent.shape) != expected_latent_shape or not torch.isfinite(latent).all():
        raise AssertionError(
            f"Unexpected Wan latent: shape={tuple(latent.shape)}, expected={expected_latent_shape}, "
            f"dtype={latent.dtype}"
        )

    decode_vae_started = time.perf_counter()
    with torch.inference_mode():
        reconstruction = tokenizer.decode(latent)
    torch.cuda.synchronize(device)
    decode_vae_seconds = time.perf_counter() - decode_vae_started
    if tuple(reconstruction.shape) != tuple(video.shape) or not torch.isfinite(reconstruction).all():
        raise AssertionError(
            f"Unexpected Wan reconstruction: shape={tuple(reconstruction.shape)}, expected={tuple(video.shape)}"
        )

    mae = (reconstruction.float() - video.float()).abs().mean()
    if not torch.isfinite(mae):
        raise AssertionError("Wan reconstruction MAE is non-finite.")

    result = {
        "status": "passed",
        "scope": "single_gpu_real_decode_wan22_encode_decode",
        "does_not_cover": [
            "cosmos3_nano_dcp_load",
            "transformer_forward_backward",
            "fsdp",
            "optimizer",
            "checkpoint_save_resume",
        ],
        "cuda": {
            "device_name": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            **_cuda_snapshot(),
        },
        "dataset": {
            "split": args.split,
            "windows": len(dataset),
            "sample_index": sample_index,
            "caption": str(sample["ai_caption"]),
            "decode_seconds": round(decode_seconds, 6),
            "source_video_shape": list(source_video_u8.shape),
            "production_video_shape": list(video_u8.shape),
        },
        "wan_vae": {
            "path": str(args.wan_vae),
            "bytes": args.wan_vae.stat().st_size,
            "load_seconds": round(load_seconds, 6),
            "encode_seconds": round(encode_seconds, 6),
            "decode_seconds": round(decode_vae_seconds, 6),
            "latent_shape": list(latent.shape),
            "latent_dtype": str(latent.dtype),
            "reconstruction_mae": float(mae),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
