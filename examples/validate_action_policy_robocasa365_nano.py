# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offline preflight for the RoboCasa365 Cosmos3-Nano policy recipe.

This check deliberately stops before model construction.  It validates the
structured configuration at both supported single-node world sizes, checks
the staged model/tokenizer artifacts, and exercises the production LeRobot
video decoder on deterministic samples from each dataset split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.action.datasets.robocasa365_segment_dataset import (
    RoboCasa365SegmentDataset,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from examples.h_cluster.cwam_formal_training_evidence import protocol_for_world_size


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wan-vae", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer", type=Path, required=True)
    parser.add_argument(
        "--toml",
        type=Path,
        default=Path("examples/toml/sft_config/action_policy_robocasa365_nano.toml"),
    )
    parser.add_argument("--samples-per-split", type=int, default=3)
    return parser.parse_args()


def _require_file(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Expected a non-empty file: {path}")
    return size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_dataset_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.dataset_root / "dataset_manifest.json"
    success_path = args.dataset_root / "_SUCCESS.json"
    _require_file(manifest_path)
    _require_file(success_path)
    manifest = json.loads(manifest_path.read_text())
    success = json.loads(success_path.read_text())
    if success.get("status") != "complete":
        raise RuntimeError(f"Dataset conversion is not complete: {success.get('status')!r}")
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != success.get("dataset_manifest_sha256"):
        raise RuntimeError(
            "dataset_manifest.json checksum mismatch: "
            f"expected={success.get('dataset_manifest_sha256')!r}, actual={manifest_sha!r}"
        )

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("dataset_manifest.json has no artifact file inventory.")
    total_bytes = 0
    for relative, contract in sorted(files.items()):
        path = args.dataset_root / relative
        size = _require_file(path)
        expected_size = int(contract["bytes"])
        if size != expected_size:
            raise RuntimeError(
                f"Dataset artifact size mismatch for {relative}: expected={expected_size}, actual={size}"
            )
        actual_sha = _sha256(path)
        if actual_sha != contract["sha256"]:
            raise RuntimeError(
                f"Dataset artifact checksum mismatch for {relative}: expected={contract['sha256']}, actual={actual_sha}"
            )
        total_bytes += size
    return {
        "dataset_id": manifest.get("dataset_id"),
        "manifest_sha256": manifest_sha,
        "verified_files": len(files),
        "verified_bytes": total_bytes,
    }


def _validate_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    metadata_size = _require_file(args.checkpoint / "model/.metadata")
    checkpoint_json_size = _require_file(args.checkpoint / "checkpoint.json")
    config_size = _require_file(args.checkpoint / "model/config.json")
    distcp = sorted((args.checkpoint / "model").glob("*.distcp"))
    if not distcp:
        raise ValueError(f"No DCP shards found under {args.checkpoint / 'model'}")
    shard_sizes = [_require_file(path) for path in distcp]

    qwen_files = {
        name: _require_file(args.qwen_tokenizer / name)
        for name in (
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }
    vae_size = _require_file(args.wan_vae)
    return {
        "dataset": _validate_dataset_artifacts(args),
        "checkpoint": {
            "path": str(args.checkpoint),
            "metadata_bytes": metadata_size,
            "checkpoint_json_bytes": checkpoint_json_size,
            "config_bytes": config_size,
            "distcp_shards": len(distcp),
            "distcp_bytes": sum(shard_sizes),
        },
        "qwen_tokenizer": {"path": str(args.qwen_tokenizer), "files": qwen_files},
        "wan_vae": {"path": str(args.wan_vae), "bytes": vae_size},
    }


def _validate_config(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["ROBOCASA365_ROOT"] = str(args.dataset_root)
    os.environ["ROBOCASA365_SOURCE_ROOT"] = str(args.source_root)
    os.environ["BASE_CHECKPOINT_PATH"] = str(args.checkpoint)
    os.environ["WAN_VAE_PATH"] = str(args.wan_vae)
    os.environ["QWEN3_VL_TOKENIZER_PATH"] = str(args.qwen_tokenizer)

    worlds: dict[str, Any] = {}
    for world_size in (6, 8):
        protocol = protocol_for_world_size(world_size)
        config = load_experiment_from_toml(
            args.toml,
            extra_overrides=[
                f"model.config.parallelism.data_parallel_shard_degree={world_size}",
                f"optimizer.lr={protocol.learning_rate}",
                f"scheduler.warm_up_steps=[{protocol.warm_up_steps}]",
                f"scheduler.cycle_lengths=[{protocol.cycle_length}]",
                f"trainer.max_iter={protocol.max_iter}",
                f"trainer.max_val_iter={protocol.max_val_iter}",
                f"trainer.validation_iter={protocol.validation_iter}",
                f"checkpoint.save_iter={protocol.save_iter}",
                f"job.name=preflight_{world_size}gpu",
            ],
        )
        shard_degree = int(config.model.config.parallelism.data_parallel_shard_degree)
        if shard_degree != world_size:
            raise AssertionError(f"DP shard override failed: requested={world_size}, resolved={shard_degree}")
        train_dataset = config.dataloader_train.dataloader.datasets.robocasa365.dataset
        val_dataset = config.dataloader_val.dataloader.datasets.robocasa365.dataset
        if str(train_dataset.caption_mode) != "policy" or str(val_dataset.caption_mode) != "policy":
            raise AssertionError("Policy training/validation must use causal task-only captions.")
        if str(train_dataset.split) != "train" or str(val_dataset.split) != "val":
            raise AssertionError("Unexpected train/validation split routing.")
        if bool(train_dataset.append_idle_frames) or bool(val_dataset.append_idle_frames):
            raise AssertionError("Causal policy prompts must not include target-derived idle-frame metadata.")
        if bool(config.dataloader_train.restart_on_iter):
            raise AssertionError("The infinite training stream must not reset on each iter(loader) call.")
        if not bool(config.dataloader_train.dataloader.in_order):
            raise AssertionError("RoboCasa365 training requires stable cross-worker ordering.")
        if not bool(config.dataloader_val.restart_on_iter):
            raise AssertionError("Validation must reset to its fixed-seed stream at every checkpoint.")
        if not bool(config.trainer.run_validation) or not bool(config.trainer.run_validation_on_start):
            raise AssertionError("Validation and the step-0 baseline must both be enabled.")
        if int(config.trainer.max_iter) != protocol.max_iter:
            raise AssertionError(f"Unexpected {protocol.experiment_family} max_iter.")
        if int(config.trainer.max_val_iter) != protocol.max_val_iter:
            raise AssertionError(f"Unexpected {protocol.experiment_family} max_val_iter.")
        if int(config.trainer.validation_iter) != protocol.validation_iter:
            raise AssertionError(f"Unexpected {protocol.experiment_family} validation interval.")
        if int(config.checkpoint.save_iter) != protocol.save_iter:
            raise AssertionError(f"Unexpected {protocol.experiment_family} checkpoint interval.")
        if int(config.scheduler.warm_up_steps[0]) != protocol.warm_up_steps:
            raise AssertionError(f"Unexpected {protocol.experiment_family} warmup.")
        if int(config.scheduler.cycle_lengths[0]) != protocol.cycle_length:
            raise AssertionError(f"Unexpected {protocol.experiment_family} scheduler cycle.")
        if abs(float(config.optimizer.lr) - protocol.learning_rate) > 1.0e-12:
            raise AssertionError(f"Unexpected {protocol.experiment_family} learning rate.")
        if str(config.job.wandb_mode) != "offline":
            raise AssertionError("H-cluster policy recipe must use offline W&B.")
        if str(config.checkpoint.load_path) != str(args.checkpoint):
            raise AssertionError("The configured DCP path does not match the staged checkpoint.")
        if str(config.model.config.vlm_config.tokenizer.pretrained_model_name) != str(args.qwen_tokenizer):
            raise AssertionError("The configured Qwen processor path is not local.")
        worlds[str(world_size)] = {
            "experiment_family": protocol.experiment_family,
            "dp_shard_degree": shard_degree,
            "global_batch_per_optimizer_step": protocol.global_batch,
            "learning_rate": float(config.optimizer.lr),
            "grad_accum_iter": int(config.trainer.grad_accum_iter),
            "max_iter": int(config.trainer.max_iter),
            "max_val_iter": int(config.trainer.max_val_iter),
            "validation_iter": int(config.trainer.validation_iter),
            "checkpoint_save_iter": int(config.checkpoint.save_iter),
            "scheduler_warm_up_steps": list(config.scheduler.warm_up_steps),
            "scheduler_cycle_lengths": list(config.scheduler.cycle_lengths),
            "caption_mode": str(train_dataset.caption_mode),
            "include_idle_frames": bool(train_dataset.append_idle_frames),
            "training_in_order": bool(config.dataloader_train.dataloader.in_order),
            "validation_restart_on_iter": bool(config.dataloader_val.restart_on_iter),
            "wandb_mode": str(config.job.wandb_mode),
        }
    return worlds


def _sample_indices(length: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError(f"--samples-per-split must be positive, got {count}.")
    count = min(count, length)
    if count == 1:
        return [length // 2]
    return sorted({round(i * (length - 1) / (count - 1)) for i in range(count)})


def _validate_decode(args: argparse.Namespace) -> dict[str, Any]:
    config = load_experiment_from_toml(
        args.toml,
        extra_overrides=[
            "model.config.parallelism.data_parallel_shard_degree=8",
            "job.name=preflight_decode",
        ],
    )
    pipeline = ActionTransformPipeline(
        tokenizer_config=config.model.config.vlm_config.tokenizer,
        cfg_dropout_rate=0.0,
        max_action_dim=int(config.model.config.max_action_dim),
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=False,
        format_prompt_as_json=True,
    )

    decoded: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        dataset = RoboCasa365SegmentDataset(
            root=str(args.dataset_root),
            source_root=str(args.source_root),
            split=split,
            caption_mode="policy",
            video_backend="lerobot",
            resample_on_decode_error=False,
        )
        samples: list[dict[str, Any]] = []
        for index in _sample_indices(len(dataset), args.samples_per_split):
            started = time.perf_counter()
            sample = dataset[index]
            decode_seconds = time.perf_counter() - started
            video = sample["video"]
            action = sample["action"]
            if tuple(video.shape) != (3, 17, 256, 512) or video.dtype != torch.uint8:
                raise AssertionError(
                    f"Unexpected {split} video contract: shape={tuple(video.shape)}, dtype={video.dtype}"
                )
            if tuple(action.shape) != (16, 10) or not torch.isfinite(action).all():
                raise AssertionError(f"Unexpected {split} action contract.")
            caption = str(sample["ai_caption"])
            if not caption or "Current action:" in caption:
                raise AssertionError(f"Non-causal or empty caption in {split}: {caption!r}")
            sample_url = str(sample.get("__url__", ""))
            sample_key = str(sample.get("__key__", ""))
            if not sample_url.startswith("robocasa365://") or sample_key != f"{index:09d}":
                raise AssertionError(
                    f"Invalid stable sample identity for {split} index {index}: url={sample_url!r}, key={sample_key!r}"
                )

            transformed = pipeline(sample, None, action_normalizer=dataset.get_action_normalizer())
            prompt = json.loads(transformed["ai_caption"])
            prompt_text = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
            if "idle_frame" in prompt or any("idle_frame" in action_prompt for action_prompt in prompt["actions"]):
                raise AssertionError(f"Target-derived idle_frame leaked into the {split} policy prompt.")
            if "Current action:" in prompt_text:
                raise AssertionError(f"Future action caption leaked into the {split} policy prompt.")
            token_ids = transformed.get("text_token_ids")
            if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1 or token_ids.numel() == 0:
                raise AssertionError(f"Qwen tokenizer produced invalid token IDs for {split} index {index}.")
            if tuple(transformed["action_raw"].shape) != (16, 10):
                raise AssertionError(f"Unexpected transformed raw action shape for {split} index {index}.")
            if tuple(transformed["action"].shape) != (16, int(config.model.config.max_action_dim)):
                raise AssertionError(f"Unexpected padded action shape for {split} index {index}.")
            if not torch.isfinite(transformed["action"]).all():
                raise AssertionError(f"Non-finite normalized action for {split} index {index}.")
            samples.append(
                {
                    "sample_index": index,
                    "decode_seconds": round(decode_seconds, 6),
                    "video_shape": list(video.shape),
                    "action_shape": list(action.shape),
                    "text_tokens": int(token_ids.numel()),
                    "caption": caption,
                    "sample_url": sample_url,
                    "sample_key": sample_key,
                }
            )
        decoded[split] = {
            "dataset_windows": len(dataset),
            "samples": samples,
        }
    return decoded


def main() -> None:
    args = _parse_args()
    result = {
        "artifacts": _validate_artifacts(args),
        "config": _validate_config(args),
        "production_decode": _validate_decode(args),
        "status": "passed",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
