# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU gate for one real production-sized RoboCasa365 training batch.

The gate loads the unmodified train-loader recipe from the structured SFT
TOML, initializes a one-rank Gloo process group, instantiates the production
``PackingDataLoader``, and emits exactly one packed batch.  It intentionally
does not construct the model or move tensors to CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch

_PRODUCTION_SAMPLE_COUNT = 128
# The dataset decodes a 256x512 concatenated view.  With ``resolution=None``,
# VideoResize selects the "256" tier and maps that 2:1 input to a 192x320
# production canvas (160x320 resized content plus bottom reflection padding).
_PRODUCTION_VIDEO_SHAPE = (3, 17, 192, 320)
_PRODUCTION_VIDEO_LATENT_SHAPE = (48, 5, 10, 20)
_PRODUCTION_ACTION_SHAPE = (16, 64)
_PRODUCTION_RAW_ACTION_SHAPE = (16, 10)
_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


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
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional machine-readable evidence path; written atomically after all checks pass.",
    )
    return parser.parse_args()


def _require_path(path: Path, *, kind: str) -> None:
    if kind == "file":
        valid = path.is_file() and path.stat().st_size > 0
    elif kind == "directory":
        valid = path.is_dir()
    else:
        raise ValueError(f"Unsupported path kind: {kind!r}")
    if not valid:
        raise FileNotFoundError(f"Expected a valid {kind}: {path}")


def _configure_runtime(args: argparse.Namespace) -> None:
    """Set every environment interpolation used by the production TOML."""
    _require_path(args.dataset_root, kind="directory")
    _require_path(args.source_root, kind="directory")
    _require_path(args.checkpoint, kind="directory")
    _require_path(args.wan_vae, kind="file")
    _require_path(args.qwen_tokenizer, kind="directory")
    _require_path(args.toml, kind="file")

    # cosmos_framework.utils.flags is fixed when first imported below.
    os.environ["COSMOS_DEVICE"] = "cpu"
    os.environ["COSMOS_INTERNAL"] = "0"
    os.environ["COSMOS_TRAINING"] = "1"
    os.environ["ROBOCASA365_ROOT"] = str(args.dataset_root)
    os.environ["ROBOCASA365_SOURCE_ROOT"] = str(args.source_root)
    os.environ["BASE_CHECKPOINT_PATH"] = str(args.checkpoint)
    os.environ["WAN_VAE_PATH"] = str(args.wan_vae)
    os.environ["QWEN3_VL_TOKENIZER_PATH"] = str(args.qwen_tokenizer)
    # This CPU-only gate can run in an SSH workspace without a usable
    # ancillary-FD channel. The production trainer keeps PyTorch's default.
    torch.multiprocessing.set_sharing_strategy("file_system")


@contextmanager
def _single_process_gloo() -> Iterator[None]:
    """Provide the one-rank distributed context required by the real loader."""
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable; cannot instantiate the production loader.")

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        backend = str(torch.distributed.get_backend()).lower()
        if world_size != 1 or not backend.endswith("gloo"):
            raise RuntimeError(
                "An incompatible process group is already initialized: "
                f"backend={backend!r}, world_size={world_size}; expected one-rank Gloo."
            )
        yield
        return

    with tempfile.TemporaryDirectory(prefix="robocasa365-batch-gate-") as temp_dir:
        rendezvous = Path(temp_dir) / "gloo-rendezvous"
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=rendezvous.as_uri(),
            rank=0,
            world_size=1,
            timeout=timedelta(minutes=5),
        )
        try:
            yield
        finally:
            torch.distributed.destroy_process_group()


def _tensor_bytes(value: Any) -> int:
    """Return the recursively summed logical bytes of all tensors in ``value``."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _require_sequence(batch: Mapping[str, Any], key: str, expected_count: int) -> Sequence[Any]:
    if key not in batch:
        raise AssertionError(f"Packed batch is missing required key {key!r}.")
    values = batch[key]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise AssertionError(f"Packed batch key {key!r} must be a sequence, got {type(values).__name__}.")
    if len(values) != expected_count:
        raise AssertionError(f"Packed batch key {key!r} has {len(values)} entries; expected exactly {expected_count}.")
    return values


def _require_single_tensor_items(
    batch: Mapping[str, Any],
    key: str,
    expected_count: int,
) -> list[torch.Tensor]:
    """Validate the production ``list[list[Tensor]]`` packing contract."""
    values = _require_sequence(batch, key, expected_count)
    tensors: list[torch.Tensor] = []
    for index, item in enumerate(values):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes, bytearray))
            or len(item) != 1
            or not isinstance(item[0], torch.Tensor)
        ):
            raise AssertionError(
                f"Packed batch {key}[{index}] must contain exactly one tensor, got {type(item).__name__}."
            )
        tensors.append(item[0])
    return tensors


def _require_finite(tensor: torch.Tensor, *, label: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise AssertionError(f"Packed batch contains non-finite values in {label}.")


def _validate_prompts(batch: Mapping[str, Any], expected_count: int) -> int:
    captions = _require_sequence(batch, "ai_caption", expected_count)
    cfg_dropped_count = 0
    for index, caption in enumerate(captions):
        if not isinstance(caption, str):
            raise AssertionError(f"ai_caption[{index}] must be a string, got {type(caption).__name__}.")
        if caption == "":
            # The production train recipe deliberately applies 10% CFG dropout.
            cfg_dropped_count += 1
            continue
        if "Current action:" in caption or '"idle_frame"' in caption:
            raise AssertionError(f"Target-derived information leaked into ai_caption[{index}].")
        try:
            prompt = json.loads(caption)
        except json.JSONDecodeError as error:
            raise AssertionError(f"ai_caption[{index}] is not valid JSON.") from error
        actions = prompt.get("actions") if isinstance(prompt, dict) else None
        if not isinstance(actions, list) or not actions:
            raise AssertionError(f"ai_caption[{index}] has no structured action description.")
    return cfg_dropped_count


def _validate_sample_identities(batch: Mapping[str, Any], expected_count: int) -> int:
    urls = _require_sequence(batch, "__url__", expected_count)
    keys = _require_sequence(batch, "__key__", expected_count)
    identities: list[tuple[str, str]] = []
    for index, (url, key) in enumerate(zip(urls, keys)):
        if not isinstance(url, str) or not url.startswith("robocasa365://"):
            raise AssertionError(f"Invalid stable __url__ at sample {index}: {url!r}.")
        if not isinstance(key, str) or len(key) != 9 or not key.isdecimal():
            raise AssertionError(f"Invalid stable __key__ at sample {index}: {key!r}.")
        identities.append((url, key))
    unique_count = len(set(identities))
    if unique_count != expected_count:
        raise AssertionError(
            "Packed batch contains duplicate stable (__url__, __key__) identities: "
            f"unique={unique_count}, expected={expected_count}."
        )
    return unique_count


def _validate_sequence_plans(batch: Mapping[str, Any], expected_count: int) -> None:
    modes = _require_sequence(batch, "mode", expected_count)
    plans = _require_sequence(batch, "sequence_plan", expected_count)
    for index, (mode, plan) in enumerate(zip(modes, plans)):
        if mode != "policy":
            raise AssertionError(f"mode[{index}]={mode!r}; production policy training requires 'policy'.")
        flags = (
            bool(getattr(plan, "has_text", False)),
            bool(getattr(plan, "has_vision", False)),
            bool(getattr(plan, "has_action", False)),
        )
        if flags != (True, True, True):
            raise AssertionError(f"sequence_plan[{index}] has invalid text/vision/action flags: {flags}.")
        if list(getattr(plan, "condition_frame_indexes_vision", [])) != [0]:
            raise AssertionError(f"sequence_plan[{index}] must condition only the first vision frame.")
        if list(getattr(plan, "condition_frame_indexes_action", [])):
            raise AssertionError(f"sequence_plan[{index}] must supervise every action step.")


def _sample_for_token_estimate(batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    media_keys = [key for key in ("images", "video", "video_latent") if key in batch]
    if len(media_keys) != 1:
        raise AssertionError(
            "Packed batch must contain exactly one vision input for token estimation; "
            f"got {media_keys}."
        )
    sample = {
        key: batch[key][index]
        for key in ("text_token_ids", media_keys[0], "action")
    }
    for key in (
        "enable_per_camera_vae_encoding",
        "sample_n_views",
        "num_video_frames_per_view",
    ):
        if key in batch:
            values = batch[key]
            sample[key] = values[index] if isinstance(values, Sequence) else values
    return sample


def _validate_packed_batch(
    loader: Any,
    batch: Mapping[str, Any],
    *,
    expected_sample_count: int,
    expected_video_shape: tuple[int, ...],
    expected_video_latent_shape: tuple[int, ...] | None = None,
    expected_action_shape: tuple[int, ...],
    expected_raw_action_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Validate one packed action-policy batch and return JSON-safe metrics."""
    if not isinstance(batch, Mapping):
        raise AssertionError(f"Expected a mapping batch, got {type(batch).__name__}.")
    if expected_sample_count <= 0:
        raise ValueError(f"expected_sample_count must be positive, got {expected_sample_count}.")

    vision_keys = [key for key in ("images", "video", "video_latent") if key in batch]
    if len(vision_keys) != 1:
        raise AssertionError(
            "Packed batch must contain exactly one of 'images', 'video', or 'video_latent'; "
            f"got {vision_keys}."
        )
    vision_key = vision_keys[0]
    if vision_key == "images":
        raise AssertionError("RoboCasa365 policy training does not support an image-only packed batch.")
    vision_tensors = _require_single_tensor_items(batch, vision_key, expected_sample_count)
    actions = _require_single_tensor_items(batch, "action", expected_sample_count)
    raw_actions = _require_single_tensor_items(batch, "action_raw", expected_sample_count)
    text_tokens = _require_single_tensor_items(batch, "text_token_ids", expected_sample_count)

    if vision_key == "video":
        for index, video in enumerate(vision_tensors):
            if tuple(video.shape) != expected_video_shape or video.dtype != torch.uint8:
                raise AssertionError(
                    f"video[{index}] has shape={tuple(video.shape)}, dtype={video.dtype}; "
                    f"expected shape={expected_video_shape}, dtype=torch.uint8."
                )
            _require_finite(video, label=f"video[{index}]")
    else:
        if expected_video_latent_shape is None:
            raise AssertionError("Cached packed batches require an expected video_latent shape.")
        for index, latent in enumerate(vision_tensors):
            if tuple(latent.shape) != expected_video_latent_shape or latent.dtype != torch.bfloat16:
                raise AssertionError(
                    f"video_latent[{index}] has shape={tuple(latent.shape)}, dtype={latent.dtype}; "
                    f"expected shape={expected_video_latent_shape}, dtype=torch.bfloat16."
                )
            _require_finite(latent, label=f"video_latent[{index}]")

    for index, (action, raw_action) in enumerate(zip(actions, raw_actions)):
        if tuple(action.shape) != expected_action_shape or not action.is_floating_point():
            raise AssertionError(
                f"action[{index}] has shape={tuple(action.shape)}, dtype={action.dtype}; "
                f"expected floating shape={expected_action_shape}."
            )
        if tuple(raw_action.shape) != expected_raw_action_shape or not raw_action.is_floating_point():
            raise AssertionError(
                f"action_raw[{index}] has shape={tuple(raw_action.shape)}, dtype={raw_action.dtype}; "
                f"expected floating shape={expected_raw_action_shape}."
            )
        _require_finite(action, label=f"action[{index}]")
        _require_finite(raw_action, label=f"action_raw[{index}]")

    text_token_counts: list[int] = []
    for index, token_ids in enumerate(text_tokens):
        if token_ids.ndim != 1 or token_ids.numel() == 0 or token_ids.dtype not in _INTEGER_DTYPES:
            raise AssertionError(
                f"text_token_ids[{index}] must be a non-empty 1D integer tensor, "
                f"got shape={tuple(token_ids.shape)}, dtype={token_ids.dtype}."
            )
        _require_finite(token_ids, label=f"text_token_ids[{index}]")
        text_token_counts.append(int(token_ids.numel()))

    cfg_dropped_count = _validate_prompts(batch, expected_sample_count)
    unique_identity_count = _validate_sample_identities(batch, expected_sample_count)
    _validate_sequence_plans(batch, expected_sample_count)

    compute_tokens = getattr(loader, "_compute_num_tokens_per_sample", None)
    if not callable(compute_tokens):
        raise AssertionError("Loader does not expose PackingDataLoader._compute_num_tokens_per_sample.")
    per_sample_tokens: list[int] = []
    for index in range(expected_sample_count):
        token_count = int(compute_tokens(_sample_for_token_estimate(batch, index)))
        if token_count <= 0:
            raise AssertionError(f"Estimated token count for sample {index} is not positive: {token_count}.")
        per_sample_tokens.append(token_count)
    estimated_tokens = sum(per_sample_tokens)

    if "_num_samples" in batch and int(batch["_num_samples"]) != expected_sample_count:
        raise AssertionError(
            f"Loader-reported _num_samples={batch['_num_samples']!r}; expected {expected_sample_count}."
        )
    if "_num_tokens" in batch and int(batch["_num_tokens"]) != estimated_tokens:
        raise AssertionError(f"Loader-reported _num_tokens={batch['_num_tokens']!r}; recomputed {estimated_tokens}.")

    return {
        "sample_count": expected_sample_count,
        "estimated_tokens": estimated_tokens,
        "estimated_tokens_min_per_sample": min(per_sample_tokens),
        "estimated_tokens_max_per_sample": max(per_sample_tokens),
        "tensor_bytes": _tensor_bytes(batch),
        "text_tokens": sum(text_token_counts),
        "text_tokens_min_per_sample": min(text_token_counts),
        "text_tokens_max_per_sample": max(text_token_counts),
        "cfg_dropped_caption_count": cfg_dropped_count,
        "unique_url_key_count": unique_identity_count,
        "vision_input_key": vision_key,
        "video_shape": list(expected_video_shape) if vision_key == "video" else None,
        "video_latent_shape": list(expected_video_latent_shape) if vision_key == "video_latent" else None,
        "video_latent_dtype": str(vision_tensors[0].dtype) if vision_key == "video_latent" else None,
        "action_shape": list(expected_action_shape),
        "raw_action_shape": list(expected_raw_action_shape),
    }


def _validate_loader_config(config: Any) -> dict[str, Any]:
    train = config.dataloader_train
    max_samples = int(train.max_samples_per_batch)
    max_sequence = train.max_sequence_length
    if max_samples != _PRODUCTION_SAMPLE_COUNT:
        raise AssertionError(
            f"Production max_samples_per_batch drifted: expected {_PRODUCTION_SAMPLE_COUNT}, got {max_samples}."
        )
    if max_sequence is not None:
        raise AssertionError(f"Production max_sequence_length must be None, got {max_sequence!r}.")

    inner = train.dataloader
    dataset = inner.datasets.robocasa365.dataset
    if int(inner.batch_size) != 1:
        raise AssertionError(f"Inner train DataLoader batch_size must be 1, got {inner.batch_size!r}.")
    if str(dataset.split) != "train" or str(dataset.mode) != "policy" or str(dataset.caption_mode) != "policy":
        raise AssertionError("Resolved train dataset is not the causal RoboCasa365 policy split.")
    if bool(dataset.append_idle_frames):
        raise AssertionError("Target-derived idle-frame metadata must be disabled for policy training.")
    if not bool(dataset.iterable_shuffle) or not bool(inner.in_order):
        raise AssertionError("Production train loader must preserve the configured stable shuffled stream.")
    latent_cache_mode = str(dataset.latent_cache_mode)
    latent_cache_root = dataset.latent_cache_root
    action_relevance_enabled = bool(config.model.config.get("action_relevance_enabled", False))
    if action_relevance_enabled:
        if latent_cache_mode != "required" or not latent_cache_root:
            raise AssertionError("S1 production validation requires the audited latent cache.")
        tokenizer_target = str(config.model.config.tokenizer.get("_target_", ""))
        if not tokenizer_target.endswith(".PrecomputedWan2pt2LatentTokenizer"):
            raise AssertionError("S1 production validation requires the metadata-only Wan tokenizer.")

    return {
        "max_samples_per_batch": max_samples,
        "max_sequence_length": None,
        "inner_batch_size": int(inner.batch_size),
        "num_workers": int(inner.num_workers),
        "persistent_workers": bool(inner.persistent_workers),
        "split": str(dataset.split),
        "dataset_mode": str(dataset.mode),
        "caption_mode": str(dataset.caption_mode),
        "iterable_shuffle": bool(dataset.iterable_shuffle),
        "in_order": bool(inner.in_order),
        "latent_cache_mode": latent_cache_mode,
        "latent_cache_root": latent_cache_root,
        "metadata_only_video_tokenizer": action_relevance_enabled,
    }


def main() -> None:
    args = _parse_args()
    _configure_runtime(args)

    # Import only after the CPU/training flags and all TOML interpolations are fixed.
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader
    from cosmos_framework.utils.lazy_config import instantiate

    total_started = time.perf_counter()
    config_started = time.perf_counter()
    config = load_experiment_from_toml(args.toml)
    config_seconds = time.perf_counter() - config_started
    resolved_loader = _validate_loader_config(config)

    with _single_process_gloo():
        instantiate_started = time.perf_counter()
        loader = instantiate(config.dataloader_train)
        instantiate_seconds = time.perf_counter() - instantiate_started
        if not isinstance(loader, PackingDataLoader):
            raise AssertionError(f"Resolved train loader is {type(loader).__name__}, expected PackingDataLoader.")
        if loader.max_samples_per_batch != _PRODUCTION_SAMPLE_COUNT or loader.max_sequence_length is not None:
            raise AssertionError(
                "Instantiated PackingDataLoader does not match the 128-sample, no-token-cap production contract."
            )

        batch_started = time.perf_counter()
        batch = next(iter(loader))
        batch_seconds = time.perf_counter() - batch_started
        batch_metrics = _validate_packed_batch(
            loader,
            batch,
            expected_sample_count=_PRODUCTION_SAMPLE_COUNT,
            expected_video_shape=_PRODUCTION_VIDEO_SHAPE,
            expected_video_latent_shape=_PRODUCTION_VIDEO_LATENT_SHAPE,
            expected_action_shape=_PRODUCTION_ACTION_SHAPE,
            expected_raw_action_shape=_PRODUCTION_RAW_ACTION_SHAPE,
        )

    elapsed_seconds = time.perf_counter() - total_started
    result = {
        "status": "passed",
        "scope": "single_process_cpu_real_production_train_batch",
        "does_not_cover": [
            "cosmos3_nano_model_load",
            "post_model_packing_actual_token_lengths",
            "multi_rank_per_microbatch_contract",
            "gpu_memory",
            "forward_backward",
            "optimizer",
            "fsdp",
            "checkpoint_save_resume",
        ],
        "toml": str(args.toml),
        "gloo_world_size": 1,
        "loader": resolved_loader,
        "sample_count": batch_metrics["sample_count"],
        "estimated_tokens": batch_metrics["estimated_tokens"],
        "tensor_bytes": batch_metrics["tensor_bytes"],
        "elapsed_seconds": round(elapsed_seconds, 6),
        "timing": {
            "config_seconds": round(config_seconds, 6),
            "instantiate_and_prewarm_seconds": round(instantiate_seconds, 6),
            "batch_seconds": round(batch_seconds, 6),
        },
        "batch": batch_metrics,
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        temporary.write_text(serialized)
        temporary.replace(args.output_json)
    print(serialized, end="")


if __name__ == "__main__":
    main()
