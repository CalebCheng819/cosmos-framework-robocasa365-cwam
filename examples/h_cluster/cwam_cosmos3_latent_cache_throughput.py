#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fail-closed utilities for the CWAM Cosmos3 latent-cache throughput probe.

The production shell wrappers keep orchestration in Bash.  This module owns the
parts that are easier to make deterministic and unit-testable in Python:

* validate the complete RoboCasa365 Wan-latent cache contract;
* copy that cache to node-local NVMe on distributed rank 0 and synchronize all
  eight ranks before training starts;
* capture scheduler/node/eight-H200 evidence;
* summarize per-cell iteration timing and GPU telemetry; and
* remove probe-owned checkpoint payloads after their metrics are durable.

It intentionally has no top-level torch import so all non-distributed commands
and tests remain pure CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import shutil
import socket
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
READY_FILENAME = "_CWAM_CACHE_READY.json"
LOWER_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
OFFICIAL_WAN22_VAE_SHA256 = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
METADATA_ONLY_VIDEO_TOKENIZER_TARGET = (
    "cosmos_framework.model.generator.tokenizers.precomputed_wan2pt2."
    "PrecomputedWan2pt2LatentTokenizer"
)
CUDA_OOM_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("torch.cuda.OutOfMemoryError", re.compile(r"torch\.cuda\.OutOfMemoryError")),
    ("CUDA out of memory", re.compile(r"CUDA out of memory", re.IGNORECASE)),
    ("CUDA error: out of memory", re.compile(r"CUDA error:\s*out of memory", re.IGNORECASE)),
)
CELL_ID_PATTERN = re.compile(
    r"^workers-(?P<workers>[1-9][0-9]*)__storage-(?P<storage>gpfs|nvme)"
    r"__ac-(?P<activation_checkpointing>selective|none)$"
)


def make_throughput_probe_callback(
    *,
    output_path: str,
    warmup_steps: int = 10,
    measure_steps: int = 50,
    expected_world_size: int = 8,
    expected_grad_accum_iter: int = 2,
) -> Any:
    """Construct the runtime-only optimizer-step wall-clock callback.

    The returned callback starts each step immediately before the first
    dataloader fetch and stops it after the optimizer boundary and a CUDA
    synchronize.  Therefore it includes all ``grad_accum_iter`` loader waits,
    host-to-device work, forward/backward, communication, and optimizer work.
    Rank 0 writes one consolidated NDJSON row per optimizer step; the row keeps
    every rank's own timing plus the trainer's ``dataloader_train`` timer.

    Imports are deliberately local so cache validation and unit tests do not
    require torch or the Cosmos training stack.
    """

    import torch
    import torch.distributed as dist

    from cosmos_framework.utils.callback import Callback

    class _ThroughputProbeCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.output_path = Path(output_path)
            self.warmup_steps = int(warmup_steps)
            self.measure_steps = int(measure_steps)
            self.expected_world_size = int(expected_world_size)
            self.expected_grad_accum_iter = int(expected_grad_accum_iter)
            self.initial_iteration: int | None = None
            self.active_iteration: int | None = None
            self.step_start_monotonic_ns: int | None = None
            self.step_start_unix_ns: int | None = None
            self.loader_start_ns: int | None = None
            self.loader_wait_ns = 0
            self.loader_calls = 0
            self.timer_dataloader_values: list[float] = []
            self.microbatch_trace: list[dict[str, Any]] = []
            self.microbatch_text_tokens: list[Any] = []

        def on_train_start(self, model: Any, iteration: int = 0) -> None:
            del model
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if world_size != self.expected_world_size:
                raise RuntimeError(
                    f"throughput callback requires world_size={self.expected_world_size}, got {world_size}"
                )
            resolved_grad_accum = int(self.config.trainer.grad_accum_iter)
            if resolved_grad_accum != self.expected_grad_accum_iter:
                raise RuntimeError(
                    "throughput callback requires "
                    f"grad_accum_iter={self.expected_grad_accum_iter}, got {resolved_grad_accum}"
                )
            self.initial_iteration = int(iteration)
            if dist.get_rank() == 0:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    self.output_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                )
                os.close(descriptor)
            dist.barrier()

        def on_before_dataloading(self, iteration: int = 0) -> None:
            current = int(iteration)
            if self.active_iteration is None:
                self.active_iteration = current
                self.step_start_monotonic_ns = time.monotonic_ns()
                self.step_start_unix_ns = time.time_ns()
                self.loader_wait_ns = 0
                self.loader_calls = 0
                self.timer_dataloader_values = []
                self.microbatch_trace = []
                self.microbatch_text_tokens = []
            elif self.active_iteration != current:
                raise RuntimeError(
                    "throughput callback observed a new optimizer iteration before closing the prior one: "
                    f"active={self.active_iteration}, current={current}"
                )
            if self.loader_start_ns is not None:
                raise RuntimeError("throughput callback observed nested dataloader hooks")
            self.loader_start_ns = time.monotonic_ns()

        def on_after_dataloading(self, iteration: int = 0) -> None:
            if self.loader_start_ns is None or self.active_iteration != int(iteration):
                raise RuntimeError("throughput callback observed an unmatched dataloader end")
            self.loader_wait_ns += time.monotonic_ns() - self.loader_start_ns
            self.loader_calls += 1
            timer_results = self.trainer.training_timer.results.get("dataloader_train", [])
            if not timer_results:
                raise RuntimeError("trainer timer/dataloader_train did not append before on_after_dataloading")
            self.timer_dataloader_values.append(float(timer_results[-1]))
            self.loader_start_ns = None

        def on_training_step_batch_end(
            self,
            model: Any,
            data_batch: Mapping[str, Any],
            output_batch: Mapping[str, Any],
            loss: Any,
            iteration: int = 0,
        ) -> None:
            del model, loss
            if self.active_iteration != int(iteration):
                raise RuntimeError("throughput callback received batch trace outside its active optimizer step")
            urls = data_batch.get("__url__")
            keys = data_batch.get("__key__")
            if (
                not isinstance(urls, (list, tuple))
                or not isinstance(keys, (list, tuple))
                or len(urls) != len(keys)
                or not urls
                or any(not isinstance(url, str) or not isinstance(key, str) for url, key in zip(urls, keys))
            ):
                raise RuntimeError("throughput callback requires ordered string (__url__, __key__) identities")
            identities = list(zip(urls, keys))
            identity_bytes = json.dumps(
                identities,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()

            def scalar_int(name: str) -> int:
                value = output_batch.get(name)
                if hasattr(value, "numel") and value.numel() == 1:
                    return int(value.item())
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
                raise RuntimeError(f"throughput callback requires scalar output metric {name}")

            self.microbatch_trace.append(
                {
                    "microbatch_index": len(self.microbatch_trace),
                    "identity_count": len(identities),
                    "identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
                    "und_token_length": scalar_int("und_token_length"),
                    "gen_token_length": scalar_int("gen_token_length"),
                    "vision_token_length": scalar_int("vision_token_length"),
                    "action_token_length": scalar_int("action_token_length"),
                }
            )
            if "text_token_ids" not in data_batch:
                raise RuntimeError("throughput callback requires text_token_ids for content trace validation")
            self.microbatch_text_tokens.append(data_batch["text_token_ids"])

        def on_training_step_end(
            self,
            model: Any,
            data_batch: Mapping[str, Any],
            output_batch: Mapping[str, Any],
            loss: Any,
            iteration: int = 0,
        ) -> None:
            del model, data_batch, output_batch, loss
            completed_iteration = int(iteration)
            if self.initial_iteration is None or self.active_iteration is None:
                raise RuntimeError("throughput callback step ended without an active timing window")
            if completed_iteration != self.active_iteration + 1:
                raise RuntimeError(
                    f"throughput callback iteration mismatch: active={self.active_iteration}, "
                    f"completed={completed_iteration}"
                )
            if self.loader_start_ns is not None:
                raise RuntimeError("throughput callback step ended during a dataloader interval")
            if self.loader_calls != self.expected_grad_accum_iter:
                raise RuntimeError(
                    f"throughput callback expected {self.expected_grad_accum_iter} dataloader calls, "
                    f"observed {self.loader_calls}"
                )
            if len(self.microbatch_trace) != self.expected_grad_accum_iter:
                raise RuntimeError(
                    f"throughput callback expected {self.expected_grad_accum_iter} traced microbatches, "
                    f"observed {len(self.microbatch_trace)}"
                )
            torch.cuda.synchronize()
            end_monotonic_ns = time.monotonic_ns()
            end_unix_ns = time.time_ns()

            def hash_token_tree(value: Any) -> tuple[str, int]:
                digest = hashlib.sha256()
                token_count = 0

                def update(item: Any) -> None:
                    nonlocal token_count
                    if isinstance(item, torch.Tensor):
                        tensor = item.detach().contiguous().cpu()
                        digest.update(b"T")
                        digest.update(str(tensor.dtype).encode())
                        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
                        digest.update(tensor.numpy().tobytes(order="C"))
                        token_count += int(tensor.numel())
                        return
                    if isinstance(item, (list, tuple)):
                        digest.update(b"L")
                        digest.update(len(item).to_bytes(8, "little"))
                        for nested in item:
                            update(nested)
                        return
                    raise RuntimeError(
                        "throughput callback text_token_ids must be a tensor or nested tensor sequence, "
                        f"got {type(item).__name__}"
                    )

                update(value)
                return digest.hexdigest(), token_count

            if len(self.microbatch_text_tokens) != self.expected_grad_accum_iter:
                raise RuntimeError("throughput callback did not retain text tokens for every microbatch")
            for trace_record, token_values in zip(self.microbatch_trace, self.microbatch_text_tokens):
                token_sha256, token_count = hash_token_tree(token_values)
                trace_record["text_token_ids_sha256"] = token_sha256
                trace_record["text_token_ids_count"] = token_count
            timer_values = list(self.timer_dataloader_values)
            if len(timer_values) != self.expected_grad_accum_iter:
                raise RuntimeError(
                    "trainer timer/dataloader_train did not record every gradient-accumulation microbatch: "
                    f"expected={self.expected_grad_accum_iter}, observed={len(timer_values)}"
                )
            local_record = {
                "rank": dist.get_rank(),
                "start_monotonic_ns": self.step_start_monotonic_ns,
                "end_monotonic_ns": end_monotonic_ns,
                "start_unix_ns": self.step_start_unix_ns,
                "end_unix_ns": end_unix_ns,
                "outer_step_seconds": (
                    end_monotonic_ns - int(self.step_start_monotonic_ns)
                )
                / 1e9,
                "callback_loader_wait_seconds": self.loader_wait_ns / 1e9,
                "timer_dataloader_train_seconds": float(sum(timer_values)),
                "timer_dataloader_train_microbatches": len(timer_values),
                "microbatches": self.microbatch_trace,
            }
            gathered: list[dict[str, Any] | None] = [None] * self.expected_world_size
            dist.all_gather_object(gathered, local_record)
            if dist.get_rank() == 0:
                per_rank = [record for record in gathered if record is not None]
                if len(per_rank) != self.expected_world_size:
                    raise RuntimeError("throughput callback did not gather every rank")
                relative_step = completed_iteration - self.initial_iteration
                total_steps = self.warmup_steps + self.measure_steps
                if relative_step < 1 or relative_step > total_steps:
                    raise RuntimeError(
                        f"throughput callback relative step {relative_step} is outside 1..{total_steps}"
                    )
                consolidated = {
                    "schema_version": SCHEMA_VERSION,
                    "completed_iteration": completed_iteration,
                    "relative_optimizer_step": relative_step,
                    "window": "warmup" if relative_step <= self.warmup_steps else "measure",
                    "start_unix_ns": min(int(record["start_unix_ns"]) for record in per_rank),
                    "end_unix_ns": max(int(record["end_unix_ns"]) for record in per_rank),
                    "outer_step_seconds": (
                        max(int(record["end_monotonic_ns"]) for record in per_rank)
                        - min(int(record["start_monotonic_ns"]) for record in per_rank)
                    )
                    / 1e9,
                    "timer_dataloader_train_seconds": max(
                        float(record["timer_dataloader_train_seconds"]) for record in per_rank
                    ),
                    "callback_loader_wait_seconds": max(
                        float(record["callback_loader_wait_seconds"]) for record in per_rank
                    ),
                    "workers_per_rank": int(os.environ["ROBOCASA365_TRAIN_NUM_WORKERS"]),
                    "node_total_workers": (
                        int(os.environ["ROBOCASA365_TRAIN_NUM_WORKERS"]) * self.expected_world_size
                    ),
                    "grad_accum_iter": self.expected_grad_accum_iter,
                    "world_size": self.expected_world_size,
                    "per_rank": sorted(per_rank, key=lambda record: int(record["rank"])),
                }
                trace_contract = [
                    {
                        "rank": int(record["rank"]),
                        "microbatches": record["microbatches"],
                    }
                    for record in sorted(per_rank, key=lambda record: int(record["rank"]))
                ]
                consolidated["trace_identity_count"] = sum(
                    int(microbatch["identity_count"])
                    for rank_record in trace_contract
                    for microbatch in rank_record["microbatches"]
                )
                consolidated["trace_identity_sha256"] = hashlib.sha256(
                    json.dumps(
                        trace_contract,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                encoded = json.dumps(consolidated, sort_keys=True) + "\n"
                with self.output_path.open("a") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            self.active_iteration = None
            self.step_start_monotonic_ns = None
            self.step_start_unix_ns = None
            self.loader_wait_ns = 0
            self.loader_calls = 0
            self.timer_dataloader_values = []
            self.microbatch_trace = []
            self.microbatch_text_tokens = []

    return _ThroughputProbeCallback()


class ContractError(ValueError):
    """Raised when an operational artifact violates the probe contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or LOWER_SHA256_PATTERN.fullmatch(value) is None:
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty relative POSIX path")
    if "\\" in value:
        raise ContractError(f"{label} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError(f"{label} must be a normalized relative path")
    normalized = path.as_posix()
    if normalized != value:
        raise ContractError(f"{label} must be normalized (got {value!r})")
    return normalized


def _load_json_file(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {label} at {path}: {error}") from error
    return _require_mapping(value, label)


def _write_json_atomic(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_cache(root_value: str | Path) -> dict[str, Any]:
    """Validate every mapping and shard hash in one completed cache."""

    root = Path(root_value)
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"cache root must be a directory and not a symlink: {root}")
    root = root.resolve(strict=True)
    try:
        from cosmos_framework.data.generator.action.datasets.robocasa365_latent_cache import (
            validate_published_cache_files,
        )

        shared = validate_published_cache_files(root, verify_shard_checksums=True)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ContractError(f"shared RoboCasa365 latent-cache validation failed: {error}") from error
    manifest = _require_mapping(shared.get("manifest"), "shared cache manifest")
    vae = _require_mapping(manifest.get("vae"), "manifest.vae")
    wan_vae_sha256 = _require_sha256(vae.get("weights_sha256"), "manifest.vae.weights_sha256")
    if wan_vae_sha256 != OFFICIAL_WAN22_VAE_SHA256:
        raise ContractError(
            "cache Wan VAE digest does not match the audited official Wan2.2 VAE digest"
        )
    raw_files = shared.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ContractError("shared cache validator returned no file inventory")
    checked_files: list[dict[str, Any]] = []
    total_bytes = 0
    for index, raw_item in enumerate(raw_files):
        item = _require_mapping(raw_item, f"shared cache files[{index}]")
        relative = _safe_relative_path(item.get("path"), f"shared cache files[{index}].path")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"cache file must be a regular non-symlink file: {path}")
        resolved = path.resolve(strict=True)
        if root not in resolved.parents:
            raise ContractError(f"cache file escapes the cache root: {relative}")
        role = str(item.get("role"))
        size_bytes = _require_int(item.get("size_bytes"), f"shared cache files[{index}].size_bytes")
        if path.stat().st_size != size_bytes:
            raise ContractError(
                f"{role} size mismatch for {relative}: "
                f"expected {size_bytes}, got {path.stat().st_size}"
            )
        expected_sha = _require_sha256(item.get("sha256"), f"shared cache files[{index}].sha256")
        if item.get("checksum_verified") is not True:
            raise ContractError(f"shared cache validator did not verify {relative}")
        total_bytes += size_bytes
        checked_files.append(
            {
                "role": role,
                "path": relative,
                "size_bytes": size_bytes,
                "sha256": expected_sha,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "validated_at": _utc_now(),
        "root": str(root),
        "cache_id": shared["cache_id"],
        "dataset_id": manifest["dataset"]["dataset_id"],
        "manifest_sha256": shared["manifest_sha256"],
        "wan_vae_sha256": wan_vae_sha256,
        "file_count": len(checked_files),
        "shard_count": sum(item["role"] == "shard" for item in checked_files),
        "total_bytes": total_bytes,
        "files": checked_files,
    }


def _fixed_trace_dataset_contract(
    dataset_root_value: str | Path,
    source_root_value: str | Path,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root_value)
    source_root = Path(source_root_value)
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ContractError(
            f"fixed trace dataset root must be a real non-symlink directory: {dataset_root}"
        )
    if source_root.is_symlink() or not source_root.is_dir():
        raise ContractError(
            f"fixed trace source root must be a real non-symlink directory: {source_root}"
        )
    dataset_root = dataset_root.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    manifest_path = dataset_root / "dataset_manifest.json"
    success_path = dataset_root / "_SUCCESS.json"
    split_path = dataset_root / "splits" / "train.parquet"
    manifest = _load_json_file(manifest_path, "RoboCasa365 dataset manifest")
    _load_json_file(success_path, "RoboCasa365 dataset completion marker")
    if split_path.is_symlink() or not split_path.is_file():
        raise ContractError(f"RoboCasa365 train split must be a regular non-symlink file: {split_path}")
    return {
        "schema_version": 1,
        "dataset_id": manifest.get("dataset_id"),
        "dataset_root": str(dataset_root),
        "source_root": str(source_root),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "dataset_success_sha256": _sha256_file(success_path),
        "train_split_sha256": _sha256_file(split_path),
        "split": "train",
        "fps": 20.0,
        "chunk_length": 16,
        "mode": "policy",
        "camera_mode": "concat_view",
        "image_size": 256,
        "caption_mode": "policy",
        "action_normalization": "quantile_rot",
        "video_backend": "lerobot",
        "episode_shuffle_seed": 42,
    }


def _build_fixed_trace_dataset(dataset_root: str | Path, source_root: str | Path) -> Any:
    try:
        from cosmos_framework.data.generator.action.datasets.robocasa365_segment_dataset import (
            RoboCasa365SegmentDataset,
        )
    except ImportError as error:
        raise ContractError(
            "fixed trace preparation requires the Cosmos training environment"
        ) from error
    return RoboCasa365SegmentDataset(
        root=str(dataset_root),
        source_root=str(source_root),
        split="train",
        fps=20.0,
        chunk_length=16,
        mode="policy",
        camera_mode="concat_view",
        image_size=256,
        caption_mode="policy",
        action_normalization="quantile_rot",
        video_backend="lerobot",
        decode_retries=8,
        resample_on_decode_error=False,
        validate_manifest=True,
        latent_cache_root=None,
        latent_cache_mode="disabled",
    )


def _shuffle_blocks_digest(dataset: Any) -> str:
    blocks = [[int(start), int(length)] for start, length in dataset.get_shuffle_blocks()]
    return hashlib.sha256(json.dumps(blocks, separators=(",", ":")).encode()).hexdigest()


def validate_fixed_index_trace(
    path_value: str | Path,
    *,
    expected_dataset: Any | None = None,
    expected_dataset_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the fixed 8-rank index/RNG replay contract used by every cell."""

    path = Path(path_value)
    if path.suffix != ".npy" or path.is_symlink() or not path.is_file():
        raise ContractError(f"fixed index trace must be a regular non-symlink .npy file: {path}")
    path = path.resolve(strict=True)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = _load_json_file(manifest_path, "fixed index trace manifest")
    exact_fields: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "array_path": path.name,
        "dtype": "int64",
        "shape": [8, 15552],
        "shuffle_seed": 42,
        "source_world_size": 8,
        "baseline_num_workers": 4,
        "indices_per_rank": 15552,
        "batch_size_per_rank": 128,
        "grad_accum_iter": 2,
        "warmup_steps": 10,
        "measure_steps": 50,
        "total_optimizer_steps": 60,
        "prefetch_factor": 4,
        "in_order": True,
        "transform_seed": 42,
        "algorithm": "baseline-worker-interleave-v1",
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in exact_fields.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ContractError(f"fixed index trace manifest mismatch: {mismatches}")
    expected_array_sha = _require_sha256(manifest.get("array_sha256"), "trace manifest.array_sha256")
    actual_array_sha = _sha256_file(path)
    if actual_array_sha != expected_array_sha:
        raise ContractError("fixed index trace array SHA-256 mismatch")
    _require_sha256(manifest.get("shuffle_blocks_sha256"), "trace manifest.shuffle_blocks_sha256")
    dataset_length = _require_int(manifest.get("dataset_length"), "trace manifest.dataset_length", minimum=1)
    try:
        import numpy as np
    except ImportError as error:
        raise ContractError("fixed index trace validation requires numpy in COSMOS_ENV") from error
    # GPFS on H does not support NumPy mmap reliably; this trace is only about
    # 1 MiB, so validate it with an ordinary bounded read.
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.dtype("int64") or list(array.shape) != [8, 15552]:
        raise ContractError(f"fixed index trace array contract mismatch: dtype={array.dtype}, shape={array.shape}")
    minimum = int(array.min())
    maximum = int(array.max())
    if minimum < 0 or maximum >= dataset_length:
        raise ContractError(
            f"fixed index trace contains out-of-range indices: min={minimum}, max={maximum}, "
            f"dataset_length={dataset_length}"
        )
    if expected_dataset is not None:
        if len(expected_dataset) != dataset_length:
            raise ContractError(
                "fixed index trace dataset length does not match the active RoboCasa365 dataset: "
                f"trace={dataset_length}, active={len(expected_dataset)}"
            )
        actual_blocks_sha256 = _shuffle_blocks_digest(expected_dataset)
        if actual_blocks_sha256 != manifest["shuffle_blocks_sha256"]:
            raise ContractError(
                "fixed index trace shuffle blocks do not match the active RoboCasa365 train split"
            )
    if expected_dataset_contract is not None:
        actual_contract = _require_mapping(
            manifest.get("dataset_contract"),
            "trace manifest.dataset_contract",
        )
        if dict(actual_contract) != dict(expected_dataset_contract):
            raise ContractError(
                "fixed index trace dataset contract does not match the active RoboCasa365 inputs"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "validated_at": _utc_now(),
        "path": str(path),
        "array_sha256": actual_array_sha,
        "manifest_path": str(manifest_path.resolve(strict=True)),
        "manifest_sha256": _sha256_file(manifest_path),
        "shape": [8, 15552],
        "dtype": "int64",
        "dataset_length": dataset_length,
        "minimum_index": minimum,
        "maximum_index": maximum,
        "contract": exact_fields,
        "dataset_contract": manifest.get("dataset_contract"),
    }


def prepare_fixed_index_trace(
    path_value: str | Path,
    dataset_root_value: str | Path,
    source_root_value: str | Path,
) -> dict[str, Any]:
    """Generate an absent fixed trace or validate one against the active dataset."""

    path = Path(path_value)
    if path.suffix != ".npy" or path.is_symlink():
        raise ContractError(f"fixed index trace target must be a non-symlink .npy path: {path}")
    manifest_path = path.with_suffix(".manifest.json")
    array_exists = path.exists()
    manifest_exists = manifest_path.exists()
    if array_exists != manifest_exists:
        raise ContractError(
            "refusing a partial fixed trace publication; array and manifest must both exist or both be absent"
        )
    dataset_contract = _fixed_trace_dataset_contract(dataset_root_value, source_root_value)
    dataset = _build_fixed_trace_dataset(dataset_root_value, source_root_value)
    preparation = "validated_existing"
    if not array_exists:
        try:
            from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
                write_action_fixed_index_trace,
            )
        except ImportError as error:
            raise ContractError(
                "fixed trace generation requires the Cosmos training environment"
            ) from error
        write_action_fixed_index_trace(
            dataset,
            path,
            world_size=8,
            baseline_num_workers=4,
            seed=42,
            transform_seed=42,
            indices_per_rank=15_552,
            batch_size_per_rank=128,
            grad_accum_iter=2,
            warmup_steps=10,
            measure_steps=50,
            prefetch_factor=4,
            in_order=True,
            dataset_contract=dataset_contract,
        )
        preparation = "generated"
    payload = validate_fixed_index_trace(
        path,
        expected_dataset=dataset,
        expected_dataset_contract=dataset_contract,
    )
    payload["preparation"] = preparation
    return payload


def stage_cache(source_value: str | Path, destination_value: str | Path) -> dict[str, Any]:
    """Copy one validated cache into a new destination and publish atomically."""

    source = Path(source_value).resolve(strict=True)
    destination = Path(destination_value)
    if destination.exists() or destination.is_symlink():
        raise ContractError(f"NVMe cache destination already exists: {destination}")
    source_report = validate_cache(source)
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    if destination_parent.is_symlink() or not destination_parent.is_dir():
        raise ContractError(f"NVMe cache destination parent is unsafe: {destination_parent}")
    stage = destination_parent / f".incomplete-{destination.name}-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o755)
    try:
        for item in source_report["files"]:
            relative = item["path"]
            source_file = source / relative
            destination_file = stage / relative
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, destination_file)
            os.chmod(destination_file, 0o444)
        destination_report = validate_cache(stage)
        if destination_report["manifest_sha256"] != source_report["manifest_sha256"]:
            raise ContractError("staged cache manifest SHA-256 differs from the source")
        ready = {
            "schema_version": SCHEMA_VERSION,
            "status": "ready",
            "published_at": _utc_now(),
            "source_root": str(source),
            "destination_root": str(destination.resolve(strict=False)),
            "cache_id": source_report["cache_id"],
            "manifest_sha256": source_report["manifest_sha256"],
            "file_count": source_report["file_count"],
            "shard_count": source_report["shard_count"],
            "total_bytes": source_report["total_bytes"],
        }
        _write_json_atomic(stage / READY_FILENAME, ready, exclusive=True)
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    result = dict(source_report)
    result.update(
        {
            "status": "staged",
            "source_root": str(source),
            "root": str(destination.resolve(strict=True)),
            "ready_path": str(destination / READY_FILENAME),
        }
    )
    return result


def _distributed_stage(args: argparse.Namespace) -> int:
    try:
        import torch.distributed as dist
    except ImportError as error:
        raise ContractError("stage-nvme requires the training environment's torch installation") from error

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    if world_size != 8 or local_world_size != 8 or rank not in range(8):
        raise ContractError(
            "stage-nvme requires torchrun with WORLD_SIZE=LOCAL_WORLD_SIZE=8 "
            f"(got world={world_size}, local_world={local_world_size}, rank={rank})"
        )
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    failure_path = evidence_dir / "nvme-stage-failure.json"
    rank0_path = evidence_dir / "nvme-stage-rank0.json"
    # Rank 0 verifies ~100 GiB on GPFS, copies it to local NVMe, and verifies
    # the destination while ranks 1-7 wait at the barrier.  The default Gloo
    # collective timeout is too short for that deliberately exhaustive I/O.
    dist.init_process_group(backend="gloo", timeout=timedelta(hours=48))
    try:
        if rank == 0:
            try:
                report = stage_cache(args.source, args.destination)
                _write_json_atomic(rank0_path, report, exclusive=True)
            except BaseException as error:  # noqa: BLE001 - synchronize the same failure to every rank.
                _write_json_atomic(
                    failure_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "failed",
                        "failed_at": _utc_now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    exclusive=True,
                )
        dist.barrier()
        if failure_path.exists():
            failure = _load_json_file(failure_path, "NVMe stage failure")
            raise ContractError(f"rank0 NVMe staging failed: {failure.get('error')}")
        ready = _load_json_file(Path(args.destination) / READY_FILENAME, READY_FILENAME)
        expected_manifest_sha = _load_json_file(rank0_path, "rank0 stage evidence")["manifest_sha256"]
        if ready.get("status") != "ready" or ready.get("manifest_sha256") != expected_manifest_sha:
            raise ContractError("rank observed an invalid NVMe ready marker after the barrier")
        _write_json_atomic(
            evidence_dir / f"nvme-stage-rank-{rank:02d}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                "observed_at": _utc_now(),
                "rank": rank,
                "world_size": world_size,
                "hostname": socket.gethostname(),
                "destination": str(Path(args.destination).resolve(strict=True)),
                "manifest_sha256": expected_manifest_sha,
            },
            exclusive=True,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


def _parse_nvidia_smi_inventory(text: str, expected_gpus: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in next(csv.reader([line]))]
        if len(fields) != 5:
            raise ContractError(f"nvidia-smi inventory line {line_number} has {len(fields)} fields, expected 5")
        index_text, uuid_text, name, memory_text, driver_version = fields
        try:
            index = int(index_text)
            memory_mib = int(float(memory_text))
        except ValueError as error:
            raise ContractError(f"invalid numeric nvidia-smi inventory line {line_number}") from error
        records.append(
            {
                "index": index,
                "uuid": uuid_text,
                "name": name,
                "memory_total_mib": memory_mib,
                "driver_version": driver_version,
            }
        )
    if len(records) != expected_gpus:
        raise ContractError(f"expected exactly {expected_gpus} visible GPUs, got {len(records)}")
    if len({record["index"] for record in records}) != expected_gpus:
        raise ContractError("GPU indices are not unique")
    if len({record["uuid"] for record in records}) != expected_gpus:
        raise ContractError("GPU UUIDs are not unique")
    unexpected = [record["name"] for record in records if "H200" not in record["name"].upper()]
    if unexpected:
        raise ContractError(f"all visible GPUs must be H200; unexpected names: {unexpected}")
    return records


def _cgroup_cpu_quota_cores() -> float | None:
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if not cpu_max.is_file():
        return None
    fields = cpu_max.read_text().strip().split()
    if len(fields) != 2 or fields[0] == "max":
        return None
    quota = int(fields[0])
    period = int(fields[1])
    if quota < 1 or period < 1:
        raise ContractError(f"invalid cgroup v2 cpu.max: {fields}")
    return quota / period


def capture_node_evidence(
    output: Path,
    expected_gpus: int,
    rjob_show_name: str,
    minimum_cpus: int,
    requested_cpus: int,
) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    gpus = _parse_nvidia_smi_inventory(completed.stdout, expected_gpus)
    affinity_cpus = len(os.sched_getaffinity(0))
    quota_cores = _cgroup_cpu_quota_cores()
    effective_cpus = min(float(affinity_cpus), quota_cores) if quota_cores is not None else float(affinity_cpus)
    if effective_cpus < minimum_cpus:
        raise ContractError(
            f"probe requires at least {minimum_cpus} effective CPUs, got {effective_cpus}"
        )
    selected_env_names = (
        "JOB_ID",
        "RJOB_NAME",
        "KUBEBRAIN_RJOB_NAME",
        "KUBEBRAIN_RJOB_UID",
        "KUBEBRAIN_NODE_NAME",
        "POD_NAME",
        "HOSTNAME",
        "CUDA_VISIBLE_DEVICES",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "captured_at": _utc_now(),
        "rjob_show_name": rjob_show_name,
        "hostname": socket.gethostname(),
        "expected_gpu_count": expected_gpus,
        "visible_gpu_count": len(gpus),
        "requested_cpu_count": requested_cpus,
        "minimum_effective_cpu_count": minimum_cpus,
        "affinity_cpu_count": affinity_cpus,
        "cgroup_cpu_quota_cores": quota_cores,
        "effective_cpu_count": effective_cpus,
        "maximum_workers_per_rank": 16,
        "maximum_node_loader_workers": 16 * expected_gpus,
        "training_rank_count": expected_gpus,
        "gpus": gpus,
        "scheduler_environment": {name: os.environ.get(name) for name in selected_env_names},
    }
    _write_json_atomic(output, payload, exclusive=True)
    return payload


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ContractError("cannot compute a percentile from an empty sequence")
    if not 0.0 <= percentile <= 1.0:
        raise ContractError("percentile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sample_gpu(output: Path, interval_seconds: int, expected_gpus: int) -> int:
    """Continuously append timestamped GPU batches until SIGTERM/SIGINT."""

    if interval_seconds < 1:
        raise ContractError("GPU sample interval must be a positive integer")
    if output.exists() or output.is_symlink():
        raise ContractError(f"GPU sample output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.close(descriptor)
    stopping = False

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    while not stopping:
        started_ns = time.time_ns()
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        finished_ns = time.time_ns()
        gpus: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(completed.stdout.splitlines(), start=1):
            fields = [field.strip() for field in next(csv.reader([raw_line]))]
            if len(fields) != 7:
                raise ContractError(
                    f"nvidia-smi GPU sampler line {line_number} has {len(fields)} fields, expected 7"
                )
            index_text, uuid_text, name, util_text, used_text, total_text, power_text = fields
            try:
                gpus.append(
                    {
                        "index": int(index_text),
                        "uuid": uuid_text,
                        "name": name,
                        "utilization_gpu_percent": float(util_text),
                        "memory_used_mib": float(used_text),
                        "memory_total_mib": float(total_text),
                        "power_draw_w": float(power_text),
                    }
                )
            except ValueError as error:
                raise ContractError(f"invalid numeric GPU sampler line {line_number}") from error
        if len(gpus) != expected_gpus or len({gpu["uuid"] for gpu in gpus}) != expected_gpus:
            raise ContractError(f"GPU sampler expected exactly {expected_gpus} unique GPUs")
        if any("H200" not in gpu["name"].upper() for gpu in gpus):
            raise ContractError("GPU sampler observed a non-H200 device")
        record = {
            "schema_version": SCHEMA_VERSION,
            "sample_started_unix_ns": started_ns,
            "sample_finished_unix_ns": finished_ns,
            "sample_unix_ns": (started_ns + finished_ns) // 2,
            "gpus": sorted(gpus, key=lambda gpu: int(gpu["index"])),
        }
        with output.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        deadline = time.monotonic() + interval_seconds
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    return 0


def _load_step_records(path: Path) -> list[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        return []
    records: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(errors="strict").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ContractError(f"invalid step NDJSON line {line_number}: {error}") from error
        records.append(_require_mapping(record, f"step NDJSON line {line_number}"))
    return records


def _gpu_samples(
    path: Path,
    *,
    measured_start_unix_ns: int | None,
    measured_end_unix_ns: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        return [], {"sample_batches": 0, "sample_rows": 0, "gpu_count": 0}
    records: list[dict[str, Any]] = []
    sample_batches = 0
    for line_number, raw_line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            batch = _require_mapping(json.loads(raw_line), f"GPU sample line {line_number}")
        except json.JSONDecodeError as error:
            raise ContractError(f"invalid GPU NDJSON line {line_number}: {error}") from error
        sample_unix_ns = _require_int(batch.get("sample_unix_ns"), f"GPU sample line {line_number}.sample_unix_ns")
        sample_started_unix_ns = _require_int(
            batch.get("sample_started_unix_ns"),
            f"GPU sample line {line_number}.sample_started_unix_ns",
        )
        sample_finished_unix_ns = _require_int(
            batch.get("sample_finished_unix_ns"),
            f"GPU sample line {line_number}.sample_finished_unix_ns",
        )
        if not sample_started_unix_ns <= sample_unix_ns <= sample_finished_unix_ns:
            raise ContractError(f"GPU sample line {line_number} has inconsistent timestamps")
        if measured_start_unix_ns is None or measured_end_unix_ns is None:
            continue
        if not (
            measured_start_unix_ns <= sample_started_unix_ns
            and sample_finished_unix_ns <= measured_end_unix_ns
        ):
            continue
        raw_gpus = batch.get("gpus")
        if not isinstance(raw_gpus, list):
            raise ContractError(f"GPU sample line {line_number}.gpus must be a list")
        sample_batches += 1
        for raw_gpu in raw_gpus:
            gpu = dict(_require_mapping(raw_gpu, f"GPU sample line {line_number} GPU"))
            gpu["sample_unix_ns"] = sample_unix_ns
            records.append(gpu)
    if not records:
        return [], {"sample_batches": sample_batches, "sample_rows": 0, "gpu_count": 0}
    uuids = sorted({record["uuid"] for record in records})
    memory_fractions = [
        record["memory_used_mib"] / record["memory_total_mib"]
        for record in records
        if record["memory_total_mib"] > 0
    ]
    summary = {
        "window_start_unix_ns": measured_start_unix_ns,
        "window_end_unix_ns": measured_end_unix_ns,
        "sample_batches": sample_batches,
        "sample_rows": len(records),
        "gpu_count": len(uuids),
        "gpu_uuids": uuids,
        "average_utilization_percent": statistics.fmean(
            record["utilization_gpu_percent"] for record in records
        ),
        "p50_utilization_percent": _percentile(
            [record["utilization_gpu_percent"] for record in records],
            0.5,
        ),
        "average_power_draw_w": statistics.fmean(record["power_draw_w"] for record in records),
        "peak_memory_used_mib": max(record["memory_used_mib"] for record in records),
        "peak_memory_used_fraction": max(memory_fractions) if memory_fractions else None,
    }
    return records, summary


def _memory_gate_summary(path: Path, *, expected_world_size: int) -> tuple[bool, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        return False, {"status": "missing", "path": str(path.resolve(strict=False))}
    payload = _load_json_file(path, "production memory gate")
    thresholds = _require_mapping(payload.get("thresholds"), "memory gate thresholds")
    contract = _require_mapping(payload.get("production_batch_contract"), "memory gate batch contract")
    raw_rows = payload.get("per_rank")
    rows = raw_rows if isinstance(raw_rows, list) else []
    required_fields = (
        "total_memory_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "driver_used_bytes",
        "peak_allocated_fraction",
        "peak_reserved_fraction",
        "driver_used_fraction",
    )
    rows_complete = (
        len(rows) == expected_world_size
        and sorted(int(row.get("rank", -1)) for row in rows) == list(range(expected_world_size))
        and all(all(field in row for field in required_fields) for row in rows)
    )
    passed = (
        payload.get("status") == "passed"
        and payload.get("violations") == []
        and payload.get("world_size") == expected_world_size
        and payload.get("start_iteration") == 1000
        and payload.get("observed_iteration") == 1001
        and thresholds.get("max_peak_reserved_fraction") == 0.94
        and thresholds.get("max_driver_used_fraction") == 0.94
        and contract.get("expected_samples_per_microbatch") == 128
        and contract.get("expected_grad_accum_iter") == 2
        and contract.get("allowed_world_sizes") == [8]
        and rows_complete
    )
    summary = {
        "status": "passed" if passed else "failed",
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256_file(path),
        "start_iteration": payload.get("start_iteration"),
        "observed_iteration": payload.get("observed_iteration"),
        "world_size": payload.get("world_size"),
        "thresholds": thresholds,
        "production_batch_contract": contract,
        "violations": payload.get("violations"),
        "per_rank": rows,
        "max_peak_allocated_fraction": (
            max(float(row["peak_allocated_fraction"]) for row in rows) if rows_complete else None
        ),
        "max_peak_reserved_fraction": (
            max(float(row["peak_reserved_fraction"]) for row in rows) if rows_complete else None
        ),
        "max_driver_used_fraction": (
            max(float(row["driver_used_fraction"]) for row in rows) if rows_complete else None
        ),
    }
    return passed, summary


def _strict_no_ac_memory_headroom_failure(
    path: Path,
    *,
    expected_world_size: int,
) -> dict[str, Any] | None:
    """Recognize only a structurally valid eight-rank threshold violation."""

    if path.is_symlink() or not path.is_file():
        return None
    try:
        payload = _load_json_file(path, "production memory gate")
        thresholds = _require_mapping(payload.get("thresholds"), "memory gate thresholds")
        contract = _require_mapping(payload.get("production_batch_contract"), "memory gate batch contract")
        raw_rows = payload.get("per_rank")
        rows = raw_rows if isinstance(raw_rows, list) else []
        required_fields = (
            "rank",
            "peak_reserved_fraction",
            "driver_used_fraction",
        )
        rows_complete = (
            len(rows) == expected_world_size
            and sorted(_require_int(row.get("rank"), "memory gate rank", minimum=0) for row in rows)
            == list(range(expected_world_size))
            and all(all(field in row for field in required_fields) for row in rows)
        )
        violations = payload.get("violations")
        structural_contract = (
            payload.get("status") == "failed"
            and payload.get("world_size") == expected_world_size
            and payload.get("start_iteration") == 1000
            and payload.get("observed_iteration") == 1001
            and thresholds.get("max_peak_reserved_fraction") == 0.94
            and thresholds.get("max_driver_used_fraction") == 0.94
            and contract.get("expected_samples_per_microbatch") == 128
            and contract.get("expected_grad_accum_iter") == 2
            and contract.get("allowed_world_sizes") == [8]
            and rows_complete
            and isinstance(violations, list)
            and len(violations) > 0
        )
        actually_over_threshold = any(
            float(row["peak_reserved_fraction"]) >= 0.94
            or float(row["driver_used_fraction"]) >= 0.94
            for row in rows
        )
        if not structural_contract or not actually_over_threshold:
            return None
        return {
            "status": "failed",
            "path": str(path.resolve(strict=True)),
            "sha256": _sha256_file(path),
            "world_size": payload.get("world_size"),
            "thresholds": thresholds,
            "production_batch_contract": contract,
            "violations": violations,
            "per_rank": rows,
            "max_peak_reserved_fraction": max(float(row["peak_reserved_fraction"]) for row in rows),
            "max_driver_used_fraction": max(float(row["driver_used_fraction"]) for row in rows),
        }
    except (ContractError, KeyError, OSError, TypeError, ValueError):
        return None


def _optional_no_ac_infeasibility(args: argparse.Namespace) -> dict[str, Any] | None:
    """Classify only the optional no-AC cell's exact OOM/headroom outcomes."""

    if args.activation_checkpointing != "none" or args.training_exit_code == 0:
        return None
    log_path = Path(args.log)
    if log_path.is_file() and not log_path.is_symlink():
        log_text = log_path.read_text(errors="replace")
        for marker, pattern in CUDA_OOM_MARKERS:
            if pattern.search(log_text):
                return {
                    "status": "infeasible_oom",
                    "reason": "no_ac_cuda_oom",
                    "canonical_marker": marker,
                    "production_batch_memory_gate": {
                        "status": "unavailable_or_not_reached",
                    },
                }
    memory_summary = _strict_no_ac_memory_headroom_failure(
        Path(args.memory_gate_json),
        expected_world_size=args.expected_gpus,
    )
    if memory_summary is not None:
        return {
            "status": "infeasible_memory_headroom",
            "reason": "no_ac_exceeded_eight_gpu_memory_gate",
            "canonical_marker": "production_batch_memory_gate_threshold_violation",
            "production_batch_memory_gate": memory_summary,
        }
    return None


def _write_infeasible_cell(args: argparse.Namespace, classification: Mapping[str, Any]) -> dict[str, Any]:
    """Persist an auditable optional-cell outcome without calling it passed."""

    log_path = Path(args.log)
    step_path = Path(args.step_ndjson)
    gpu_path = Path(args.gpu_samples)
    evidence_errors: list[str] = []
    steps: list[dict[str, Any]] = []
    trace_steps: list[dict[str, Any]] = []
    try:
        steps = _load_step_records(step_path)
        for record in steps:
            trace_steps.append(
                {
                    "relative_optimizer_step": _require_int(
                        record.get("relative_optimizer_step"),
                        "partial step relative_optimizer_step",
                        minimum=1,
                    ),
                    "completed_iteration": _require_int(
                        record.get("completed_iteration"),
                        "partial step completed_iteration",
                        minimum=1,
                    ),
                    "identity_count": _require_int(
                        record.get("trace_identity_count"),
                        "partial step trace_identity_count",
                        minimum=1,
                    ),
                    "identity_sha256": _require_sha256(
                        record.get("trace_identity_sha256"),
                        "partial step trace_identity_sha256",
                    ),
                }
            )
    except (ContractError, KeyError, OSError, TypeError, ValueError) as error:
        evidence_errors.append(f"partial_step_evidence: {type(error).__name__}: {error}")
        steps = []
        trace_steps = []
    trace_aggregate_sha256 = (
        hashlib.sha256(
            json.dumps(trace_steps, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if trace_steps
        else None
    )
    observed_start = min((int(record["start_unix_ns"]) for record in steps), default=None)
    observed_end = max((int(record["end_unix_ns"]) for record in steps), default=None)
    try:
        _, gpu_summary = _gpu_samples(
            gpu_path,
            measured_start_unix_ns=observed_start,
            measured_end_unix_ns=observed_end,
        )
    except (ContractError, KeyError, OSError, TypeError, ValueError) as error:
        evidence_errors.append(f"partial_gpu_evidence: {type(error).__name__}: {error}")
        gpu_summary = {"sample_batches": 0, "sample_rows": 0, "gpu_count": 0}

    memory_summary = classification["production_batch_memory_gate"]
    if classification["status"] == "infeasible_oom":
        try:
            _, memory_summary = _memory_gate_summary(
                Path(args.memory_gate_json),
                expected_world_size=args.expected_gpus,
            )
        except (ContractError, KeyError, OSError, TypeError, ValueError) as error:
            evidence_errors.append(f"memory_gate_evidence: {type(error).__name__}: {error}")
            memory_path = Path(args.memory_gate_json)
            memory_summary = {
                "status": "invalid",
                "path": str(memory_path.resolve(strict=False)),
                "sha256": (
                    _sha256_file(memory_path)
                    if memory_path.is_file() and not memory_path.is_symlink()
                    else None
                ),
            }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": classification["status"],
        "summarized_at": _utc_now(),
        "cell_id": args.cell_id,
        "workers_per_rank": args.workers,
        "node_total_workers": args.workers * args.expected_gpus,
        "storage": args.storage,
        "activation_checkpointing": args.activation_checkpointing,
        "training_exit_code": args.training_exit_code,
        "infeasibility_reason": classification["reason"],
        "canonical_marker": classification["canonical_marker"],
        "log": str(log_path.resolve(strict=False)),
        "log_sha256": _sha256_file(log_path) if log_path.is_file() and not log_path.is_symlink() else None,
        "optimizer_step_ndjson": str(step_path.resolve(strict=False)),
        "optimizer_step_ndjson_sha256": (
            _sha256_file(step_path) if step_path.is_file() and not step_path.is_symlink() else None
        ),
        "gpu_samples": str(gpu_path.resolve(strict=False)),
        "gpu_samples_sha256": (
            _sha256_file(gpu_path) if gpu_path.is_file() and not gpu_path.is_symlink() else None
        ),
        "evidence_parse_errors": evidence_errors,
        "timing": {
            "observed_optimizer_steps": len(steps),
            "observed_start_unix_ns": observed_start,
            "observed_end_unix_ns": observed_end,
        },
        "gpu": gpu_summary,
        "production_batch_memory_gate": memory_summary,
        "sample_trace": {
            "algorithm": (
                "sha256-canonical-ranked-microbatch-identities-text-token-contents-and-token-lengths-v1"
            ),
            "aggregate_sha256": trace_aggregate_sha256,
            "step_count": len(trace_steps),
            "per_step": trace_steps,
        },
    }
    _write_json_atomic(Path(args.output), payload, exclusive=True)
    return payload


def summarize_cell(args: argparse.Namespace) -> dict[str, Any]:
    infeasible = _optional_no_ac_infeasibility(args)
    if infeasible is not None:
        return _write_infeasible_cell(args, infeasible)
    expected_steps = args.warmup_steps + args.measure_steps
    steps = _load_step_records(Path(args.step_ndjson))
    relative_steps = [int(record.get("relative_optimizer_step", -1)) for record in steps]
    contiguous = relative_steps == list(range(1, expected_steps + 1))
    warmup = [record for record in steps if record.get("window") == "warmup"]
    measured = [record for record in steps if record.get("window") == "measure"]
    measured_seconds = [float(record["outer_step_seconds"]) for record in measured]
    measured_start_unix_ns = (
        min(int(record["start_unix_ns"]) for record in measured) if measured else None
    )
    measured_end_unix_ns = (
        max(int(record["end_unix_ns"]) for record in measured) if measured else None
    )
    trace_steps: list[dict[str, Any]] = []
    for record in steps:
        trace_steps.append(
            {
                "relative_optimizer_step": _require_int(
                    record.get("relative_optimizer_step"),
                    "step relative_optimizer_step",
                    minimum=1,
                ),
                "completed_iteration": _require_int(
                    record.get("completed_iteration"),
                    "step completed_iteration",
                    minimum=1,
                ),
                "identity_count": _require_int(
                    record.get("trace_identity_count"),
                    "step trace_identity_count",
                    minimum=1,
                ),
                "identity_sha256": _require_sha256(
                    record.get("trace_identity_sha256"),
                    "step trace_identity_sha256",
                ),
            }
        )
    trace_aggregate_sha256 = hashlib.sha256(
        json.dumps(trace_steps, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _, gpu_summary = _gpu_samples(
        Path(args.gpu_samples),
        measured_start_unix_ns=measured_start_unix_ns,
        measured_end_unix_ns=measured_end_unix_ns,
    )
    timing_complete = (
        contiguous
        and len(warmup) == args.warmup_steps
        and len(measured_seconds) == args.measure_steps
    )
    gpu_complete = (
        gpu_summary["gpu_count"] == args.expected_gpus
        and gpu_summary["sample_batches"] > 0
        and gpu_summary["sample_rows"] > 0
    )
    memory_gate_passed, memory_gate_summary = _memory_gate_summary(
        Path(args.memory_gate_json),
        expected_world_size=args.expected_gpus,
    )
    status = (
        "passed"
        if args.training_exit_code == 0 and timing_complete and gpu_complete and memory_gate_passed
        else "failed"
    )
    timing_summary: dict[str, Any] = {
        "observed_optimizer_steps": len(steps),
        "contiguous": contiguous,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "measurement_scope": (
            "rank-consolidated wall clock from before the first dataloader fetch through the "
            "GA=2 optimizer boundary and CUDA synchronize; first 10 steps discarded"
        ),
        "measured_start_unix_ns": measured_start_unix_ns,
        "measured_end_unix_ns": measured_end_unix_ns,
    }
    if measured_seconds:
        mean_seconds = statistics.fmean(measured_seconds)
        timer_dataloader_seconds = [
            float(record["timer_dataloader_train_seconds"]) for record in measured
        ]
        callback_loader_seconds = [
            float(record["callback_loader_wait_seconds"]) for record in measured
        ]
        timing_summary.update(
            {
                "first_measured_iteration": int(measured[0]["completed_iteration"]),
                "last_measured_iteration": int(measured[-1]["completed_iteration"]),
                "mean_step_seconds": mean_seconds,
                "median_step_seconds": statistics.median(measured_seconds),
                "p90_step_seconds": _percentile(measured_seconds, 0.90),
                "p95_step_seconds": _percentile(measured_seconds, 0.95),
                "min_step_seconds": min(measured_seconds),
                "max_step_seconds": max(measured_seconds),
                "optimizer_steps_per_second": 1.0 / mean_seconds,
                "mean_timer_dataloader_train_seconds": statistics.fmean(
                    timer_dataloader_seconds
                ),
                "p95_timer_dataloader_train_seconds": _percentile(
                    timer_dataloader_seconds,
                    0.95,
                ),
                "mean_callback_loader_wait_seconds": statistics.fmean(
                    callback_loader_seconds
                ),
                "mean_timer_dataloader_fraction": (
                    statistics.fmean(timer_dataloader_seconds) / mean_seconds
                ),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "summarized_at": _utc_now(),
        "cell_id": args.cell_id,
        "workers_per_rank": args.workers,
        "node_total_workers": args.workers * args.expected_gpus,
        "storage": args.storage,
        "activation_checkpointing": args.activation_checkpointing,
        "training_exit_code": args.training_exit_code,
        "log": str(Path(args.log).resolve(strict=False)),
        "log_sha256": _sha256_file(Path(args.log)) if Path(args.log).is_file() else None,
        "optimizer_step_ndjson": str(Path(args.step_ndjson).resolve(strict=False)),
        "optimizer_step_ndjson_sha256": (
            _sha256_file(Path(args.step_ndjson)) if Path(args.step_ndjson).is_file() else None
        ),
        "gpu_samples": str(Path(args.gpu_samples).resolve(strict=False)),
        "timing": timing_summary,
        "gpu": gpu_summary,
        "production_batch_memory_gate": memory_gate_summary,
        "sample_trace": {
            "algorithm": (
                "sha256-canonical-ranked-microbatch-identities-text-token-contents-and-token-lengths-v1"
            ),
            "aggregate_sha256": trace_aggregate_sha256,
            "step_count": len(trace_steps),
            "per_step": trace_steps,
        },
    }
    _write_json_atomic(Path(args.output), payload, exclusive=True)
    return payload


def summarize_cell_fail_closed(args: argparse.Namespace) -> dict[str, Any]:
    """Always persist a cell result; malformed/missing evidence is a failed cell."""

    try:
        return summarize_cell(args)
    except (ContractError, KeyError, OSError, TypeError, ValueError) as error:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "summarized_at": _utc_now(),
            "cell_id": args.cell_id,
            "workers_per_rank": args.workers,
            "node_total_workers": args.workers * args.expected_gpus,
            "storage": args.storage,
            "activation_checkpointing": args.activation_checkpointing,
            "training_exit_code": args.training_exit_code,
            "metrics_error_type": type(error).__name__,
            "metrics_error": str(error),
            "timing": {"observed_optimizer_steps": 0},
            "gpu": {"sample_batches": 0, "sample_rows": 0, "gpu_count": 0},
            "production_batch_memory_gate": {"status": "invalid_or_missing"},
            "sample_trace": {
                "algorithm": (
                    "sha256-canonical-ranked-microbatch-identities-text-token-contents-and-token-lengths-v1"
                ),
                "aggregate_sha256": None,
                "step_count": 0,
                "per_step": [],
            },
        }
        _write_json_atomic(Path(args.output), payload, exclusive=True)
        return payload


def summarize_matrix(run_root: Path, output: Path, expected_cell_ids: Sequence[str]) -> dict[str, Any]:
    cells: list[Mapping[str, Any]] = []
    missing: list[str] = []
    identity_mismatches: list[dict[str, Any]] = []
    if len(set(expected_cell_ids)) != len(expected_cell_ids):
        raise ContractError("expected cell IDs must be unique")
    for cell_id in expected_cell_ids:
        match = CELL_ID_PATTERN.fullmatch(cell_id)
        if match is None:
            raise ContractError(f"invalid expected cell ID: {cell_id!r}")
        expected_identity: dict[str, Any] = {
            "cell_id": cell_id,
            "workers_per_rank": int(match.group("workers")),
            "storage": match.group("storage"),
            "activation_checkpointing": match.group("activation_checkpointing"),
        }
        metrics_path = run_root / "cells" / cell_id / "metrics.json"
        if not metrics_path.is_file() or metrics_path.is_symlink():
            missing.append(cell_id)
            continue
        cell = _load_json_file(metrics_path, f"metrics for {cell_id}")
        observed_identity = {
            key: cell.get(key)
            for key in (
                "cell_id",
                "workers_per_rank",
                "storage",
                "activation_checkpointing",
            )
        }
        if observed_identity != expected_identity:
            identity_mismatches.append(
                {
                    "expected": expected_identity,
                    "observed": observed_identity,
                    "metrics_path": str(metrics_path.resolve(strict=True)),
                }
            )
        cells.append(cell)
    passed = [cell for cell in cells if cell.get("status") == "passed"]
    infeasible_statuses = {"infeasible_oom", "infeasible_memory_headroom"}
    infeasible = [cell for cell in cells if cell.get("status") in infeasible_statuses]
    failed = [
        cell
        for cell in cells
        if cell.get("status") not in {"passed", *infeasible_statuses}
    ]
    invalid_infeasible = [
        cell for cell in infeasible if cell.get("activation_checkpointing") != "none"
    ]
    passed_selective_keys = {
        (cell.get("workers_per_rank"), cell.get("storage"))
        for cell in passed
        if cell.get("activation_checkpointing") == "selective"
    }
    missing_selective_fallback = [
        cell
        for cell in infeasible
        if (cell.get("workers_per_rank"), cell.get("storage")) not in passed_selective_keys
    ]
    nonpassed_selective = [
        cell
        for cell in cells
        if cell.get("activation_checkpointing") != "none" and cell.get("status") != "passed"
    ]
    trace_payloads = [
        _require_mapping(cell.get("sample_trace"), f"sample trace for {cell.get('cell_id')}")
        for cell in passed
    ]
    trace_aggregates = {
        str(trace.get("aggregate_sha256"))
        for trace in trace_payloads
        if trace.get("step_count") == 60
    }
    trace_consistent = (
        bool(trace_payloads)
        and all(trace.get("step_count") == 60 for trace in trace_payloads)
        and len(trace_aggregates) == 1
    )
    matrix_complete = (
        not missing
        and len(cells) == len(expected_cell_ids)
        and not identity_mismatches
        and not failed
        and not invalid_infeasible
        and not missing_selective_fallback
        and not nonpassed_selective
        and trace_consistent
    )
    if matrix_complete and infeasible:
        matrix_status = "passed_with_no_ac_fallback"
    elif matrix_complete:
        matrix_status = "passed"
    else:
        matrix_status = "partial_or_failed"
    ranked = sorted(
        passed,
        key=lambda cell: float(_require_mapping(cell.get("timing"), "cell.timing")["mean_step_seconds"]),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": matrix_status,
        "outcome": (
            "all_cells_passed"
            if matrix_complete and not infeasible
            else "optional_no_ac_infeasible_selective_fallback"
            if matrix_complete
            else "protocol_failed"
        ),
        "summarized_at": _utc_now(),
        "run_root": str(run_root.resolve(strict=True)),
        "expected_cell_count": len(expected_cell_ids),
        "observed_cell_count": len(cells),
        "passed_cell_count": len(passed),
        "infeasible_cell_count": len(infeasible),
        "infeasible_cell_ids": [cell.get("cell_id") for cell in infeasible],
        "failed_cell_count": len(failed),
        "invalid_infeasible_cell_ids": [cell.get("cell_id") for cell in invalid_infeasible],
        "missing_selective_fallback_cell_ids": [
            cell.get("cell_id") for cell in missing_selective_fallback
        ],
        "nonpassed_selective_cell_ids": [cell.get("cell_id") for cell in nonpassed_selective],
        "missing_cell_ids": missing,
        "cell_identity_mismatches": identity_mismatches,
        "sample_trace_consistent_across_cells": trace_consistent,
        "sample_trace_aggregate_sha256": next(iter(trace_aggregates)) if trace_consistent else None,
        "best_observed_cell_id": ranked[0]["cell_id"] if ranked else None,
        "ranking_by_mean_step_seconds": [
            {
                "cell_id": cell["cell_id"],
                "mean_step_seconds": cell["timing"]["mean_step_seconds"],
                "average_gpu_utilization_percent": cell["gpu"]["average_utilization_percent"],
                "peak_memory_used_fraction": cell["gpu"]["peak_memory_used_fraction"],
            }
            for cell in ranked
        ],
        "cells": cells,
    }
    _write_json_atomic(output, payload, exclusive=True)
    return payload


def select_best_cell(
    metrics_paths: Sequence[Path],
    *,
    decision: str,
    output: Path,
) -> dict[str, Any]:
    """Apply the documented adaptive-probe selection rule."""

    candidates: list[Mapping[str, Any]] = []
    for path in metrics_paths:
        payload = _load_json_file(path, f"selection candidate {path}")
        if payload.get("status") == "passed":
            candidates.append(payload)
    if not candidates:
        raise ContractError(f"adaptive decision {decision!r} has no passed candidate")
    trace_aggregates = {
        _require_mapping(cell.get("sample_trace"), "candidate sample trace").get("aggregate_sha256")
        for cell in candidates
    }
    if len(trace_aggregates) != 1:
        raise ContractError(
            f"adaptive decision {decision!r} cannot compare cells with different sample traces"
        )
    ranked = sorted(
        candidates,
        key=lambda cell: (
            float(_require_mapping(cell.get("timing"), "candidate timing")["mean_step_seconds"]),
            str(cell["cell_id"]),
        ),
    )
    selected = ranked[0]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "selected",
        "selected_at": _utc_now(),
        "decision": decision,
        "selection_rule": (
            "minimum mean rank-consolidated optimizer-step wall seconds among passed candidates; "
            "lexicographic cell_id tie-break"
        ),
        "selected_cell_id": selected["cell_id"],
        "selected_workers_per_rank": selected["workers_per_rank"],
        "selected_node_total_workers": selected["node_total_workers"],
        "selected_storage": selected["storage"],
        "selected_activation_checkpointing": selected["activation_checkpointing"],
        "candidates": [
            {
                "cell_id": cell["cell_id"],
                "mean_step_seconds": cell["timing"]["mean_step_seconds"],
                "workers_per_rank": cell["workers_per_rank"],
                "storage": cell["storage"],
                "activation_checkpointing": cell["activation_checkpointing"],
            }
            for cell in ranked
        ],
    }
    _write_json_atomic(output, payload, exclusive=True)
    return payload


def write_cell_contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.video_tokenizer_target != METADATA_ONLY_VIDEO_TOKENIZER_TARGET:
        raise ContractError(
            "latent-cache throughput cells require the audited metadata-only video tokenizer target"
        )
    if args.activation_checkpointing == "none":
        if args.cell_timeout_seconds < 600:
            raise ContractError("the optional no-AC cell requires a timeout of at least 600 seconds")
    elif args.cell_timeout_seconds != 0:
        raise ContractError("selective-AC comparison cells must record cell_timeout_seconds=0")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "created_at": _utc_now(),
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "cell_id": args.cell_id,
        "world_size": args.world_size,
        "workers_per_rank": args.workers,
        "node_total_workers": args.workers * args.world_size,
        "storage": args.storage,
        "activation_checkpointing": args.activation_checkpointing,
        "video_tokenizer": {
            "target": args.video_tokenizer_target,
            "metadata_only": args.video_tokenizer_target == METADATA_ONLY_VIDEO_TOKENIZER_TARGET,
        },
        "cell_timeout_seconds": args.cell_timeout_seconds,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "total_steps": args.warmup_steps + args.measure_steps,
        "latent_cache_root": str(Path(args.latent_cache_root).resolve(strict=True)),
        "latent_cache_manifest_sha256": args.latent_cache_manifest_sha256,
        "fixed_index_trace_path": str(Path(args.fixed_index_trace_path).resolve(strict=True)),
        "fixed_index_trace_sha256": args.fixed_index_trace_sha256,
        "fixed_index_trace_manifest_sha256": args.fixed_index_trace_manifest_sha256,
        "wan_vae_path": str(Path(args.wan_vae_path).resolve(strict=True)),
        "wan_vae_sha256": args.wan_vae_sha256,
        "resume_mode": args.resume_mode,
        "resume_checkpoint": args.resume_checkpoint or None,
        "data_stream_resume": "restarted_not_cursor_resumed",
        "source_root": str(Path(args.source_root).resolve(strict=True)),
        "source_commit": args.source_commit,
        "snapshot_manifest_sha256": args.snapshot_manifest_sha256,
        "runner_sha256": args.runner_sha256,
        "helper_sha256": args.helper_sha256,
        "node_evidence": str(Path(args.node_evidence).resolve(strict=True)),
    }
    _write_json_atomic(Path(args.output), payload, exclusive=True)
    return payload


def purge_probe_checkpoints(cell_root_value: str | Path, run_dir_value: str | Path, output: Path) -> dict[str, Any]:
    """Delete only the checkpoint directory created inside one exclusive probe cell."""

    cell_root = Path(cell_root_value).resolve(strict=True)
    run_dir_input = Path(run_dir_value)
    if run_dir_input.is_symlink():
        raise ContractError(f"probe run directory must not be a symlink: {run_dir_input}")
    run_dir = run_dir_input.resolve(strict=False)
    if cell_root not in run_dir.parents:
        raise ContractError(f"probe run directory escapes its cell root: {run_dir}")
    expected_tail = ("cosmos3_action_robocasa365", "action_sft")
    if tuple(run_dir.parts[-3:-1]) != expected_tail:
        raise ContractError(f"unexpected probe run directory layout: {run_dir}")
    checkpoints = run_dir / "checkpoints"
    removed_files = 0
    removed_bytes = 0
    if checkpoints.exists() or checkpoints.is_symlink():
        if checkpoints.is_symlink() or not checkpoints.is_dir():
            raise ContractError(f"probe checkpoints path must be a real directory: {checkpoints}")
        for path in checkpoints.rglob("*"):
            if path.is_file() and not path.is_symlink():
                removed_files += 1
                removed_bytes += path.stat().st_size
        shutil.rmtree(checkpoints)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "removed" if removed_files else "absent",
        "recorded_at": _utc_now(),
        "cell_root": str(cell_root),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoints),
        "removed_regular_files": removed_files,
        "removed_regular_bytes": removed_bytes,
    }
    _write_json_atomic(output, payload, exclusive=True)
    return payload


def _print_payload(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _add_common_cell_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--storage", choices=("gpfs", "nvme"), required=True)
    parser.add_argument("--activation-checkpointing", choices=("selective", "none"), required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--measure-steps", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-cache", help="validate success, manifest, mappings, and shards")
    validate.add_argument("--root", required=True)
    validate.add_argument("--output")

    trace = subparsers.add_parser("validate-fixed-trace", help="validate the fixed 8-rank replay trace")
    trace.add_argument("--path", required=True)
    trace.add_argument("--output")

    prepare_trace = subparsers.add_parser(
        "prepare-fixed-trace",
        help="atomically generate or dataset-bind-validate the fixed 8-rank replay trace",
    )
    prepare_trace.add_argument("--path", required=True)
    prepare_trace.add_argument("--dataset-root", required=True)
    prepare_trace.add_argument("--source-root", required=True)
    prepare_trace.add_argument("--output")

    stage = subparsers.add_parser("stage-nvme", help="rank0 copy/validate followed by an eight-rank barrier")
    stage.add_argument("--source", required=True)
    stage.add_argument("--destination", required=True)
    stage.add_argument("--evidence-dir", required=True)

    node = subparsers.add_parser("capture-node", help="capture rjob, node, and exact eight-H200 evidence")
    node.add_argument("--output", required=True)
    node.add_argument("--expected-gpus", type=int, default=8)
    node.add_argument("--rjob-show-name", required=True)
    node.add_argument("--minimum-cpus", type=int, default=160)
    node.add_argument("--requested-cpus", type=int, default=192)

    sampler = subparsers.add_parser("sample-gpu", help="write timestamped GPU telemetry NDJSON")
    sampler.add_argument("--output", required=True)
    sampler.add_argument("--interval-seconds", type=int, default=2)
    sampler.add_argument("--expected-gpus", type=int, default=8)

    cell = subparsers.add_parser("summarize-cell", help="write one independent cell metrics JSON")
    _add_common_cell_summary_arguments(cell)
    cell.add_argument("--log", required=True)
    cell.add_argument("--step-ndjson", required=True)
    cell.add_argument("--gpu-samples", required=True)
    cell.add_argument("--memory-gate-json", required=True)
    cell.add_argument("--training-exit-code", type=int, required=True)
    cell.add_argument("--expected-gpus", type=int, default=8)
    cell.add_argument("--output", required=True)

    matrix = subparsers.add_parser("summarize-matrix", help="summarize the complete explicit matrix")
    matrix.add_argument("--run-root", required=True)
    matrix.add_argument("--expected-cell-id", action="append", required=True)
    matrix.add_argument("--output", required=True)

    select = subparsers.add_parser("select-best-cell", help="apply the recorded adaptive selection rule")
    select.add_argument("--metrics", action="append", required=True)
    select.add_argument("--decision", required=True)
    select.add_argument("--output", required=True)

    contract = subparsers.add_parser("write-cell-contract", help="write an immutable planned cell contract")
    _add_common_cell_summary_arguments(contract)
    contract.add_argument("--experiment-id", required=True)
    contract.add_argument("--run-id", required=True)
    contract.add_argument("--world-size", type=int, required=True)
    contract.add_argument("--video-tokenizer-target", required=True)
    contract.add_argument("--cell-timeout-seconds", type=int, required=True)
    contract.add_argument("--latent-cache-root", required=True)
    contract.add_argument("--latent-cache-manifest-sha256", required=True)
    contract.add_argument("--fixed-index-trace-path", required=True)
    contract.add_argument("--fixed-index-trace-sha256", required=True)
    contract.add_argument("--fixed-index-trace-manifest-sha256", required=True)
    contract.add_argument("--wan-vae-path", required=True)
    contract.add_argument("--wan-vae-sha256", required=True)
    contract.add_argument("--resume-mode", choices=("none", "r6_iter1000_model_optimizer"), required=True)
    contract.add_argument("--resume-checkpoint", default="")
    contract.add_argument("--source-root", required=True)
    contract.add_argument("--source-commit", required=True)
    contract.add_argument("--snapshot-manifest-sha256", required=True)
    contract.add_argument("--runner-sha256", required=True)
    contract.add_argument("--helper-sha256", required=True)
    contract.add_argument("--node-evidence", required=True)
    contract.add_argument("--output", required=True)

    purge = subparsers.add_parser("purge-probe-checkpoints", help="remove only probe-owned checkpoint payloads")
    purge.add_argument("--cell-root", required=True)
    purge.add_argument("--run-dir", required=True)
    purge.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-cache":
        payload = validate_cache(args.root)
        if args.output:
            _write_json_atomic(Path(args.output), payload, exclusive=True)
        _print_payload(payload)
        return 0
    if args.command == "validate-fixed-trace":
        payload = validate_fixed_index_trace(args.path)
        if args.output:
            _write_json_atomic(Path(args.output), payload, exclusive=True)
        _print_payload(payload)
        return 0
    if args.command == "prepare-fixed-trace":
        payload = prepare_fixed_index_trace(args.path, args.dataset_root, args.source_root)
        if args.output:
            _write_json_atomic(Path(args.output), payload, exclusive=True)
        _print_payload(payload)
        return 0
    if args.command == "stage-nvme":
        return _distributed_stage(args)
    if args.command == "capture-node":
        _print_payload(
            capture_node_evidence(
                Path(args.output),
                args.expected_gpus,
                args.rjob_show_name,
                args.minimum_cpus,
                args.requested_cpus,
            )
        )
        return 0
    if args.command == "sample-gpu":
        return sample_gpu(Path(args.output), args.interval_seconds, args.expected_gpus)
    if args.command == "summarize-cell":
        _print_payload(summarize_cell_fail_closed(args))
        return 0
    if args.command == "summarize-matrix":
        _print_payload(
            summarize_matrix(
                Path(args.run_root),
                Path(args.output),
                args.expected_cell_id,
            )
        )
        return 0
    if args.command == "select-best-cell":
        _print_payload(
            select_best_cell(
                [Path(value) for value in args.metrics],
                decision=args.decision,
                output=Path(args.output),
            )
        )
        return 0
    if args.command == "write-cell-contract":
        _print_payload(write_cell_contract(args))
        return 0
    if args.command == "purge-probe-checkpoints":
        _print_payload(
            purge_probe_checkpoints(
                args.cell_root,
                args.run_dir,
                Path(args.output),
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
