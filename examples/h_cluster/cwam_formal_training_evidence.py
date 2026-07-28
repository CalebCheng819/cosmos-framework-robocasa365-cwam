#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed evidence checks for CWAM/Cosmos3-Nano H-cluster runs.

The production wrapper uses this module after each expensive phase.  Keeping
the assertions in importable Python makes the formal acceptance contract unit
testable instead of hiding it in shell heredocs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_BATCH_PER_RANK = 128
_GRAD_ACCUM = 2
_ACTION_TOKENS_PER_SAMPLE = 16
_FORMAL_MIN_FREE_BYTES = 5 * 1024**4
_FORMAL_ESTIMATED_REQUIRED_BYTES = 4 * 1024**4
_FORMAL_OUTPUT_ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-formal")
_FORMAL_LOG_ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/logs/cwam-cosmos3-formal")
_FORMAL_QUOTA_GROUP = "eailabagent_gpu"
_FORMAL_EXCLUDED_NODE_TAG = "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"
_FORMAL_MOUNTS = (
    "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs",
    "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public",
)
_DCP_COMPONENTS = ("model", "optim", "scheduler", "trainer")
_LEGAL_ZERO_DCP_SHARD_RE = re.compile(r"^__[0-7]_0\.distcp$")
_CONTINUATION_INITIAL_ITERATION = 1_000
_CONTINUATION_DATA_STREAM_RESUME = "restarted_from_beginning_not_exact_loader_cursor"
_CONTINUATION_CONTRACT_KIND = "cwam_cosmos3_formal_continuation_inputs"
_PRECOMPUTED_WAN_TOKENIZER_TARGET = (
    "cosmos_framework.model.generator.tokenizers.precomputed_wan2pt2.PrecomputedWan2pt2LatentTokenizer"
)
_FORBIDDEN_LOG_MARKERS = (
    "Traceback (most recent call last):",
    "CUDA out of memory",
    "OutOfMemoryError",
    "ncclUnhandledCudaError",
    '"status": "failed"',
)
_VALIDATION_RE = re.compile(
    r"\[val\] number of unique url and key: (?P<unique>\d+) / (?P<total>\d+); "
    r"avg_final_loss: (?P<loss>[^;\s]+); unstable_count: (?P<unstable>\d+); "
    r"iteration: (?P<iteration>\d+)"
)


@dataclass(frozen=True)
class TrainingProtocol:
    """Topology-specific scientific contract for one formal experiment."""

    experiment_family: str
    world_size: int
    learning_rate: float
    max_iter: int
    warm_up_steps: int
    cycle_length: int
    validation_iter: int
    max_val_iter: int
    save_iter: int

    @property
    def global_batch(self) -> int:
        return self.world_size * _BATCH_PER_RANK * _GRAD_ACCUM

    @property
    def validation_iterations(self) -> list[int]:
        return [0, *range(self.validation_iter, self.max_iter + 1, self.validation_iter)]

    @property
    def checkpoint_iterations(self) -> list[int]:
        return list(range(self.save_iter, self.max_iter + 1, self.save_iter))


_PROTOCOLS = {
    8: TrainingProtocol(
        experiment_family="CWAM-E35",
        world_size=8,
        learning_rate=5.0e-5,
        max_iter=10_000,
        warm_up_steps=500,
        cycle_length=16_000,
        validation_iter=500,
        max_val_iter=8,
        save_iter=500,
    ),
    6: TrainingProtocol(
        experiment_family="CWAM-E35A",
        world_size=6,
        learning_rate=3.75e-5,
        max_iter=13_340,
        warm_up_steps=667,
        cycle_length=21_344,
        validation_iter=667,
        max_val_iter=11,
        save_iter=667,
    ),
}


class EvidenceError(ValueError):
    """Raised when a formal training artifact violates the run contract."""


def _load_json_and_sha256(path: Path, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise EvidenceError(f"{label} is not a regular non-symlink file: {path}")
    before = path.stat()
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{label} is not valid JSON: {path}: {error}") from error
    after = path.stat()
    identity_before = (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_before != identity_after:
        raise EvidenceError(f"{label} changed while reading: {path}")
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must contain a JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return _load_json_and_sha256(path, label)[0]


def _sha256_file(path: Path, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise EvidenceError(f"{label} is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    identity_before = (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_before != identity_after:
        raise EvidenceError(f"{label} changed while hashing: {path}")
    return digest.hexdigest()


def _write_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def _load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file() or config_path.is_symlink():
        raise EvidenceError(f"run config is missing or unsafe: {config_path}")
    try:
        value = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise EvidenceError(f"run config is not valid YAML: {config_path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"run config must contain an object: {config_path}")
    return value


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise EvidenceError(f"{label} mismatch: expected={expected!r}, actual={actual!r}")


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object, got {type(value).__name__}")
    return value


def _require_exact_keys(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise EvidenceError(
            f"{label} key set mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def protocol_for_world_size(world_size: int) -> TrainingProtocol:
    try:
        return _PROTOCOLS[world_size]
    except KeyError as error:
        raise EvidenceError(f"formal world size must be 6 or 8, got {world_size}") from error


def _validate_run_identity(
    *,
    run_dir: Path,
    expected_name: str,
    experiment_id: str,
    protocol: TrainingProtocol,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", expected_name):
        raise EvidenceError(f"run name contains unsupported characters: {expected_name!r}")
    if not experiment_id.startswith(f"{protocol.experiment_family}-"):
        raise EvidenceError(
            f"{protocol.experiment_family} protocol requires a matching experiment ID, got {experiment_id!r}"
        )
    absolute = Path(os.path.abspath(run_dir))
    resolved = run_dir.resolve(strict=True)
    if run_dir != absolute or resolved != absolute or run_dir.is_symlink() or not run_dir.is_dir():
        raise EvidenceError(f"run directory must be a canonical non-symlink directory: {run_dir}")
    _require_equal(resolved.name, expected_name, "run directory name")
    return resolved


def _validate_protocol_scalars(config: dict[str, Any], protocol: TrainingProtocol) -> None:
    optimizer = _require_mapping(config.get("optimizer"), "optimizer config")
    scheduler = _require_mapping(config.get("scheduler"), "scheduler config")
    learning_rate = float(optimizer.get("lr", float("nan")))
    if not math.isclose(learning_rate, protocol.learning_rate, rel_tol=0.0, abs_tol=1.0e-12):
        raise EvidenceError(
            f"optimizer.lr mismatch for {protocol.experiment_family}: "
            f"expected={protocol.learning_rate}, actual={learning_rate}"
        )
    _require_equal(
        scheduler.get("warm_up_steps"),
        [protocol.warm_up_steps],
        f"{protocol.experiment_family} scheduler warmup",
    )
    _require_equal(
        scheduler.get("cycle_lengths"),
        [protocol.cycle_length],
        f"{protocol.experiment_family} scheduler cycle",
    )


def _validate_callback_config(config: dict[str, Any], *, gradient_every_n: int) -> None:
    callbacks = _require_mapping(
        _require_mapping(config.get("trainer"), "trainer config").get("callbacks"),
        "trainer callbacks",
    )
    gradient = _require_mapping(callbacks.get("action_gradient_evidence"), "gradient callback")
    _require_equal(
        gradient.get("families"),
        ["action2llm", "llm2action", "action_modality_embed"],
        "action-gradient configured families",
    )
    _require_equal(gradient.get("every_n"), gradient_every_n, "action-gradient frequency")
    _require_equal(gradient.get("fail_on_first_step"), True, "action-gradient first-step policy")

    memory = _require_mapping(callbacks.get("production_batch_memory_gate"), "memory-gate callback")
    memory_contract = {
        "max_peak_reserved_fraction": 0.94,
        "max_driver_used_fraction": 0.94,
        "expected_samples_per_microbatch": _BATCH_PER_RANK,
        "expected_grad_accum_iter": _GRAD_ACCUM,
        "expected_action_tokens_per_sample": _ACTION_TOKENS_PER_SAMPLE,
        "allowed_world_sizes": [6, 8],
        "required": True,
    }
    for key, expected in memory_contract.items():
        actual = memory.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise EvidenceError(f"memory-gate callback {key} mismatch: expected={expected}, actual={actual}")
        else:
            _require_equal(actual, expected, f"memory-gate callback {key}")


def _validate_common_config(
    config: dict[str, Any],
    *,
    world_size: int,
    max_iter: int,
    max_val_iter: int,
    save_iter: int,
    train_num_workers: int = 4,
    val_num_workers: int = 2,
) -> None:
    trainer = _require_mapping(config.get("trainer"), "trainer config")
    checkpoint = _require_mapping(config.get("checkpoint"), "checkpoint config")
    model = _require_mapping(
        _require_mapping(config.get("model"), "model config").get("config"),
        "model.config",
    )
    parallelism = _require_mapping(model.get("parallelism"), "model parallelism config")
    compile_config = _require_mapping(model.get("compile"), "model compile config")
    ema_config = _require_mapping(model.get("ema"), "model EMA config")
    train_loader = _require_mapping(config.get("dataloader_train"), "train loader config")
    val_loader = _require_mapping(config.get("dataloader_val"), "validation loader config")

    contracts = (
        (trainer.get("distributed_parallelism"), "fsdp", "distributed parallelism"),
        (trainer.get("max_iter"), max_iter, "trainer.max_iter"),
        (trainer.get("max_val_iter"), max_val_iter, "trainer.max_val_iter"),
        (trainer.get("grad_accum_iter"), _GRAD_ACCUM, "trainer.grad_accum_iter"),
        (trainer.get("run_validation"), True, "trainer.run_validation"),
        (trainer.get("run_validation_on_start"), True, "trainer.run_validation_on_start"),
        (checkpoint.get("strict_resume"), True, "checkpoint.strict_resume"),
        (checkpoint.get("save_iter"), save_iter, "checkpoint.save_iter"),
        (checkpoint.get("dcp_async_mode_enabled"), False, "checkpoint.dcp_async_mode_enabled"),
        (compile_config.get("enabled"), True, "model.compile.enabled"),
        (ema_config.get("enabled"), True, "model.ema.enabled"),
        (parallelism.get("data_parallel_shard_degree"), world_size, "data_parallel_shard_degree"),
        (parallelism.get("data_parallel_replicate_degree"), 1, "data_parallel_replicate_degree"),
        (parallelism.get("context_parallel_shard_degree"), 1, "context_parallel_shard_degree"),
        (parallelism.get("cfg_parallel_shard_degree"), 1, "cfg_parallel_shard_degree"),
        (train_loader.get("max_samples_per_batch"), _BATCH_PER_RANK, "train max_samples_per_batch"),
        (train_loader.get("max_sequence_length"), None, "train max_sequence_length"),
        (val_loader.get("max_samples_per_batch"), _BATCH_PER_RANK, "val max_samples_per_batch"),
    )
    for actual, expected, label in contracts:
        _require_equal(actual, expected, label)

    train_worker = _require_mapping(train_loader.get("dataloader"), "train DataLoader config")
    val_worker = _require_mapping(val_loader.get("dataloader"), "validation DataLoader config")
    for worker, expected_workers, label in (
        (train_worker, train_num_workers, "train"),
        (val_worker, val_num_workers, "validation"),
    ):
        _require_equal(worker.get("num_workers"), expected_workers, f"{label} num_workers")
        _require_equal(worker.get("persistent_workers"), True, f"{label} persistent_workers")
        _require_equal(worker.get("prefetch_factor"), 4, f"{label} prefetch_factor")


def _validate_dcp(checkpoint_dir: Path) -> dict[str, dict[str, int]]:
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise EvidenceError(f"DCP directory is missing or unsafe: {checkpoint_dir}")
    actual_components = sorted(path.name for path in checkpoint_dir.iterdir())
    _require_equal(actual_components, sorted(_DCP_COMPONENTS), f"DCP component set under {checkpoint_dir}")
    result: dict[str, dict[str, int]] = {}
    for component in _DCP_COMPONENTS:
        component_dir = checkpoint_dir / component
        if not component_dir.is_dir() or component_dir.is_symlink():
            raise EvidenceError(f"DCP component directory is missing or unsafe: {component_dir}")
        entries = list(component_dir.iterdir())
        unsafe = [
            path.name
            for path in entries
            if path.is_symlink() or not path.is_file() or (path.name != ".metadata" and path.suffix != ".distcp")
        ]
        if unsafe:
            raise EvidenceError(f"DCP component contains unsafe or unexpected entries: {component_dir}: {unsafe}")
        metadata = component_dir / ".metadata"
        if not metadata.is_file() or metadata.is_symlink() or metadata.stat().st_size <= 0:
            raise EvidenceError(f"missing non-empty DCP metadata: {metadata}")
        all_shards = sorted(path for path in component_dir.glob("*.distcp") if path.is_file() and not path.is_symlink())
        zero_shards = [path for path in all_shards if path.stat().st_size == 0]
        illegal_zero_shards = [
            path.name for path in zero_shards if _LEGAL_ZERO_DCP_SHARD_RE.fullmatch(path.name) is None
        ]
        if illegal_zero_shards:
            raise EvidenceError(
                f"DCP component contains an illegal zero-byte shard name: {component_dir}: {illegal_zero_shards}"
            )
        shards = [path for path in all_shards if path.stat().st_size > 0]
        if not shards:
            raise EvidenceError(f"missing non-empty DCP shards: {component_dir}")
        try:
            from torch.distributed.checkpoint import FileSystemReader

            dcp_metadata = FileSystemReader(component_dir).read_metadata()
        except Exception as error:
            raise EvidenceError(f"DCP metadata is not parseable for {component_dir}: {error}") from error
        state_dict_metadata = getattr(dcp_metadata, "state_dict_metadata", None)
        storage_data = getattr(dcp_metadata, "storage_data", None)
        if not isinstance(state_dict_metadata, dict) or not state_dict_metadata:
            raise EvidenceError(f"DCP metadata has no state-dict entries: {component_dir}")
        if not isinstance(storage_data, dict) or not storage_data:
            raise EvidenceError(f"DCP metadata has no storage entries: {component_dir}")
        referenced_shards = set()
        for storage_info in storage_data.values():
            relative_path = getattr(storage_info, "relative_path", None)
            if (
                not isinstance(relative_path, str)
                or Path(relative_path).name != relative_path
                or not relative_path.endswith(".distcp")
            ):
                raise EvidenceError(
                    f"DCP metadata references an unsafe storage path under {component_dir}: {relative_path!r}"
                )
            referenced_shards.add(relative_path)
        actual_shards = {path.name for path in shards}
        _require_equal(actual_shards, referenced_shards, f"DCP metadata/storage shard set for {component_dir}")
        result[component] = {
            "metadata_bytes": metadata.stat().st_size,
            "distcp_files": len(shards),
            "distcp_bytes": sum(path.stat().st_size for path in shards),
            "zero_byte_rank_shards": [path.name for path in zero_shards],
            "state_dict_entries": len(state_dict_metadata),
            "storage_entries": len(storage_data),
        }
    return result


def _validate_latest_checkpoint(run_dir: Path, expected_iteration: int) -> tuple[Path, dict[str, dict[str, int]]]:
    latest = run_dir / "checkpoints" / "latest_checkpoint.txt"
    expected_name = f"iter_{expected_iteration:09d}"
    if not latest.is_file() or latest.is_symlink():
        raise EvidenceError(f"latest checkpoint pointer is missing or unsafe: {latest}")
    _require_equal(latest.read_text().strip(), expected_name, "latest checkpoint pointer")
    checkpoint_dir = latest.parent / expected_name
    return checkpoint_dir, _validate_dcp(checkpoint_dir)


def _require_exact_regular_files(directory: Path, expected_names: list[str], label: str) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise EvidenceError(f"{label} directory is missing or unsafe: {directory}")
    actual_names: list[str] = []
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise EvidenceError(f"{label} contains a non-regular or symlink entry: {path}")
        actual_names.append(path.name)
    _require_equal(sorted(actual_names), sorted(expected_names), f"{label} file set")


def _require_exact_checkpoint_root(checkpoint_root: Path, expected_iterations: list[int]) -> None:
    if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
        raise EvidenceError(f"checkpoint root is missing or unsafe: {checkpoint_root}")
    expected_directories = {f"iter_{iteration:09d}" for iteration in expected_iterations}
    actual_directories: set[str] = set()
    latest_seen = False
    for path in checkpoint_root.iterdir():
        if path.name == "latest_checkpoint.txt":
            if path.is_symlink() or not path.is_file():
                raise EvidenceError(f"latest checkpoint pointer is unsafe: {path}")
            latest_seen = True
            continue
        if path.is_symlink() or not path.is_dir():
            raise EvidenceError(f"checkpoint root contains an unsafe or unexpected entry: {path}")
        actual_directories.add(path.name)
    if not latest_seen:
        raise EvidenceError(f"checkpoint root has no latest_checkpoint.txt: {checkpoint_root}")
    _require_equal(actual_directories, expected_directories, "checkpoint directory set")


def _validate_log(text: str, *, required: tuple[str, ...], extra_forbidden: tuple[str, ...] = ()) -> None:
    missing = [marker for marker in required if marker not in text]
    forbidden = [marker for marker in (*_FORBIDDEN_LOG_MARKERS, *extra_forbidden) if marker in text]
    if missing or forbidden:
        raise EvidenceError(f"log contract failed: missing={missing}, forbidden={forbidden}")


def _unique_position(text: str, marker: str, label: str) -> int:
    count = text.count(marker)
    if count != 1:
        raise EvidenceError(f"{label} must occur exactly once, got {count}: {marker!r}")
    return text.index(marker)


def _validate_resume_log(text: str, *, run_dir: Path, world_size: int) -> str:
    expected_resume_path = str((run_dir / "checkpoints" / "iter_000000001").resolve())
    resume_marker = (
        f"Resuming ckpt {expected_resume_path} with keys: ['dataloader', 'model', 'optim', 'scheduler', 'trainer']"
    )
    positions = {
        "replay_warning": _unique_position(
            text,
            "WARNING: resuming model/optim/scheduler/trainer with a restarted data stream.",
            "data-stream replay warning",
        ),
        "resume": _unique_position(text, resume_marker, "resume declaration"),
        "model": _unique_position(text, "- Loading the model...", "model load marker"),
        "optimizer": _unique_position(text, "- Loading the optimizer...", "optimizer load marker"),
        "scheduler": _unique_position(text, "- Loading the scheduler...", "scheduler load marker"),
        "trainer": _unique_position(text, "- Loading the trainer...", "trainer load marker"),
        "loaded": _unique_position(
            text,
            f"Loaded checkpoint from {expected_resume_path} in iteration 1",
            "completed checkpoint-load marker",
        ),
        "fsdp": _unique_position(
            text,
            "Distributed parallelism mode: fsdp",
            "distributed-parallelism marker",
        ),
        "done": _unique_position(text, "Done with training.", "training completion marker"),
    }
    if text.count("Resuming ckpt ") != 1 or text.count("Loaded checkpoint from ") != 1:
        raise EvidenceError("resume log contains multiple checkpoint attempts")
    skip_positions = []
    for rank in range(world_size):
        marker = f"[RANK {rank}] Checkpoint {expected_resume_path}/dataloader does not exist, skip loading dataloader."
        skip_positions.append(_unique_position(text, marker, f"rank {rank} dataloader replay acknowledgement"))
    if text.count("skip loading dataloader.") != world_size:
        raise EvidenceError(
            f"resume log must acknowledge one missing dataloader state per rank, "
            f"expected={world_size}, actual={text.count('skip loading dataloader.')}"
        )
    ordered = (
        positions["replay_warning"]
        < positions["resume"]
        < min(skip_positions)
        <= max(skip_positions)
        < positions["model"]
        < positions["optimizer"]
        < positions["scheduler"]
        < positions["trainer"]
        < positions["loaded"]
        < positions["fsdp"]
        < positions["done"]
    )
    if not ordered:
        raise EvidenceError(f"resume log checkpoint state machine is out of order: {positions}")
    _validate_log(
        text,
        required=(),
        extra_forbidden=(
            "Training from scratch.",
            "Unstable val loss",
            "No dataloader checkpoint found",
            "- Loading the dataloader ",
        ),
    )
    return expected_resume_path


def _validation_summaries(text: str) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for match in _VALIDATION_RE.finditer(text):
        loss = float(match.group("loss"))
        rows.append(
            {
                "unique": int(match.group("unique")),
                "total": int(match.group("total")),
                "loss": loss,
                "unstable": int(match.group("unstable")),
                "iteration": int(match.group("iteration")),
            }
        )
    return rows


def _validate_validation_summaries(
    text: str,
    *,
    expected_iterations: list[int],
    expected_samples: int,
) -> list[dict[str, int | float]]:
    rows = _validation_summaries(text)
    actual_iterations = [int(row["iteration"]) for row in rows]
    if actual_iterations != expected_iterations:
        raise EvidenceError(
            f"validation iteration contract failed: expected={expected_iterations}, actual={actual_iterations}"
        )
    for row in rows:
        if int(row["unique"]) != expected_samples or int(row["total"]) != expected_samples:
            raise EvidenceError(
                "validation coverage mismatch at iteration "
                f"{row['iteration']}: unique={row['unique']}, total={row['total']}, "
                f"expected={expected_samples}"
            )
        loss = float(row["loss"])
        if not math.isfinite(loss) or loss <= 0:
            raise EvidenceError(f"invalid validation loss at iteration {row['iteration']}: {loss}")
        if int(row["unstable"]) != 0:
            raise EvidenceError(f"unstable validation loss at iteration {row['iteration']}: {row['unstable']}")
    return rows


def _validate_gradient_report(path: Path, *, world_size: int, iteration: int) -> dict[str, Any]:
    report = _load_json(path, f"action-gradient evidence for iteration {iteration}")
    _require_exact_keys(
        report,
        {"schema_version", "iteration", "world_size", "gradient_stage", "families", "status", "losses"},
        "action-gradient report",
    )
    _require_equal(report.get("schema_version"), 1, "action-gradient schema")
    _require_equal(report.get("status"), "passed", "action-gradient status")
    _require_equal(report.get("iteration"), iteration, "action-gradient iteration")
    _require_equal(report.get("world_size"), world_size, "action-gradient world size")
    _require_equal(
        report.get("gradient_stage"),
        "after_backward_before_callback_clipping",
        "action-gradient collection stage",
    )
    if report.get("loss_failures"):
        raise EvidenceError(f"action-gradient evidence contains loss failures: {report['loss_failures']!r}")
    families = _require_mapping(report.get("families"), "action-gradient families")
    _require_exact_keys(
        families,
        {"action2llm", "llm2action", "action_modality_embed"},
        "action-gradient family set",
    )
    for family in ("action2llm", "llm2action", "action_modality_embed"):
        stats = _require_mapping(families.get(family), f"action-gradient family {family}")
        _require_exact_keys(
            stats,
            {
                "parameter_elements",
                "trainable_elements",
                "optimizer_elements",
                "gradient_elements",
                "nonfinite_gradient_elements",
                "nonzero_gradient_elements",
                "gradient_l2_norm",
                "gradient_abs_max",
            },
            f"action-gradient stats for {family}",
        )
        parameter_elements = int(stats.get("parameter_elements", -1))
        if parameter_elements <= 0:
            raise EvidenceError(f"{family} has no parameter elements")
        for key in ("trainable_elements", "optimizer_elements", "gradient_elements"):
            _require_equal(int(stats.get(key, -1)), parameter_elements, f"{family}.{key}")
        _require_equal(int(stats.get("nonfinite_gradient_elements", -1)), 0, f"{family} nonfinite gradients")
        if int(stats.get("nonzero_gradient_elements", 0)) <= 0:
            raise EvidenceError(f"{family} has no nonzero gradients")
        gradient_norm = float(stats.get("gradient_l2_norm", float("nan")))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise EvidenceError(f"{family} has invalid gradient norm: {gradient_norm}")
        gradient_max = float(stats.get("gradient_abs_max", float("nan")))
        if not math.isfinite(gradient_max) or gradient_max <= 0:
            raise EvidenceError(f"{family} has invalid maximum gradient: {gradient_max}")
    losses = _require_mapping(report.get("losses"), "action-gradient losses")
    _require_exact_keys(
        losses,
        {
            "scope",
            "final_microbatch_total_loss_rank_mean",
            "final_microbatch_flow_matching_loss_action_rank_mean",
        },
        "action-gradient loss evidence",
    )
    _require_equal(
        losses.get("scope"),
        "final_microbatch_at_optimizer_boundary",
        "action-gradient loss scope",
    )
    for key in (
        "final_microbatch_total_loss_rank_mean",
        "final_microbatch_flow_matching_loss_action_rank_mean",
    ):
        value = float(losses.get(key, float("nan")))
        if not math.isfinite(value) or value <= 0:
            raise EvidenceError(f"invalid {key}: {value}")
    return report


def _validate_memory_report(
    path: Path,
    *,
    world_size: int,
    iteration: int,
    included_initial_validation: bool,
    included_checkpoint_save: bool,
) -> dict[str, Any]:
    report = _load_json(path, f"production memory evidence for iteration {iteration}")
    _require_exact_keys(
        report,
        {
            "schema_version",
            "status",
            "recorded_at",
            "scope",
            "included_phases",
            "start_iteration",
            "observed_iteration",
            "world_size",
            "production_batch_contract",
            "thresholds",
            "per_rank",
            "violations",
        },
        "production memory report",
    )
    _require_equal(report.get("schema_version"), 1, "production memory schema")
    _require_equal(report.get("status"), "passed", "production memory status")
    _require_equal(
        report.get("scope"),
        "first_production_optimizer_boundary_after_checkpoint_load",
        "production memory scope",
    )
    if not isinstance(report.get("recorded_at"), str) or not report["recorded_at"]:
        raise EvidenceError("production memory recorded_at must be a non-empty string")
    _require_equal(report.get("world_size"), world_size, "production memory world size")
    _require_equal(report.get("observed_iteration"), iteration, "production memory iteration")
    _require_equal(report.get("start_iteration"), iteration - 1, "production memory start iteration")
    expected_phases = [
        "resident_model_optimizer_checkpoint_state",
        f"{_GRAD_ACCUM}_training_microbatches",
        "first_optimizer_scheduler_ema_step",
    ]
    if included_initial_validation:
        expected_phases.insert(1, "initial_ema_validation")
    if included_checkpoint_save:
        expected_phases.append("first_step_dcp_save")
    _require_equal(report.get("included_phases"), expected_phases, "production memory included phases")
    _require_equal(report.get("violations"), [], "production memory violations")
    contract = _require_mapping(report.get("production_batch_contract"), "production batch contract")
    _require_exact_keys(
        contract,
        {
            "expected_samples_per_microbatch",
            "expected_grad_accum_iter",
            "expected_action_tokens_per_sample",
            "allowed_world_sizes",
        },
        "production batch contract",
    )
    _require_equal(
        contract.get("expected_samples_per_microbatch"),
        _BATCH_PER_RANK,
        "expected samples per microbatch",
    )
    _require_equal(contract.get("expected_grad_accum_iter"), _GRAD_ACCUM, "expected grad accumulation")
    _require_equal(
        contract.get("expected_action_tokens_per_sample"),
        _ACTION_TOKENS_PER_SAMPLE,
        "expected action tokens per sample",
    )
    _require_equal(contract.get("allowed_world_sizes"), [6, 8], "allowed production world sizes")
    thresholds = _require_mapping(report.get("thresholds"), "production memory thresholds")
    _require_exact_keys(
        thresholds,
        {"max_peak_reserved_fraction", "max_driver_used_fraction"},
        "production memory thresholds",
    )
    for key in ("max_peak_reserved_fraction", "max_driver_used_fraction"):
        value = float(thresholds.get(key, float("nan")))
        if not math.isclose(value, 0.94, rel_tol=0.0, abs_tol=1.0e-12):
            raise EvidenceError(f"production memory threshold changed: {key}={value}")
    rows = report.get("per_rank")
    if not isinstance(rows, list) or len(rows) != world_size:
        raise EvidenceError(f"memory evidence expected {world_size} rank rows, got {rows!r}")
    actual_ranks = sorted(int(_require_mapping(row, "per-rank memory evidence").get("rank", -1)) for row in rows)
    _require_equal(actual_ranks, list(range(world_size)), "production memory rank set")
    for row in rows:
        row = _require_mapping(row, "per-rank memory evidence")
        expected_row_keys = {
            "rank",
            "device_index",
            "total_memory_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "current_allocated_bytes",
            "current_reserved_bytes",
            "driver_free_bytes",
            "driver_used_bytes",
            "observed_microbatch_count",
            "peak_allocated_fraction",
            "peak_reserved_fraction",
            "driver_used_fraction",
        }
        observation_fields = (
            "input_sample_count",
            "unique_identity_count",
            "model_batch_size",
            "und_token_length",
            "gen_token_length",
            "vision_token_length",
            "action_token_length",
            "input_tensor_bytes",
        )
        for microbatch in range(_GRAD_ACCUM):
            expected_row_keys.update(f"microbatch_{microbatch}_{field}" for field in observation_fields)
        _require_exact_keys(row, expected_row_keys, f"production memory row for rank {row.get('rank')}")

        rank = int(row["rank"])
        _require_equal(int(row["device_index"]), rank, f"CUDA device for rank {rank}")
        total = int(row["total_memory_bytes"])
        if total <= 0:
            raise EvidenceError(f"rank {rank} has invalid total memory: {total}")
        integer_counters = (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "current_allocated_bytes",
            "current_reserved_bytes",
            "driver_free_bytes",
            "driver_used_bytes",
        )
        for key in integer_counters:
            value = int(row[key])
            if not 0 <= value <= total:
                raise EvidenceError(f"rank {rank} has invalid {key}: {value} outside [0, {total}]")
        if int(row["peak_allocated_bytes"]) < int(row["current_allocated_bytes"]):
            raise EvidenceError(f"rank {rank} peak allocated memory is below current allocated memory")
        if int(row["peak_reserved_bytes"]) < int(row["current_reserved_bytes"]):
            raise EvidenceError(f"rank {rank} peak reserved memory is below current reserved memory")
        if int(row["peak_reserved_bytes"]) < int(row["peak_allocated_bytes"]):
            raise EvidenceError(f"rank {rank} peak reserved memory is below peak allocated memory")
        if int(row["current_reserved_bytes"]) < int(row["current_allocated_bytes"]):
            raise EvidenceError(f"rank {rank} current reserved memory is below current allocated memory")
        _require_equal(
            int(row["driver_free_bytes"]) + int(row["driver_used_bytes"]),
            total,
            f"rank {rank} driver memory accounting",
        )
        expected_fractions = {
            "peak_allocated_fraction": int(row["peak_allocated_bytes"]) / total,
            "peak_reserved_fraction": int(row["peak_reserved_bytes"]) / total,
            "driver_used_fraction": int(row["driver_used_bytes"]) / total,
        }
        for key, expected_fraction in expected_fractions.items():
            actual_fraction = float(row[key])
            if not math.isfinite(actual_fraction) or not math.isclose(
                actual_fraction,
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise EvidenceError(
                    f"rank {rank} has invalid {key}: expected={expected_fraction}, actual={actual_fraction}"
                )
        if float(row.get("peak_reserved_fraction", 1.0)) >= 0.94:
            raise EvidenceError(f"rank {row.get('rank')} exceeded peak-reserved memory threshold")
        if float(row.get("driver_used_fraction", 1.0)) >= 0.94:
            raise EvidenceError(f"rank {row.get('rank')} exceeded driver-used memory threshold")
        _require_equal(row.get("observed_microbatch_count"), _GRAD_ACCUM, "observed microbatch count")
        for microbatch in range(_GRAD_ACCUM):
            prefix = f"microbatch_{microbatch}"
            for key in ("input_sample_count", "unique_identity_count", "model_batch_size"):
                _require_equal(row.get(f"{prefix}_{key}"), _BATCH_PER_RANK, f"{prefix}_{key}")
            for key in ("und_token_length", "gen_token_length", "vision_token_length"):
                if int(row.get(f"{prefix}_{key}", 0)) <= 0:
                    raise EvidenceError(f"{prefix}_{key} must be positive")
            if int(row.get(f"{prefix}_input_tensor_bytes", 0)) <= 0:
                raise EvidenceError(f"{prefix}_input_tensor_bytes must be positive")
            _require_equal(
                row.get(f"{prefix}_action_token_length"),
                _BATCH_PER_RANK * _ACTION_TOKENS_PER_SAMPLE,
                f"{prefix}_action_token_length",
            )
    return report


def _validate_sampled_memory(path: Path, *, world_size: int, label: str) -> dict[str, Any]:
    sampled_memory = _load_json(path, label)
    _require_equal(sampled_memory.get("schema_version"), 1, f"{label} schema")
    _require_equal(sampled_memory.get("status"), "passed", f"{label} status")
    _require_equal(sampled_memory.get("gpu_count"), world_size, f"{label} GPU count")
    if int(sampled_memory.get("sample_count", 0)) < world_size:
        raise EvidenceError(f"{label} has too few samples: {sampled_memory.get('sample_count')!r}")
    maximum_allowed = float(sampled_memory.get("maximum_allowed_fraction", float("nan")))
    maximum_observed = float(sampled_memory.get("max_memory_used_fraction", float("nan")))
    if not math.isclose(maximum_allowed, 0.94, rel_tol=0.0, abs_tol=1.0e-12):
        raise EvidenceError(f"{label} threshold changed: {maximum_allowed}")
    if not math.isfinite(maximum_observed) or not 0 <= maximum_observed < maximum_allowed:
        raise EvidenceError(f"invalid {label} peak: {maximum_observed}")
    per_gpu = _require_mapping(sampled_memory.get("per_gpu"), f"{label} per-GPU evidence")
    if len(per_gpu) != world_size:
        raise EvidenceError(f"{label} expected {world_size} per-GPU rows, got {len(per_gpu)}")
    indices = sorted(int(_require_mapping(row, f"{label} GPU row").get("index", -1)) for row in per_gpu.values())
    _require_equal(indices, list(range(world_size)), f"{label} GPU index set")
    for uuid, raw_row in per_gpu.items():
        row = _require_mapping(raw_row, f"{label} GPU {uuid}")
        if not isinstance(uuid, str) or not uuid:
            raise EvidenceError(f"{label} contains an invalid GPU UUID: {uuid!r}")
        if "H200" not in str(row.get("name", "")):
            raise EvidenceError(f"{label} contains a non-H200 GPU: {row.get('name')!r}")
        if int(row.get("sample_count", 0)) < 1:
            raise EvidenceError(f"{label} GPU {uuid} has no samples")
        peak = float(row.get("max_memory_used_fraction", float("nan")))
        if not math.isfinite(peak) or not 0 <= peak < maximum_allowed:
            raise EvidenceError(f"{label} GPU {uuid} has invalid peak memory fraction: {peak}")
    return sampled_memory


def verify_probe(
    *,
    run_dir: Path,
    log_path: Path,
    cpu_batch_path: Path,
    sampled_memory_path: Path,
    world_size: int,
    inductor_cache: Path,
    experiment_id: str,
    formal_run_name: str,
) -> dict[str, Any]:
    protocol = protocol_for_world_size(world_size)
    _require_equal(formal_run_name, experiment_id, "formal run name")
    probe_run_name = f"{formal_run_name}-prod128-probe"
    run_dir = _validate_run_identity(
        run_dir=run_dir,
        expected_name=probe_run_name,
        experiment_id=experiment_id,
        protocol=protocol,
    )
    config = _load_config(run_dir)
    _validate_common_config(config, world_size=world_size, max_iter=1, max_val_iter=1, save_iter=1)
    _validate_protocol_scalars(config, protocol)
    _require_equal(
        config["trainer"].get("validation_iter"),
        protocol.validation_iter,
        "production-probe validation frequency",
    )
    _validate_callback_config(config, gradient_every_n=1)

    cpu_batch = _load_json(cpu_batch_path, "CPU production-batch evidence")
    _require_equal(cpu_batch.get("status"), "passed", "CPU production-batch status")
    _require_equal(cpu_batch.get("sample_count"), _BATCH_PER_RANK, "CPU production-batch sample count")
    if int(cpu_batch.get("estimated_tokens", 0)) <= 0 or int(cpu_batch.get("tensor_bytes", 0)) <= 0:
        raise EvidenceError("CPU production-batch token/tensor evidence is not positive")

    memory_path = run_dir / "production_batch_memory_gate" / "iteration_000000001.json"
    gradient_path = run_dir / "action_gradient_evidence" / "iteration_000000001.json"
    _require_exact_regular_files(
        memory_path.parent,
        [memory_path.name],
        "production-probe memory evidence",
    )
    _require_exact_regular_files(
        gradient_path.parent,
        [gradient_path.name],
        "production-probe gradient evidence",
    )
    _validate_memory_report(
        memory_path,
        world_size=world_size,
        iteration=1,
        included_initial_validation=True,
        included_checkpoint_save=True,
    )
    _validate_gradient_report(gradient_path, world_size=world_size, iteration=1)
    _require_exact_checkpoint_root(run_dir / "checkpoints", [1])
    checkpoint_dir, components = _validate_latest_checkpoint(run_dir, 1)

    cache_files = [
        path
        for path in inductor_cache.rglob("*")
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    ]
    if not cache_files:
        raise EvidenceError("compile-on production probe produced no non-empty Inductor cache files")

    if not log_path.is_file() or log_path.is_symlink():
        raise EvidenceError(f"production-probe log is missing or unsafe: {log_path}")
    if not log_path.is_file() or log_path.is_symlink():
        raise EvidenceError(f"resume-probe log is missing or unsafe: {log_path}")
    text = log_path.read_text(errors="replace")
    _validate_log(
        text,
        required=(
            "Distributed parallelism mode: fsdp",
            "validation: Switched to EMA weights",
            "validation: Restored training weights",
            "Done with training.",
        ),
        extra_forbidden=("Unstable val loss",),
    )
    validation = _validate_validation_summaries(
        text,
        expected_iterations=[0],
        expected_samples=world_size * _BATCH_PER_RANK,
    )
    _validate_sampled_memory(
        sampled_memory_path,
        world_size=world_size,
        label="production-probe sampled GPU-memory evidence",
    )

    return {
        "schema_version": 1,
        "status": "passed",
        "experiment_family": protocol.experiment_family,
        "experiment_id": experiment_id,
        "formal_run_name": formal_run_name,
        "probe_run_name": probe_run_name,
        "world_size": world_size,
        "global_batch_per_optimizer_step": protocol.global_batch,
        "run_dir": str(run_dir),
        "cpu_batch": str(cpu_batch_path),
        "model_memory_gate": str(memory_path),
        "sampled_gpu_memory": str(sampled_memory_path),
        "action_gradient_evidence": str(gradient_path),
        "checkpoint": str(checkpoint_dir),
        "dcp_components": components,
        "validation": validation,
        "inductor_cache_files": len(cache_files),
        "inductor_cache_bytes": sum(path.stat().st_size for path in cache_files),
    }


def verify_resume(
    *,
    run_dir: Path,
    log_path: Path,
    world_size: int,
    experiment_id: str,
    formal_run_name: str,
    sampled_memory_path: Path,
) -> dict[str, Any]:
    protocol = protocol_for_world_size(world_size)
    _require_equal(formal_run_name, experiment_id, "formal run name")
    probe_run_name = f"{formal_run_name}-prod128-probe"
    run_dir = _validate_run_identity(
        run_dir=run_dir,
        expected_name=probe_run_name,
        experiment_id=experiment_id,
        protocol=protocol,
    )
    config = _load_config(run_dir)
    _validate_common_config(config, world_size=world_size, max_iter=2, max_val_iter=1, save_iter=1)
    _validate_protocol_scalars(config, protocol)
    _require_equal(
        config["trainer"].get("validation_iter"),
        protocol.validation_iter,
        "resume-probe validation frequency",
    )
    _validate_callback_config(config, gradient_every_n=1)
    memory_path = run_dir / "production_batch_memory_gate" / "iteration_000000002.json"
    gradient_path = run_dir / "action_gradient_evidence" / "iteration_000000002.json"
    _require_exact_regular_files(
        memory_path.parent,
        ["iteration_000000001.json", "iteration_000000002.json"],
        "resume-probe memory evidence",
    )
    _require_exact_regular_files(
        gradient_path.parent,
        ["iteration_000000001.json", "iteration_000000002.json"],
        "resume-probe gradient evidence",
    )
    _validate_memory_report(
        memory_path,
        world_size=world_size,
        iteration=2,
        included_initial_validation=False,
        included_checkpoint_save=True,
    )
    _validate_gradient_report(gradient_path, world_size=world_size, iteration=2)
    _validate_sampled_memory(
        sampled_memory_path,
        world_size=world_size,
        label="resume-probe sampled GPU-memory evidence",
    )
    _require_exact_checkpoint_root(run_dir / "checkpoints", [1, 2])
    checkpoint_dir, components = _validate_latest_checkpoint(run_dir, 2)
    _validate_dcp(run_dir / "checkpoints" / "iter_000000001")

    if not log_path.is_file() or log_path.is_symlink():
        raise EvidenceError(f"formal log is missing or unsafe: {log_path}")
    text = log_path.read_text(errors="replace")
    expected_resume_path = _validate_resume_log(text, run_dir=run_dir, world_size=world_size)
    if _validation_summaries(text):
        raise EvidenceError("resume probe unexpectedly ran validation after iteration 1")
    return {
        "schema_version": 1,
        "status": "passed",
        "experiment_family": protocol.experiment_family,
        "experiment_id": experiment_id,
        "formal_run_name": formal_run_name,
        "probe_run_name": probe_run_name,
        "world_size": world_size,
        "run_dir": str(run_dir),
        "resumed_from": expected_resume_path,
        "data_stream_resume": "restarted_from_beginning_not_exact_loader_cursor",
        "model_memory_gate": str(memory_path),
        "action_gradient_evidence": str(gradient_path),
        "sampled_gpu_memory": str(sampled_memory_path),
        "checkpoint": str(checkpoint_dir),
        "dcp_components": components,
    }


def verify_storage(
    *,
    path: Path,
    minimum_free_bytes: int,
    estimated_required_bytes: int,
) -> dict[str, Any]:
    _require_equal(minimum_free_bytes, _FORMAL_MIN_FREE_BYTES, "formal minimum free storage")
    _require_equal(
        estimated_required_bytes,
        _FORMAL_ESTIMATED_REQUIRED_BYTES,
        "formal estimated required storage",
    )
    if not path.is_dir() or path.is_symlink():
        raise EvidenceError(f"storage gate path is missing or unsafe: {path}")
    absolute = Path(os.path.abspath(path))
    resolved = path.resolve(strict=True)
    if path != absolute or resolved != absolute:
        raise EvidenceError(f"storage gate path must not traverse symlinks: requested={absolute}, resolved={resolved}")
    _require_equal(resolved, _FORMAL_OUTPUT_ROOT, "formal output root")
    stat = os.statvfs(path)
    available = int(stat.f_bavail * stat.f_frsize)
    total = int(stat.f_blocks * stat.f_frsize)
    if available < minimum_free_bytes:
        raise EvidenceError(
            f"insufficient free storage under {path}: available={available}, minimum={minimum_free_bytes}"
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "path": str(resolved),
        "filesystem_total_bytes": total,
        "filesystem_available_bytes": available,
        "minimum_free_bytes": minimum_free_bytes,
        "estimated_required_bytes": estimated_required_bytes,
        "headroom_after_estimate_bytes": available - estimated_required_bytes,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _parse_utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError(f"{label} is not a valid ISO-8601 timestamp: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EvidenceError(f"{label} must include an explicit UTC offset: {value!r}")
    return parsed


def _validate_storage_report(
    report: dict[str, Any],
    *,
    expected_path: Path,
    owner_acquired_at: datetime,
) -> datetime:
    _require_exact_keys(
        report,
        {
            "schema_version",
            "status",
            "path",
            "filesystem_total_bytes",
            "filesystem_available_bytes",
            "minimum_free_bytes",
            "estimated_required_bytes",
            "headroom_after_estimate_bytes",
            "recorded_at",
        },
        "formal storage preflight",
    )
    _require_equal(report.get("schema_version"), 1, "storage-preflight schema version")
    _require_equal(report.get("status"), "passed", "storage-preflight status")
    _require_equal(
        Path(str(report.get("path"))).resolve(strict=True),
        expected_path.resolve(strict=True),
        "storage-preflight output root",
    )
    numeric_fields = (
        "filesystem_total_bytes",
        "filesystem_available_bytes",
        "minimum_free_bytes",
        "estimated_required_bytes",
        "headroom_after_estimate_bytes",
    )
    for field in numeric_fields:
        if type(report.get(field)) is not int:
            raise EvidenceError(f"storage-preflight {field} must be an integer: {report.get(field)!r}")
    total = report["filesystem_total_bytes"]
    available = report["filesystem_available_bytes"]
    minimum = report["minimum_free_bytes"]
    estimated = report["estimated_required_bytes"]
    headroom = report["headroom_after_estimate_bytes"]
    _require_equal(minimum, _FORMAL_MIN_FREE_BYTES, "storage-preflight minimum free bytes")
    _require_equal(
        estimated,
        _FORMAL_ESTIMATED_REQUIRED_BYTES,
        "storage-preflight estimated required bytes",
    )
    if total <= 0 or not minimum <= available <= total:
        raise EvidenceError(
            f"storage-preflight capacity arithmetic is invalid: total={total}, available={available}, minimum={minimum}"
        )
    _require_equal(headroom, available - estimated, "storage-preflight estimated headroom")
    recorded_at = _parse_utc_timestamp(report.get("recorded_at"), "storage-preflight recorded_at")
    if recorded_at < owner_acquired_at:
        raise EvidenceError(
            "storage preflight predates attempt ownership: "
            f"owner={owner_acquired_at.isoformat()}, storage={recorded_at.isoformat()}"
        )
    return recorded_at


def _expected_gradient_iterations(max_iter: int) -> list[int]:
    iterations = {1, *range(50, max_iter + 1, 50), max_iter}
    return sorted(iterations)


def verify_formal(
    *,
    run_dir: Path,
    log_path: Path,
    world_size: int,
    probe_assertions: Path,
    resume_assertions: Path,
    experiment_id: str,
    formal_manifest: Path,
    formal_manifest_sha256: str,
    owner_path: Path,
    provenance_path: Path,
    formal_run_name: str,
    storage_preflight_path: Path,
    submission_reservation_path: Path,
    sampled_memory_path: Path,
) -> dict[str, Any]:
    protocol = protocol_for_world_size(world_size)
    _require_equal(formal_run_name, experiment_id, "formal run name")
    run_dir = _validate_run_identity(
        run_dir=run_dir,
        expected_name=formal_run_name,
        experiment_id=experiment_id,
        protocol=protocol,
    )
    config = _load_config(run_dir)
    _validate_common_config(
        config,
        world_size=world_size,
        max_iter=protocol.max_iter,
        max_val_iter=protocol.max_val_iter,
        save_iter=protocol.save_iter,
    )
    _validate_protocol_scalars(config, protocol)
    _require_equal(config["trainer"].get("validation_iter"), protocol.validation_iter, "validation frequency")
    _validate_callback_config(config, gradient_every_n=50)

    memory_dir = run_dir / "production_batch_memory_gate"
    memory_files = sorted(path.name for path in memory_dir.glob("iteration_*.json"))
    _require_equal(memory_files, ["iteration_000000001.json"], "formal memory-gate file set")
    _require_exact_regular_files(memory_dir, memory_files, "formal memory evidence")
    memory_path = memory_dir / memory_files[0]
    _validate_memory_report(
        memory_path,
        world_size=world_size,
        iteration=1,
        included_initial_validation=True,
        included_checkpoint_save=False,
    )

    gradient_dir = run_dir / "action_gradient_evidence"
    expected_gradient_iterations = _expected_gradient_iterations(protocol.max_iter)
    expected_gradient_files = [f"iteration_{iteration:09d}.json" for iteration in expected_gradient_iterations]
    actual_gradient_files = sorted(path.name for path in gradient_dir.glob("iteration_*.json"))
    _require_equal(actual_gradient_files, expected_gradient_files, "formal action-gradient file set")
    _require_exact_regular_files(gradient_dir, expected_gradient_files, "formal action-gradient evidence")
    for iteration in expected_gradient_iterations:
        _validate_gradient_report(
            gradient_dir / f"iteration_{iteration:09d}.json",
            world_size=world_size,
            iteration=iteration,
        )

    expected_checkpoint_iterations = protocol.checkpoint_iterations
    checkpoint_root = run_dir / "checkpoints"
    _require_exact_checkpoint_root(checkpoint_root, expected_checkpoint_iterations)
    all_checkpoint_bytes = 0
    for iteration in expected_checkpoint_iterations:
        component_stats = _validate_dcp(checkpoint_root / f"iter_{iteration:09d}")
        all_checkpoint_bytes += sum(
            stats["metadata_bytes"] + stats["distcp_bytes"] for stats in component_stats.values()
        )
    final_checkpoint, final_components = _validate_latest_checkpoint(run_dir, protocol.max_iter)

    expected_log_path = _FORMAL_LOG_ROOT / f"{formal_run_name}.formal.log"
    literal_log_path = Path(os.path.abspath(log_path))
    if (
        log_path != literal_log_path
        or log_path != expected_log_path
        or log_path.is_symlink()
        or not log_path.is_file()
        or log_path.resolve(strict=True) != expected_log_path
    ):
        raise EvidenceError(
            f"formal log must be the exact canonical non-symlink path: expected={expected_log_path}, actual={log_path}"
        )
    text = log_path.read_text(errors="replace")
    _validate_log(
        text,
        required=("Distributed parallelism mode: fsdp", "Done with training."),
        extra_forbidden=("Unstable val loss",),
    )
    validation = _validate_validation_summaries(
        text,
        expected_iterations=protocol.validation_iterations,
        expected_samples=world_size * protocol.max_val_iter * _BATCH_PER_RANK,
    )
    _validate_sampled_memory(
        sampled_memory_path,
        world_size=world_size,
        label="formal-run sampled GPU-memory evidence",
    )
    probe, probe_sha256 = _load_json_and_sha256(probe_assertions, "production-probe assertions")
    resume, resume_sha256 = _load_json_and_sha256(resume_assertions, "resume-probe assertions")
    provenance, provenance_sha256 = _load_json_and_sha256(provenance_path, "formal provenance")
    owner, owner_sha256 = _load_json_and_sha256(owner_path, "formal attempt owner")
    storage, storage_sha256 = _load_json_and_sha256(storage_preflight_path, "formal storage preflight")
    reservation, reservation_sha256 = _load_json_and_sha256(
        submission_reservation_path,
        "formal submission reservation",
    )
    if not re.fullmatch(r"[0-9a-f]{64}", formal_manifest_sha256):
        raise EvidenceError(f"formal manifest SHA-256 is invalid: {formal_manifest_sha256!r}")
    formal_manifest_payload, actual_formal_manifest_sha256 = _load_json_and_sha256(
        formal_manifest,
        "formal manifest",
    )
    _require_equal(
        actual_formal_manifest_sha256,
        formal_manifest_sha256,
        "formal manifest SHA-256",
    )
    _require_equal(formal_manifest_payload.get("experiment_id"), experiment_id, "formal manifest experiment")
    for label, payload in (("production probe", probe), ("resume probe", resume), ("provenance", provenance)):
        _require_equal(payload.get("schema_version"), 1, f"{label} schema version")
        _require_equal(payload.get("status"), "passed", f"{label} status")
    for label, payload in (("production probe", probe), ("resume probe", resume)):
        _require_equal(payload.get("world_size"), world_size, f"{label} world size")
        _require_equal(
            payload.get("experiment_id"),
            experiment_id,
            f"{label} experiment",
        )
    _require_equal(provenance.get("nproc_per_node"), world_size, "provenance world size")
    expected_probe_run_dir = str(run_dir.parent / f"{formal_run_name}-prod128-probe")
    for label, payload in (("production probe", probe), ("resume probe", resume)):
        _require_equal(
            payload.get("experiment_family"),
            protocol.experiment_family,
            f"{label} experiment family",
        )
        _require_equal(payload.get("formal_run_name"), formal_run_name, f"{label} formal run name")
        _require_equal(payload.get("probe_run_name"), f"{formal_run_name}-prod128-probe", f"{label} probe run name")
        _require_equal(payload.get("run_dir"), expected_probe_run_dir, f"{label} run directory")
    _require_equal(resume.get("resumed_from"), f"{expected_probe_run_dir}/checkpoints/iter_000000001", "resume source")
    _require_equal(
        resume.get("data_stream_resume"),
        "restarted_from_beginning_not_exact_loader_cursor",
        "resume data-stream boundary",
    )
    _require_equal(owner.get("status"), "owned", "attempt-owner status")
    _require_equal(owner.get("schema_version"), 1, "attempt-owner schema version")
    _require_equal(owner.get("experiment_id"), experiment_id, "attempt-owner experiment")
    _require_equal(owner.get("run_name"), formal_run_name, "attempt-owner run name")
    _require_equal(owner.get("world_size"), world_size, "attempt-owner world size")
    _require_equal(owner.get("formal_manifest_sha256"), formal_manifest_sha256, "attempt-owner manifest SHA-256")
    _require_equal(provenance.get("experiment_id"), experiment_id, "provenance experiment")
    _require_equal(provenance.get("formal_manifest_sha256"), formal_manifest_sha256, "provenance manifest SHA-256")
    literal_formal_manifest = Path(os.path.abspath(formal_manifest))
    _require_equal(formal_manifest, literal_formal_manifest, "literal formal-manifest path")
    _require_equal(
        Path(str(provenance.get("formal_manifest"))),
        formal_manifest,
        "provenance literal formal-manifest path",
    )
    _require_equal(formal_manifest.resolve(strict=True), formal_manifest, "canonical formal-manifest path")
    output_root = run_dir.parents[2]
    _require_equal(output_root, _FORMAL_OUTPUT_ROOT, "formal output root")
    evidence_dir = output_root / "formal_evidence" / formal_run_name
    path_security_path = evidence_dir / "path-security.json"
    path_security, path_security_sha256 = _load_json_and_sha256(
        path_security_path,
        "formal path-security receipt",
    )
    expected_evidence_paths = (
        (probe_assertions, evidence_dir / "probe-assertions.json"),
        (resume_assertions, evidence_dir / "resume-assertions.json"),
        (owner_path, evidence_dir / "owner.json"),
        (provenance_path, evidence_dir / "provenance.json"),
        (storage_preflight_path, evidence_dir / "formal-storage-preflight.json"),
        (sampled_memory_path, evidence_dir / "formal-gpu-memory.json"),
        (path_security_path, evidence_dir / "path-security.json"),
    )
    canonical_actual_paths = [actual.resolve(strict=True) for actual, _ in expected_evidence_paths]
    if len(set(canonical_actual_paths)) != len(canonical_actual_paths):
        raise EvidenceError("formal evidence paths must resolve to distinct regular files")
    for (actual_path, expected_path), canonical_actual in zip(
        expected_evidence_paths,
        canonical_actual_paths,
        strict=True,
    ):
        _require_equal(
            actual_path,
            Path(os.path.abspath(actual_path)),
            f"literal formal evidence path for {expected_path.name}",
        )
        _require_equal(
            actual_path,
            expected_path,
            f"exact formal evidence path for {expected_path.name}",
        )
        _require_equal(
            canonical_actual,
            expected_path,
            f"formal evidence path for {expected_path.name}",
        )
    for key, expected in {
        "schema_version": 1,
        "ok": True,
        "operation": "create_directory_exclusive",
        "root": str(output_root),
        "target": str(evidence_dir),
        "relative_path": f"formal_evidence/{formal_run_name}",
        "created": True,
        "exclusive": True,
        "strictly_within_root": True,
        "resolved_strict": True,
        "symlink_policy": "reject_every_existing_component",
        "target_kind": "directory",
        "target_state": "existing",
    }.items():
        _require_equal(path_security.get(key), expected, f"path-security receipt {key}")
    target_identity = _require_mapping(
        path_security.get("target_identity"),
        "path-security target identity",
    )
    evidence_metadata = evidence_dir.stat()
    for key, expected in {
        "device": int(evidence_metadata.st_dev),
        "inode": int(evidence_metadata.st_ino),
        "kind": "directory",
        "mode": f"{evidence_metadata.st_mode & 0o777:04o}",
    }.items():
        _require_equal(target_identity.get(key), expected, f"path-security target identity {key}")
    expected_storage_path = output_root.resolve(strict=True)
    _require_exact_keys(
        owner,
        {
            "schema_version",
            "status",
            "acquired_at",
            "experiment_id",
            "run_name",
            "rjob_name",
            "scheduler_rjob_name",
            "scheduler_rjob_uid",
            "scheduler_replica_name",
            "scheduler_pod_name",
            "scheduler_receipt",
            "scheduler_receipt_sha256",
            "hostname",
            "node_name",
            "shell_pid",
            "world_size",
            "container_image",
            "source_manifest_sha256",
            "formal_manifest_sha256",
        },
        "formal attempt owner",
    )
    owner_acquired_at = _parse_utc_timestamp(owner.get("acquired_at"), "attempt-owner acquired_at")
    storage_recorded_at = _validate_storage_report(
        storage,
        expected_path=expected_storage_path,
        owner_acquired_at=owner_acquired_at,
    )
    formal_log_mtime = datetime.fromtimestamp(log_path.stat().st_mtime, timezone.utc)
    if storage_recorded_at > formal_log_mtime:
        raise EvidenceError(
            "formal storage preflight was not recorded before the formal log started: "
            f"storage={storage_recorded_at.isoformat()}, log_mtime={formal_log_mtime.isoformat()}"
        )

    expected_reservation_path = output_root / "formal_submission_reservations" / experiment_id / "submission.json"
    _require_equal(
        Path(os.path.abspath(submission_reservation_path)),
        expected_reservation_path,
        "literal formal submission-reservation path",
    )
    _require_equal(
        submission_reservation_path.resolve(strict=True),
        expected_reservation_path.resolve(strict=True),
        "formal submission-reservation path",
    )
    _require_equal(
        submission_reservation_path.stat().st_mode & 0o777,
        0o444,
        "formal submission-reservation mode",
    )
    _require_exact_keys(
        reservation,
        {
            "schema_version",
            "status",
            "created_at",
            "accepted_at",
            "experiment_id",
            "run_name",
            "rjob_name",
            "scheduler_rjob_name",
            "world_size",
            "container_image",
            "formal_manifest_sha256",
            "source_commit",
            "workflow_snapshot_sha256",
            "source_manifest_sha256",
            "formal_output_root",
            "minimum_free_bytes",
            "estimated_required_bytes",
            "quota_group",
            "cpu",
            "memory_mib",
            "negative_tag",
            "task_type",
            "private_machine",
            "priority",
            "restart_policy",
            "backoff_limit",
            "submission_log",
            "submission_log_sha256",
            "scheduler_expected_contract",
            "scheduler_expected_contract_sha256",
            "scheduler_capture_log",
            "scheduler_capture_log_sha256",
            "scheduler_receipt",
            "scheduler_receipt_sha256",
            "scheduler_text_receipt",
            "scheduler_text_receipt_sha256",
        },
        "formal submission reservation",
    )
    _require_equal(reservation.get("schema_version"), 2, "submission-reservation schema version")
    _require_equal(reservation.get("status"), "accepted", "submission-reservation status")
    reservation_created_at = _parse_utc_timestamp(
        reservation.get("created_at"),
        "submission-reservation created_at",
    )
    reservation_accepted_at = _parse_utc_timestamp(
        reservation.get("accepted_at"),
        "submission-reservation accepted_at",
    )
    if not reservation_created_at <= reservation_accepted_at <= owner_acquired_at:
        raise EvidenceError(
            "submission/ownership timestamps are out of order: "
            f"created={reservation_created_at.isoformat()}, "
            f"accepted={reservation_accepted_at.isoformat()}, "
            f"owner={owner_acquired_at.isoformat()}"
        )
    expected_cpu = 128 if world_size == 8 else 96
    expected_memory_mib = 1_500_000 if world_size == 8 else 1_125_000
    reservation_bindings = {
        "experiment_id": experiment_id,
        "run_name": formal_run_name,
        "rjob_name": owner.get("rjob_name"),
        "world_size": world_size,
        "container_image": provenance.get("container_image"),
        "formal_manifest_sha256": formal_manifest_sha256,
        "source_commit": provenance.get("source_commit"),
        "workflow_snapshot_sha256": provenance.get("workflow_snapshot_sha256"),
        "source_manifest_sha256": provenance.get("source_manifest_sha256"),
        "formal_output_root": str(output_root),
        "minimum_free_bytes": _FORMAL_MIN_FREE_BYTES,
        "estimated_required_bytes": _FORMAL_ESTIMATED_REQUIRED_BYTES,
        "quota_group": _FORMAL_QUOTA_GROUP,
        "cpu": expected_cpu,
        "memory_mib": expected_memory_mib,
        "negative_tag": _FORMAL_EXCLUDED_NODE_TAG,
        "task_type": "normal",
        "private_machine": "group",
        "priority": 9,
        "restart_policy": "never",
        "backoff_limit": 1,
    }
    for key, expected in reservation_bindings.items():
        _require_equal(reservation.get(key), expected, f"submission-reservation {key}")
    _require_equal(
        provenance.get("storage"),
        {
            "output_root": str(output_root),
            "minimum_free_bytes": _FORMAL_MIN_FREE_BYTES,
            "estimated_required_bytes": _FORMAL_ESTIMATED_REQUIRED_BYTES,
        },
        "provenance storage contract",
    )
    _require_equal(
        provenance.get("rjob"),
        {
            "quota_group": _FORMAL_QUOTA_GROUP,
            "cpu": expected_cpu,
            "memory_mib": expected_memory_mib,
            "negative_tag": _FORMAL_EXCLUDED_NODE_TAG,
            "task_type": "normal",
            "private_machine": "group",
            "priority": 9,
            "restart_policy": "never",
            "backoff_limit": 1,
        },
        "provenance rjob contract",
    )
    scheduler_rjob_name = reservation.get("scheduler_rjob_name")
    if (
        not isinstance(scheduler_rjob_name, str)
        or re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", scheduler_rjob_name) is None
        or not scheduler_rjob_name.startswith(f"{owner.get('rjob_name')}-")
    ):
        raise EvidenceError(f"invalid scheduler rjob name in reservation: {scheduler_rjob_name!r}")
    reservation_dir = expected_reservation_path.parent
    submission_log_path = reservation_dir / "rjob-submit.log"
    scheduler_expected_contract_path = reservation_dir / "rjob-expected-contract.json"
    scheduler_capture_log_path = reservation_dir / "rjob-scheduler-capture.log"
    scheduler_receipt_path = reservation_dir / "rjob-scheduler-receipt.json"
    scheduler_text_receipt_path = reservation_dir / "rjob-get.log"
    _require_equal(reservation.get("submission_log"), str(submission_log_path), "submission-log path")
    _require_equal(
        reservation.get("scheduler_expected_contract"),
        str(scheduler_expected_contract_path),
        "scheduler expected-contract path",
    )
    _require_equal(
        reservation.get("scheduler_capture_log"),
        str(scheduler_capture_log_path),
        "scheduler capture-log path",
    )
    _require_equal(
        reservation.get("scheduler_receipt"),
        str(scheduler_receipt_path),
        "scheduler-receipt path",
    )
    _require_equal(
        reservation.get("scheduler_text_receipt"),
        str(scheduler_text_receipt_path),
        "scheduler text-receipt path",
    )
    _require_equal(
        _sha256_file(submission_log_path, "rjob submission log"),
        reservation.get("submission_log_sha256"),
        "submission-log SHA-256",
    )
    _require_equal(
        _sha256_file(scheduler_expected_contract_path, "scheduler expected contract"),
        reservation.get("scheduler_expected_contract_sha256"),
        "scheduler expected-contract SHA-256",
    )
    _require_equal(
        _sha256_file(scheduler_capture_log_path, "scheduler capture log"),
        reservation.get("scheduler_capture_log_sha256"),
        "scheduler capture-log SHA-256",
    )
    scheduler_receipt_payload, scheduler_receipt_sha256 = _load_json_and_sha256(
        scheduler_receipt_path,
        "normalized scheduler receipt",
    )
    _require_equal(
        scheduler_receipt_sha256,
        reservation.get("scheduler_receipt_sha256"),
        "scheduler-receipt SHA-256",
    )
    _require_equal(
        _sha256_file(scheduler_text_receipt_path, "rjob scheduler text receipt"),
        reservation.get("scheduler_text_receipt_sha256"),
        "scheduler text-receipt SHA-256",
    )
    for immutable_path, label in (
        (submission_log_path, "submission log"),
        (scheduler_expected_contract_path, "scheduler expected contract"),
        (scheduler_capture_log_path, "scheduler capture log"),
        (scheduler_receipt_path, "scheduler receipt"),
        (scheduler_text_receipt_path, "scheduler text receipt"),
    ):
        _require_equal(immutable_path.stat().st_mode & 0o777, 0o444, f"{label} mode")
    submission_log = submission_log_path.read_text(errors="replace")
    scheduler_text_receipt = scheduler_text_receipt_path.read_text(errors="replace")
    if submission_log.count(f"created rjob_name: {scheduler_rjob_name}") != 1:
        raise EvidenceError("submission log does not bind exactly one scheduler rjob name")
    if f"rjob {scheduler_rjob_name} (showname={owner.get('rjob_name')}):" not in scheduler_text_receipt:
        raise EvidenceError("scheduler text receipt does not bind actual rjob name to requested showname")

    scheduler_expected = _load_json(
        scheduler_expected_contract_path,
        "scheduler expected contract",
    )
    _require_exact_keys(
        scheduler_expected,
        {
            "schema_version",
            "scheduler_rjob_name",
            "showname",
            "quota_group",
            "task_type",
            "priority",
            "backoff_limit",
            "task_name",
            "replicas",
            "image",
            "cpu",
            "memory_mib",
            "gpu",
            "negative_tags",
            "private_machine",
            "restart_policy",
            "termination_grace_period_seconds",
            "mounts",
            "snapshot_root",
            "wrapper_path",
            "experiment_id",
            "command_sha256",
        },
        "scheduler expected contract",
    )
    expected_scheduler_contract = {
        "schema_version": 1,
        "scheduler_rjob_name": scheduler_rjob_name,
        "showname": owner.get("rjob_name"),
        "quota_group": _FORMAL_QUOTA_GROUP,
        "task_type": "normal",
        "priority": 9,
        "backoff_limit": 1,
        "task_name": "t0",
        "replicas": 1,
        "image": provenance.get("container_image"),
        "cpu": expected_cpu,
        "memory_mib": expected_memory_mib,
        "gpu": world_size,
        "negative_tags": [_FORMAL_EXCLUDED_NODE_TAG],
        "private_machine": "group",
        "restart_policy": "Never",
        "termination_grace_period_seconds": 60,
        "mounts": list(_FORMAL_MOUNTS),
        "snapshot_root": provenance.get("snapshot_root"),
        "wrapper_path": (
            f"{provenance.get('snapshot_root')}/examples/h_cluster/run_cwam_cosmos3_nano_production_then_formal.sh"
        ),
        "experiment_id": experiment_id,
        "command_sha256": scheduler_expected.get("command_sha256"),
    }
    _require_equal(
        scheduler_expected,
        expected_scheduler_contract,
        "scheduler expected-contract payload",
    )
    _require_equal(
        scheduler_receipt_payload.get("schema_version"),
        1,
        "normalized scheduler-receipt schema",
    )
    _require_equal(
        scheduler_receipt_payload.get("kind"),
        "cwam_formal_rjob_scheduler_receipt",
        "normalized scheduler-receipt kind",
    )
    _require_equal(
        scheduler_receipt_payload.get("validation"),
        {"status": "passed", "expected_contract_schema_version": 1},
        "normalized scheduler-receipt validation",
    )
    scheduler_metadata = _require_mapping(
        scheduler_receipt_payload.get("metadata"),
        "normalized scheduler-receipt metadata",
    )
    _require_equal(scheduler_metadata.get("name"), scheduler_rjob_name, "scheduler metadata name")
    _require_equal(scheduler_metadata.get("showname"), owner.get("rjob_name"), "scheduler showname")
    scheduler_rjob_uid = scheduler_metadata.get("uid")
    if not isinstance(scheduler_rjob_uid, str) or not scheduler_rjob_uid:
        raise EvidenceError("normalized scheduler metadata UID must be non-empty")
    scheduler_spec = _require_mapping(
        scheduler_receipt_payload.get("spec"),
        "normalized scheduler-receipt spec",
    )
    for key, expected in {
        "quota_group": _FORMAL_QUOTA_GROUP,
        "task_type": "normal",
        "priority": 9,
        "backoff_limit": 1,
    }.items():
        _require_equal(scheduler_spec.get(key), expected, f"normalized scheduler spec {key}")
    scheduler_task = _require_mapping(scheduler_spec.get("task"), "normalized scheduler task")
    for key, expected in {
        "name": "t0",
        "replicas": 1,
        "negative_tags": [_FORMAL_EXCLUDED_NODE_TAG],
        "private_machine": "group",
        "restart_policy": "Never",
        "termination_grace_period_seconds": 60,
        "mounts": list(_FORMAL_MOUNTS),
    }.items():
        _require_equal(scheduler_task.get(key), expected, f"normalized scheduler task {key}")
    scheduler_container = _require_mapping(
        scheduler_task.get("container"),
        "normalized scheduler container",
    )
    _require_equal(
        scheduler_container.get("image"),
        provenance.get("container_image"),
        "normalized scheduler image",
    )
    _require_equal(
        scheduler_container.get("resources"),
        {
            "requests_equal_core_limits": True,
            "cpu": expected_cpu,
            "memory_mib": expected_memory_mib,
            "gpu": world_size,
            "platform_injected_limit_keys": [],
        },
        "normalized scheduler resources",
    )
    _require_equal(
        scheduler_container.get("snapshot_root"),
        provenance.get("snapshot_root"),
        "normalized scheduler snapshot",
    )
    _require_equal(
        scheduler_container.get("experiment_id"),
        experiment_id,
        "normalized scheduler experiment",
    )
    template_command_for_hash = scheduler_container.get("command")
    if (
        not isinstance(template_command_for_hash, list)
        or len(template_command_for_hash) != 3
        or template_command_for_hash[:2] != ["bash", "-lc"]
        or not isinstance(template_command_for_hash[2], str)
    ):
        raise EvidenceError("normalized scheduler command cannot be hashed")
    observed_command_sha256 = hashlib.sha256(template_command_for_hash[2].encode("utf-8")).hexdigest()
    _require_equal(
        scheduler_expected.get("command_sha256"),
        observed_command_sha256,
        "scheduler formal-command SHA-256",
    )
    pod_names = scheduler_receipt_payload.get("pod_names")
    replica_names = scheduler_receipt_payload.get("replica_names")
    if not isinstance(pod_names, list) or len(pod_names) != 1:
        raise EvidenceError("normalized scheduler receipt must bind exactly one pod hostname")
    if not isinstance(replica_names, list) or len(replica_names) != 1:
        raise EvidenceError("normalized scheduler receipt must bind exactly one Replica CR")
    replicas = scheduler_receipt_payload.get("replicas")
    if not isinstance(replicas, list) or len(replicas) != 1:
        raise EvidenceError("normalized scheduler receipt must contain exactly one expanded Replica")
    replica = _require_mapping(replicas[0], "normalized expanded Replica")
    _require_equal(
        replica.get("replica_name"),
        replica_names[0],
        "normalized expanded Replica name",
    )
    _require_equal(
        replica.get("pod_name"),
        pod_names[0],
        "normalized expanded Replica pod name",
    )
    _require_equal(
        replica.get("pod_hostname"),
        pod_names[0],
        "normalized expanded Replica pod hostname",
    )
    if not isinstance(pod_names[0], str) or not pod_names[0].endswith("-0"):
        raise EvidenceError("normalized expanded Replica pod hostname has the wrong replica suffix")
    hostname_prefix = pod_names[0][:-2]
    _require_equal(
        replica.get("runtime_identity"),
        {
            "host_network": True,
            "dns_policy": "ClusterFirstWithHostNet",
            "hostname_prefix": hostname_prefix,
            "node_name_field_path": "spec.nodeName",
            "rjob_name": scheduler_rjob_name,
            "rjob_uid": scheduler_rjob_uid,
            "replica_name": replica_names[0],
            "quota_group": _FORMAL_QUOTA_GROUP,
        },
        "normalized expanded Replica runtime identity",
    )
    worker_spec = _require_mapping(
        replica.get("worker_spec"),
        "normalized expanded worker spec",
    )
    expected_worker_spec_keys = {
        "image",
        "resources",
        "restart_policy",
        "termination_grace_period_seconds",
        "mounts",
        "command",
        "worker_init_prefix",
    }
    if set(worker_spec) != expected_worker_spec_keys:
        raise EvidenceError(
            "normalized expanded worker spec key mismatch: "
            f"missing={sorted(expected_worker_spec_keys - set(worker_spec))}, "
            f"extra={sorted(set(worker_spec) - expected_worker_spec_keys)}"
        )
    _require_equal(
        worker_spec.get("image"),
        provenance.get("container_image"),
        "normalized expanded worker image",
    )
    _require_equal(
        worker_spec.get("restart_policy"),
        "Never",
        "normalized expanded worker restart policy",
    )
    _require_equal(
        worker_spec.get("termination_grace_period_seconds"),
        60,
        "normalized expanded worker termination grace",
    )
    _require_equal(
        worker_spec.get("mounts"),
        list(_FORMAL_MOUNTS),
        "normalized expanded worker mounts",
    )
    worker_resources = _require_mapping(
        worker_spec.get("resources"),
        "normalized expanded worker resources",
    )
    injected_limit_keys = worker_resources.get("platform_injected_limit_keys")
    allowed_injected_limit_keys = {
        "mellanox.com/mlnx_rdma",
        "rdma/mlnx_shared",
    }
    if (
        not isinstance(injected_limit_keys, list)
        or injected_limit_keys != sorted(set(injected_limit_keys))
        or not set(injected_limit_keys).issubset(allowed_injected_limit_keys)
    ):
        raise EvidenceError("normalized expanded worker has unsupported injected resource limits")
    _require_equal(
        worker_resources,
        {
            "requests_equal_core_limits": True,
            "cpu": expected_cpu,
            "memory_mib": expected_memory_mib,
            "gpu": world_size,
            "platform_injected_limit_keys": injected_limit_keys,
        },
        "normalized expanded worker resources",
    )
    template_command = scheduler_container.get("command")
    worker_command = worker_spec.get("command")
    worker_init_prefix = worker_spec.get("worker_init_prefix")
    audited_worker_init_prefixes = [
        [],
        [
            "/kubebrain/worker-init",
            "run",
            "--enable-pid-monitor",
            "--",
        ],
        [
            "/kubebrain/worker-init",
            "run",
            "--enable-pid-monitor",
            "--metrics-addr=",
            "--",
        ],
    ]
    if not isinstance(template_command, list) or not all(isinstance(item, str) for item in template_command):
        raise EvidenceError("normalized scheduler template command is invalid")
    if worker_init_prefix not in audited_worker_init_prefixes:
        raise EvidenceError("normalized expanded worker-init prefix mismatch")
    _require_equal(
        worker_command,
        worker_init_prefix + template_command,
        "normalized expanded worker command",
    )
    for key, expected in {
        "scheduler_rjob_name": scheduler_rjob_name,
        "scheduler_rjob_uid": scheduler_rjob_uid,
        "scheduler_replica_name": replica_names[0],
        "scheduler_pod_name": pod_names[0],
        "scheduler_receipt": str(scheduler_receipt_path),
        "scheduler_receipt_sha256": scheduler_receipt_sha256,
    }.items():
        _require_equal(owner.get(key), expected, f"attempt-owner {key}")
    owner_hostname = owner.get("hostname")
    owner_node_name = owner.get("node_name")
    if not isinstance(owner_hostname, str) or not owner_hostname:
        raise EvidenceError("attempt-owner hostname must be non-empty")
    if not isinstance(owner_node_name, str) or not owner_node_name:
        raise EvidenceError("attempt-owner node_name must be non-empty")
    _require_equal(
        owner_hostname,
        owner_node_name,
        "host-network attempt-owner hostname versus node_name",
    )
    _require_equal(
        owner.get("container_image"),
        provenance.get("container_image"),
        "attempt-owner container image",
    )
    _require_equal(
        owner.get("source_manifest_sha256"),
        provenance.get("source_manifest_sha256"),
        "attempt-owner source-manifest SHA-256",
    )

    return {
        "schema_version": 1,
        "status": "training_completed",
        "experiment_family": protocol.experiment_family,
        "experiment_id": experiment_id,
        "formal_run_name": formal_run_name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "world_size": world_size,
        "global_batch_per_optimizer_step": protocol.global_batch,
        "run_dir": str(run_dir),
        "formal_log": str(log_path),
        "probe_assertions": str(probe_assertions),
        "resume_assertions": str(resume_assertions),
        "first_step_memory_gate": str(memory_path),
        "action_gradient_reports": len(expected_gradient_files),
        "validation_reports": validation,
        "checkpoint_count": len(expected_checkpoint_iterations),
        "all_checkpoint_bytes": all_checkpoint_bytes,
        "final_checkpoint": str(final_checkpoint),
        "final_dcp_components": final_components,
        "formal_manifest": str(formal_manifest),
        "formal_manifest_sha256": actual_formal_manifest_sha256,
        "probe_assertions_sha256": probe_sha256,
        "resume_assertions_sha256": resume_sha256,
        "attempt_owner": str(owner_path),
        "attempt_owner_sha256": owner_sha256,
        "provenance": str(provenance_path),
        "provenance_sha256": provenance_sha256,
        "storage_preflight": str(storage_preflight_path),
        "storage_preflight_sha256": storage_sha256,
        "submission_reservation": str(submission_reservation_path),
        "submission_reservation_sha256": reservation_sha256,
        "sampled_gpu_memory": str(sampled_memory_path),
        "sampled_gpu_memory_sha256": _sha256_file(
            sampled_memory_path,
            "formal-run sampled GPU-memory evidence",
        ),
        "path_security": str(path_security_path),
        "path_security_sha256": path_security_sha256,
        "container_image": provenance["container_image"],
        "scheduler_rjob_name": scheduler_rjob_name,
        "scheduler_rjob_uid": scheduler_rjob_uid,
        "scheduler_replica_name": replica_names[0],
        "scheduler_pod_name": pod_names[0],
        "exact_data_resume": False,
        "production_topology_resume_passed": True,
        "control_success_claim": False,
    }


def _expected_continuation_gradient_iterations(
    *,
    initial_iteration: int,
    max_iter: int,
    every_n: int,
) -> list[int]:
    """Return callback boundaries reached after a non-zero checkpoint resume."""

    if not 0 <= initial_iteration < max_iter or every_n < 1:
        raise EvidenceError(
            f"invalid continuation gradient range: initial={initial_iteration}, max={max_iter}, every_n={every_n}"
        )
    first_periodic = ((initial_iteration // every_n) + 1) * every_n
    iterations = set(range(first_periodic, max_iter + 1, every_n))
    iterations.add(max_iter)
    return sorted(iterations)


def _validate_continuation_resume_log(
    text: str,
    *,
    seed_link: Path,
    initial_iteration: int,
) -> None:
    expected_resume_path = str(seed_link)
    markers = {
        "warning": "WARNING: resuming model/optim/scheduler/trainer with a restarted data stream.",
        "resume": (f"Resuming ckpt {expected_resume_path} with keys: ['model', 'optim', 'scheduler', 'trainer']"),
        "model": "- Loading the model...",
        "optimizer": "- Loading the optimizer...",
        "scheduler": "- Loading the scheduler...",
        "trainer": "- Loading the trainer...",
        "loaded": (f"Loaded checkpoint from {expected_resume_path} in iteration {initial_iteration}"),
        "fsdp": "Distributed parallelism mode: fsdp",
        "done": "Done with training.",
    }
    positions = {
        label: _unique_position(text, marker, f"formal continuation {label} marker")
        for label, marker in markers.items()
    }
    if not (
        positions["warning"]
        < positions["resume"]
        < positions["model"]
        < positions["optimizer"]
        < positions["scheduler"]
        < positions["trainer"]
        < positions["loaded"]
        < positions["fsdp"]
        < positions["done"]
    ):
        raise EvidenceError(f"formal continuation checkpoint state machine is out of order: {positions}")
    if text.count("Resuming ckpt ") != 1 or text.count("Loaded checkpoint from ") != 1:
        raise EvidenceError("formal continuation log contains multiple checkpoint attempts")
    _validate_log(
        text,
        required=(),
        extra_forbidden=(
            "Training from scratch.",
            "Unstable val loss",
            "skip loading dataloader.",
            "- Loading the dataloader",
        ),
    )


def _validate_continuation_checkpoint_root(
    *,
    run_dir: Path,
    resume_checkpoint: Path,
    initial_iteration: int,
    max_iter: int,
    save_iter: int,
) -> tuple[Path, dict[str, dict[str, int]], int, int]:
    checkpoint_root = run_dir / "checkpoints"
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise EvidenceError(f"continuation checkpoint root is missing or unsafe: {checkpoint_root}")
    latest = checkpoint_root / "latest_checkpoint.txt"
    if latest.is_symlink() or not latest.is_file():
        raise EvidenceError(f"continuation latest-checkpoint pointer is missing or unsafe: {latest}")
    expected_final_name = f"iter_{max_iter:09d}"
    _require_equal(latest.read_text().strip(), expected_final_name, "continuation latest checkpoint")

    seed_name = f"iter_{initial_iteration:09d}"
    seed_link = checkpoint_root / seed_name
    if not seed_link.is_symlink():
        raise EvidenceError(f"continuation seed checkpoint must remain a symlink: {seed_link}")
    _require_equal(
        seed_link.resolve(strict=True),
        resume_checkpoint.resolve(strict=True),
        "continuation seed checkpoint target",
    )
    expected_new_iterations = list(range(initial_iteration + save_iter, max_iter + 1, save_iter))
    expected_entries = {
        "latest_checkpoint.txt",
        seed_name,
        *(f"iter_{iteration:09d}" for iteration in expected_new_iterations),
    }
    _require_equal(
        {path.name for path in checkpoint_root.iterdir()},
        expected_entries,
        "continuation checkpoint-root entry set",
    )

    seed_components = _validate_dcp(resume_checkpoint)
    all_checkpoint_bytes = sum(stats["metadata_bytes"] + stats["distcp_bytes"] for stats in seed_components.values())
    for iteration in expected_new_iterations:
        checkpoint = checkpoint_root / f"iter_{iteration:09d}"
        if checkpoint.is_symlink():
            raise EvidenceError(f"new continuation checkpoint must not be a symlink: {checkpoint}")
        components = _validate_dcp(checkpoint)
        all_checkpoint_bytes += sum(stats["metadata_bytes"] + stats["distcp_bytes"] for stats in components.values())
    final_checkpoint = checkpoint_root / expected_final_name
    final_components = _validate_dcp(final_checkpoint)
    return (
        final_checkpoint,
        final_components,
        len(expected_new_iterations) + 1,
        all_checkpoint_bytes,
    )


def verify_formal_continuation(
    *,
    run_dir: Path,
    log_path: Path,
    world_size: int,
    experiment_id: str,
    formal_run_name: str,
    continuation_contract_path: Path,
    continuation_contract_sha256: str,
    submission_reservation_path: Path,
) -> dict[str, Any]:
    """Validate a formal 8-H200 continuation from the exact R6 iter-1000 state."""

    if world_size != 8:
        raise EvidenceError("formal continuation is defined only for world_size=8")
    if not re.fullmatch(r"[0-9a-f]{64}", continuation_contract_sha256):
        raise EvidenceError("continuation contract SHA-256 must be lowercase hexadecimal")
    protocol = protocol_for_world_size(world_size)
    _require_equal(formal_run_name, experiment_id, "formal continuation run name")
    run_dir = _validate_run_identity(
        run_dir=run_dir,
        expected_name=formal_run_name,
        experiment_id=experiment_id,
        protocol=protocol,
    )
    contract, observed_contract_sha256 = _load_json_and_sha256(
        continuation_contract_path,
        "formal continuation contract",
    )
    _require_equal(
        observed_contract_sha256,
        continuation_contract_sha256,
        "formal continuation contract SHA-256",
    )
    for key, expected in {
        "schema_version": 1,
        "status": "passed",
        "kind": _CONTINUATION_CONTRACT_KIND,
        "experiment_id": experiment_id,
        "run_name": formal_run_name,
        "world_size": world_size,
    }.items():
        _require_equal(contract.get(key), expected, f"formal continuation contract {key}")
    cache = _require_mapping(contract.get("cache"), "formal continuation cache")
    throughput = _require_mapping(
        contract.get("throughput"),
        "formal continuation throughput selection",
    )
    resume = _require_mapping(contract.get("resume"), "formal continuation resume")
    continuation_protocol = _require_mapping(
        contract.get("protocol"),
        "formal continuation protocol",
    )
    submission = _require_mapping(
        contract.get("submission"),
        "formal continuation submission binding",
    )
    expected_submission_keys = {
        "reservation_path",
        "reservation_sha256",
        "submission_nonce",
        "command_sha256",
        "container_image",
        "resources",
        "mounts",
        "store_host_nvme",
        "scheduler_rjob_name",
        "scheduler_rjob_uid",
        "scheduler_replica_name",
        "scheduler_pod_name",
        "artifacts",
        "scheduler_source_objects",
    }
    if set(submission) != expected_submission_keys:
        raise EvidenceError("formal continuation submission-binding schema mismatch")
    expected_reservation_path = (
        _FORMAL_OUTPUT_ROOT / "formal_continuation_submission_reservations" / experiment_id / "submission.json"
    )
    _require_equal(
        submission_reservation_path,
        expected_reservation_path,
        "formal continuation submission-reservation path",
    )
    _require_equal(
        submission.get("reservation_path"),
        str(expected_reservation_path),
        "formal continuation contract submission-reservation path",
    )
    if (
        submission_reservation_path.is_symlink()
        or not submission_reservation_path.is_file()
        or submission_reservation_path.stat().st_mode & 0o777 != 0o444
    ):
        raise EvidenceError(
            "formal continuation submission reservation must remain mode-0444 regular non-symlink evidence"
        )
    reservation, reservation_sha256 = _load_json_and_sha256(
        submission_reservation_path,
        "formal continuation submission reservation",
    )
    _require_equal(
        reservation_sha256,
        submission.get("reservation_sha256"),
        "formal continuation submission-reservation SHA-256",
    )
    for key, expected in {
        "schema_version": 2,
        "kind": "cwam_cosmos3_formal_continuation_submission_reservation",
        "status": "accepted",
        "experiment_id": experiment_id,
        "run_name": formal_run_name,
        "submission_nonce": submission.get("submission_nonce"),
        "command_sha256": submission.get("command_sha256"),
        "container_image": submission.get("container_image"),
        "mounts": submission.get("mounts"),
        "scheduler_rjob_name": submission.get("scheduler_rjob_name"),
        "scheduler_rjob_uid": submission.get("scheduler_rjob_uid"),
        "scheduler_replica_name": submission.get("scheduler_replica_name"),
        "scheduler_pod_name": submission.get("scheduler_pod_name"),
    }.items():
        _require_equal(
            reservation.get(key),
            expected,
            f"formal continuation submission reservation {key}",
        )
    for label in (
        "reservation_sha256",
        "submission_nonce",
        "command_sha256",
    ):
        value = submission.get(label)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise EvidenceError(f"formal continuation submission {label} is invalid")
    artifact_bindings = _require_mapping(
        submission.get("artifacts"),
        "formal continuation submission artifact bindings",
    )
    _require_equal(
        set(artifact_bindings),
        {
            "submission_log",
            "scheduler_expected_contract",
            "scheduler_capture_log",
            "scheduler_structured_receipt",
            "scheduler_text_readback",
        },
        "formal continuation submission artifact set",
    )
    for label, raw_descriptor in artifact_bindings.items():
        descriptor = _require_mapping(
            raw_descriptor,
            f"formal continuation submission artifact {label}",
        )
        _require_equal(
            set(descriptor),
            {"path", "size_bytes", "sha256"},
            f"formal continuation submission artifact {label} schema",
        )
        path = Path(str(descriptor.get("path")))
        if (
            path.parent != submission_reservation_path.parent
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_mode & 0o777 != 0o444
        ):
            raise EvidenceError(f"formal continuation submission artifact {label} is unsafe")
        _require_equal(
            path.stat().st_size,
            descriptor.get("size_bytes"),
            f"formal continuation submission artifact {label} size",
        )
        _require_equal(
            _sha256_file(
                path,
                f"formal continuation submission artifact {label}",
            ),
            descriptor.get("sha256"),
            f"formal continuation submission artifact {label} SHA-256",
        )
    expected_protocol = {
        "initial_iteration": _CONTINUATION_INITIAL_ITERATION,
        "max_iter": protocol.max_iter,
        "resume_existing_run": True,
        "allow_data_stream_replay": True,
        "data_stream_resume": _CONTINUATION_DATA_STREAM_RESUME,
        "checkpoint_load_training_state": True,
        "checkpoint_keys_to_skip_loading": [],
        "checkpoint_keys_not_to_resume": ["dataloader"],
        "checkpoint_strict_resume": True,
        "fixed_index_trace_path": None,
    }
    _require_equal(
        continuation_protocol,
        expected_protocol,
        "formal continuation protocol payload",
    )
    workers = throughput.get("workers_per_rank")
    if workers not in {4, 8, 12, 16}:
        raise EvidenceError(f"invalid selected continuation worker count: {workers!r}")
    storage = throughput.get("storage")
    if storage not in {"gpfs", "nvme"}:
        raise EvidenceError(f"invalid selected continuation storage: {storage!r}")
    activation_checkpointing = throughput.get("activation_checkpointing")
    if activation_checkpointing not in {"selective", "none"}:
        raise EvidenceError(
            f"invalid selected continuation activation-checkpointing mode: {activation_checkpointing!r}"
        )
    expected_submission_cpu = 160 if workers in {12, 16} else 128
    _require_equal(
        submission.get("resources"),
        {
            "cpu": expected_submission_cpu,
            "memory_mib": 1_500_000,
            "gpu": 8,
        },
        "formal continuation scheduler resources",
    )
    submission_host_nvme = _require_mapping(
        submission.get("store_host_nvme"),
        "formal continuation host-NVMe binding",
    )
    _require_equal(
        submission_host_nvme.get("enabled"),
        storage == "nvme",
        "formal continuation host-NVMe selection",
    )
    if storage == "nvme":
        _require_equal(
            submission_host_nvme,
            {
                "enabled": True,
                "annotation": ("kubebrain.brainpp.cn/writeread-hostpath-mounts"),
                "value": "/nvme:/mnt/nvme1n1,/cfs:/mnt/cfs",
                "container_target": "/nvme",
                "host_source": "/mnt/nvme1n1",
            },
            "formal continuation host-NVMe scheduler binding",
        )
    if throughput.get("best_observed_cell_id") != (
        f"workers-{workers}__storage-{storage}__ac-{activation_checkpointing}"
    ):
        raise EvidenceError("continuation throughput fields do not match best_observed_cell_id")
    benchmark_artifacts = _require_mapping(
        throughput.get("benchmark_artifacts"),
        "formal continuation throughput benchmark artifacts",
    )
    _require_equal(
        benchmark_artifacts.get("gpfs_latent_cache_root"),
        cache.get("root"),
        "formal continuation throughput/cache root binding",
    )
    _require_equal(
        benchmark_artifacts.get("latent_cache_manifest_sha256"),
        cache.get("manifest_sha256"),
        "formal continuation throughput/cache manifest binding",
    )
    _require_equal(
        benchmark_artifacts.get("resume_checkpoint"),
        resume.get("path"),
        "formal continuation throughput/R6 checkpoint binding",
    )
    for label, value in (
        ("throughput source root", benchmark_artifacts.get("source_root")),
        ("throughput Wan VAE path", benchmark_artifacts.get("wan_vae_path")),
    ):
        if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
            raise EvidenceError(f"{label} is invalid: {value!r}")
    benchmark_source_commit = benchmark_artifacts.get("source_commit")
    if not isinstance(benchmark_source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", benchmark_source_commit) is None:
        raise EvidenceError(f"throughput source commit is invalid: {benchmark_source_commit!r}")
    for label, value in (
        ("cache _SUCCESS SHA-256", cache.get("success_sha256")),
        ("cache manifest SHA-256", cache.get("manifest_sha256")),
        ("throughput matrix SHA-256", throughput.get("matrix_summary_sha256")),
        ("R6 checkpoint fingerprint", resume.get("fingerprint_sha256")),
        ("throughput terminal receipt SHA-256", benchmark_artifacts.get("run_exit_sha256")),
        ("throughput fixed trace SHA-256", benchmark_artifacts.get("fixed_index_trace_sha256")),
        (
            "throughput fixed trace manifest SHA-256",
            benchmark_artifacts.get("fixed_index_trace_manifest_sha256"),
        ),
        ("throughput node evidence SHA-256", benchmark_artifacts.get("node_evidence_sha256")),
        (
            "throughput source snapshot manifest SHA-256",
            benchmark_artifacts.get("snapshot_manifest_sha256"),
        ),
        ("throughput Wan VAE SHA-256", benchmark_artifacts.get("wan_vae_sha256")),
        ("throughput runner SHA-256", benchmark_artifacts.get("runner_sha256")),
        ("throughput helper SHA-256", benchmark_artifacts.get("helper_sha256")),
    ):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise EvidenceError(f"{label} is invalid: {value!r}")
    benchmark_cells = benchmark_artifacts.get("cells")
    if not isinstance(benchmark_cells, list) or not benchmark_cells:
        raise EvidenceError("formal continuation has no artifact-bound throughput cells")
    for index, cell in enumerate(benchmark_cells):
        cell = _require_mapping(
            cell,
            f"formal continuation throughput artifact cell {index}",
        )
        for key in ("contract_sha256", "metrics_sha256"):
            value = cell.get(key)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise EvidenceError(f"formal continuation throughput cell {index} {key} is invalid")
    component_reports = _require_mapping(
        resume.get("component_reports"),
        "formal continuation R6 component reports",
    )
    _require_equal(
        set(component_reports),
        {"model", "optim", "scheduler", "trainer"},
        "formal continuation R6 component-report set",
    )
    for component, raw_report in component_reports.items():
        report = _require_mapping(
            raw_report,
            f"formal continuation R6 {component} report",
        )
        torch_metadata = _require_mapping(
            report.get("torch_dcp_metadata"),
            f"formal continuation R6 {component} Torch DCP metadata",
        )
        _require_equal(
            torch_metadata.get("status"),
            "passed",
            f"formal continuation R6 {component} Torch DCP metadata status",
        )

    config = _load_config(run_dir)
    _validate_common_config(
        config,
        world_size=world_size,
        max_iter=protocol.max_iter,
        max_val_iter=protocol.max_val_iter,
        save_iter=protocol.save_iter,
        train_num_workers=int(workers),
        val_num_workers=2,
    )
    _validate_protocol_scalars(config, protocol)
    _require_equal(
        config["trainer"].get("validation_iter"),
        protocol.validation_iter,
        "continuation validation frequency",
    )
    _validate_callback_config(config, gradient_every_n=50)
    checkpoint_config = _require_mapping(config.get("checkpoint"), "checkpoint config")
    for key, expected in {
        "load_training_state": True,
        "keys_to_skip_loading": [],
        "keys_not_to_resume": ["dataloader"],
        "strict_resume": True,
    }.items():
        _require_equal(
            checkpoint_config.get(key),
            expected,
            f"formal continuation checkpoint.{key}",
        )
    model_config = _require_mapping(
        _require_mapping(config.get("model"), "model config").get("config"),
        "model.config",
    )
    activation_config = _require_mapping(
        model_config.get("activation_checkpointing"),
        "model activation-checkpointing config",
    )
    _require_equal(
        activation_config.get("mode"),
        activation_checkpointing,
        "formal continuation activation-checkpointing selection",
    )
    tokenizer = _require_mapping(model_config.get("tokenizer"), "model tokenizer config")
    _require_equal(
        tokenizer.get("_target_"),
        _PRECOMPUTED_WAN_TOKENIZER_TARGET,
        "formal continuation metadata-only tokenizer",
    )
    train_loader = _require_mapping(config.get("dataloader_train"), "train loader config")
    train_worker = _require_mapping(train_loader.get("dataloader"), "train worker config")
    datasets = _require_mapping(train_worker.get("datasets"), "train dataset mapping")
    dataset_wrapper = _require_mapping(datasets.get("robocasa365"), "RoboCasa365 dataset wrapper")
    dataset = _require_mapping(dataset_wrapper.get("dataset"), "RoboCasa365 train dataset")
    expected_cache_root = (
        Path(str(cache.get("root")))
        if storage == "gpfs"
        else Path("/nvme/.cwam-cosmos3-formal-continuation") / formal_run_name / "cache"
    )
    _require_equal(
        Path(str(dataset.get("latent_cache_root"))),
        expected_cache_root,
        "formal continuation runtime latent-cache root",
    )
    _require_equal(dataset.get("latent_cache_mode"), "required", "formal continuation cache mode")
    _require_equal(
        dataset.get("fixed_index_trace_path"),
        None,
        "formal continuation fixed-index trace",
    )

    resume_checkpoint = Path(str(resume.get("path")))
    if resume_checkpoint.name != "iter_000001000":
        raise EvidenceError("formal continuation resume checkpoint is not iter_000001000")
    seed_link = run_dir / "checkpoints" / "iter_000001000"
    seed_receipt_path = run_dir / "continuation-seed.json"
    seed_receipt, seed_receipt_sha256 = _load_json_and_sha256(
        seed_receipt_path,
        "formal continuation seed receipt",
    )
    for key, expected in {
        "schema_version": 1,
        "status": "prepared",
        "run_name": formal_run_name,
        "run_dir": str(run_dir),
        "seed_checkpoint_link": str(seed_link),
        "seed_checkpoint_target": str(resume_checkpoint),
        "seed_checkpoint_fingerprint_algorithm": resume.get("fingerprint_algorithm"),
        "seed_checkpoint_fingerprint_sha256": resume.get("fingerprint_sha256"),
        "continuation_contract": str(continuation_contract_path),
        "continuation_contract_sha256": continuation_contract_sha256,
        "submission_reservation": str(submission_reservation_path),
        "submission_reservation_sha256": reservation_sha256,
        "submission_nonce": submission["submission_nonce"],
        "scheduler_rjob_name": submission["scheduler_rjob_name"],
        "scheduler_rjob_uid": submission["scheduler_rjob_uid"],
        "scheduler_replica_name": submission["scheduler_replica_name"],
        "scheduler_pod_name": submission["scheduler_pod_name"],
    }.items():
        _require_equal(seed_receipt.get(key), expected, f"continuation seed receipt {key}")

    memory_dir = run_dir / "production_batch_memory_gate"
    expected_memory_name = "iteration_000001001.json"
    _require_exact_regular_files(
        memory_dir,
        [expected_memory_name],
        "formal continuation memory evidence",
    )
    memory_path = memory_dir / expected_memory_name
    _validate_memory_report(
        memory_path,
        world_size=world_size,
        iteration=_CONTINUATION_INITIAL_ITERATION + 1,
        included_initial_validation=False,
        included_checkpoint_save=False,
    )

    gradient_iterations = _expected_continuation_gradient_iterations(
        initial_iteration=_CONTINUATION_INITIAL_ITERATION,
        max_iter=protocol.max_iter,
        every_n=50,
    )
    gradient_files = [f"iteration_{iteration:09d}.json" for iteration in gradient_iterations]
    gradient_dir = run_dir / "action_gradient_evidence"
    _require_exact_regular_files(
        gradient_dir,
        gradient_files,
        "formal continuation action-gradient evidence",
    )
    for iteration in gradient_iterations:
        _validate_gradient_report(
            gradient_dir / f"iteration_{iteration:09d}.json",
            world_size=world_size,
            iteration=iteration,
        )

    final_checkpoint, final_components, checkpoint_count, all_checkpoint_bytes = _validate_continuation_checkpoint_root(
        run_dir=run_dir,
        resume_checkpoint=resume_checkpoint,
        initial_iteration=_CONTINUATION_INITIAL_ITERATION,
        max_iter=protocol.max_iter,
        save_iter=protocol.save_iter,
    )
    expected_log_path = _FORMAL_LOG_ROOT / f"{formal_run_name}.continuation.log"
    if (
        log_path != Path(os.path.abspath(log_path))
        or log_path != expected_log_path
        or log_path.is_symlink()
        or not log_path.is_file()
        or log_path.resolve(strict=True) != expected_log_path
    ):
        raise EvidenceError(
            "formal continuation log must be the exact canonical non-symlink path: "
            f"expected={expected_log_path}, actual={log_path}"
        )
    text = log_path.read_text(errors="replace")
    _validate_continuation_resume_log(
        text,
        seed_link=seed_link,
        initial_iteration=_CONTINUATION_INITIAL_ITERATION,
    )
    # Trainer.run_validation_on_start is intentionally gated on iteration == 0;
    # an iter-1000 resume first validates at the next periodic boundary.
    validation_iterations = list(
        range(
            _CONTINUATION_INITIAL_ITERATION + protocol.validation_iter,
            protocol.max_iter + 1,
            protocol.validation_iter,
        )
    )
    validation = _validate_validation_summaries(
        text,
        expected_iterations=validation_iterations,
        expected_samples=world_size * protocol.max_val_iter * _BATCH_PER_RANK,
    )
    return {
        "schema_version": 1,
        "status": "training_completed",
        "mode": "continuation",
        "experiment_family": protocol.experiment_family,
        "experiment_id": experiment_id,
        "formal_run_name": formal_run_name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "world_size": world_size,
        "global_batch_per_optimizer_step": protocol.global_batch,
        "initial_iteration": _CONTINUATION_INITIAL_ITERATION,
        "final_iteration": protocol.max_iter,
        "run_dir": str(run_dir),
        "formal_log": str(log_path),
        "workers_per_rank": workers,
        "storage": storage,
        "activation_checkpointing": activation_checkpointing,
        "best_observed_cell_id": throughput["best_observed_cell_id"],
        "cache_success_sha256": cache["success_sha256"],
        "cache_manifest_sha256": cache["manifest_sha256"],
        "throughput_matrix_sha256": throughput["matrix_summary_sha256"],
        "throughput_source_root": benchmark_artifacts["source_root"],
        "throughput_source_commit": benchmark_source_commit,
        "throughput_snapshot_manifest_sha256": benchmark_artifacts["snapshot_manifest_sha256"],
        "throughput_wan_vae_path": benchmark_artifacts["wan_vae_path"],
        "throughput_wan_vae_sha256": benchmark_artifacts["wan_vae_sha256"],
        "throughput_runner_sha256": benchmark_artifacts["runner_sha256"],
        "throughput_helper_sha256": benchmark_artifacts["helper_sha256"],
        "resume_checkpoint": str(resume_checkpoint),
        "resume_checkpoint_fingerprint_algorithm": resume["fingerprint_algorithm"],
        "resume_checkpoint_fingerprint_sha256": resume["fingerprint_sha256"],
        "continuation_contract": str(continuation_contract_path),
        "continuation_contract_sha256": continuation_contract_sha256,
        "submission_reservation": str(submission_reservation_path),
        "submission_reservation_sha256": reservation_sha256,
        "submission_nonce": submission["submission_nonce"],
        "scheduler_rjob_name": submission["scheduler_rjob_name"],
        "scheduler_rjob_uid": submission["scheduler_rjob_uid"],
        "scheduler_replica_name": submission["scheduler_replica_name"],
        "scheduler_pod_name": submission["scheduler_pod_name"],
        "continuation_seed_receipt": str(seed_receipt_path),
        "continuation_seed_receipt_sha256": seed_receipt_sha256,
        "first_step_memory_gate": str(memory_path),
        "action_gradient_reports": len(gradient_files),
        "validation_reports": validation,
        "checkpoint_count_including_seed": checkpoint_count,
        "all_checkpoint_bytes_including_seed": all_checkpoint_bytes,
        "final_checkpoint": str(final_checkpoint),
        "final_dcp_components": final_components,
        "exact_data_resume": False,
        "data_stream_resume": _CONTINUATION_DATA_STREAM_RESUME,
        "fixed_index_trace_used": False,
        "control_success_claim": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    storage = subparsers.add_parser("verify-storage")
    storage.add_argument("--path", type=Path, required=True)
    storage.add_argument("--minimum-free-bytes", type=int, required=True)
    storage.add_argument("--estimated-required-bytes", type=int, required=True)
    storage.add_argument("--output-json", type=Path, required=True)

    probe = subparsers.add_parser("verify-probe")
    probe.add_argument("--run-dir", type=Path, required=True)
    probe.add_argument("--log", type=Path, required=True)
    probe.add_argument("--cpu-batch", type=Path, required=True)
    probe.add_argument("--sampled-memory", type=Path, required=True)
    probe.add_argument("--world-size", type=int, choices=(6, 8), required=True)
    probe.add_argument("--inductor-cache", type=Path, required=True)
    probe.add_argument("--experiment-id", required=True)
    probe.add_argument("--formal-run-name", required=True)
    probe.add_argument("--output-json", type=Path, required=True)

    resume = subparsers.add_parser("verify-resume")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--log", type=Path, required=True)
    resume.add_argument("--world-size", type=int, choices=(6, 8), required=True)
    resume.add_argument("--experiment-id", required=True)
    resume.add_argument("--formal-run-name", required=True)
    resume.add_argument("--sampled-memory", type=Path, required=True)
    resume.add_argument("--output-json", type=Path, required=True)

    formal = subparsers.add_parser("verify-formal")
    formal.add_argument(
        "--mode",
        choices=("fresh", "continuation"),
        default="fresh",
        help="fresh keeps the original production-probe contract; continuation resumes R6 iter 1000",
    )
    formal.add_argument("--run-dir", type=Path, required=True)
    formal.add_argument("--log", type=Path, required=True)
    formal.add_argument("--world-size", type=int, choices=(6, 8), required=True)
    formal.add_argument("--formal-run-name", required=True)
    formal.add_argument("--probe-assertions", type=Path)
    formal.add_argument("--resume-assertions", type=Path)
    formal.add_argument("--experiment-id", required=True)
    formal.add_argument("--formal-manifest", type=Path)
    formal.add_argument("--formal-manifest-sha256")
    formal.add_argument("--owner", type=Path)
    formal.add_argument("--provenance", type=Path)
    formal.add_argument("--storage-preflight", type=Path)
    formal.add_argument("--submission-reservation", type=Path)
    formal.add_argument("--sampled-memory", type=Path)
    formal.add_argument("--continuation-contract", type=Path)
    formal.add_argument("--continuation-contract-sha256")
    formal.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "verify-storage":
            payload = verify_storage(
                path=args.path,
                minimum_free_bytes=args.minimum_free_bytes,
                estimated_required_bytes=args.estimated_required_bytes,
            )
        elif args.command == "verify-probe":
            payload = verify_probe(
                run_dir=args.run_dir,
                log_path=args.log,
                cpu_batch_path=args.cpu_batch,
                sampled_memory_path=args.sampled_memory,
                world_size=args.world_size,
                inductor_cache=args.inductor_cache,
                experiment_id=args.experiment_id,
                formal_run_name=args.formal_run_name,
            )
        elif args.command == "verify-resume":
            payload = verify_resume(
                run_dir=args.run_dir,
                log_path=args.log,
                world_size=args.world_size,
                experiment_id=args.experiment_id,
                formal_run_name=args.formal_run_name,
                sampled_memory_path=args.sampled_memory,
            )
        elif args.mode == "continuation":
            if (
                args.continuation_contract is None
                or args.continuation_contract_sha256 is None
                or args.submission_reservation is None
            ):
                raise EvidenceError(
                    "continuation mode requires --continuation-contract and "
                    "--continuation-contract-sha256 and --submission-reservation"
                )
            payload = verify_formal_continuation(
                run_dir=args.run_dir,
                log_path=args.log,
                world_size=args.world_size,
                experiment_id=args.experiment_id,
                formal_run_name=args.formal_run_name,
                continuation_contract_path=args.continuation_contract,
                continuation_contract_sha256=args.continuation_contract_sha256,
                submission_reservation_path=args.submission_reservation,
            )
        else:
            fresh_required = {
                "--probe-assertions": args.probe_assertions,
                "--resume-assertions": args.resume_assertions,
                "--formal-manifest": args.formal_manifest,
                "--formal-manifest-sha256": args.formal_manifest_sha256,
                "--owner": args.owner,
                "--provenance": args.provenance,
                "--storage-preflight": args.storage_preflight,
                "--submission-reservation": args.submission_reservation,
                "--sampled-memory": args.sampled_memory,
            }
            missing = sorted(name for name, value in fresh_required.items() if value is None)
            if missing:
                raise EvidenceError("fresh verify-formal mode is missing required arguments: " + ", ".join(missing))
            payload = verify_formal(
                run_dir=args.run_dir,
                log_path=args.log,
                world_size=args.world_size,
                probe_assertions=args.probe_assertions,
                resume_assertions=args.resume_assertions,
                experiment_id=args.experiment_id,
                formal_manifest=args.formal_manifest,
                formal_manifest_sha256=args.formal_manifest_sha256,
                owner_path=args.owner,
                provenance_path=args.provenance,
                formal_run_name=args.formal_run_name,
                storage_preflight_path=args.storage_preflight,
                submission_reservation_path=args.submission_reservation,
                sampled_memory_path=args.sampled_memory,
            )
        _write_json(payload, args.output_json)
    except (EvidenceError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
