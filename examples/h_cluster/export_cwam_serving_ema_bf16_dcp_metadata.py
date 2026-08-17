#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export an EMA-only BF16 DCP checkpoint with metadata-based validation.

The exporter streams one DCP tensor chunk at a time. It validates the source
and target using paths, sizes, DCP metadata, tensor shapes and dtypes, and a
full semantic source-to-target readback. No digest or hash chain is produced.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import io
import json
import os
import pickle
import shutil
import stat
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader, _StorageInfo
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torch.distributed.checkpoint.planner import MetadataIndex

from cosmos_framework.checkpoint.dcp import CustomLoadPlanner
from examples.h_cluster import export_cwam_serving_ema_bf16_dcp as common

EMA_PREFIX = "net_ema."
REGULAR_PREFIX = "net."
COPY_BUFFER_BYTES = 8 * 1024 * 1024


class ExportError(RuntimeError):
    """Raised when a source, conversion, or publication gate fails."""


@dataclasses.dataclass(frozen=True)
class Expectations:
    ema_key_count: int
    regular_key_count: int
    shard_count: int
    source_ema_logical_bytes: int
    target_logical_bytes: int


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _require_regular(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ExportError(f"{label} does not exist: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ExportError(f"{label} must be a regular non-symlink file: {path}")
    return metadata


def _require_directory(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ExportError(f"{label} does not exist: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ExportError(f"{label} must be a directory without symlink indirection: {path}")
    return metadata


def _canonical_existing(path: Path, *, directory: bool, label: str) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    resolved = lexical.resolve(strict=True)
    if lexical != resolved:
        raise ExportError(f"{label} must be canonical: provided={lexical}, resolved={resolved}")
    if directory:
        _require_directory(resolved, label)
    else:
        _require_regular(resolved, label)
    return resolved


def _canonical_new(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    parent = _canonical_existing(lexical.parent, directory=True, label=f"{label} parent")
    result = parent / lexical.name
    if result.exists() or result.is_symlink():
        raise ExportError(f"{label} already exists: {result}")
    return result


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _load_metadata(path: Path) -> Metadata:
    try:
        with path.open("rb") as stream:
            metadata = pickle.load(stream)  # noqa: S301 - trusted local training artifact.
    except Exception as error:
        raise ExportError(f"cannot load DCP metadata {path}: {error}") from error
    if not isinstance(metadata, Metadata):
        raise ExportError(f"unexpected DCP metadata type: {type(metadata)!r}")
    if not isinstance(metadata.storage_data, dict) or not metadata.storage_data:
        raise ExportError("source DCP has no filesystem storage metadata")
    return metadata


def _source_inventory(source_model: Path) -> dict[str, int]:
    inventory: dict[str, int] = {}
    for path in sorted(source_model.iterdir(), key=lambda item: item.name):
        metadata = _require_regular(path, f"source entry {path.name}")
        if path.name in inventory:
            raise ExportError(f"duplicate source entry: {path.name}")
        inventory[path.name] = metadata.st_size
    if ".metadata" not in inventory:
        raise ExportError("source model is missing .metadata")
    return inventory


def _logical_bytes(value: TensorStorageMetadata) -> int:
    return int(value.size.numel()) * value.properties.dtype.itemsize


def _validate_source(
    metadata: Metadata,
    inventory: Mapping[str, int],
    expectations: Expectations,
) -> tuple[list[str], list[str], dict[str, list[tuple[MetadataIndex, _StorageInfo]]]]:
    state = metadata.state_dict_metadata
    ema_keys = sorted(key for key in state if key.startswith(EMA_PREFIX))
    regular_keys = sorted(key for key in state if key.startswith(REGULAR_PREFIX))
    other_keys = sorted(set(state) - set(ema_keys) - set(regular_keys))
    if other_keys:
        raise ExportError(f"unexpected model key prefixes: {other_keys[:10]}")
    if len(ema_keys) != expectations.ema_key_count:
        raise ExportError(f"EMA key count mismatch: expected={expectations.ema_key_count}, actual={len(ema_keys)}")
    if len(regular_keys) != expectations.regular_key_count:
        raise ExportError(
            f"regular key count mismatch: expected={expectations.regular_key_count}, actual={len(regular_keys)}"
        )
    if [key.removeprefix(EMA_PREFIX) for key in ema_keys] != [
        key.removeprefix(REGULAR_PREFIX) for key in regular_keys
    ]:
        raise ExportError("regular and EMA key suffix sets differ")

    source_logical_bytes = 0
    target_logical_bytes = 0
    for ema_key in ema_keys:
        ema_value = state[ema_key]
        regular_value = state[f"{REGULAR_PREFIX}{ema_key.removeprefix(EMA_PREFIX)}"]
        if not isinstance(ema_value, TensorStorageMetadata) or not isinstance(
            regular_value, TensorStorageMetadata
        ):
            raise ExportError(f"source key pair is not tensor metadata: {ema_key}")
        if ema_value.properties.dtype != torch.float32:
            raise ExportError(f"EMA tensor is not FP32: {ema_key}")
        if regular_value.properties.dtype != torch.bfloat16:
            raise ExportError(f"regular tensor is not BF16: {ema_key}")
        if ema_value.size != regular_value.size:
            raise ExportError(f"regular and EMA shapes differ: {ema_key}")
        source_logical_bytes += _logical_bytes(ema_value)
        target_logical_bytes += int(ema_value.size.numel()) * torch.bfloat16.itemsize
    if source_logical_bytes != expectations.source_ema_logical_bytes:
        raise ExportError(
            "source EMA logical-byte mismatch: "
            f"expected={expectations.source_ema_logical_bytes}, actual={source_logical_bytes}"
        )
    if target_logical_bytes != expectations.target_logical_bytes:
        raise ExportError(
            f"target logical-byte mismatch: expected={expectations.target_logical_bytes}, actual={target_logical_bytes}"
        )

    storage_by_file: dict[str, list[tuple[MetadataIndex, _StorageInfo]]] = defaultdict(list)
    observed: set[MetadataIndex] = set()
    for index, storage in metadata.storage_data.items():
        if not isinstance(index, MetadataIndex) or not isinstance(storage, _StorageInfo):
            raise ExportError("unsupported source storage metadata types")
        if not index.fqn.startswith(EMA_PREFIX):
            continue
        if storage.relative_path not in inventory:
            raise ExportError(f"storage references an absent source shard: {storage.relative_path}")
        if storage.transform_descriptors not in (None, (), []):
            raise ExportError(f"transformed DCP storage is unsupported: {index}")
        if storage.offset < 0 or storage.length <= 0:
            raise ExportError(f"invalid source storage interval: {index}")
        if storage.offset + storage.length > inventory[storage.relative_path]:
            raise ExportError(f"out-of-bounds source storage interval: {index}")
        storage_by_file[storage.relative_path].append((index, storage))
        observed.add(index)

    shard_names = sorted(storage_by_file)
    if len(shard_names) != expectations.shard_count:
        raise ExportError(f"EMA shard count mismatch: expected={expectations.shard_count}, actual={len(shard_names)}")
    for entries in storage_by_file.values():
        entries.sort(key=lambda item: (item[1].offset, item[0].fqn, item[0].index or -1))
        previous_end = 0
        for index, storage in entries:
            if storage.offset < previous_end:
                raise ExportError(f"overlapping source storage entries: {index}")
            previous_end = storage.offset + storage.length
    expected_chunks = sum(len(state[key].chunks) for key in ema_keys)
    if len(observed) != expected_chunks:
        raise ExportError(f"EMA storage count mismatch: expected={expected_chunks}, actual={len(observed)}")
    return ema_keys, shard_names, storage_by_file


def _progress(phase: str, completed: int, total: int, started: float, shard: str) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = completed / elapsed
    print(
        json.dumps(
            {
                "event": "progress",
                "phase": phase,
                "shard": shard,
                "completed_bytes": completed,
                "total_bytes": total,
                "percent": round(100.0 * completed / total, 3) if total else 100.0,
                "bytes_per_second": round(rate, 3),
                "eta_seconds": round((total - completed) / rate, 3) if rate else None,
                "at": _utc_now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _convert_chunks(
    source_model: Path,
    target_model: Path,
    source_metadata: Metadata,
    shard_names: Sequence[str],
    storage_by_file: Mapping[str, Sequence[tuple[MetadataIndex, _StorageInfo]]],
) -> tuple[dict[MetadataIndex, _StorageInfo], dict[MetadataIndex, tuple[int, ...]]]:
    total = sum(storage.length for entries in storage_by_file.values() for _, storage in entries)
    target_storage: dict[MetadataIndex, _StorageInfo] = {}
    shapes: dict[MetadataIndex, tuple[int, ...]] = {}
    completed = 0
    started = time.monotonic()
    last_progress = started
    for shard_name in shard_names:
        source_path = source_model / shard_name
        target_path = target_model / shard_name
        source_identity = _identity(_require_regular(source_path, f"source shard {shard_name}"))
        with source_path.open("rb", buffering=0) as source_stream, target_path.open("xb", buffering=0) as target_stream:
            for index, storage in storage_by_file[shard_name]:
                source_stream.seek(storage.offset)
                payload = source_stream.read(storage.length)
                if len(payload) != storage.length:
                    raise ExportError(f"short source read: {index}")
                tensor = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
                del payload
                if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                    raise ExportError(f"source EMA chunk is not a CPU FP32 tensor: {index}")
                expected_shape = common._source_chunk_shape(source_metadata, index)
                if tuple(tensor.shape) != expected_shape:
                    raise ExportError(f"source chunk shape mismatch: {index}")
                target_tensor = tensor.to(dtype=torch.bfloat16)
                del tensor
                offset = target_stream.tell()
                torch.save(target_tensor, target_stream)
                del target_tensor
                length = target_stream.tell() - offset
                if length <= 0:
                    raise ExportError(f"empty target chunk: {index}")
                target_storage[index] = _StorageInfo(shard_name, offset, length, None)
                shapes[index] = expected_shape
                completed += storage.length
                now = time.monotonic()
                if now - last_progress >= 60:
                    _progress("convert", completed, total, started, shard_name)
                    last_progress = now
            target_stream.flush()
            os.fsync(target_stream.fileno())
        if _identity(_require_regular(source_path, f"source shard {shard_name}")) != source_identity:
            raise ExportError(f"source shard changed during conversion: {source_path}")
        _progress("convert", completed, total, started, shard_name)
    return target_storage, shapes


def _read_tensor(stream: object, offset: int, length: int, label: str) -> torch.Tensor:
    stream.seek(offset)
    payload = stream.read(length)
    if len(payload) != length:
        raise ExportError(f"short serialized tensor read: {label}")
    tensor = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise ExportError(f"serialized value is not a tensor: {label}")
    return tensor


def _semantic_readback(
    source_model: Path,
    target_model: Path,
    shard_names: Sequence[str],
    source_storage: Mapping[str, Sequence[tuple[MetadataIndex, _StorageInfo]]],
    target_storage: Mapping[MetadataIndex, _StorageInfo],
    shapes: Mapping[MetadataIndex, tuple[int, ...]],
) -> tuple[list[dict[str, object]], int]:
    total = sum((target_model / name).stat().st_size for name in shard_names)
    completed = 0
    verified = 0
    started = time.monotonic()
    files: list[dict[str, object]] = []
    for shard_name in shard_names:
        target_path = target_model / shard_name
        target_identity = _identity(_require_regular(target_path, f"target shard {shard_name}"))
        expected_target_offset = 0
        with (source_model / shard_name).open("rb", buffering=0) as source_stream, target_path.open(
            "rb", buffering=0
        ) as target_stream:
            for index, source_info in source_storage[shard_name]:
                target_info = target_storage[index]
                if target_info.offset != expected_target_offset:
                    raise ExportError(f"target shard contains a gap or overlap: {index}")
                source_tensor = _read_tensor(
                    source_stream, source_info.offset, source_info.length, f"source {index}"
                )
                target_tensor = _read_tensor(
                    target_stream, target_info.offset, target_info.length, f"target {index}"
                )
                if source_tensor.dtype != torch.float32 or target_tensor.dtype != torch.bfloat16:
                    raise ExportError(f"source or target dtype mismatch: {index}")
                if tuple(target_tensor.shape) != shapes[index]:
                    raise ExportError(f"target shape mismatch: {index}")
                if not torch.equal(source_tensor.to(dtype=torch.bfloat16), target_tensor):
                    raise ExportError(f"target tensor differs from the BF16 EMA source: {index}")
                expected_target_offset += target_info.length
                completed += target_info.length
                verified += 1
            if target_stream.read(1):
                raise ExportError(f"target shard contains trailing bytes: {target_path}")
        metadata = _require_regular(target_path, f"target shard {shard_name}")
        if _identity(metadata) != target_identity or metadata.st_size != expected_target_offset:
            raise ExportError(f"target shard changed or has an indexed-size mismatch: {target_path}")
        files.append({"relative_path": shard_name, "size_bytes": metadata.st_size})
        _progress("semantic_readback", completed, total, started, shard_name)
    return files, verified


def _planner_samples(
    source_model: Path,
    target_model: Path,
    source_metadata: Metadata,
    target_metadata: Metadata,
    sample_keys: Sequence[str],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for ema_key in sample_keys:
        if not ema_key.startswith(EMA_PREFIX) or ema_key not in target_metadata.state_dict_metadata:
            raise ExportError(f"sample key is missing from target: {ema_key}")
        target_value = target_metadata.state_dict_metadata[ema_key]
        regular_key = f"{REGULAR_PREFIX}{ema_key.removeprefix(EMA_PREFIX)}"
        explicit_source = common._load_one_tensor(
            checkpoint_path=source_model,
            key=ema_key,
            metadata=source_metadata,
            planner=None,
        ).to(dtype=torch.bfloat16)
        explicit_target = common._load_one_tensor(
            checkpoint_path=target_model,
            key=ema_key,
            metadata=target_metadata,
            planner=None,
        )
        planner_target = {regular_key: torch.empty(tuple(target_value.size), dtype=torch.bfloat16)}
        dcp.load(
            state_dict=planner_target,
            storage_reader=FileSystemReader(str(target_model)),
            planner=CustomLoadPlanner(load_ema_to_reg=True),
        )
        if not torch.equal(explicit_source, explicit_target) or not torch.equal(
            explicit_source, planner_target[regular_key]
        ):
            raise ExportError(f"CustomLoadPlanner sample validation failed: {ema_key}")
        results.append(
            {
                "ema_key": ema_key,
                "regular_target_key": regular_key,
                "shape": list(explicit_target.shape),
                "dtype": str(explicit_target.dtype),
                "source_matches_target": True,
                "planner_selects_ema": True,
            }
        )
    return results


def _copy_and_compare(source: Path, target: Path) -> int:
    source_identity = _identity(_require_regular(source, "source config"))
    with source.open("rb") as source_stream, target.open("xb") as target_stream:
        shutil.copyfileobj(source_stream, target_stream, length=COPY_BUFFER_BYTES)
        target_stream.flush()
        os.fsync(target_stream.fileno())
    if _identity(_require_regular(source, "source config")) != source_identity:
        raise ExportError("source config changed while copying")
    with source.open("rb") as source_stream, target.open("rb") as target_stream:
        while True:
            source_chunk = source_stream.read(COPY_BUFFER_BYTES)
            target_chunk = target_stream.read(COPY_BUFFER_BYTES)
            if source_chunk != target_chunk:
                raise ExportError("copied config differs from source")
            if not source_chunk:
                break
    return target.stat().st_size


def _seal(root: Path) -> None:
    directories: list[Path] = []
    for current_root, directory_names, file_names in os.walk(root):
        current = Path(current_root)
        directories.append(current)
        for name in directory_names:
            _require_directory(current / name, "artifact directory")
        for name in file_names:
            path = current / name
            _require_regular(path, "artifact file")
            path.chmod(0o444)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)


def export_checkpoint(
    source_model: Path,
    source_config: Path,
    final_artifact: Path,
    expectations: Expectations,
    sample_keys: Sequence[str],
    source_run_id: str,
    source_git_commit: str,
    seal: bool = True,
) -> dict[str, object]:
    source_model = _canonical_existing(source_model, directory=True, label="source model")
    source_config = _canonical_existing(source_config, directory=False, label="source config")
    final_artifact = _canonical_new(final_artifact, "final artifact")
    final_model = final_artifact / "model"
    stage = final_artifact.parent / f".{final_artifact.name}.incomplete-{uuid.uuid4().hex[:10]}"
    started_at = _utc_now()
    started = time.monotonic()
    print(json.dumps({"event": "export_start", "source": str(source_model), "stage": str(stage)}), flush=True)
    try:
        stage.mkdir(mode=0o700)
        target_model = stage / "model"
        target_model.mkdir(mode=0o700)
        inventory = _source_inventory(source_model)
        source_identities = {
            name: _identity(_require_regular(source_model / name, f"source file {name}")) for name in inventory
        }
        source_metadata = _load_metadata(source_model / ".metadata")
        ema_keys, shard_names, source_storage = _validate_source(source_metadata, inventory, expectations)
        target_storage, shapes = _convert_chunks(
            source_model, target_model, source_metadata, shard_names, source_storage
        )
        target_metadata = common._target_metadata(
            source_metadata=source_metadata,
            ema_keys=ema_keys,
            target_storage=target_storage,
            final_model=final_model,
        )
        common._write_metadata(target_model / ".metadata", target_metadata)
        shard_files, verified_chunks = _semantic_readback(
            source_model,
            target_model,
            shard_names,
            source_storage,
            target_storage,
            shapes,
        )
        expected_chunks = sum(len(value.chunks) for value in target_metadata.state_dict_metadata.values())
        if verified_chunks != expected_chunks:
            raise ExportError(f"readback chunk count mismatch: expected={expected_chunks}, actual={verified_chunks}")
        samples = _planner_samples(
            source_model, target_model, source_metadata, target_metadata, sample_keys
        )
        for name, identity in source_identities.items():
            if _identity(_require_regular(source_model / name, f"source file {name}")) != identity:
                raise ExportError(f"source file changed during export: {name}")

        config_size = _copy_and_compare(source_config, stage / "config.yaml")
        model_files = [
            {"relative_path": ".metadata", "size_bytes": (target_model / ".metadata").stat().st_size},
            *shard_files,
        ]
        artifact_manifest = {
            "schema_version": 1,
            "kind": "cwam_serving_checkpoint_size_manifest",
            "checkpoint_path": str(final_model),
            "files": model_files,
            "file_count": len(model_files),
            "total_physical_bytes": sum(int(record["size_bytes"]) for record in model_files),
        }
        _write_json(stage / "serving-checkpoint-size-manifest.json", artifact_manifest)
        receipt = {
            "schema_version": 1,
            "kind": "cwam_serving_ema_bf16_metadata_receipt",
            "status": "succeeded",
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "source": {
                "run_id": source_run_id,
                "git_commit": source_git_commit,
                "model_path": str(source_model),
                "file_count": len(inventory),
                "total_physical_bytes": sum(inventory.values()),
                "config_path": str(source_config),
                "config_size_bytes": config_size,
            },
            "selection": {
                "included_prefix": EMA_PREFIX,
                "excluded_prefix": REGULAR_PREFIX,
                "source_dtype": str(torch.float32),
                "target_dtype": str(torch.bfloat16),
                "key_count": len(ema_keys),
                "chunk_count": verified_chunks,
                "source_logical_bytes": expectations.source_ema_logical_bytes,
                "target_logical_bytes": expectations.target_logical_bytes,
            },
            "validation": {
                "source_file_identities_stable": True,
                "full_semantic_readback": True,
                "custom_load_planner_samples": samples,
            },
            "target": {
                "model_path": str(final_model),
                "shard_count": len(shard_names),
                "file_count": len(model_files),
                "total_physical_bytes": artifact_manifest["total_physical_bytes"],
            },
            "runtime": {"python": sys.version, "torch": torch.__version__},
        }
        _write_json(stage / "ema-bf16-metadata-receipt.json", receipt)
        if seal:
            _seal(stage)
        os.replace(stage, final_artifact)
        result = {
            "status": "succeeded",
            "final_artifact_path": str(final_artifact),
            "final_model_path": str(final_model),
            "key_count": len(ema_keys),
            "chunk_count": verified_chunks,
            "shard_count": len(shard_names),
            "target_logical_bytes": expectations.target_logical_bytes,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        print(json.dumps({"event": "export_complete", **result}, sort_keys=True), flush=True)
        return result
    except Exception:
        print(json.dumps({"event": "export_failed", "stage": str(stage), "at": _utc_now()}), file=sys.stderr)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--final-artifact", type=Path, required=True)
    parser.add_argument("--expected-ema-key-count", type=int, required=True)
    parser.add_argument("--expected-regular-key-count", type=int, required=True)
    parser.add_argument("--expected-shard-count", type=int, required=True)
    parser.add_argument("--expected-source-ema-logical-bytes", type=int, required=True)
    parser.add_argument("--expected-target-logical-bytes", type=int, required=True)
    parser.add_argument("--sample-key", action="append", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-git-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    expectations = Expectations(
        ema_key_count=args.expected_ema_key_count,
        regular_key_count=args.expected_regular_key_count,
        shard_count=args.expected_shard_count,
        source_ema_logical_bytes=args.expected_source_ema_logical_bytes,
        target_logical_bytes=args.expected_target_logical_bytes,
    )
    try:
        export_checkpoint(
            source_model=args.source_model,
            source_config=args.source_config,
            final_artifact=args.final_artifact,
            expectations=expectations,
            sample_keys=args.sample_key,
            source_run_id=args.source_run_id,
            source_git_commit=args.source_git_commit,
        )
    except (ExportError, OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
