#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export a streaming, serving-only EMA BF16 PyTorch DCP checkpoint.

The training checkpoint contains two complete trees:

* ``net.*``: regular BF16 training weights;
* ``net_ema.*``: FP32 EMA weights.

Inference needs only one of them.  This exporter copies only ``net_ema.*``,
converts every storage chunk to BF16, and preserves the ``net_ema.*`` names so
``CustomLoadPlanner(load_ema_to_reg=True)`` remains the explicit selection
mechanism.  It never materializes a full model: one serialized DCP chunk is
loaded, converted, written, and released at a time.

Publication is fail-closed.  Work is written below a sibling
``.incomplete-*`` directory, every target chunk is then read back and checked,
and only a completely verified, read-only artifact is atomically renamed to
the requested final path.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import stat
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader, _StorageInfo
from torch.distributed.checkpoint.metadata import (
    Metadata,
    StorageMeta,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import MetadataIndex

from cosmos_framework.checkpoint.dcp import CustomLoadPlanner

SCHEMA_VERSION = 1
EMA_PREFIX = "net_ema."
REGULAR_PREFIX = "net."
HASH_CHUNK_BYTES = 8 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EmaBf16ExportError(RuntimeError):
    """Raised when a source, conversion, verification, or publication gate fails."""


@dataclasses.dataclass(frozen=True)
class ExportExpectations:
    """Pinned source facts that prevent exporting the wrong checkpoint."""

    origin_manifest_sha256: str
    origin_tree_sha256: str
    source_metadata_sha256: str
    source_config_sha256: str
    ema_key_count: int
    regular_key_count: int
    shard_count: int
    source_ema_logical_bytes: int
    target_logical_bytes: int


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> tuple[int, str]:
    metadata_before = _require_regular_file(path, str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    metadata_after = _require_regular_file(path, str(path))
    if _stable_identity(metadata_before) != _stable_identity(metadata_after):
        raise EmaBf16ExportError(f"file changed while hashing: {path}")
    return metadata_before.st_size, digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    byte_view = contiguous.view(torch.uint8).numpy()
    return hashlib.sha256(byte_view).hexdigest()


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise EmaBf16ExportError(f"{label} does not exist: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EmaBf16ExportError(f"{label} must be a non-symlink regular file: {path}")
    return metadata


def _require_directory(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise EmaBf16ExportError(f"{label} does not exist: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise EmaBf16ExportError(f"{label} must be a non-symlink directory: {path}")
    return metadata


def _canonical_existing_path(path: Path, *, directory: bool, label: str) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as error:
        raise EmaBf16ExportError(f"{label} does not exist: {lexical}") from error
    if lexical != resolved:
        raise EmaBf16ExportError(
            f"{label} must be canonical and contain no symlink components: "
            f"provided={lexical}, resolved={resolved}"
        )
    if directory:
        _require_directory(resolved, label)
    else:
        _require_regular_file(resolved, label)
    return resolved


def _canonical_new_path(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    parent = _canonical_existing_path(lexical.parent, directory=True, label=f"{label} parent")
    canonical = parent / lexical.name
    if ".incomplete" in canonical.name:
        raise EmaBf16ExportError(f"{label} final name must not contain '.incomplete': {canonical}")
    if canonical.exists() or canonical.is_symlink():
        raise EmaBf16ExportError(f"{label} already exists; refusing to overwrite: {canonical}")
    return canonical


def _require_sha256(value: str, label: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise EmaBf16ExportError(f"{label} must be a lowercase SHA-256, got {value!r}")
    return value


def _load_json_no_duplicates(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EmaBf16ExportError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EmaBf16ExportError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise EmaBf16ExportError(f"JSON root must be an object: {path}")
    return value


def _validate_origin_manifest(
    *,
    source_model: Path,
    origin_manifest_path: Path,
    expectations: ExportExpectations,
) -> tuple[dict[str, dict[str, object]], str]:
    manifest_size, manifest_sha = _hash_file(origin_manifest_path)
    del manifest_size
    if manifest_sha != expectations.origin_manifest_sha256:
        raise EmaBf16ExportError(
            "origin manifest SHA-256 mismatch: "
            f"expected={expectations.origin_manifest_sha256}, actual={manifest_sha}"
        )

    manifest = _load_json_no_duplicates(origin_manifest_path)
    required = {"schema_version", "kind", "checkpoint_path", "files", "tree_sha256"}
    if set(manifest) != required:
        raise EmaBf16ExportError(
            f"origin manifest keys mismatch: expected={sorted(required)}, actual={sorted(manifest)}"
        )
    if manifest["schema_version"] != 1 or manifest["kind"] != "cwam_serving_checkpoint_manifest":
        raise EmaBf16ExportError("origin manifest schema or kind is not the pinned serving-manifest format")
    if manifest["checkpoint_path"] != str(source_model):
        raise EmaBf16ExportError(
            f"origin manifest checkpoint path mismatch: expected={source_model}, actual={manifest['checkpoint_path']}"
        )
    if manifest["tree_sha256"] != expectations.origin_tree_sha256:
        raise EmaBf16ExportError(
            "origin tree SHA-256 mismatch: "
            f"expected={expectations.origin_tree_sha256}, actual={manifest['tree_sha256']}"
        )
    raw_files = manifest["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise EmaBf16ExportError("origin manifest files must be a non-empty list")

    records: list[dict[str, object]] = []
    by_path: dict[str, dict[str, object]] = {}
    for raw_record in raw_files:
        if not isinstance(raw_record, dict) or set(raw_record) != {"relative_path", "size_bytes", "sha256"}:
            raise EmaBf16ExportError(f"invalid origin manifest file record: {raw_record!r}")
        relative_path = raw_record["relative_path"]
        size_bytes = raw_record["size_bytes"]
        sha256 = raw_record["sha256"]
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or Path(relative_path).as_posix() != relative_path
            or ".." in Path(relative_path).parts
        ):
            raise EmaBf16ExportError(f"invalid origin relative path: {relative_path!r}")
        if relative_path in by_path:
            raise EmaBf16ExportError(f"duplicate origin manifest path: {relative_path}")
        if type(size_bytes) is not int or size_bytes < 0:
            raise EmaBf16ExportError(f"invalid origin size for {relative_path}: {size_bytes!r}")
        if not isinstance(sha256, str):
            raise EmaBf16ExportError(f"invalid origin SHA-256 for {relative_path}: {sha256!r}")
        _require_sha256(sha256, f"origin file {relative_path} SHA-256")
        record = {
            "relative_path": relative_path,
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
        records.append(record)
        by_path[relative_path] = record

    if [record["relative_path"] for record in records] != sorted(by_path):
        raise EmaBf16ExportError("origin manifest files are not sorted")
    if _canonical_digest(records) != expectations.origin_tree_sha256:
        raise EmaBf16ExportError("origin manifest file records do not reproduce the pinned tree SHA-256")

    actual_names = sorted(path.name for path in source_model.iterdir())
    if actual_names != sorted(by_path):
        raise EmaBf16ExportError(
            f"source model file set differs from origin manifest: expected={sorted(by_path)}, actual={actual_names}"
        )
    total_bytes = sum(int(record["size_bytes"]) for record in by_path.values())
    completed_bytes = 0
    hash_started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "source_full_hash_start",
                "file_count": len(by_path),
                "total_bytes": total_bytes,
                "at": _utc_now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for file_index, (relative_path, record) in enumerate(sorted(by_path.items()), start=1):
        if "/" in relative_path:
            raise EmaBf16ExportError(f"source DCP must be flat, found nested manifest path: {relative_path}")
        path = source_model / relative_path
        metadata = _require_regular_file(path, f"source file {relative_path}")
        if metadata.st_size != record["size_bytes"]:
            raise EmaBf16ExportError(
                f"source file size mismatch for {relative_path}: "
                f"expected={record['size_bytes']}, actual={metadata.st_size}"
            )
        actual_size, actual_sha = _hash_file(path)
        if actual_size != record["size_bytes"] or actual_sha != record["sha256"]:
            raise EmaBf16ExportError(
                f"source file SHA-256 mismatch for {relative_path}: "
                f"expected={record['sha256']}, actual={actual_sha}"
            )
        completed_bytes += actual_size
        _progress(
            phase="source_full_hash",
            completed_bytes=completed_bytes,
            total_bytes=total_bytes,
            started=hash_started,
            extra={
                "verified_file": relative_path,
                "file_index": file_index,
                "file_count": len(by_path),
            },
        )

    metadata_record = by_path.get(".metadata")
    if metadata_record is None or metadata_record["sha256"] != expectations.source_metadata_sha256:
        raise EmaBf16ExportError("origin manifest does not bind the expected source .metadata SHA-256")
    return by_path, manifest_sha


def _load_source_metadata(path: Path) -> Metadata:
    try:
        with path.open("rb") as stream:
            metadata = pickle.load(stream)  # noqa: S301 - trusted, SHA-pinned PyTorch DCP metadata.
    except Exception as error:
        raise EmaBf16ExportError(f"cannot load source DCP metadata {path}: {error}") from error
    if not isinstance(metadata, Metadata):
        raise EmaBf16ExportError(f"source .metadata is not a PyTorch Metadata object: {type(metadata)!r}")
    if not isinstance(metadata.storage_data, dict) or not metadata.storage_data:
        raise EmaBf16ExportError("source DCP metadata has no filesystem storage_data")
    return metadata


def _logical_bytes(metadata: TensorStorageMetadata) -> int:
    return int(metadata.size.numel()) * metadata.properties.dtype.itemsize


def _validate_source_metadata(
    metadata: Metadata,
    origin_files: Mapping[str, Mapping[str, object]],
    expectations: ExportExpectations,
) -> tuple[list[str], list[str], dict[str, list[tuple[MetadataIndex, _StorageInfo]]]]:
    state_metadata = metadata.state_dict_metadata
    if not isinstance(state_metadata, dict) or not state_metadata:
        raise EmaBf16ExportError("source DCP state_dict_metadata is empty")
    ema_keys = sorted(key for key in state_metadata if key.startswith(EMA_PREFIX))
    regular_keys = sorted(key for key in state_metadata if key.startswith(REGULAR_PREFIX))
    other_keys = sorted(set(state_metadata) - set(ema_keys) - set(regular_keys))
    if other_keys:
        raise EmaBf16ExportError(f"source model DCP contains unexpected key prefixes: {other_keys[:10]}")
    if len(ema_keys) != expectations.ema_key_count:
        raise EmaBf16ExportError(
            f"EMA key count mismatch: expected={expectations.ema_key_count}, actual={len(ema_keys)}"
        )
    if len(regular_keys) != expectations.regular_key_count:
        raise EmaBf16ExportError(
            f"regular key count mismatch: expected={expectations.regular_key_count}, actual={len(regular_keys)}"
        )
    if [key.removeprefix(EMA_PREFIX) for key in ema_keys] != [
        key.removeprefix(REGULAR_PREFIX) for key in regular_keys
    ]:
        raise EmaBf16ExportError("regular and EMA key suffix sets differ")

    source_ema_bytes = 0
    target_bytes = 0
    for key in ema_keys:
        value = state_metadata[key]
        regular_value = state_metadata[f"{REGULAR_PREFIX}{key.removeprefix(EMA_PREFIX)}"]
        if not isinstance(value, TensorStorageMetadata) or not isinstance(regular_value, TensorStorageMetadata):
            raise EmaBf16ExportError(f"source key pair is not tensor metadata: {key}")
        if value.properties.dtype != torch.float32:
            raise EmaBf16ExportError(f"EMA tensor must be FP32, got {value.properties.dtype} for {key}")
        if regular_value.properties.dtype != torch.bfloat16:
            raise EmaBf16ExportError(
                f"regular tensor must be BF16, got {regular_value.properties.dtype} for {key}"
            )
        if value.size != regular_value.size:
            raise EmaBf16ExportError(f"regular and EMA tensor sizes differ for {key}")
        source_ema_bytes += _logical_bytes(value)
        target_bytes += int(value.size.numel()) * torch.bfloat16.itemsize
    if source_ema_bytes != expectations.source_ema_logical_bytes:
        raise EmaBf16ExportError(
            "source EMA logical-byte mismatch: "
            f"expected={expectations.source_ema_logical_bytes}, actual={source_ema_bytes}"
        )
    if target_bytes != expectations.target_logical_bytes:
        raise EmaBf16ExportError(
            f"target logical-byte mismatch: expected={expectations.target_logical_bytes}, actual={target_bytes}"
        )

    storage_by_file: dict[str, list[tuple[MetadataIndex, _StorageInfo]]] = defaultdict(list)
    observed_indexes: set[MetadataIndex] = set()
    for index, storage in metadata.storage_data.items():
        if not isinstance(index, MetadataIndex) or not isinstance(storage, _StorageInfo):
            raise EmaBf16ExportError(
                f"unsupported source storage metadata types: index={type(index)!r}, storage={type(storage)!r}"
            )
        if not index.fqn.startswith(EMA_PREFIX):
            continue
        if index.fqn not in state_metadata:
            raise EmaBf16ExportError(f"storage index references unknown EMA key: {index.fqn}")
        if storage.transform_descriptors not in (None, (), []):
            raise EmaBf16ExportError(f"transformed DCP storage is unsupported for {index.fqn}")
        if storage.relative_path not in origin_files:
            raise EmaBf16ExportError(f"storage references a file absent from origin manifest: {storage.relative_path}")
        storage_by_file[storage.relative_path].append((index, storage))
        observed_indexes.add(index)

    shard_names = sorted(storage_by_file)
    if len(shard_names) != expectations.shard_count:
        raise EmaBf16ExportError(
            f"EMA shard count mismatch: expected={expectations.shard_count}, actual={len(shard_names)}"
        )
    for name, entries in storage_by_file.items():
        if name == ".metadata" or "/" in name:
            raise EmaBf16ExportError(f"invalid DCP shard path: {name}")
        entries.sort(key=lambda item: (item[1].offset, item[0].fqn, item[0].index or -1))
        previous_end = 0
        source_size = int(origin_files[name]["size_bytes"])
        for index, storage in entries:
            if storage.offset < 0 or storage.length <= 0 or storage.offset + storage.length > source_size:
                raise EmaBf16ExportError(f"out-of-bounds source storage for {index}: {storage}")
            if storage.offset < previous_end:
                raise EmaBf16ExportError(f"overlapping EMA storage entries in {name}")
            previous_end = storage.offset + storage.length

    expected_chunk_count = sum(
        len(state_metadata[key].chunks) for key in ema_keys if isinstance(state_metadata[key], TensorStorageMetadata)
    )
    if len(observed_indexes) != expected_chunk_count:
        raise EmaBf16ExportError(
            f"EMA storage-index count mismatch: expected={expected_chunk_count}, actual={len(observed_indexes)}"
        )
    return ema_keys, shard_names, storage_by_file


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _progress(
    *,
    phase: str,
    completed_bytes: int,
    total_bytes: int,
    started: float,
    extra: Mapping[str, object] | None = None,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = completed_bytes / elapsed
    remaining = max(total_bytes - completed_bytes, 0)
    event: dict[str, object] = {
        "event": "progress",
        "phase": phase,
        "completed_bytes": completed_bytes,
        "total_bytes": total_bytes,
        "percent": round(100.0 * completed_bytes / total_bytes, 3) if total_bytes else 100.0,
        "elapsed_seconds": round(elapsed, 3),
        "bytes_per_second": round(rate, 3),
        "eta_seconds": round(remaining / rate, 3) if rate else None,
        "at": _utc_now(),
    }
    if extra:
        event.update(extra)
    print(json.dumps(event, sort_keys=True), flush=True)


def _source_chunk_shape(
    metadata: Metadata,
    index: MetadataIndex,
) -> tuple[int, ...]:
    tensor_metadata = metadata.state_dict_metadata[index.fqn]
    if not isinstance(tensor_metadata, TensorStorageMetadata):
        raise EmaBf16ExportError(f"non-tensor metadata for {index.fqn}")
    matches = [
        chunk
        for chunk in tensor_metadata.chunks
        if tuple(chunk.offsets) == tuple(index.offset or ())
        or (index.index is not None and tensor_metadata.chunks.index(chunk) == index.index)
    ]
    if len(matches) != 1:
        raise EmaBf16ExportError(f"cannot resolve one metadata chunk for {index}")
    return tuple(int(value) for value in matches[0].sizes)


def _convert_chunks(
    *,
    source_model: Path,
    target_model: Path,
    source_metadata: Metadata,
    shard_names: Sequence[str],
    storage_by_file: Mapping[str, Sequence[tuple[MetadataIndex, _StorageInfo]]],
    total_source_bytes: int,
) -> tuple[
    dict[MetadataIndex, _StorageInfo],
    dict[str, list[dict[str, object]]],
    dict[MetadataIndex, dict[str, object]],
]:
    target_storage: dict[MetadataIndex, _StorageInfo] = {}
    key_chunks: dict[str, list[dict[str, object]]] = defaultdict(list)
    chunk_records: dict[MetadataIndex, dict[str, object]] = {}
    completed = 0
    started = time.monotonic()
    last_progress = started

    for shard_index, shard_name in enumerate(shard_names, start=1):
        source_path = source_model / shard_name
        target_path = target_model / shard_name
        source_identity = _stable_identity(_require_regular_file(source_path, f"source shard {shard_name}"))
        with source_path.open("rb", buffering=0) as source_stream, target_path.open("xb", buffering=0) as target_stream:
            for metadata_index, storage in storage_by_file[shard_name]:
                source_stream.seek(storage.offset)
                payload = source_stream.read(storage.length)
                if len(payload) != storage.length:
                    raise EmaBf16ExportError(
                        f"short source read for {metadata_index}: expected={storage.length}, actual={len(payload)}"
                    )
                source_serialized_sha = _sha256_bytes(payload)
                try:
                    source_tensor = torch.load(
                        io.BytesIO(payload),
                        map_location="cpu",
                        weights_only=True,
                    )
                except Exception as error:
                    raise EmaBf16ExportError(
                        f"cannot deserialize source DCP chunk {metadata_index}: {error}"
                    ) from error
                del payload
                if not isinstance(source_tensor, torch.Tensor):
                    raise EmaBf16ExportError(f"source DCP chunk is not a tensor: {metadata_index}")
                if source_tensor.device.type != "cpu" or source_tensor.dtype != torch.float32:
                    raise EmaBf16ExportError(
                        f"source EMA chunk must be CPU FP32 for {metadata_index}, "
                        f"got device={source_tensor.device}, dtype={source_tensor.dtype}"
                    )
                expected_shape = _source_chunk_shape(source_metadata, metadata_index)
                if tuple(source_tensor.shape) != expected_shape:
                    raise EmaBf16ExportError(
                        f"source chunk shape mismatch for {metadata_index}: "
                        f"expected={expected_shape}, actual={tuple(source_tensor.shape)}"
                    )

                target_tensor = source_tensor.to(dtype=torch.bfloat16)
                del source_tensor
                tensor_sha = _tensor_sha256(target_tensor)
                target_offset = target_stream.tell()
                torch.save(target_tensor, target_stream)
                target_end = target_stream.tell()
                target_length = target_end - target_offset
                del target_tensor
                if target_length <= 0:
                    raise EmaBf16ExportError(f"empty serialized target chunk for {metadata_index}")

                target_info = _StorageInfo(
                    relative_path=shard_name,
                    offset=target_offset,
                    length=target_length,
                    transform_descriptors=None,
                )
                target_storage[metadata_index] = target_info
                record: dict[str, object] = {
                    "chunk_index": metadata_index.index,
                    "global_offset": list(metadata_index.offset or ()),
                    "shape": list(expected_shape),
                    "source_storage": {
                        "relative_path": storage.relative_path,
                        "offset": storage.offset,
                        "length": storage.length,
                        "serialized_sha256": source_serialized_sha,
                    },
                    "target_storage": {
                        "relative_path": shard_name,
                        "offset": target_offset,
                        "length": target_length,
                        "serialized_sha256": None,
                    },
                    "bf16_tensor_sha256": tensor_sha,
                }
                key_chunks[metadata_index.fqn].append(record)
                chunk_records[metadata_index] = record
                completed += storage.length
                now = time.monotonic()
                if now - last_progress >= 60:
                    _progress(
                        phase="convert",
                        completed_bytes=completed,
                        total_bytes=total_source_bytes,
                        started=started,
                        extra={"shard": shard_name, "shard_index": shard_index, "shard_count": len(shard_names)},
                    )
                    last_progress = now
            target_stream.flush()
            os.fsync(target_stream.fileno())
        if _stable_identity(_require_regular_file(source_path, f"source shard {shard_name}")) != source_identity:
            raise EmaBf16ExportError(f"source shard changed during conversion: {source_path}")
        _progress(
            phase="convert",
            completed_bytes=completed,
            total_bytes=total_source_bytes,
            started=started,
            extra={"completed_shard": shard_name, "shard_index": shard_index, "shard_count": len(shard_names)},
        )
    return target_storage, key_chunks, chunk_records


def _target_metadata(
    *,
    source_metadata: Metadata,
    ema_keys: Sequence[str],
    target_storage: Mapping[MetadataIndex, _StorageInfo],
    final_model: Path,
) -> Metadata:
    state_metadata: dict[str, TensorStorageMetadata] = {}
    for key in ema_keys:
        value = source_metadata.state_dict_metadata[key]
        if not isinstance(value, TensorStorageMetadata):
            raise EmaBf16ExportError(f"EMA key is not tensor metadata: {key}")
        state_metadata[key] = dataclasses.replace(
            value,
            properties=dataclasses.replace(value.properties, dtype=torch.bfloat16),
        )

    planner_data = source_metadata.planner_data
    if planner_data is not None:
        if not isinstance(planner_data, dict):
            raise EmaBf16ExportError(f"unsupported planner_data type: {type(planner_data)!r}")
        planner_data = {key: planner_data[key] for key in ema_keys}

    if source_metadata.storage_meta is None:
        storage_meta = StorageMeta(
            checkpoint_id=str(final_model),
            save_id=str(uuid.uuid4()),
            load_id=None,
        )
    else:
        storage_meta = dataclasses.replace(
            source_metadata.storage_meta,
            checkpoint_id=str(final_model),
            save_id=str(uuid.uuid4()),
            load_id=None,
        )
    return Metadata(
        state_dict_metadata=state_metadata,
        planner_data=planner_data,
        storage_data=dict(target_storage),
        storage_meta=storage_meta,
        version=source_metadata.version,
    )


def _write_metadata(path: Path, metadata: Metadata) -> None:
    temporary = path.with_name(".metadata.tmp")
    with temporary.open("xb") as stream:
        pickle.dump(metadata, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _readback_chunks(
    *,
    target_model: Path,
    shard_names: Sequence[str],
    target_storage: Mapping[MetadataIndex, _StorageInfo],
    chunk_records: Mapping[MetadataIndex, dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    by_file: dict[str, list[tuple[MetadataIndex, _StorageInfo]]] = defaultdict(list)
    for index, storage in target_storage.items():
        by_file[storage.relative_path].append((index, storage))
    for entries in by_file.values():
        entries.sort(key=lambda item: item[1].offset)

    total_bytes = sum((target_model / name).stat().st_size for name in shard_names)
    completed = 0
    started = time.monotonic()
    last_progress = started
    files: list[dict[str, object]] = []
    verified_chunks = 0

    for shard_index, shard_name in enumerate(shard_names, start=1):
        path = target_model / shard_name
        identity = _stable_identity(_require_regular_file(path, f"target shard {shard_name}"))
        file_digest = hashlib.sha256()
        expected_offset = 0
        with path.open("rb", buffering=0) as stream:
            for metadata_index, storage in by_file[shard_name]:
                if storage.offset != expected_offset:
                    raise EmaBf16ExportError(
                        f"target shard has a gap or overlap at {metadata_index}: "
                        f"expected_offset={expected_offset}, actual={storage.offset}"
                    )
                payload = stream.read(storage.length)
                if len(payload) != storage.length:
                    raise EmaBf16ExportError(
                        f"short target read for {metadata_index}: expected={storage.length}, actual={len(payload)}"
                    )
                file_digest.update(payload)
                serialized_sha = _sha256_bytes(payload)
                record = chunk_records[metadata_index]
                target_record = record["target_storage"]
                if not isinstance(target_record, dict):
                    raise AssertionError("internal target-storage record is not a dictionary")
                target_record["serialized_sha256"] = serialized_sha
                try:
                    tensor = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
                except Exception as error:
                    raise EmaBf16ExportError(
                        f"cannot deserialize target DCP chunk {metadata_index}: {error}"
                    ) from error
                del payload
                if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
                    raise EmaBf16ExportError(f"target chunk is not a BF16 tensor: {metadata_index}")
                expected_shape = tuple(record["shape"])
                if tuple(tensor.shape) != expected_shape:
                    raise EmaBf16ExportError(
                        f"target shape mismatch for {metadata_index}: "
                        f"expected={expected_shape}, actual={tuple(tensor.shape)}"
                    )
                actual_tensor_sha = _tensor_sha256(tensor)
                del tensor
                if actual_tensor_sha != record["bf16_tensor_sha256"]:
                    raise EmaBf16ExportError(
                        f"target semantic SHA-256 mismatch for {metadata_index}: "
                        f"expected={record['bf16_tensor_sha256']}, actual={actual_tensor_sha}"
                    )
                expected_offset += storage.length
                completed += storage.length
                verified_chunks += 1
                now = time.monotonic()
                if now - last_progress >= 60:
                    _progress(
                        phase="full_readback",
                        completed_bytes=completed,
                        total_bytes=total_bytes,
                        started=started,
                        extra={"shard": shard_name, "shard_index": shard_index, "shard_count": len(shard_names)},
                    )
                    last_progress = now
            trailing = stream.read(1)
            if trailing:
                raise EmaBf16ExportError(f"target shard contains bytes beyond the final indexed chunk: {path}")
        metadata = _require_regular_file(path, f"target shard {shard_name}")
        if _stable_identity(metadata) != identity:
            raise EmaBf16ExportError(f"target shard changed during readback: {path}")
        if expected_offset != metadata.st_size:
            raise EmaBf16ExportError(
                f"target shard indexed size mismatch: indexed={expected_offset}, file={metadata.st_size}"
            )
        files.append(
            {
                "relative_path": shard_name,
                "size_bytes": metadata.st_size,
                "sha256": file_digest.hexdigest(),
            }
        )
        _progress(
            phase="full_readback",
            completed_bytes=completed,
            total_bytes=total_bytes,
            started=started,
            extra={"completed_shard": shard_name, "shard_index": shard_index, "shard_count": len(shard_names)},
        )
    return files, verified_chunks


def _load_one_tensor(
    *,
    checkpoint_path: Path,
    key: str,
    metadata: Metadata,
    planner: dcp.LoadPlanner | None,
) -> torch.Tensor:
    value = metadata.state_dict_metadata[key]
    if not isinstance(value, TensorStorageMetadata):
        raise EmaBf16ExportError(f"sample key is not tensor metadata: {key}")
    target = {key: torch.empty(tuple(value.size), dtype=value.properties.dtype)}
    dcp.load(
        state_dict=target,
        storage_reader=FileSystemReader(str(checkpoint_path)),
        planner=planner,
    )
    return target[key]


def _planner_sample_validation(
    *,
    source_model: Path,
    target_model: Path,
    source_metadata: Metadata,
    target_metadata: Metadata,
    sample_keys: Sequence[str],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for ema_key in sample_keys:
        if not ema_key.startswith(EMA_PREFIX) or ema_key not in target_metadata.state_dict_metadata:
            raise EmaBf16ExportError(f"sample key is absent or not EMA-prefixed: {ema_key}")
        regular_key = f"{REGULAR_PREFIX}{ema_key.removeprefix(EMA_PREFIX)}"
        target_value = target_metadata.state_dict_metadata[ema_key]
        if not isinstance(target_value, TensorStorageMetadata):
            raise EmaBf16ExportError(f"sample target is not tensor metadata: {ema_key}")

        explicit_source = _load_one_tensor(
            checkpoint_path=source_model,
            key=ema_key,
            metadata=source_metadata,
            planner=None,
        ).to(dtype=torch.bfloat16)
        explicit_target = _load_one_tensor(
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
        selected = planner_target[regular_key]
        if not torch.equal(explicit_target, explicit_source) or not torch.equal(selected, explicit_source):
            raise EmaBf16ExportError(f"planner sample validation failed for {ema_key}")
        results.append(
            {
                "ema_key": ema_key,
                "regular_target_key": regular_key,
                "shape": list(explicit_target.shape),
                "dtype": str(explicit_target.dtype),
                "bf16_tensor_sha256": _tensor_sha256(explicit_target),
                "explicit_target_matches_source_ema_cast": True,
                "custom_load_planner_matches_source_ema_cast": True,
            }
        )
    return results


def _target_manifest(final_model: Path, model_files: Sequence[Mapping[str, object]]) -> dict[str, object]:
    files = sorted((dict(record) for record in model_files), key=lambda record: str(record["relative_path"]))
    return {
        "schema_version": 1,
        "kind": "cwam_serving_checkpoint_manifest",
        "checkpoint_path": str(final_model),
        "files": files,
        "tree_sha256": _canonical_digest(files),
    }


def _write_key_manifest(
    *,
    path: Path,
    source_model: Path,
    final_model: Path,
    origin_manifest_path: Path,
    origin_manifest_sha: str,
    expectations: ExportExpectations,
    source_metadata: Metadata,
    key_chunks: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    keys: list[dict[str, object]] = []
    for key in sorted(key_chunks):
        metadata = source_metadata.state_dict_metadata[key]
        if not isinstance(metadata, TensorStorageMetadata):
            raise EmaBf16ExportError(f"EMA key is not tensor metadata: {key}")
        chunks = sorted(
            (dict(chunk) for chunk in key_chunks[key]),
            key=lambda chunk: (tuple(chunk["global_offset"]), int(chunk["chunk_index"] or -1)),
        )
        keys.append(
            {
                "key": key,
                "shape": list(metadata.size),
                "source_dtype": str(metadata.properties.dtype),
                "target_dtype": str(torch.bfloat16),
                "source_logical_bytes": _logical_bytes(metadata),
                "target_logical_bytes": int(metadata.size.numel()) * torch.bfloat16.itemsize,
                "chunk_count": len(chunks),
                "chunks": chunks,
                "chunk_tree_sha256": _canonical_digest(chunks),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "cwam_serving_ema_bf16_key_manifest",
        "created_at": _utc_now(),
        "source_model_path": str(source_model),
        "source_metadata_sha256": expectations.source_metadata_sha256,
        "origin_manifest_path": str(origin_manifest_path),
        "origin_manifest_sha256": origin_manifest_sha,
        "origin_tree_sha256": expectations.origin_tree_sha256,
        "target_model_path": str(final_model),
        "selection": "net_ema.* only; key names preserved",
        "conversion": "torch.float32 to torch.bfloat16, one DCP storage chunk at a time",
        "key_count": len(keys),
        "source_logical_bytes": sum(int(record["source_logical_bytes"]) for record in keys),
        "target_logical_bytes": sum(int(record["target_logical_bytes"]) for record in keys),
        "keys": keys,
        "key_tree_sha256": _canonical_digest(keys),
    }
    _write_bytes_atomic(path, _canonical_json_bytes(manifest))
    return manifest


def _copy_config(source: Path, target: Path, expected_sha256: str) -> tuple[int, str]:
    size, actual_sha = _hash_file(source)
    if actual_sha != expected_sha256:
        raise EmaBf16ExportError(
            f"source config SHA-256 mismatch: expected={expected_sha256}, actual={actual_sha}"
        )
    source_identity = _stable_identity(_require_regular_file(source, "source config"))
    with source.open("rb") as source_stream, target.open("xb") as target_stream:
        shutil.copyfileobj(source_stream, target_stream, length=HASH_CHUNK_BYTES)
        target_stream.flush()
        os.fsync(target_stream.fileno())
    if _stable_identity(_require_regular_file(source, "source config")) != source_identity:
        raise EmaBf16ExportError(f"source config changed while copying: {source}")
    copied_size, copied_sha = _hash_file(target)
    if copied_size != size or copied_sha != actual_sha:
        raise EmaBf16ExportError("copied config does not match its source")
    return copied_size, copied_sha


def _seal_tree(root: Path) -> None:
    files: list[Path] = []
    directories: list[Path] = []
    for current_root, directory_names, file_names in os.walk(root):
        current = Path(current_root)
        directories.append(current)
        for directory_name in directory_names:
            path = current / directory_name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise EmaBf16ExportError(f"artifact contains a non-directory or symlink: {path}")
        for file_name in file_names:
            path = current / file_name
            _require_regular_file(path, "artifact file")
            files.append(path)
    for path in files:
        path.chmod(0o444)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)


def _verify_sealed_tree(root: Path) -> None:
    for current_root, directory_names, file_names in os.walk(root):
        current = Path(current_root)
        metadata = _require_directory(current, "published artifact directory")
        if stat.S_IMODE(metadata.st_mode) != 0o555:
            raise EmaBf16ExportError(f"published directory is not mode 0555: {current}")
        for name in directory_names:
            _require_directory(current / name, "published artifact directory")
        for name in file_names:
            path = current / name
            metadata = _require_regular_file(path, "published artifact file")
            if stat.S_IMODE(metadata.st_mode) != 0o444:
                raise EmaBf16ExportError(f"published file is not mode 0444: {path}")


def export_ema_bf16_checkpoint(
    *,
    source_model: Path,
    source_config: Path,
    origin_manifest_path: Path,
    final_artifact: Path,
    expectations: ExportExpectations,
    sample_keys: Sequence[str],
    seal: bool = True,
) -> dict[str, object]:
    """Run the complete conversion, verification, and atomic publication."""

    for field in (
        "origin_manifest_sha256",
        "origin_tree_sha256",
        "source_metadata_sha256",
        "source_config_sha256",
    ):
        _require_sha256(getattr(expectations, field), field)
    if any(
        value <= 0
        for value in (
            expectations.ema_key_count,
            expectations.regular_key_count,
            expectations.shard_count,
            expectations.source_ema_logical_bytes,
            expectations.target_logical_bytes,
        )
    ):
        raise EmaBf16ExportError("all numeric expectations must be positive")

    source_model = _canonical_existing_path(source_model, directory=True, label="source model")
    source_config = _canonical_existing_path(source_config, directory=False, label="source config")
    origin_manifest_path = _canonical_existing_path(
        origin_manifest_path,
        directory=False,
        label="origin manifest",
    )
    final_artifact = _canonical_new_path(final_artifact, "final artifact")
    final_model = final_artifact / "model"
    stage = final_artifact.parent / (
        f".{final_artifact.name}.incomplete-{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:10]}"
    )
    if stage.exists() or stage.is_symlink():
        raise EmaBf16ExportError(f"generated staging path unexpectedly exists: {stage}")

    exporter_path = Path(__file__).resolve(strict=True)
    exporter_source = exporter_path.read_bytes()
    exporter_sha = _sha256_bytes(exporter_source)
    started_at = _utc_now()
    started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "export_start",
                "source_model": str(source_model),
                "final_artifact": str(final_artifact),
                "staging_artifact": str(stage),
                "started_at": started_at,
                "exporter_sha256": exporter_sha,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        stage.mkdir(mode=0o700)
        target_model = stage / "model"
        target_model.mkdir(mode=0o700)
        origin_files, origin_manifest_sha = _validate_origin_manifest(
            source_model=source_model,
            origin_manifest_path=origin_manifest_path,
            expectations=expectations,
        )
        source_metadata = _load_source_metadata(source_model / ".metadata")
        ema_keys, shard_names, storage_by_file = _validate_source_metadata(
            source_metadata,
            origin_files,
            expectations,
        )
        total_source_storage_bytes = sum(
            storage.length for entries in storage_by_file.values() for _, storage in entries
        )
        target_storage, key_chunks, chunk_records = _convert_chunks(
            source_model=source_model,
            target_model=target_model,
            source_metadata=source_metadata,
            shard_names=shard_names,
            storage_by_file=storage_by_file,
            total_source_bytes=total_source_storage_bytes,
        )
        target_metadata = _target_metadata(
            source_metadata=source_metadata,
            ema_keys=ema_keys,
            target_storage=target_storage,
            final_model=final_model,
        )
        _write_metadata(target_model / ".metadata", target_metadata)
        _, target_metadata_sha = _hash_file(target_model / ".metadata")

        shard_files, verified_chunk_count = _readback_chunks(
            target_model=target_model,
            shard_names=shard_names,
            target_storage=target_storage,
            chunk_records=chunk_records,
        )
        expected_chunk_count = sum(len(chunks) for chunks in key_chunks.values())
        if verified_chunk_count != expected_chunk_count:
            raise EmaBf16ExportError(
                f"full readback chunk count mismatch: expected={expected_chunk_count}, actual={verified_chunk_count}"
            )
        sample_results = _planner_sample_validation(
            source_model=source_model,
            target_model=target_model,
            source_metadata=source_metadata,
            target_metadata=target_metadata,
            sample_keys=sample_keys,
        )

        model_files: list[dict[str, object]] = [
            {
                "relative_path": ".metadata",
                "size_bytes": (target_model / ".metadata").stat().st_size,
                "sha256": target_metadata_sha,
            },
            *shard_files,
        ]
        target_manifest = _target_manifest(final_model, model_files)
        target_manifest_path = stage / "serving-checkpoint-manifest.json"
        _write_bytes_atomic(target_manifest_path, _canonical_json_bytes(target_manifest))
        _, target_manifest_sha = _hash_file(target_manifest_path)

        key_manifest_path = stage / "ema-bf16-key-manifest.json"
        key_manifest = _write_key_manifest(
            path=key_manifest_path,
            source_model=source_model,
            final_model=final_model,
            origin_manifest_path=origin_manifest_path,
            origin_manifest_sha=origin_manifest_sha,
            expectations=expectations,
            source_metadata=source_metadata,
            key_chunks=key_chunks,
        )
        _, key_manifest_sha = _hash_file(key_manifest_path)

        validation: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "cwam_serving_ema_bf16_validation",
            "created_at": _utc_now(),
            "status": "passed",
            "target_model_path": str(final_model),
            "full_readback": {
                "status": "passed",
                "verified_key_count": len(ema_keys),
                "verified_chunk_count": verified_chunk_count,
                "verified_shard_count": len(shard_names),
                "every_serialized_chunk_sha256_recorded": True,
                "every_bf16_tensor_sha256_matched": True,
                "every_target_file_sha256_recorded": True,
                "checkpoint_tree_sha256": target_manifest["tree_sha256"],
            },
            "custom_load_planner_samples": sample_results,
        }
        validation_path = stage / "ema-bf16-validation.json"
        _write_bytes_atomic(validation_path, _canonical_json_bytes(validation))
        _, validation_sha = _hash_file(validation_path)

        copied_config_size, copied_config_sha = _copy_config(
            source_config,
            stage / "config.yaml",
            expectations.source_config_sha256,
        )
        (stage / "exporter-source.py").write_bytes(exporter_source)
        with (stage / "exporter-source.py").open("rb") as stream:
            os.fsync(stream.fileno())

        receipt: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "cwam_serving_ema_bf16_export_receipt",
            "status": "succeeded",
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "publication": {
                "method": "sibling incomplete directory followed by atomic rename",
                "final_artifact_path": str(final_artifact),
                "final_model_path": str(final_model),
                "read_only": seal,
                "original_source_modified": False,
            },
            "source": {
                "model_path": str(source_model),
                "metadata_sha256": expectations.source_metadata_sha256,
                "file_count": len(origin_files),
                "full_file_sha256_readback": True,
                "origin_manifest_path": str(origin_manifest_path),
                "origin_manifest_sha256": origin_manifest_sha,
                "origin_tree_sha256": expectations.origin_tree_sha256,
                "config_path": str(source_config),
                "config_size_bytes": copied_config_size,
                "config_sha256": copied_config_sha,
            },
            "selection": {
                "included_prefix": EMA_PREFIX,
                "excluded_prefix": REGULAR_PREFIX,
                "key_names_preserved": True,
                "source_dtype": str(torch.float32),
                "target_dtype": str(torch.bfloat16),
                "key_count": len(ema_keys),
                "chunk_count": verified_chunk_count,
                "source_logical_bytes": key_manifest["source_logical_bytes"],
                "target_logical_bytes": key_manifest["target_logical_bytes"],
            },
            "target": {
                "shard_count": len(shard_names),
                "metadata_sha256": target_metadata_sha,
                "checkpoint_tree_sha256": target_manifest["tree_sha256"],
                "serving_manifest_sha256": target_manifest_sha,
                "key_manifest_sha256": key_manifest_sha,
                "validation_sha256": validation_sha,
            },
            "exporter": {
                "source_path": "exporter-source.py",
                "sha256": exporter_sha,
                "python": sys.version,
                "torch": torch.__version__,
            },
        }
        receipt_path = stage / "ema-bf16-export-receipt.json"
        _write_bytes_atomic(receipt_path, _canonical_json_bytes(receipt))

        checksum_records: list[tuple[str, str]] = []
        for current_root, _, file_names in os.walk(stage):
            current = Path(current_root)
            for name in sorted(file_names):
                path = current / name
                relative = path.relative_to(stage).as_posix()
                if relative == "SHA256SUMS":
                    continue
                _, sha256 = _hash_file(path)
                checksum_records.append((relative, sha256))
        checksum_records.sort()
        sums = "".join(f"{sha256}  {relative}\n" for relative, sha256 in checksum_records).encode()
        _write_bytes_atomic(stage / "SHA256SUMS", sums)

        _fsync_directory(target_model)
        _fsync_directory(stage)
        if seal:
            _seal_tree(stage)
        _fsync_directory(final_artifact.parent)
        os.replace(stage, final_artifact)
        _fsync_directory(final_artifact.parent)
        if seal:
            _verify_sealed_tree(final_artifact)

        published_receipt = final_artifact / "ema-bf16-export-receipt.json"
        _, published_receipt_sha = _hash_file(published_receipt)
        result: dict[str, object] = {
            "status": "succeeded",
            "final_artifact_path": str(final_artifact),
            "final_model_path": str(final_model),
            "receipt_path": str(published_receipt),
            "receipt_sha256": published_receipt_sha,
            "checkpoint_tree_sha256": target_manifest["tree_sha256"],
            "target_logical_bytes": key_manifest["target_logical_bytes"],
            "key_count": len(ema_keys),
            "chunk_count": verified_chunk_count,
            "shard_count": len(shard_names),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "read_only": seal,
        }
        print(json.dumps({"event": "export_complete", **result}, sort_keys=True), flush=True)
        return result
    except Exception:
        print(
            json.dumps(
                {
                    "event": "export_failed",
                    "staging_artifact": str(stage),
                    "final_artifact": str(final_artifact),
                    "at": _utc_now(),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--origin-manifest", type=Path, required=True)
    parser.add_argument("--final-artifact", type=Path, required=True)
    parser.add_argument("--expected-origin-manifest-sha256", required=True)
    parser.add_argument("--expected-origin-tree-sha256", required=True)
    parser.add_argument("--expected-source-metadata-sha256", required=True)
    parser.add_argument("--expected-source-config-sha256", required=True)
    parser.add_argument("--expected-ema-key-count", type=int, required=True)
    parser.add_argument("--expected-regular-key-count", type=int, required=True)
    parser.add_argument("--expected-shard-count", type=int, required=True)
    parser.add_argument("--expected-source-ema-logical-bytes", type=int, required=True)
    parser.add_argument("--expected-target-logical-bytes", type=int, required=True)
    parser.add_argument(
        "--sample-key",
        action="append",
        required=True,
        help="Exact net_ema.* key for end-to-end source/target/CustomLoadPlanner verification; repeatable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    expectations = ExportExpectations(
        origin_manifest_sha256=args.expected_origin_manifest_sha256,
        origin_tree_sha256=args.expected_origin_tree_sha256,
        source_metadata_sha256=args.expected_source_metadata_sha256,
        source_config_sha256=args.expected_source_config_sha256,
        ema_key_count=args.expected_ema_key_count,
        regular_key_count=args.expected_regular_key_count,
        shard_count=args.expected_shard_count,
        source_ema_logical_bytes=args.expected_source_ema_logical_bytes,
        target_logical_bytes=args.expected_target_logical_bytes,
    )
    try:
        export_ema_bf16_checkpoint(
            source_model=args.source_model,
            source_config=args.source_config,
            origin_manifest_path=args.origin_manifest,
            final_artifact=args.final_artifact,
            expectations=expectations,
            sample_keys=args.sample_key,
        )
    except EmaBf16ExportError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
