# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build a compact first-frame sidecar from a published RoboCasa365 Wan cache."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_SOURCE_SHAPE = (48, 5, 30, 52)
_OUTPUT_SHAPE = (48, 15, 26)
_OUTPUT_RECORD_NBYTES = 48 * 15 * 26 * 2


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _validate_parent(parent_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], int, int, int]:
    manifest = _read_json(parent_root / "manifest.json")
    success = _read_json(parent_root / "_SUCCESS.json")
    if success.get("status") != "complete":
        raise RuntimeError("Parent Wan latent cache is not complete")
    cache_id = manifest.get("cache_id")
    if not isinstance(cache_id, str) or not cache_id:
        raise ValueError("Parent manifest.cache_id must be a non-empty string")
    if success.get("cache_id") != cache_id:
        raise RuntimeError("Parent success marker and manifest use different cache IDs")

    latent = manifest.get("latent")
    if not isinstance(latent, dict):
        raise ValueError("Parent manifest.latent must be an object")
    if latent.get("shape") != list(_SOURCE_SHAPE) or latent.get("storage_dtype") != "bfloat16":
        raise ValueError(
            "Visual history sidecar requires BF16 parent latents with shape "
            f"{list(_SOURCE_SHAPE)}, got shape={latent.get('shape')!r}, "
            f"dtype={latent.get('storage_dtype')!r}"
        )
    source_record_nbytes = int(latent.get("record_nbytes", -1))
    source_stride = int(latent.get("record_stride_bytes", -1))
    expected_source_nbytes = 48 * 5 * 30 * 52 * 2
    if source_record_nbytes != expected_source_nbytes or source_stride < source_record_nbytes:
        raise ValueError("Parent fixed-record byte layout is inconsistent")

    mapping_gate = manifest.get("mapping_gate")
    if not isinstance(mapping_gate, dict) or mapping_gate.get("status") != "passed":
        raise RuntimeError("Parent cache mapping gate has not passed")
    num_records = int(mapping_gate.get("unique_records", -1))
    if num_records < 1:
        raise ValueError("Parent mapping_gate.unique_records must be positive")

    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("Parent manifest.shards must be non-empty")
    shards: list[dict[str, Any]] = []
    expected_first = 0
    for expected_shard_id, raw in enumerate(raw_shards):
        if not isinstance(raw, dict):
            raise ValueError("Parent shard metadata must be objects")
        shard_id = int(raw.get("shard_id", -1))
        first = int(raw.get("first_ordinal", -1))
        count = int(raw.get("num_records", -1))
        relative_path = raw.get("path")
        if (
            shard_id != expected_shard_id
            or first != expected_first
            or count < 1
            or not isinstance(relative_path, str)
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise ValueError(f"Invalid parent shard metadata: {raw!r}")
        source_path = (parent_root / relative_path).resolve()
        if parent_root.resolve() not in source_path.parents:
            raise ValueError(f"Parent shard escapes cache root: {relative_path!r}")
        expected_size = count * source_stride
        if not source_path.is_file() or source_path.stat().st_size != expected_size:
            raise RuntimeError(f"Parent shard size mismatch: {source_path}")
        shards.append(
            {
                "shard_id": shard_id,
                "first_ordinal": first,
                "num_records": count,
                "source_path": source_path,
            }
        )
        expected_first += count
    if expected_first != num_records:
        raise RuntimeError(f"Parent shards cover {expected_first} records, expected {num_records}")
    return manifest, shards, num_records, source_record_nbytes, source_stride


def _pool_chunk(payload: bytes, *, count: int, source_record_nbytes: int, source_stride: int) -> bytes:
    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(count, source_stride)
    compact = raw[:, :source_record_nbytes].contiguous()
    source = compact.view(torch.bfloat16).reshape(count, *_SOURCE_SHAPE)
    pooled = F.avg_pool2d(source[:, :, 0].float(), kernel_size=2, stride=2).to(torch.bfloat16)
    if tuple(pooled.shape[1:]) != _OUTPUT_SHAPE or not torch.isfinite(pooled).all():
        raise RuntimeError(f"Invalid pooled visual history chunk shape={tuple(pooled.shape)}")
    return pooled.contiguous().view(torch.uint8).numpy().tobytes()


def _build_shard(
    shard: dict[str, Any],
    *,
    output_root: Path,
    chunk_records: int,
    source_record_nbytes: int,
    source_stride: int,
) -> dict[str, Any]:
    shard_id = int(shard["shard_id"])
    count = int(shard["num_records"])
    relative_path = Path("shards") / "shape-48x15x26" / f"shard-{shard_id:06d}.bin"
    destination = output_root / relative_path
    expected_size = count * _OUTPUT_RECORD_NBYTES
    if destination.is_file() and destination.stat().st_size == expected_size:
        print(f"resume shard={shard_id:06d} records={count}", flush=True)
    else:
        partial = destination.with_suffix(".bin.partial")
        source_fd = os.open(Path(shard["source_path"]), os.O_RDONLY)
        try:
            with partial.open("wb") as stream:
                for offset_records in range(0, count, chunk_records):
                    current_count = min(chunk_records, count - offset_records)
                    offset_bytes = offset_records * source_stride
                    payload_size = current_count * source_stride
                    payload = os.pread(source_fd, payload_size, offset_bytes)
                    if len(payload) != payload_size:
                        raise RuntimeError(f"Short parent read shard={shard_id} offset_records={offset_records}")
                    stream.write(
                        _pool_chunk(
                            payload,
                            count=current_count,
                            source_record_nbytes=source_record_nbytes,
                            source_stride=source_stride,
                        )
                    )
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(source_fd)
        if partial.stat().st_size != expected_size:
            raise RuntimeError(f"Visual history shard size mismatch after build: {partial}")
        os.replace(partial, destination)
        print(f"built shard={shard_id:06d} records={count}", flush=True)
    return {
        "shard_id": shard_id,
        "first_ordinal": int(shard["first_ordinal"]),
        "num_records": count,
        "path": relative_path.as_posix(),
        "size_bytes": expected_size,
    }


def build_cache(parent_root: Path, output_root: Path, *, chunk_records: int, workers: int = 1) -> dict[str, Any]:
    if chunk_records < 1:
        raise ValueError("chunk_records must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    parent_root = parent_root.resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve(strict=True)
    parent, parent_shards, num_records, source_record_nbytes, source_stride = _validate_parent(parent_root)

    output_shard_root = output_root / "shards" / "shape-48x15x26"
    output_shard_root.mkdir(parents=True, exist_ok=True)
    build_one = partial(
        _build_shard,
        output_root=output_root,
        chunk_records=chunk_records,
        source_record_nbytes=source_record_nbytes,
        source_stride=source_stride,
    )
    if workers == 1:
        output_shards = [build_one(shard) for shard in parent_shards]
    else:
        torch.set_num_threads(1)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="visual-cache") as executor:
            output_shards = list(executor.map(build_one, parent_shards))

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "parent_cache_id": parent["cache_id"],
        "num_records": num_records,
        "storage_dtype": "bfloat16",
        "shape": list(_OUTPUT_SHAPE),
        "pooling": {"source_temporal_index": 0, "spatial": "avg_pool2d_2x2"},
        "record_nbytes": _OUTPUT_RECORD_NBYTES,
        "shards": output_shards,
    }
    _atomic_write_json(output_root / "manifest.json", manifest)
    _atomic_write_json(
        output_root / "_SUCCESS.json",
        {
            "schema_version": 1,
            "status": "complete",
            "manifest_path": "manifest.json",
            "parent_cache_id": parent["cache_id"],
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-records", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_cache(
        args.parent_cache_root,
        args.output_root,
        chunk_records=args.chunk_records,
        workers=args.workers,
    )
    print(
        f"published records={manifest['num_records']} shards={len(manifest['shards'])} "
        f"root={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
