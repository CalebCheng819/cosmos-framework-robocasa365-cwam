#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Encode and decode manifest-aligned XOR deltas for CWAM DCP shards.

The target key manifest defines serialized chunk ranges in each target shard.
For a target range with the same key, shape, chunk index, global offset, and
serialized length in the base checkpoint, the synthetic base uses that base
range. All gaps and unmatched target ranges use zero bytes. The gzip stream is
the XOR between the complete target shard and this deterministic synthetic
base, so decoding still reconstructs and verifies every byte of the target.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_RE = re.compile(r"^__[0-9]+_0[.]distcp$")
_DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024


class DeltaCodecError(RuntimeError):
    """Raised when a delta input, mapping, stream, or output fails closed."""


@dataclass(frozen=True)
class FileDescriptor:
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class StorageRange:
    relative_path: str
    offset: int
    length: int


@dataclass(frozen=True)
class Segment:
    target_offset: int
    length: int
    base_storage: StorageRange | None
    key: str | None
    chunk_index: int | None


@dataclass(frozen=True)
class DeltaPlan:
    target_shard: str
    target_size_bytes: int
    segments: tuple[Segment, ...]
    matched_bytes: int
    unmatched_chunk_bytes: int
    gap_bytes: int
    matched_chunks: int
    unmatched_chunks: int


class _DigestWriter:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size_bytes = 0

    def write(self, payload: bytes) -> int:
        written = self.stream.write(payload)
        if written != len(payload):
            raise DeltaCodecError("compressed output accepted a short write")
        self.digest.update(payload)
        self.size_bytes += len(payload)
        return written

    def flush(self) -> None:
        self.stream.flush()


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _validate_sha256(value: str, label: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise DeltaCodecError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_directory(path: Path, label: str) -> Path:
    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        metadata = supplied.lstat()
        canonical = supplied.resolve(strict=True)
    except FileNotFoundError:
        raise DeltaCodecError(f"{label} does not exist: {supplied}") from None
    if supplied != canonical or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise DeltaCodecError(f"{label} must be a canonical non-symlink directory: {supplied}")
    return canonical


def _stable_descriptor(path: Path, label: str, expected_sha256: str | None = None) -> FileDescriptor:
    supplied = Path(os.path.abspath(path.expanduser()))
    try:
        before = supplied.lstat()
    except FileNotFoundError:
        raise DeltaCodecError(f"{label} is missing: {supplied}") from None
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise DeltaCodecError(f"{label} must be a regular non-symlink file: {supplied}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(supplied, flags)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        opened = os.fstat(descriptor)
        if _identity(before) != _identity(opened):
            raise DeltaCodecError(f"{label} changed while opening: {supplied}")
        while payload := os.read(descriptor, _DEFAULT_BLOCK_SIZE):
            digest.update(payload)
            size_bytes += len(payload)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = supplied.lstat()
    if not (_identity(before) == _identity(opened) == _identity(after) == _identity(final)):
        raise DeltaCodecError(f"{label} changed while hashing: {supplied}")
    observed = FileDescriptor(size_bytes=size_bytes, sha256=digest.hexdigest())
    if expected_sha256 is not None and observed.sha256 != _validate_sha256(expected_sha256, f"{label} expected hash"):
        raise DeltaCodecError(f"{label} SHA-256 mismatch: expected={expected_sha256}, actual={observed.sha256}")
    return observed


def _load_manifest(path: Path, expected_sha256: str, label: str) -> tuple[dict[str, Any], FileDescriptor]:
    descriptor = _stable_descriptor(path, label, expected_sha256)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DeltaCodecError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(value, dict) or value.get("kind") != "cwam_serving_ema_bf16_key_manifest":
        raise DeltaCodecError(f"{label} has an invalid kind")
    keys = value.get("keys")
    if not isinstance(keys, list) or value.get("key_count") != len(keys):
        raise DeltaCodecError(f"{label} key closure is invalid")
    return value, descriptor


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DeltaCodecError(f"{label} must be an integer >= {minimum}")
    return value


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise DeltaCodecError(f"{label} must be an integer list")
    return tuple(_integer(item, f"{label} item") for item in value)


def _storage(value: object, label: str) -> StorageRange:
    if not isinstance(value, Mapping):
        raise DeltaCodecError(f"{label} must be an object")
    relative_path = value.get("relative_path")
    if not isinstance(relative_path, str) or _SHARD_RE.fullmatch(relative_path) is None:
        raise DeltaCodecError(f"{label} relative_path is unsafe")
    return StorageRange(
        relative_path=relative_path,
        offset=_integer(value.get("offset"), f"{label} offset"),
        length=_integer(value.get("length"), f"{label} length", minimum=1),
    )


def _key_records(
    manifest: Mapping[str, Any], label: str
) -> dict[str, tuple[tuple[int, ...], dict[tuple[Any, ...], StorageRange]]]:
    result: dict[str, tuple[tuple[int, ...], dict[tuple[Any, ...], StorageRange]]] = {}
    for key_index, raw_key in enumerate(manifest["keys"]):
        if not isinstance(raw_key, Mapping):
            raise DeltaCodecError(f"{label} key {key_index} must be an object")
        key = raw_key.get("key")
        if not isinstance(key, str) or not key or key in result:
            raise DeltaCodecError(f"{label} key {key_index} name is invalid or duplicated")
        shape = _integer_tuple(raw_key.get("shape"), f"{label} key {key} shape")
        chunks = raw_key.get("chunks")
        if not isinstance(chunks, list) or raw_key.get("chunk_count") != len(chunks):
            raise DeltaCodecError(f"{label} key {key} chunk closure is invalid")
        mapped: dict[tuple[Any, ...], StorageRange] = {}
        for position, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                raise DeltaCodecError(f"{label} key {key} chunk {position} must be an object")
            chunk_index = _integer(chunk.get("chunk_index"), f"{label} key {key} chunk index")
            global_offset = _integer_tuple(
                chunk.get("global_offset"),
                f"{label} key {key} chunk {chunk_index} global offset",
            )
            identity = (chunk_index, global_offset, shape)
            if identity in mapped:
                raise DeltaCodecError(f"{label} key {key} has a duplicated chunk identity")
            mapped[identity] = _storage(
                chunk.get("target_storage"),
                f"{label} key {key} chunk {chunk_index} target storage",
            )
        result[key] = (shape, mapped)
    return result


def build_delta_plan(
    *,
    base_manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    target_shard: str,
    target_size_bytes: int,
) -> DeltaPlan:
    if _SHARD_RE.fullmatch(target_shard) is None:
        raise DeltaCodecError("target shard name is unsafe")
    target_size_bytes = _integer(target_size_bytes, "target shard size", minimum=1)
    base = _key_records(base_manifest, "base key manifest")
    target = _key_records(target_manifest, "target key manifest")
    ranges: list[Segment] = []
    matched_bytes = 0
    unmatched_bytes = 0
    matched_chunks = 0
    unmatched_chunks = 0
    for key, (target_shape, chunks) in target.items():
        base_value = base.get(key)
        base_shape, base_chunks = base_value if base_value is not None else ((), {})
        for identity, target_storage in chunks.items():
            if target_storage.relative_path != target_shard:
                continue
            base_storage = base_chunks.get(identity) if base_shape == target_shape else None
            if base_storage is not None and base_storage.length == target_storage.length:
                matched_bytes += target_storage.length
                matched_chunks += 1
            else:
                base_storage = None
                unmatched_bytes += target_storage.length
                unmatched_chunks += 1
            ranges.append(
                Segment(
                    target_offset=target_storage.offset,
                    length=target_storage.length,
                    base_storage=base_storage,
                    key=key,
                    chunk_index=int(identity[0]),
                )
            )
    if not ranges:
        raise DeltaCodecError(f"target key manifest has no ranges for {target_shard}")
    ranges.sort(key=lambda item: (item.target_offset, item.length, item.key or ""))
    complete: list[Segment] = []
    cursor = 0
    gap_bytes = 0
    for item in ranges:
        if item.target_offset < cursor:
            raise DeltaCodecError(f"target chunk ranges overlap in {target_shard} at offset {item.target_offset}")
        if item.target_offset + item.length > target_size_bytes:
            raise DeltaCodecError(f"target chunk range exceeds {target_shard} size")
        if item.target_offset > cursor:
            gap = item.target_offset - cursor
            complete.append(Segment(cursor, gap, None, None, None))
            gap_bytes += gap
        complete.append(item)
        cursor = item.target_offset + item.length
    if cursor < target_size_bytes:
        gap = target_size_bytes - cursor
        complete.append(Segment(cursor, gap, None, None, None))
        gap_bytes += gap
    if sum(item.length for item in complete) != target_size_bytes:
        raise DeltaCodecError("delta plan does not cover the complete target shard")
    return DeltaPlan(
        target_shard=target_shard,
        target_size_bytes=target_size_bytes,
        segments=tuple(complete),
        matched_bytes=matched_bytes,
        unmatched_chunk_bytes=unmatched_bytes,
        gap_bytes=gap_bytes,
        matched_chunks=matched_chunks,
        unmatched_chunks=unmatched_chunks,
    )


def _open_storage(root: Path, storage: StorageRange, handles: dict[str, BinaryIO]) -> BinaryIO:
    path = root / storage.relative_path
    key = str(path)
    if key in handles:
        return handles[key]
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise DeltaCodecError(f"base shard is missing: {path}") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DeltaCodecError(f"base shard must be a regular non-symlink file: {path}")
    if storage.offset + storage.length > metadata.st_size:
        raise DeltaCodecError(f"base range exceeds shard size: {path}")
    stream = path.open("rb", buffering=0)
    handles[key] = stream
    return stream


def _read_exact(stream: BinaryIO, size_bytes: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size_bytes
    while remaining:
        payload = stream.read(remaining)
        if not payload:
            raise DeltaCodecError(f"short read from {label}: missing {remaining} bytes")
        chunks.append(payload)
        remaining -= len(payload)
    return b"".join(chunks)


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise DeltaCodecError("XOR operands have different lengths")
    if not left:
        return b""
    return (int.from_bytes(left, "little") ^ int.from_bytes(right, "little")).to_bytes(len(left), "little")


def _blocks(length: int, block_size: int) -> Iterator[tuple[int, int]]:
    offset = 0
    while offset < length:
        size = min(block_size, length - offset)
        yield offset, size
        offset += size


def _plan_receipt(plan: DeltaPlan) -> dict[str, Any]:
    return {
        "target_shard": plan.target_shard,
        "target_size_bytes": plan.target_size_bytes,
        "segment_count": len(plan.segments),
        "matched_chunks": plan.matched_chunks,
        "unmatched_chunks": plan.unmatched_chunks,
        "matched_bytes": plan.matched_bytes,
        "unmatched_chunk_bytes": plan.unmatched_chunk_bytes,
        "gap_bytes": plan.gap_bytes,
        "synthetic_base_fraction": plan.matched_bytes / plan.target_size_bytes,
    }


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> FileDescriptor:
    supplied = Path(os.path.abspath(path.expanduser()))
    if os.path.lexists(supplied):
        raise DeltaCodecError(f"receipt output already exists: {supplied}")
    supplied.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = supplied.with_name(f".{supplied.name}.incomplete-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, supplied)
        os.unlink(temporary)
    finally:
        os.close(descriptor)
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return _stable_descriptor(supplied, "delta receipt")


def _output_stream(path: str) -> tuple[BinaryIO, Path | None, int | None]:
    if path == "-":
        return sys.stdout.buffer, None, None
    supplied = Path(os.path.abspath(Path(path).expanduser()))
    if os.path.lexists(supplied):
        raise DeltaCodecError(f"delta output already exists: {supplied}")
    supplied.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(supplied, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "wb", closefd=False), supplied, descriptor


def encode_delta(
    *,
    base_model_root: Path,
    target_model_root: Path,
    plan: DeltaPlan,
    expected_target: FileDescriptor,
    output: str,
    compression_level: int,
    block_size: int,
) -> tuple[FileDescriptor, FileDescriptor]:
    base_root = _canonical_directory(base_model_root, "base model root")
    target_root = _canonical_directory(target_model_root, "target model root")
    target_path = target_root / plan.target_shard
    if expected_target.size_bytes != plan.target_size_bytes:
        raise DeltaCodecError("expected target size differs from the delta plan")
    if not 1 <= compression_level <= 9:
        raise DeltaCodecError("compression level must be between 1 and 9")
    block_size = _integer(block_size, "block size", minimum=4096)
    output_stream, output_path, output_descriptor = _output_stream(output)
    compressed = _DigestWriter(output_stream)
    target_digest = hashlib.sha256()
    target_bytes = 0
    base_handles: dict[str, BinaryIO] = {}
    try:
        try:
            target_metadata = target_path.lstat()
        except FileNotFoundError:
            raise DeltaCodecError(f"target shard is missing: {target_path}") from None
        if stat.S_ISLNK(target_metadata.st_mode) or not stat.S_ISREG(target_metadata.st_mode):
            raise DeltaCodecError(f"target shard must be a regular non-symlink file: {target_path}")
        if target_metadata.st_size != expected_target.size_bytes:
            raise DeltaCodecError("target shard size differs from the expected descriptor")
        with target_path.open("rb", buffering=0) as target_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=compression_level,
                fileobj=compressed,
                mtime=0,
            ) as archive:
                for segment in plan.segments:
                    if target_stream.tell() != segment.target_offset:
                        raise DeltaCodecError("target stream offset differs from the delta plan")
                    base_stream = None
                    if segment.base_storage is not None:
                        base_stream = _open_storage(base_root, segment.base_storage, base_handles)
                    for relative_offset, size in _blocks(segment.length, block_size):
                        target_payload = _read_exact(target_stream, size, "target shard")
                        target_digest.update(target_payload)
                        target_bytes += len(target_payload)
                        if base_stream is None:
                            delta = target_payload
                        else:
                            base_stream.seek(segment.base_storage.offset + relative_offset)
                            base_payload = _read_exact(base_stream, size, "base shard")
                            delta = _xor_bytes(target_payload, base_payload)
                        archive.write(delta)
        output_stream.flush()
        if output_descriptor is not None:
            os.fsync(output_descriptor)
        observed_target = FileDescriptor(target_bytes, target_digest.hexdigest())
        if observed_target != expected_target:
            raise DeltaCodecError(
                f"target shard descriptor mismatch: expected={expected_target}, actual={observed_target}"
            )
        observed_delta = FileDescriptor(compressed.size_bytes, compressed.digest.hexdigest())
        if output_path is not None:
            if output_path.stat().st_size != observed_delta.size_bytes:
                raise DeltaCodecError("delta output size changed after encoding")
            os.chmod(output_path, 0o444)
        return observed_delta, observed_target
    except Exception:
        if output_path is not None and os.path.lexists(output_path):
            os.unlink(output_path)
        raise
    finally:
        for stream in base_handles.values():
            stream.close()
        if output_descriptor is not None:
            output_stream.close()
            os.close(output_descriptor)


def decode_delta(
    *,
    base_model_root: Path,
    plan: DeltaPlan,
    delta_path: Path,
    expected_delta: FileDescriptor,
    expected_target: FileDescriptor,
    output_path: Path,
    block_size: int,
) -> FileDescriptor:
    base_root = _canonical_directory(base_model_root, "base model root")
    block_size = _integer(block_size, "block size", minimum=4096)
    if expected_target.size_bytes != plan.target_size_bytes:
        raise DeltaCodecError("expected target size differs from the delta plan")
    observed_delta = _stable_descriptor(delta_path, "compressed XOR delta", expected_delta.sha256)
    if observed_delta.size_bytes != expected_delta.size_bytes:
        raise DeltaCodecError(
            f"compressed XOR delta size mismatch: expected={expected_delta.size_bytes}, "
            f"actual={observed_delta.size_bytes}"
        )
    supplied = Path(os.path.abspath(output_path.expanduser()))
    if os.path.lexists(supplied):
        raise DeltaCodecError(f"decoded output already exists: {supplied}")
    supplied.parent.mkdir(parents=True, exist_ok=True)
    output_descriptor = os.open(supplied, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest = hashlib.sha256()
    written = 0
    base_handles: dict[str, BinaryIO] = {}
    try:
        with os.fdopen(output_descriptor, "wb", closefd=False) as output_stream:
            with delta_path.open("rb", buffering=0) as compressed_stream:
                with gzip.GzipFile(mode="rb", fileobj=compressed_stream) as archive:
                    for segment in plan.segments:
                        base_stream = None
                        if segment.base_storage is not None:
                            base_stream = _open_storage(base_root, segment.base_storage, base_handles)
                        for relative_offset, size in _blocks(segment.length, block_size):
                            delta = _read_exact(archive, size, "uncompressed XOR delta")
                            if base_stream is None:
                                target_payload = delta
                            else:
                                base_stream.seek(segment.base_storage.offset + relative_offset)
                                base_payload = _read_exact(base_stream, size, "base shard")
                                target_payload = _xor_bytes(delta, base_payload)
                            output_stream.write(target_payload)
                            digest.update(target_payload)
                            written += len(target_payload)
                    if archive.read(1):
                        raise DeltaCodecError("uncompressed XOR delta contains trailing bytes")
            output_stream.flush()
            os.fsync(output_stream.fileno())
        observed_target = FileDescriptor(written, digest.hexdigest())
        if observed_target != expected_target:
            raise DeltaCodecError(
                f"decoded target descriptor mismatch: expected={expected_target}, actual={observed_target}"
            )
        if supplied.stat().st_size != observed_target.size_bytes:
            raise DeltaCodecError("decoded output size changed after reconstruction")
        os.chmod(supplied, 0o444)
        return observed_target
    except Exception:
        if os.path.lexists(supplied):
            os.unlink(supplied)
        raise
    finally:
        for stream in base_handles.values():
            stream.close()
        os.close(output_descriptor)


def _manifest_and_plan(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], FileDescriptor, FileDescriptor, DeltaPlan]:
    base_manifest, base_descriptor = _load_manifest(
        args.base_key_manifest,
        args.expected_base_key_manifest_sha256,
        "base key manifest",
    )
    target_manifest, target_descriptor = _load_manifest(
        args.target_key_manifest,
        args.expected_target_key_manifest_sha256,
        "target key manifest",
    )
    plan = build_delta_plan(
        base_manifest=base_manifest,
        target_manifest=target_manifest,
        target_shard=args.target_shard,
        target_size_bytes=args.expected_target_size,
    )
    return base_manifest, target_manifest, base_descriptor, target_descriptor, plan


def _receipt_base(
    *,
    mode: str,
    args: argparse.Namespace,
    base_descriptor: FileDescriptor,
    target_descriptor: FileDescriptor,
    plan: DeltaPlan,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "cwam_ema_manifest_xor_delta_receipt",
        "status": "passed",
        "mode": mode,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_key_manifest": {
            "path": str(args.base_key_manifest),
            "size_bytes": base_descriptor.size_bytes,
            "sha256": base_descriptor.sha256,
        },
        "target_key_manifest": {
            "path": str(args.target_key_manifest),
            "size_bytes": target_descriptor.size_bytes,
            "sha256": target_descriptor.sha256,
        },
        "base_model_root": str(args.base_model_root),
        "mapping": _plan_receipt(plan),
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-key-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-key-manifest-sha256", required=True)
    parser.add_argument("--target-key-manifest", type=Path, required=True)
    parser.add_argument("--expected-target-key-manifest-sha256", required=True)
    parser.add_argument("--base-model-root", type=Path, required=True)
    parser.add_argument("--target-shard", required=True)
    parser.add_argument("--expected-target-size", type=int, required=True)
    parser.add_argument("--expected-target-sha256", required=True)
    parser.add_argument("--block-size", type=int, default=_DEFAULT_BLOCK_SIZE)
    parser.add_argument("--receipt-output", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    encode = subparsers.add_parser("encode")
    _add_common_arguments(encode)
    encode.add_argument("--target-model-root", type=Path, required=True)
    encode.add_argument("--output", required=True, help="delta output path or '-' for stdout")
    encode.add_argument("--compression-level", type=int, default=1)
    decode = subparsers.add_parser("decode")
    _add_common_arguments(decode)
    decode.add_argument("--delta", type=Path, required=True)
    decode.add_argument("--expected-delta-size", type=int, required=True)
    decode.add_argument("--expected-delta-sha256", required=True)
    decode.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _, _, base_descriptor, target_descriptor, plan = _manifest_and_plan(args)
        expected_target = FileDescriptor(
            _integer(args.expected_target_size, "expected target size", minimum=1),
            _validate_sha256(args.expected_target_sha256, "expected target hash"),
        )
        receipt = _receipt_base(
            mode=args.command,
            args=args,
            base_descriptor=base_descriptor,
            target_descriptor=target_descriptor,
            plan=plan,
        )
        if args.command == "encode":
            observed_delta, observed_target = encode_delta(
                base_model_root=args.base_model_root,
                target_model_root=args.target_model_root,
                plan=plan,
                expected_target=expected_target,
                output=args.output,
                compression_level=args.compression_level,
                block_size=args.block_size,
            )
            receipt.update(
                {
                    "compression": {"format": "gzip", "level": args.compression_level, "mtime": 0},
                    "target": observed_target.__dict__,
                    "delta": observed_delta.__dict__,
                    "output": args.output,
                }
            )
        else:
            expected_delta = FileDescriptor(
                _integer(args.expected_delta_size, "expected delta size", minimum=1),
                _validate_sha256(args.expected_delta_sha256, "expected delta hash"),
            )
            observed_target = decode_delta(
                base_model_root=args.base_model_root,
                plan=plan,
                delta_path=args.delta,
                expected_delta=expected_delta,
                expected_target=expected_target,
                output_path=args.output,
                block_size=args.block_size,
            )
            receipt.update(
                {
                    "compression": {"format": "gzip"},
                    "target": observed_target.__dict__,
                    "delta": expected_delta.__dict__,
                    "input": str(args.delta),
                    "output": str(args.output),
                }
            )
        receipt_descriptor = _write_receipt(args.receipt_output, receipt)
    except (DeltaCodecError, OSError, ValueError) as error:
        raise SystemExit(f"manifest XOR delta failed: {error}") from error
    print(
        json.dumps(
            {
                "status": "passed",
                "receipt": str(args.receipt_output),
                "receipt_sha256": receipt_descriptor.sha256,
            },
            sort_keys=True,
        ),
        file=sys.stderr if args.command == "encode" and args.output == "-" else sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
