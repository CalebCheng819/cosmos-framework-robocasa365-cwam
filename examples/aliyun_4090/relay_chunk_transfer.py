# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Move large rclone objects through an authenticated HTTPS relay in chunks."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote


COPY_BLOCK_BYTES = 8 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class ObjectSpec:
    source: str
    relay_prefix: PurePosixPath
    expected_bytes: int
    destination: Path


@dataclasses.dataclass(frozen=True)
class ChunkSpec:
    object_spec: ObjectSpec
    index: int
    offset: int
    count: int

    @property
    def relay_path(self) -> PurePosixPath:
        return PurePosixPath(f"{self.object_spec.relay_prefix}.parts") / f"{self.index:06d}"


def read_manifest(path: Path) -> list[ObjectSpec]:
    objects: list[ObjectSpec] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 4:
                raise ValueError(f"Manifest line {line_number} must have four tab-separated fields.")
            source, relay_prefix_raw, expected_raw, destination_raw = fields
            relay_prefix = PurePosixPath(relay_prefix_raw)
            if (
                not source
                or not relay_prefix.parts
                or relay_prefix.is_absolute()
                or any(part in {"", ".", ".."} for part in relay_prefix.parts)
                or not expected_raw.isdigit()
                or int(expected_raw) <= 0
                or not destination_raw
            ):
                raise ValueError(f"Manifest line {line_number} is invalid.")
            objects.append(
                ObjectSpec(
                    source=source,
                    relay_prefix=relay_prefix,
                    expected_bytes=int(expected_raw),
                    destination=Path(destination_raw).expanduser().absolute(),
                )
            )
    if not objects:
        raise ValueError("Manifest contains no objects.")
    if len({str(item.relay_prefix) for item in objects}) != len(objects):
        raise ValueError("Manifest relay prefixes must be unique.")
    if len({str(item.destination) for item in objects}) != len(objects):
        raise ValueError("Manifest destinations must be unique.")
    return objects


def split_chunks(objects: list[ObjectSpec], chunk_bytes: int) -> list[ChunkSpec]:
    if chunk_bytes <= 0:
        raise ValueError("Chunk size must be positive.")
    chunks: list[ChunkSpec] = []
    for item in objects:
        offset = 0
        index = 0
        while offset < item.expected_bytes:
            count = min(chunk_bytes, item.expected_bytes - offset)
            chunks.append(ChunkSpec(item, index, offset, count))
            offset += count
            index += 1
    return chunks


def _run(command: list[str], *, stdout=subprocess.PIPE) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, stdout=stdout, stderr=subprocess.PIPE, check=False)


def _parse_head_size(payload: bytes) -> tuple[int, int | None]:
    blocks = payload.replace(b"\r\n", b"\n").split(b"\n\n")
    for block in reversed(blocks):
        lines = [line for line in block.splitlines() if line]
        if not lines or not lines[0].startswith(b"HTTP/"):
            continue
        fields = lines[0].split()
        if len(fields) < 2 or not fields[1].isdigit():
            continue
        headers: dict[bytes, bytes] = {}
        for line in lines[1:]:
            key, separator, value = line.partition(b":")
            if separator:
                headers[key.strip().lower()] = value.strip()
        raw_size = headers.get(b"content-length")
        size = int(raw_size) if raw_size is not None and raw_size.isdigit() else None
        return int(fields[1]), size
    raise RuntimeError("curl HEAD response did not contain a parseable HTTP status.")


def _url(base_url: str, relay_path: PurePosixPath) -> str:
    encoded = "/".join(quote(part, safe="") for part in relay_path.parts)
    return f"{base_url.rstrip('/')}/{encoded}"


def _curl_transport_args(args: argparse.Namespace) -> list[str]:
    return ["--insecure"] if args.curl_insecure else []


def _head_size(args: argparse.Namespace, url: str) -> int | None:
    result = _run(
        [
            str(args.curl),
            "--config",
            str(args.curl_config),
            *_curl_transport_args(args),
            "--silent",
            "--show-error",
            "--head",
            "--dump-header",
            "-",
            "--output",
            os.devnull,
            url,
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-1000:])
    status, size = _parse_head_size(result.stdout)
    if status == 404:
        return None
    if status != 200 or size is None:
        raise RuntimeError(f"Unexpected relay HEAD response: status={status}, size={size}.")
    return size


def _state_path(state_dir: Path, chunk: ChunkSpec, action: str) -> Path:
    relative = Path(*chunk.relay_path.parts)
    return state_dir / action / relative.parent / f"{relative.name}.json"


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _record(state_dir: Path, chunk: ChunkSpec, action: str, status: str) -> None:
    _write_state(
        _state_path(state_dir, chunk, action),
        {
            "schema_version": 1,
            "action": action,
            "status": status,
            "source": chunk.object_spec.source,
            "relay_path": str(chunk.relay_path),
            "destination": str(chunk.object_spec.destination),
            "object_expected_bytes": chunk.object_spec.expected_bytes,
            "chunk_index": chunk.index,
            "offset": chunk.offset,
            "expected_bytes": chunk.count,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )


def _source_size(rclone: Path, rclone_config: Path, source: str) -> int:
    result = _run([str(rclone), "--config", str(rclone_config), "size", source, "--json"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-1000:])
    payload = json.loads(result.stdout)
    if int(payload.get("count", -1)) != 1:
        raise RuntimeError(f"Expected one source object: {source}: {payload}")
    return int(payload["bytes"])


def _upload_chunk(args: argparse.Namespace, chunk: ChunkSpec) -> str:
    url = _url(args.relay_url, chunk.relay_path)
    remote_size = _head_size(args, url)
    if remote_size == chunk.count:
        _record(args.state_dir, chunk, "upload", "complete")
        return f"upload_skip_complete\t{chunk.relay_path}\t{chunk.count}"
    if remote_size is not None:
        raise RuntimeError(f"Relay chunk has wrong size: {chunk.relay_path}: {remote_size}.")

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"cwam-relay-{chunk.index:06d}-", suffix=".chunk", dir=args.temp_dir, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as output:
            result = _run(
                [
                    str(args.rclone),
                    "--config",
                    str(args.rclone_config),
                    "cat",
                    chunk.object_spec.source,
                    "--offset",
                    str(chunk.offset),
                    "--count",
                    str(chunk.count),
                    "--retries",
                    "3",
                    "--low-level-retries",
                    "5",
                ],
                stdout=output,
            )
        actual = temporary.stat().st_size
        if result.returncode != 0 or actual != chunk.count:
            error = result.stderr.decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(
                f"Source range failed: {chunk.object_spec.source} offset={chunk.offset} "
                f"expected={chunk.count} actual={actual} return_code={result.returncode}: {error}"
            )
        result = _run(
            [
                str(args.curl),
                "--config",
                str(args.curl_config),
                *_curl_transport_args(args),
                "--silent",
                "--show-error",
                "--request",
                "PUT",
                "--upload-file",
                str(temporary),
                "--retry",
                "2",
                "--retry-all-errors",
                "--output",
                os.devnull,
                "--write-out",
                "%{http_code}",
                url,
            ]
        )
        status = result.stdout.decode("ascii", errors="replace")
        if result.returncode != 0 or status != "201":
            error = result.stderr.decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"Relay PUT failed: status={status}, return_code={result.returncode}: {error}")
        if _head_size(args, url) != chunk.count:
            raise RuntimeError(f"Relay chunk verification failed: {chunk.relay_path}.")
        _record(args.state_dir, chunk, "upload", "complete")
        return f"upload_complete\t{chunk.relay_path}\t{chunk.count}"
    finally:
        temporary.unlink(missing_ok=True)


def _chunk_destination(staging_dir: Path, chunk: ChunkSpec) -> Path:
    return staging_dir / "relay-parts" / Path(*chunk.relay_path.parts)


def _download_chunk(args: argparse.Namespace, chunk: ChunkSpec) -> str:
    url = _url(args.relay_url, chunk.relay_path)
    remote_size = _head_size(args, url)
    if remote_size != chunk.count:
        raise RuntimeError(
            f"Relay chunk is unavailable or wrong-sized: {chunk.relay_path}: "
            f"expected={chunk.count}, actual={remote_size}."
        )
    destination = _chunk_destination(args.staging_dir, chunk)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != chunk.count:
            raise RuntimeError(f"Local chunk has wrong size: {destination}.")
        _record(args.state_dir, chunk, "download", "complete")
        return f"download_skip_complete\t{chunk.relay_path}\t{chunk.count}"
    temporary = destination.with_name(f".{destination.name}.downloading-{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            result = _run(
                [
                    str(args.curl),
                    "--config",
                    str(args.curl_config),
                    *_curl_transport_args(args),
                    "--silent",
                    "--show-error",
                    "--location",
                    "--retry",
                    "2",
                    "--retry-all-errors",
                    url,
                ],
                stdout=output,
            )
        actual = temporary.stat().st_size
        if result.returncode != 0 or actual != chunk.count:
            error = result.stderr.decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(
                f"Relay GET failed: {chunk.relay_path}: expected={chunk.count}, actual={actual}, "
                f"return_code={result.returncode}: {error}"
            )
        os.replace(temporary, destination)
        _record(args.state_dir, chunk, "download", "complete")
        return f"download_complete\t{chunk.relay_path}\t{chunk.count}"
    finally:
        temporary.unlink(missing_ok=True)


def _run_parallel(worker, args: argparse.Namespace, chunks: list[ChunkSpec]) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = [executor.submit(worker, args, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


def upload(args: argparse.Namespace, objects: list[ObjectSpec], chunks: list[ChunkSpec]) -> None:
    for item in objects:
        actual = _source_size(args.rclone, args.rclone_config, item.source)
        if actual != item.expected_bytes:
            raise RuntimeError(
                f"Source size mismatch: {item.source}: expected={item.expected_bytes}, actual={actual}."
            )
    _run_parallel(_upload_chunk, args, chunks)


def _assemble(args: argparse.Namespace, item: ObjectSpec, chunks: list[ChunkSpec]) -> None:
    item.destination.parent.mkdir(parents=True, exist_ok=True)
    if item.destination.is_symlink() or item.destination.parent.is_symlink():
        raise RuntimeError(f"Destination and parent must not be symlinks: {item.destination}.")
    if item.destination.exists():
        actual = item.destination.stat().st_size
        if actual != item.expected_bytes:
            raise RuntimeError(
                f"Completed destination has wrong size: {item.destination}: "
                f"expected={item.expected_bytes}, actual={actual}."
            )
        return
    partial = item.destination.with_name(f".{item.destination.name}.assembling-{os.getpid()}")
    try:
        with partial.open("wb") as output:
            for chunk in sorted(chunks, key=lambda value: value.index):
                source = _chunk_destination(args.staging_dir, chunk)
                if source.stat().st_size != chunk.count:
                    raise RuntimeError(f"Cannot assemble wrong-sized chunk: {source}.")
                with source.open("rb") as input_handle:
                    while block := input_handle.read(COPY_BLOCK_BYTES):
                        output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if partial.stat().st_size != item.expected_bytes:
            raise RuntimeError(f"Assembled destination has wrong size: {partial}.")
        os.replace(partial, item.destination)
    finally:
        partial.unlink(missing_ok=True)


def download(args: argparse.Namespace, objects: list[ObjectSpec], chunks: list[ChunkSpec]) -> None:
    _run_parallel(_download_chunk, args, chunks)
    for item in objects:
        item_chunks = [chunk for chunk in chunks if chunk.object_spec == item]
        _assemble(args, item, item_chunks)
        print(f"assemble_complete\t{item.destination}\t{item.expected_bytes}", flush=True)
        if args.delete_parts_after_assemble:
            for chunk in item_chunks:
                _chunk_destination(args.staging_dir, chunk).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("upload", "download"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--relay-url", required=True)
    parser.add_argument("--curl", type=Path, default=Path("/usr/bin/curl"))
    parser.add_argument("--curl-config", type=Path, required=True)
    parser.add_argument("--curl-insecure", action="store_true")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--rclone", type=Path)
    parser.add_argument("--rclone-config", type=Path)
    parser.add_argument("--temp-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--delete-parts-after-assemble", action="store_true")
    args = parser.parse_args()
    if args.parallelism <= 0 or args.chunk_bytes <= 0:
        parser.error("parallelism and chunk size must be positive")
    if not args.curl.is_file() or not os.access(args.curl, os.X_OK):
        parser.error(f"curl is not executable: {args.curl}")
    if not args.curl_config.is_file():
        parser.error(f"curl config is not readable: {args.curl_config}")
    if args.mode == "upload":
        if args.rclone is None or args.rclone_config is None:
            parser.error("upload requires --rclone and --rclone-config")
    if args.mode == "download" and args.staging_dir is None:
        parser.error("download requires --staging-dir")
    return args


def main() -> None:
    args = parse_args()
    objects = read_manifest(args.manifest)
    chunks = split_chunks(objects, args.chunk_bytes)
    if args.mode == "upload":
        upload(args, objects, chunks)
    else:
        download(args, objects, chunks)


if __name__ == "__main__":
    main()
