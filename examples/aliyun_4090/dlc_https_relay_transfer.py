#!/usr/bin/env python3
"""Probe or copy immutable artifacts from an authenticated HTTPS relay to OSSFS."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import pathlib
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


MIB = 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_token(path: pathlib.Path) -> str:
    metadata = os.stat(path, follow_symlinks=False)
    require(path.is_file() and not path.is_symlink(), f"token is not a regular file: {path}")
    require(metadata.st_mode & 0o077 == 0, f"token permissions are too broad: {path}")
    token = path.read_text(encoding="utf-8").strip()
    require(len(token) >= 32 and "\n" not in token and "\r" not in token, "invalid relay token")
    return token


def ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def relay_url(base_url: str, relative: str) -> str:
    require(relative and not relative.startswith("/"), f"invalid relay path: {relative}")
    parts = pathlib.PurePosixPath(relative).parts
    require(all(part not in {"", ".", ".."} for part in parts), f"invalid relay path: {relative}")
    return base_url.rstrip("/") + "/" + "/".join(urllib.parse.quote(part, safe="") for part in parts)


def request_range(
    *,
    url: str,
    token: str,
    start: int,
    end: int,
    destination: pathlib.Path | None,
    timeout: float,
) -> int:
    expected = end - start + 1
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Range": f"bytes={start}-{end}"},
        method="GET",
    )
    temporary: pathlib.Path | None = None
    output = None
    try:
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw = tempfile.mkstemp(prefix=f".{destination.name}.downloading-", dir=destination.parent)
            temporary = pathlib.Path(raw)
            output = os.fdopen(descriptor, "wb")
        received = 0
        with urllib.request.urlopen(request, context=ssl_context(), timeout=timeout) as response:
            require(response.status == 206, f"expected HTTP 206, observed {response.status}: {url}")
            require(int(response.headers["Content-Length"]) == expected, f"range length mismatch: {url}")
            while True:
                block = response.read(8 * MIB)
                if not block:
                    break
                received += len(block)
                if output is not None:
                    output.write(block)
        require(received == expected, f"download size mismatch: expected={expected} observed={received}: {url}")
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
            output.close()
            output = None
            require(temporary is not None, "internal temporary path missing")
            os.replace(temporary, destination)
            temporary = None
        return received
    finally:
        if output is not None:
            output.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_oss_root(path: pathlib.Path) -> None:
    mount = subprocess.check_output(["findmnt", "-n", "-T", str(path), "-o", "TARGET"], text=True).strip()
    require(mount == "/oss-chengjuntao", f"destination is not on /oss-chengjuntao: {mount}")
    require(os.access(path if path.exists() else path.parent, os.W_OK), f"OSS destination is not writable: {path}")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    token = load_token(args.token_file)
    url = relay_url(args.base_url, args.relay_path)
    started_at = utc_now()
    start_time = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                request_range,
                url=url,
                token=token,
                start=index * args.bytes_per_range,
                end=(index + 1) * args.bytes_per_range - 1,
                destination=None,
                timeout=args.timeout,
            )
            for index in range(args.workers)
        ]
        sizes = [future.result() for future in futures]
    elapsed = time.monotonic() - start_time
    total = sum(sizes)
    return {
        "schema_version": "cwam_https_relay_probe_v1",
        "status": "SUCCEEDED",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "relay_path": args.relay_path,
        "workers": args.workers,
        "bytes_per_range": args.bytes_per_range,
        "bytes_received": total,
        "elapsed_seconds": elapsed,
        "aggregate_mib_per_second": total / elapsed / MIB,
    }


@dataclass(frozen=True)
class Part:
    object_name: str
    object_size: int
    index: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def load_inventory(path: pathlib.Path) -> dict[str, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict) and value, "inventory must be a non-empty JSON object")
    inventory: dict[str, int] = {}
    for name, size in value.items():
        require(isinstance(name, str) and not name.startswith("/"), f"invalid inventory path: {name}")
        require(all(part not in {"", ".", ".."} for part in pathlib.PurePosixPath(name).parts), f"invalid path: {name}")
        require(type(size) is int and size > 0, f"invalid size for {name}")
        inventory[name] = size
    return inventory


def part_path(parts_root: pathlib.Path, part: Part) -> pathlib.Path:
    return parts_root.joinpath(*pathlib.PurePosixPath(part.object_name).parts, f"part-{part.index:06d}")


def all_parts(inventory: dict[str, int], part_bytes: int) -> list[Part]:
    result: list[Part] = []
    for name, size in inventory.items():
        for index, start in enumerate(range(0, size, part_bytes)):
            result.append(Part(name, size, index, start, min(size - 1, start + part_bytes - 1)))
    return result


def download_part(args: argparse.Namespace, token: str, parts_root: pathlib.Path, part: Part) -> str:
    destination = part_path(parts_root, part)
    if destination.is_file() and not destination.is_symlink() and destination.stat().st_size == part.size:
        return "reused"
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    request_range(
        url=relay_url(args.base_url, f"{args.relay_prefix.rstrip('/')}/{part.object_name}"),
        token=token,
        start=part.start,
        end=part.end,
        destination=destination,
        timeout=args.timeout,
    )
    require(destination.stat().st_size == part.size, f"persisted part size mismatch: {destination}")
    return "downloaded"


def merge_object(destination_root: pathlib.Path, parts_root: pathlib.Path, name: str, size: int, parts: list[Part]) -> None:
    destination = destination_root.joinpath(*pathlib.PurePosixPath(name).parts)
    if destination.is_file() and not destination.is_symlink() and destination.stat().st_size == size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.assembling")
    temporary.unlink(missing_ok=True)
    written = 0
    try:
        with temporary.open("wb") as output:
            for part in parts:
                source = part_path(parts_root, part)
                require(source.stat().st_size == part.size, f"part changed before merge: {source}")
                with source.open("rb") as stream:
                    while True:
                        block = stream.read(8 * MIB)
                        if not block:
                            break
                        output.write(block)
                        written += len(block)
            output.flush()
            os.fsync(output.fileno())
        require(written == size and temporary.stat().st_size == size, f"merged size mismatch: {name}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_transfer(args: argparse.Namespace) -> dict[str, Any]:
    verify_oss_root(args.destination_root)
    token = load_token(args.token_file)
    inventory = load_inventory(args.inventory)
    parts_root = args.destination_root.parent / f".{args.destination_root.name}.transfer-parts"
    parts = all_parts(inventory, args.part_bytes)
    started_at = utc_now()
    start_time = time.monotonic()
    counters = {"downloaded": 0, "reused": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_part, args, token, parts_root, part) for part in parts]
        for future in concurrent.futures.as_completed(futures):
            state = future.result()
            counters[state] += 1
            complete = counters["downloaded"] + counters["reused"]
            if complete % 16 == 0 or complete == len(parts):
                print(f"parts_complete={complete}/{len(parts)} downloaded={counters['downloaded']} reused={counters['reused']}", flush=True)

    grouped: dict[str, list[Part]] = {name: [] for name in inventory}
    for part in parts:
        grouped[part.object_name].append(part)
    for name, size in inventory.items():
        print(f"assembling={name} bytes={size}", flush=True)
        merge_object(args.destination_root, parts_root, name, size, grouped[name])

    observed = {
        path.relative_to(args.destination_root).as_posix(): path.stat().st_size
        for path in args.destination_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    require(observed == inventory, f"published inventory mismatch: observed={observed}")
    elapsed = time.monotonic() - start_time
    receipt = {
        "schema_version": "cwam_https_relay_transfer_v1",
        "status": "SUCCEEDED",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "relay_prefix": args.relay_prefix,
        "destination_root": str(args.destination_root),
        "inventory": inventory,
        "workers": args.workers,
        "part_bytes": args.part_bytes,
        "part_count": len(parts),
        "downloaded_part_count": counters["downloaded"],
        "reused_part_count": counters["reused"],
        "total_bytes": sum(inventory.values()),
        "elapsed_seconds": elapsed,
        "end_to_end_mib_per_second": sum(inventory.values()) / elapsed / MIB,
    }
    write_json(args.receipt, receipt)
    for path in sorted(parts_root.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    parts_root.rmdir()
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="mode", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--base-url", required=True)
    probe.add_argument("--relay-path", required=True)
    probe.add_argument("--token-file", type=pathlib.Path, required=True)
    probe.add_argument("--receipt", type=pathlib.Path, required=True)
    probe.add_argument("--workers", type=int, default=32)
    probe.add_argument("--bytes-per-range", type=int, default=2 * MIB)
    probe.add_argument("--timeout", type=float, default=180.0)

    transfer = subparsers.add_parser("transfer")
    transfer.add_argument("--base-url", required=True)
    transfer.add_argument("--relay-prefix", required=True)
    transfer.add_argument("--token-file", type=pathlib.Path, required=True)
    transfer.add_argument("--inventory", type=pathlib.Path, required=True)
    transfer.add_argument("--destination-root", type=pathlib.Path, required=True)
    transfer.add_argument("--receipt", type=pathlib.Path, required=True)
    transfer.add_argument("--workers", type=int, default=64)
    transfer.add_argument("--part-bytes", type=int, default=64 * MIB)
    transfer.add_argument("--timeout", type=float, default=900.0)
    return result


def main() -> int:
    args = parser().parse_args()
    require(args.workers > 0, "workers must be positive")
    if args.mode == "probe":
        require(args.bytes_per_range > 0, "bytes-per-range must be positive")
        value = run_probe(args)
        verify_oss_root(args.receipt)
        write_json(args.receipt, value)
    else:
        require(args.part_bytes > 0, "part-bytes must be positive")
        value = run_transfer(args)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
