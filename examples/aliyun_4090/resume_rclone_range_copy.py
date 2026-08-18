# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Resume a large rclone object copy without replaying completed byte ranges."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fetch_chunk(
    *,
    rclone: Path,
    config: Path,
    source: str,
    offset: int,
    count: int,
    temporary: Path,
) -> tuple[int, str]:
    command = [
        str(rclone),
        "--config",
        str(config),
        "cat",
        source,
        "--offset",
        str(offset),
        "--count",
        str(count),
        "--retries",
        "1",
        "--low-level-retries",
        "1",
    ]
    with temporary.open("wb") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, check=False)
    return result.returncode, result.stderr.decode("utf-8", errors="replace")


def _adopt_partial(destination: Path, partial: Path) -> None:
    if partial.exists():
        legacy = list(destination.parent.glob(f"{destination.name}.*.r3.partial"))
        if legacy:
            raise RuntimeError(f"Both deterministic and legacy partial files exist for {destination}.")
        return
    legacy = list(destination.parent.glob(f"{destination.name}.*.r3.partial"))
    if len(legacy) > 1:
        raise RuntimeError(f"Multiple legacy partial files exist for {destination}: {legacy}")
    if legacy:
        os.replace(legacy[0], partial)


def copy(args: argparse.Namespace) -> dict[str, object]:
    destination = args.destination.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.parent.is_symlink():
        raise RuntimeError("Destination and its parent must not be symlinks.")
    if args.expected_bytes <= 0 or args.chunk_bytes <= 0 or args.retry_delay_seconds < 0:
        raise ValueError("Expected size/chunk size must be positive and retry delay cannot be negative.")

    state_path = args.state_path.expanduser().absolute()
    lock_path = state_path.with_name(f"{state_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            actual = destination.stat().st_size
            if actual != args.expected_bytes:
                raise RuntimeError(
                    f"Completed destination has wrong size: expected={args.expected_bytes}, actual={actual}."
                )
            result = _state(args, completed_bytes=actual, attempts=0, status="complete")
            _write_state(state_path, result)
            return result

        partial = destination.with_name(f"{destination.name}.range.partial")
        _adopt_partial(destination, partial)
        if not partial.exists():
            partial.touch(mode=0o600)
        current = partial.stat().st_size
        if current > args.expected_bytes:
            raise RuntimeError(
                f"Partial destination exceeds expected size: expected={args.expected_bytes}, actual={current}."
            )
        if current != args.expected_bytes:
            aligned = current - current % args.chunk_bytes
            if aligned != current:
                os.truncate(partial, aligned)
                current = aligned

        attempts = 0
        while current < args.expected_bytes:
            count = min(args.chunk_bytes, args.expected_bytes - current)
            attempts += 1
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.{current}.",
                suffix=".chunk",
                dir=args.temp_dir,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            try:
                return_code, stderr = _fetch_chunk(
                    rclone=args.rclone,
                    config=args.config,
                    source=args.source,
                    offset=current,
                    count=count,
                    temporary=temporary,
                )
                fetched = temporary.stat().st_size
                if return_code != 0 or fetched != count:
                    state = _state(
                        args,
                        completed_bytes=current,
                        attempts=attempts,
                        status="retrying",
                        last_error=f"return_code={return_code}, fetched={fetched}, stderr={stderr[-1000:]}",
                    )
                    _write_state(state_path, state)
                    if args.max_attempts and attempts >= args.max_attempts:
                        raise RuntimeError(str(state["last_error"]))
                    time.sleep(args.retry_delay_seconds)
                    continue
                with partial.open("ab") as output, temporary.open("rb") as input_handle:
                    while block := input_handle.read(8 * 1024 * 1024):
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                current += count
                if partial.stat().st_size != current:
                    raise RuntimeError(f"Append size mismatch for {partial} at offset {current}.")
                _write_state(
                    state_path,
                    _state(args, completed_bytes=current, attempts=attempts, status="copying"),
                )
            finally:
                temporary.unlink(missing_ok=True)

        os.replace(partial, destination)
        result = _state(args, completed_bytes=current, attempts=attempts, status="complete")
        _write_state(state_path, result)
        return result


def _state(
    args: argparse.Namespace,
    *,
    completed_bytes: int,
    attempts: int,
    status: str,
    last_error: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "source": args.source,
        "destination": str(args.destination.expanduser().absolute()),
        "expected_bytes": args.expected_bytes,
        "completed_bytes": completed_bytes,
        "chunk_bytes": args.chunk_bytes,
        "attempts": attempts,
        "last_error": last_error,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rclone", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--max-attempts", type=int, default=0, help="Zero retries forever.")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(copy(parse_args()), indent=2, sort_keys=True))
