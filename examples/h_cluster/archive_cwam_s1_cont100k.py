#!/usr/bin/env python3
"""Archive complete 10k-step DCP checkpoints to OSS with size verification."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COMPONENTS = ("model", "optim", "scheduler", "trainer")
STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_name(step: int) -> str:
    return f"iter_{step:09d}"


def checkpoint_ready(path: Path) -> bool:
    return path.is_dir() and all((path / component / ".metadata").is_file() for component in REQUIRED_COMPONENTS)


def tree_stats(path: Path) -> tuple[int, int]:
    files = 0
    size = 0
    for root, _, names in os.walk(path):
        for name in names:
            candidate = Path(root) / name
            if candidate.is_file():
                files += 1
                size += candidate.stat().st_size
    return files, size


def run_rclone(arguments: list[str], retries: int, retry_seconds: int) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(["rclone", *arguments], check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_seconds)
    assert last_error is not None
    raise last_error


def archive_checkpoint(
    checkpoint: Path,
    remote_root: str,
    receipts_dir: Path,
    experiment_id: str,
    job_id: str,
    source_commit: str,
    retries: int,
    retry_seconds: int,
) -> Path:
    name = checkpoint.name
    remote_checkpoint = f"{remote_root.rstrip('/')}/checkpoints/{name}"
    run_rclone(
        [
            "copy",
            str(checkpoint),
            remote_checkpoint,
            "--transfers=16",
            "--checkers=32",
            "--multi-thread-streams=8",
            "--retries=3",
            "--low-level-retries=10",
        ],
        retries,
        retry_seconds,
    )
    run_rclone(
        ["check", str(checkpoint), remote_checkpoint, "--size-only", "--one-way", "--checkers=32"],
        retries,
        retry_seconds,
    )
    files, size = tree_stats(checkpoint)
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "experiment_id": experiment_id,
        "job_id": job_id,
        "source_commit": source_commit,
        "checkpoint": name,
        "step": int(name.removeprefix("iter_")),
        "source": str(checkpoint),
        "destination": remote_checkpoint,
        "files": files,
        "bytes": size,
        "verification": "rclone check --size-only --one-way",
        "verified_at": utc_now(),
    }
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / f"{name}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_pointer = receipts_dir / "latest_checkpoint.txt"
    latest_pointer.write_text(name + "\n", encoding="utf-8")
    remote_receipts = f"{remote_root.rstrip('/')}/receipts"
    run_rclone(["copyto", str(receipt_path), f"{remote_receipts}/{receipt_path.name}"], retries, retry_seconds)
    run_rclone(["copyto", str(receipt_path), f"{remote_receipts}/latest.json"], retries, retry_seconds)
    run_rclone(
        ["copyto", str(latest_pointer), f"{remote_root.rstrip('/')}/checkpoints/latest_checkpoint.txt"],
        retries,
        retry_seconds,
    )
    return receipt_path


def prune_verified(checkpoints_dir: Path, receipts_dir: Path, keep_local: int) -> None:
    verified = sorted(
        path
        for path in checkpoints_dir.glob("iter_[0-9]*")
        if (receipts_dir / f"{path.name}.json").is_file()
    )
    for path in verified[:-keep_local] if keep_local else verified:
        subprocess.run(["rm", "-rf", "--one-file-system", str(path)], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", required=True, type=Path)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--receipts-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--job-id", default="unknown")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--start-step", type=int, default=20000)
    parser.add_argument("--end-step", type=int, default=100000)
    parser.add_argument("--interval", type=int, default=10000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--keep-local", type=int, default=2)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--retry-seconds", type=int, default=30)
    return parser.parse_args()


def request_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.start_step % args.interval or args.end_step % args.interval:
        raise SystemExit("start-step and end-step must be aligned to interval")
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    pending = list(range(args.start_step, args.end_step + 1, args.interval))
    while pending and not STOP_REQUESTED:
        step = pending[0]
        checkpoint = args.checkpoints_dir / checkpoint_name(step)
        receipt = args.receipts_dir / f"{checkpoint.name}.json"
        if receipt.is_file():
            pending.pop(0)
            continue
        if checkpoint_ready(checkpoint):
            print(f"ARCHIVE_START step={step} source={checkpoint}", flush=True)
            archive_checkpoint(
                checkpoint,
                args.remote_root,
                args.receipts_dir,
                args.experiment_id,
                args.job_id,
                args.source_commit,
                args.retries,
                args.retry_seconds,
            )
            print(f"ARCHIVE_VERIFIED step={step}", flush=True)
            pending.pop(0)
            prune_verified(args.checkpoints_dir, args.receipts_dir, args.keep_local)
            continue
        time.sleep(args.poll_seconds)
    return 0 if not pending else 2


if __name__ == "__main__":
    raise SystemExit(main())
