#!/usr/bin/env python3
"""Archive DCP checkpoints to OSS and compact verified checkpoint windows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COMPONENTS = ("model", "optim", "scheduler", "trainer")
COMPLETE_MARKER = ".cwam_checkpoint_complete.json"
STOP_REQUESTED = False
STOP_REQUESTED_AT: float | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_name(step: int) -> str:
    return f"iter_{step:09d}"


def checkpoint_ready(path: Path, require_completion_marker: bool = False) -> bool:
    if not path.is_dir() or not all(
        (path / component / ".metadata").is_file() for component in REQUIRED_COMPONENTS
    ):
        return False
    if not require_completion_marker:
        return True
    marker_path = path / COMPLETE_MARKER
    if not marker_path.is_file() or marker_path.is_symlink():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    components = marker.get("components")
    return (
        marker.get("schema_version") == 1
        and marker.get("status") == "complete"
        and marker.get("checkpoint") == path.name
        and marker.get("iteration") == int(path.name.removeprefix("iter_"))
        and isinstance(components, list)
        and set(REQUIRED_COMPONENTS).issubset(components)
    )


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


def validated_tree_stats(path: Path) -> tuple[int, int]:
    for component in REQUIRED_COMPONENTS:
        component_dir = path / component
        metadata = component_dir / ".metadata"
        if metadata.is_symlink() or not metadata.is_file() or metadata.stat().st_size <= 0:
            raise RuntimeError(f"invalid DCP metadata: {metadata}")
        shards = list(component_dir.glob("*.distcp"))
        if not shards or any(shard.is_symlink() or not shard.is_file() for shard in shards):
            raise RuntimeError(f"invalid DCP shards: {component_dir}")
    for root, directories, names in os.walk(path):
        for name in [*directories, *names]:
            if (Path(root) / name).is_symlink():
                raise RuntimeError(f"checkpoint tree contains a symlink: {Path(root) / name}")
    return tree_stats(path)


def checkpoint_is_stable(
    path: Path,
    observations: dict[Path, tuple[tuple[int, int], float]],
    stability_seconds: int,
    now: float | None = None,
) -> bool:
    current_time = time.monotonic() if now is None else now
    stats = validated_tree_stats(path)
    previous = observations.get(path)
    if previous is None or previous[0] != stats:
        observations[path] = (stats, current_time)
        return stability_seconds <= 0
    return current_time - previous[1] >= stability_seconds


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_heartbeat(path: Path, phase: str, pending: set[int], detail: str = "") -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "status": "alive" if phase not in {"complete", "stopped"} else phase,
            "phase": phase,
            "pending_steps": sorted(pending),
            "detail": detail,
            "updated_at": utc_now(),
            "pid": os.getpid(),
        },
    )


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


def run_rclone_json(arguments: list[str], retries: int, retry_seconds: int) -> dict[str, object]:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ["rclone", *arguments],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            value = json.loads(result.stdout)
            if not isinstance(value, dict):
                raise RuntimeError("rclone JSON response is not an object")
            return value
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
    remote_stats = run_rclone_json(["size", "--json", remote_checkpoint], retries, retry_seconds)
    if remote_stats.get("count") != files or remote_stats.get("bytes") != size:
        raise RuntimeError(
            "remote checkpoint statistics differ after rclone check: "
            f"local=({files}, {size}) remote=({remote_stats.get('count')}, {remote_stats.get('bytes')})"
        )
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
        "verification": "rclone check --size-only --one-way plus independent count/bytes comparison",
        "verified_at": utc_now(),
    }
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / f"{name}.json"
    write_json_atomic(receipt_path, receipt)
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


def intermediate_steps_for_boundary(
    boundary_step: int,
    archive_interval: int,
    compact_interval: int,
    minimum_step: int = 0,
) -> list[int]:
    if compact_interval <= 0 or boundary_step % compact_interval:
        return []
    if archive_interval <= 0 or compact_interval % archive_interval:
        raise ValueError("compact interval must be a multiple of archive interval")
    first_step = boundary_step - compact_interval + archive_interval
    return [step for step in range(first_step, boundary_step, archive_interval) if step >= minimum_step]


def load_verified_receipt(receipt_path: Path, checkpoint: str, destination: str) -> dict[str, object]:
    if not receipt_path.is_file():
        raise RuntimeError(f"verified receipt missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "verified":
        raise RuntimeError(f"receipt is not verified: {receipt_path}")
    if receipt.get("checkpoint") != checkpoint or receipt.get("destination") != destination:
        raise RuntimeError(f"receipt target mismatch: {receipt_path}")
    return receipt


def compact_verified_window(
    boundary_step: int,
    checkpoints_dir: Path,
    remote_root: str,
    receipts_dir: Path,
    archive_interval: int,
    compact_interval: int,
    experiment_id: str,
    job_id: str,
    source_commit: str,
    retries: int,
    retry_seconds: int,
    minimum_step: int = 0,
) -> Path | None:
    intermediate_steps = intermediate_steps_for_boundary(
        boundary_step,
        archive_interval,
        compact_interval,
        minimum_step,
    )
    if not intermediate_steps:
        return None

    boundary_name = checkpoint_name(boundary_step)
    remote_checkpoints = f"{remote_root.rstrip('/')}/checkpoints"
    load_verified_receipt(
        receipts_dir / f"{boundary_name}.json",
        boundary_name,
        f"{remote_checkpoints}/{boundary_name}",
    )
    compaction_receipt = receipts_dir / f"compaction_at_{boundary_name}.json"
    if compaction_receipt.is_file():
        return compaction_receipt

    deletion_plan: list[tuple[int, str, Path]] = []
    for step in intermediate_steps:
        name = checkpoint_name(step)
        remote_checkpoint = f"{remote_checkpoints}/{name}"
        load_verified_receipt(receipts_dir / f"{name}.json", name, remote_checkpoint)
        deletion_plan.append((step, remote_checkpoint, checkpoints_dir / name))

    for step, remote_checkpoint, local_checkpoint in deletion_plan:
        run_rclone(["delete", remote_checkpoint, "--rmdirs"], retries, retry_seconds)
        if local_checkpoint.exists():
            shutil.rmtree(local_checkpoint)
        print(f"CHECKPOINT_COMPACTED step={step} retained={boundary_step}", flush=True)

    receipt = {
        "schema_version": 1,
        "status": "verified",
        "experiment_id": experiment_id,
        "job_id": job_id,
        "source_commit": source_commit,
        "archive_interval": archive_interval,
        "compact_interval": compact_interval,
        "retained_checkpoint": boundary_name,
        "retained_step": boundary_step,
        "deleted_checkpoints": [checkpoint_name(step) for step in intermediate_steps],
        "verification_gate": "retained checkpoint archived with rclone check --size-only --one-way",
        "compacted_at": utc_now(),
    }
    receipts_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(compaction_receipt, receipt)
    run_rclone(
        [
            "copyto",
            str(compaction_receipt),
            f"{remote_root.rstrip('/')}/receipts/{compaction_receipt.name}",
        ],
        retries,
        retry_seconds,
    )
    return compaction_receipt


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
    parser.add_argument("--compact-interval", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stability-seconds", type=int, default=30)
    parser.add_argument("--drain-seconds", type=int, default=300)
    parser.add_argument("--heartbeat-path", type=Path)
    parser.add_argument("--allow-legacy-metadata-only", action="store_true")
    parser.add_argument("--keep-local", type=int, default=2)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--retry-seconds", type=int, default=30)
    return parser.parse_args()


def request_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED, STOP_REQUESTED_AT
    STOP_REQUESTED = True
    STOP_REQUESTED_AT = time.monotonic()


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.start_step % args.interval or args.end_step % args.interval:
        raise SystemExit("start-step and end-step must be aligned to interval")
    if args.compact_interval < 0 or (
        args.compact_interval and args.compact_interval % args.interval
    ):
        raise SystemExit("compact-interval must be zero or a multiple of interval")
    if args.poll_seconds <= 0 or args.stability_seconds < 0 or args.drain_seconds < 0:
        raise SystemExit("poll-seconds must be positive; stability-seconds and drain-seconds cannot be negative")
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.receipts_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = args.heartbeat_path or args.receipts_dir / "archiver-heartbeat.json"
    pending = set(range(args.start_step, args.end_step + 1, args.interval))
    observations: dict[Path, tuple[tuple[int, int], float]] = {}
    require_completion_marker = not args.allow_legacy_metadata_only
    print(
        "ARCHIVER_READY "
        f"start={args.start_step} end={args.end_step} interval={args.interval} "
        f"require_completion_marker={str(require_completion_marker).lower()}",
        flush=True,
    )
    write_heartbeat(heartbeat_path, "ready", pending)

    while pending:
        made_progress = False
        ready_or_stabilizing = False
        for step in sorted(pending):
            checkpoint = args.checkpoints_dir / checkpoint_name(step)
            receipt = args.receipts_dir / f"{checkpoint.name}.json"
            if receipt.is_file():
                try:
                    compact_verified_window(
                        step,
                        args.checkpoints_dir,
                        args.remote_root,
                        args.receipts_dir,
                        args.interval,
                        args.compact_interval,
                        args.experiment_id,
                        args.job_id,
                        args.source_commit,
                        args.retries,
                        args.retry_seconds,
                        args.start_step,
                    )
                except RuntimeError as exc:
                    print(f"COMPACTION_DEFERRED step={step} reason={exc}", flush=True)
                    continue
                pending.remove(step)
                made_progress = True
                continue
            if not checkpoint_ready(checkpoint, require_completion_marker):
                observations.pop(checkpoint, None)
                continue
            ready_or_stabilizing = True
            try:
                stable = checkpoint_is_stable(
                    checkpoint,
                    observations,
                    args.stability_seconds,
                )
            except (OSError, RuntimeError) as exc:
                observations.pop(checkpoint, None)
                print(f"CHECKPOINT_REJECTED step={step} reason={exc}", flush=True)
                continue
            if not stable:
                continue
            print(f"ARCHIVE_START step={step} source={checkpoint}", flush=True)
            write_heartbeat(heartbeat_path, "archiving", pending, f"step={step}")
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
            made_progress = True

        prune_verified(args.checkpoints_dir, args.receipts_dir, args.keep_local)
        if not pending:
            write_heartbeat(heartbeat_path, "complete", pending)
            return 0
        if STOP_REQUESTED:
            assert STOP_REQUESTED_AT is not None
            drain_elapsed = time.monotonic() - STOP_REQUESTED_AT
            if not ready_or_stabilizing or drain_elapsed >= args.drain_seconds:
                write_heartbeat(heartbeat_path, "stopped", pending, f"drain_elapsed={drain_elapsed:.1f}")
                print(f"ARCHIVER_DRAIN_COMPLETE pending={sorted(pending)}", flush=True)
                return 0
        write_heartbeat(
            heartbeat_path,
            "waiting" if not made_progress else "scanning",
            pending,
        )
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
