#!/usr/bin/env python3
"""Runtime checks and durable progress records for the matched H fast eval."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


EXPECTED_PARALLELISM = {
    "dp_shard_size": 4,
    "dp_replicate_size": 1,
    "cp_size": 1,
    "cfgp_size": 1,
}


class ContractError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any], *, replace: bool = False) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ContractError(f"refusing to overwrite artifact: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_ndjson(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def process_is_descendant(pid: int, ancestor: int) -> bool:
    current = pid
    seen: set[int] = set()
    while current > 1 and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        try:
            current = int(Path(f"/proc/{current}/stat").read_text().split()[3])
        except (OSError, ValueError, IndexError):
            return False
    return current == ancestor


def receive_metadata(uri: str, timeout_seconds: float) -> dict[str, Any]:
    import msgpack
    import websockets.sync.client

    with websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        open_timeout=min(timeout_seconds, 10.0),
        close_timeout=10,
        ping_interval=20,
        ping_timeout=30,
    ) as connection:
        wire = connection.recv(timeout=min(timeout_seconds, 60.0))
    if not isinstance(wire, bytes):
        raise ContractError("policy metadata is not binary msgpack")
    value = msgpack.unpackb(wire, raw=False)
    if not isinstance(value, dict):
        raise ContractError("policy metadata is not an object")
    return value


def gpu_inventory() -> list[dict[str, str]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    result = []
    for line in output.splitlines():
        index, name, memory = (part.strip() for part in line.split(",", 2))
        result.append({"index": index, "name": name, "memory_total_mib": memory})
    return result


def sanitized(value: Any) -> Any:
    """Exclude digest-like fields emitted internally by the serving stack."""
    if isinstance(value, dict):
        return {
            key: sanitized(item)
            for key, item in value.items()
            if not any(marker in key.lower() for marker in ("sha256", "hash", "digest", "merkle", "weight_tree"))
            and key != "checkpoint_weight_receipt"
        }
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    return value


def capture_runtime(args: argparse.Namespace) -> dict[str, Any]:
    deadline = time.monotonic() + args.timeout_seconds
    metadata = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            os.kill(args.launcher_pid, 0)
            metadata = receive_metadata(args.uri, max(1.0, deadline - time.monotonic()))
            break
        except ProcessLookupError:
            raise ContractError("torchrun exited before policy readiness") from None
        except Exception as error:
            last_error = error
            time.sleep(2)
    if metadata is None:
        raise ContractError(f"policy readiness timeout: {last_error}")
    expected = {
        "launch_nonce": args.launch_nonce,
        "distributed_world_size": 4,
        "parallelism": EXPECTED_PARALLELISM,
        "checkpoint": str(args.checkpoint.resolve()),
        "config_file": str(args.config.resolve()),
        "action_stats_path": str(args.action_stats.resolve()),
        "wan2pt2_vae_path": str(args.vae.resolve()),
        "qwen_tokenizer_path": str(args.tokenizer.resolve()),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ContractError(f"server metadata mismatch for {key}: {metadata.get(key)!r}")
    if not isinstance(metadata.get("checkpoint_weight_receipt"), dict):
        raise ContractError("server omitted checkpoint weight receipt")
    server_pid = metadata.get("server_pid")
    if type(server_pid) is not int or not process_is_descendant(server_pid, args.launcher_pid):
        raise ContractError("rank-0 server is not a torchrun descendant")
    environment = Path(f"/proc/{args.launcher_pid}/environ").read_bytes().split(b"\0")
    visible = next((item.split(b"=", 1)[1].decode() for item in environment if item.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
    if visible != args.visible_devices:
        raise ContractError("CUDA_VISIBLE_DEVICES differs from the declared rank group")
    gpus = gpu_inventory()
    if len(gpus) != args.expected_gpu_count or any("H200" not in gpu["name"] for gpu in gpus):
        raise ContractError(f"expected {args.expected_gpu_count} visible H200 GPUs, got {gpus}")
    value = {
        "schema_version": "metadata_no_hash_v1",
        "kind": "cwam_h_matched_fast_runtime",
        "status": "passed",
        "captured_at_utc": utc_now(),
        "platform": "H cluster",
        "hostname": socket.gethostname(),
        "launcher_pid": args.launcher_pid,
        "server_pid": server_pid,
        "uri": args.uri,
        "server_label": args.server_label,
        "visible_devices": args.visible_devices,
        "launch_nonce": args.launch_nonce,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": {"path": str(args.config.resolve()), "size_bytes": args.config.stat().st_size},
        "action_stats": {"path": str(args.action_stats.resolve()), "size_bytes": args.action_stats.stat().st_size},
        "vae": {"path": str(args.vae.resolve()), "size_bytes": args.vae.stat().st_size},
        "tokenizer": str(args.tokenizer.resolve()),
        "source_revision": args.source_revision,
        "resources": {"cpu_affinity_count": len(os.sched_getaffinity(0)), "gpus": gpus},
        "policy_server": sanitized(metadata),
        "validation": {
            "four_rank_fsdp_server": True,
            "rank0_is_torchrun_descendant": True,
            "exact_input_paths_match_server": True,
            "checkpoint_weight_receipt_present": True,
        },
    }
    write_json(args.output, value)
    return value


def shutdown_server(args: argparse.Namespace) -> dict[str, Any]:
    import msgpack
    import websockets.sync.client

    runtime = load_object(args.runtime_receipt)
    if runtime.get("launch_nonce") != args.launch_nonce or runtime.get("uri") != args.uri:
        raise ContractError("shutdown inputs differ from runtime receipt")
    with websockets.sync.client.connect(args.uri, compression=None, max_size=None, open_timeout=10, close_timeout=10) as connection:
        wire = connection.recv(timeout=args.timeout_seconds)
        metadata = msgpack.unpackb(wire, raw=False)
        if metadata.get("launch_nonce") != args.launch_nonce:
            raise ContractError("live server nonce changed")
        request = {"__cwam_control__": "shutdown", "launch_nonce": args.launch_nonce}
        connection.send(msgpack.packb(request, use_bin_type=True))
        ack = msgpack.unpackb(connection.recv(timeout=args.timeout_seconds), raw=False)
    if ack.get("cwam_control") != "shutdown_ack" or ack.get("launch_nonce") != args.launch_nonce:
        raise ContractError(f"invalid shutdown acknowledgement: {ack!r}")
    value = {
        "schema_version": "metadata_no_hash_v1",
        "kind": "cwam_h_controlled_shutdown",
        "status": "passed",
        "completed_at_utc": utc_now(),
        "uri": args.uri,
        "server_label": runtime.get("server_label"),
        "ack": sanitized(ack),
    }
    write_json(args.output, value)
    return value


def task_state(task: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"task": task, "completed": 0, "successes": 0, "complete": False, "result_path": str(path)}
    root = load_object(path)
    value = root.get(task)
    if not isinstance(value, dict):
        raise ContractError(f"missing task {task} in {path}")
    episodes = value.get("episode_results")
    if not isinstance(episodes, list):
        raise ContractError(f"missing episode_results for {task}")
    successes = 0
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict) or type(episode.get("success")) is not bool:
            raise ContractError(f"invalid {task} episode {index}")
        successes += int(episode["success"])
    return {
        "task": task,
        "completed": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "complete": value.get("complete") is True,
        "result_path": str(path.resolve()),
    }


def parse_task_specs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        task, separator, raw_path = value.partition("=")
        if not separator or not task or not raw_path:
            raise ContractError(f"invalid task spec: {value}")
        result.append((task, Path(raw_path)))
    if len({task for task, _ in result}) != len(result):
        raise ContractError("duplicate task spec")
    return result


def watch_progress(args: argparse.Namespace) -> dict[str, Any]:
    specs = parse_task_specs(args.task_spec)
    previous = None
    while True:
        states = None
        try:
            states = [task_state(task, path) for task, path in specs]
        except (OSError, json.JSONDecodeError):
            pass
        if states is not None:
            signature = [(state["completed"], state["successes"], state["complete"]) for state in states]
            if signature != previous:
                event = {
                    "schema_version": "metadata_no_hash_v1",
                    "kind": "cwam_h_matched_fast_progress",
                    "recorded_at_utc": utc_now(),
                    "experiment_id": args.experiment_id,
                    "logical_run_id": args.logical_run_id,
                    "model_label": args.model_label,
                    "tasks": states,
                    "aggregate": {
                        "completed": sum(state["completed"] for state in states),
                        "successes": sum(state["successes"] for state in states),
                    },
                }
                append_ndjson(args.output, event)
                previous = signature
        alive = False
        for pid in args.pid:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                pass
        if not alive:
            break
        time.sleep(args.interval_seconds)
    return {"status": "finished", "output": str(args.output), "last_signature": previous}


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    states = [task_state(task, path) for task, path in parse_task_specs(args.task_spec)]
    if any(state["completed"] != args.expected_episodes or not state["complete"] for state in states):
        raise ContractError(f"incomplete task results: {states}")
    for path in args.runtime_receipt + args.shutdown_receipt:
        if load_object(path).get("status") != "passed":
            raise ContractError(f"failed runtime/control receipt: {path}")
    total = sum(state["completed"] for state in states)
    successes = sum(state["successes"] for state in states)
    value = {
        "schema_version": "metadata_no_hash_v1",
        "kind": "cwam_h_matched_fast_evaluation_summary",
        "status": "complete",
        "completed_at_utc": utc_now(),
        "experiment_id": args.experiment_id,
        "logical_run_id": args.logical_run_id,
        "model_label": args.model_label,
        "source_revision": args.source_revision,
        "checkpoint": str(args.checkpoint.resolve()),
        "node": args.node,
        "rjob_name": args.rjob_name,
        "protocol": {
            "episodes_per_task": args.expected_episodes,
            "max_steps": 400,
            "open_loop_horizon": 12,
            "policy_sampler": "unipc",
            "policy_steps": 30,
            "policy_guidance": 1.0,
            "base_seed": 42,
        },
        "tasks": states,
        "aggregate": {"episodes": total, "successes": successes, "success_rate": successes / total},
        "runtime_receipts": [str(path.resolve()) for path in args.runtime_receipt],
        "shutdown_receipts": [str(path.resolve()) for path in args.shutdown_receipt],
        "progress_log": str(args.progress_log.resolve()),
    }
    write_json(args.output, value)
    with args.csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("task", "successes", "episodes", "success_rate"))
        writer.writeheader()
        for state in states:
            writer.writerow({"task": state["task"], "successes": state["successes"], "episodes": state["completed"], "success_rate": state["success_rate"]})
    return value


def write_status(args: argparse.Namespace) -> dict[str, Any]:
    value = {
        "schema_version": "metadata_no_hash_v1",
        "experiment_id": args.experiment_id,
        "logical_run_id": args.logical_run_id,
        "status": args.status,
        "detail": args.detail,
        "updated_at_utc": utc_now(),
    }
    write_json(args.output, value, replace=True)
    append_ndjson(args.timeline, value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runtime = commands.add_parser("capture-runtime")
    runtime.add_argument("--uri", required=True)
    runtime.add_argument("--output", type=Path, required=True)
    runtime.add_argument("--timeout-seconds", type=float, default=1800)
    runtime.add_argument("--launcher-pid", type=int, required=True)
    runtime.add_argument("--server-label", required=True)
    runtime.add_argument("--visible-devices", required=True)
    runtime.add_argument("--launch-nonce", required=True)
    runtime.add_argument("--checkpoint", type=Path, required=True)
    runtime.add_argument("--config", type=Path, required=True)
    runtime.add_argument("--action-stats", type=Path, required=True)
    runtime.add_argument("--vae", type=Path, required=True)
    runtime.add_argument("--tokenizer", type=Path, required=True)
    runtime.add_argument("--source-revision", required=True)
    runtime.add_argument("--expected-gpu-count", type=int, choices=(6, 8), required=True)
    shutdown = commands.add_parser("shutdown-server")
    shutdown.add_argument("--uri", required=True)
    shutdown.add_argument("--output", type=Path, required=True)
    shutdown.add_argument("--timeout-seconds", type=float, default=60)
    shutdown.add_argument("--launch-nonce", required=True)
    shutdown.add_argument("--runtime-receipt", type=Path, required=True)
    progress = commands.add_parser("watch-progress")
    progress.add_argument("--experiment-id", required=True)
    progress.add_argument("--logical-run-id", required=True)
    progress.add_argument("--model-label", required=True)
    progress.add_argument("--task-spec", action="append", required=True)
    progress.add_argument("--pid", type=int, action="append", required=True)
    progress.add_argument("--interval-seconds", type=float, default=15)
    progress.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--experiment-id", required=True)
    summary.add_argument("--logical-run-id", required=True)
    summary.add_argument("--model-label", required=True)
    summary.add_argument("--source-revision", required=True)
    summary.add_argument("--checkpoint", type=Path, required=True)
    summary.add_argument("--node", required=True)
    summary.add_argument("--rjob-name", required=True)
    summary.add_argument("--task-spec", action="append", required=True)
    summary.add_argument("--expected-episodes", type=int, default=20)
    summary.add_argument("--runtime-receipt", type=Path, action="append", required=True)
    summary.add_argument("--shutdown-receipt", type=Path, action="append", required=True)
    summary.add_argument("--progress-log", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--csv-output", type=Path, required=True)
    status = commands.add_parser("write-status")
    status.add_argument("--experiment-id", required=True)
    status.add_argument("--logical-run-id", required=True)
    status.add_argument("--status", required=True)
    status.add_argument("--detail", default="")
    status.add_argument("--output", type=Path, required=True)
    status.add_argument("--timeline", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "capture-runtime":
            result = capture_runtime(args)
        elif args.command == "shutdown-server":
            result = shutdown_server(args)
        elif args.command == "watch-progress":
            result = watch_progress(args)
        elif args.command == "summarize":
            result = summarize(args)
        else:
            result = write_status(args)
    except (ContractError, OSError, json.JSONDecodeError, subprocess.SubprocessError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
