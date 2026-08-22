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


def expected_parallelism(world_size: int) -> dict[str, int]:
    return {
        "dp_shard_size": world_size,
        "dp_replicate_size": 1,
        "cp_size": 1,
        "cfgp_size": 1,
    }


class ContractError(RuntimeError):
    pass


def unpack_msgpack_object(wire: Any, label: str) -> dict[str, Any]:
    if isinstance(wire, str):
        detail = wire.rstrip() or "empty text frame"
        raise ContractError(f"{label} returned a server-side text error: {detail}")
    if not isinstance(wire, bytes):
        raise ContractError(f"{label} is not binary msgpack: {type(wire).__name__}")
    import msgpack

    value = msgpack.unpackb(wire, raw=False)
    if not isinstance(value, dict):
        raise ContractError(f"{label} is not an object")
    return value


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
    return unpack_msgpack_object(wire, "policy metadata")


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


def resolved_server_checkpoint(path: Path) -> Path:
    """Mirror Cosmos inference argument normalization for local checkpoints."""
    resolved = path.resolve()
    return resolved if resolved.name == "model" else (resolved / "model").resolve()


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
            raise ContractError("policy launcher exited before readiness") from None
        except Exception as error:
            last_error = error
            time.sleep(2)
    if metadata is None:
        raise ContractError(f"policy readiness timeout: {last_error}")
    checkpoint = resolved_server_checkpoint(args.checkpoint)
    expected = {
        "launch_nonce": args.launch_nonce,
        "distributed_world_size": args.expected_policy_world_size,
        "parallelism": expected_parallelism(args.expected_policy_world_size),
        "checkpoint": str(checkpoint),
        "config_file": str(args.config.resolve()),
        "action_stats_path": str(args.action_stats.resolve()),
        "wan2pt2_vae_path": str(args.vae.resolve()),
        "qwen_tokenizer_path": str(args.tokenizer.resolve()),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ContractError(
                f"server metadata mismatch for {key}: expected={value!r}, actual={metadata.get(key)!r}"
            )
    if not isinstance(metadata.get("checkpoint_weight_receipt"), dict):
        raise ContractError("server omitted checkpoint weight receipt")
    server_pid = metadata.get("server_pid")
    if type(server_pid) is not int or not process_is_descendant(server_pid, args.launcher_pid):
        raise ContractError("policy server is not a descendant of the recorded launcher")
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
        "checkpoint": str(checkpoint),
        "config": {"path": str(args.config.resolve()), "size_bytes": args.config.stat().st_size},
        "action_stats": {"path": str(args.action_stats.resolve()), "size_bytes": args.action_stats.stat().st_size},
        "vae": {"path": str(args.vae.resolve()), "size_bytes": args.vae.stat().st_size},
        "tokenizer": str(args.tokenizer.resolve()),
        "source_revision": args.source_revision,
        "resources": {"cpu_affinity_count": len(os.sched_getaffinity(0)), "gpus": gpus},
        "policy_server": sanitized(metadata),
        "validation": {
            "pure_fsdp_server": True,
            "expected_policy_world_size": args.expected_policy_world_size,
            "server_is_launcher_descendant": True,
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
        metadata = unpack_msgpack_object(wire, "shutdown metadata")
        if metadata.get("launch_nonce") != args.launch_nonce:
            raise ContractError("live server nonce changed")
        request = {"__cwam_control__": "shutdown", "launch_nonce": args.launch_nonce}
        connection.send(msgpack.packb(request, use_bin_type=True))
        ack = unpack_msgpack_object(
            connection.recv(timeout=args.timeout_seconds),
            "shutdown acknowledgement",
        )
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


def parse_resume_fragment_specs(values: list[str]) -> list[tuple[str, int, int, Path]]:
    result = []
    for value in values:
        descriptor, separator, raw_path = value.partition("=")
        parts = descriptor.split(":")
        if not separator or len(parts) != 3 or not raw_path:
            raise ContractError(f"invalid resume fragment spec: {value}")
        task, raw_start, raw_stop = parts
        try:
            start = int(raw_start)
            stop = int(raw_stop)
        except ValueError as error:
            raise ContractError(f"invalid resume fragment range: {value}") from error
        if not task or start < 0 or stop <= start:
            raise ContractError(f"invalid resume fragment range: {value}")
        result.append((task, start, stop, Path(raw_path)))
    keys = [(task, start, stop) for task, start, stop, _ in result]
    if len(set(keys)) != len(keys):
        raise ContractError("duplicate resume fragment spec")
    return result


def fragment_episode_rows(
    task: str,
    path: Path,
    start: int,
    stop: int,
    expected_episodes: int,
    *,
    require_complete: bool,
) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        if require_complete:
            raise ContractError(f"missing fragment result: {path}")
        return [], False
    root = load_object(path)
    value = root.get(task)
    if not isinstance(value, dict):
        raise ContractError(f"missing fragment task {task} in {path}")
    expected_metadata = {
        "episode_range_start": start,
        "episode_range_stop": stop,
        "target_episodes": expected_episodes,
        "episodes": stop - start,
    }
    for key, expected in expected_metadata.items():
        if value.get(key) != expected:
            raise ContractError(
                f"fragment metadata mismatch for {task} {key}: expected={expected}, actual={value.get(key)!r}"
            )
    episodes = value.get("episode_results")
    if not isinstance(episodes, list) or len(episodes) > stop - start:
        raise ContractError(f"invalid fragment episode_results for {task}: {path}")
    for offset, episode in enumerate(episodes):
        expected_index = start + offset
        if (
            not isinstance(episode, dict)
            or type(episode.get("success")) is not bool
            or type(episode.get("episode_index")) is not int
            or episode["episode_index"] != expected_index
        ):
            raise ContractError(f"invalid {task} fragment episode at absolute index {expected_index}")
    complete = value.get("complete") is True
    if complete and len(episodes) != stop - start:
        raise ContractError(f"fragment marked complete before its range ended: {path}")
    if require_complete and not complete:
        raise ContractError(f"fragment is incomplete: {path}")
    return episodes, complete


def resume_task_state(
    task: str,
    source: Path,
    fragments: list[tuple[int, int, Path]],
    expected_episodes: int,
) -> dict[str, Any]:
    root = load_object(source)
    value = root.get(task)
    if not isinstance(value, dict):
        raise ContractError(f"missing resume task {task} in {source}")
    source_episodes = value.get("episode_results")
    if not isinstance(source_episodes, list) or len(source_episodes) > expected_episodes:
        raise ContractError(f"invalid resume source episode_results for {task}")
    indexed: dict[int, dict[str, Any]] = {}
    for index, episode in enumerate(source_episodes):
        if not isinstance(episode, dict) or type(episode.get("success")) is not bool:
            raise ContractError(f"invalid resume source episode {task}:{index}")
        indexed[index] = episode
    fragment_complete = []
    fragment_paths = []
    for start, stop, path in fragments:
        if stop > expected_episodes:
            raise ContractError(f"fragment range exceeds target episodes: {task}:{start}:{stop}")
        episodes, complete = fragment_episode_rows(
            task, path, start, stop, expected_episodes, require_complete=False
        )
        fragment_complete.append(complete)
        fragment_paths.append(str(path))
        for episode in episodes:
            index = episode["episode_index"]
            if index in indexed:
                raise ContractError(f"duplicate absolute episode index for {task}: {index}")
            indexed[index] = episode
    successes = sum(int(episode["success"]) for episode in indexed.values())
    complete = (
        sorted(indexed) == list(range(expected_episodes))
        and bool(fragment_complete)
        and all(fragment_complete)
    )
    return {
        "task": task,
        "completed": len(indexed),
        "successes": successes,
        "success_rate": successes / len(indexed) if indexed else 0.0,
        "complete": complete,
        "result_path": str(source.resolve()),
        "fragment_paths": fragment_paths,
    }


def load_baseline(path: Path, tasks: list[str]) -> dict[str, Any]:
    if path.suffix == ".ndjson":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records or not isinstance(records[-1], dict):
            raise ContractError(f"baseline progress record is empty: {path}")
        root = records[-1]
        if root.get("kind") != "cwam_h_matched_fast_progress":
            raise ContractError(f"unexpected baseline progress kind: {root.get('kind')!r}")
        terminal_path = path.parent.parent / "status.json"
        terminal = load_object(terminal_path) if terminal_path.exists() else None
        record_kind = "rollout_complete_progress"
    else:
        root = load_object(path)
        if root.get("status") != "complete":
            raise ContractError(f"baseline summary is not complete: {path}")
        terminal_path = path.parent / "status.json"
        terminal = load_object(terminal_path) if terminal_path.exists() else None
        record_kind = "terminal_summary"
    aggregate = root.get("aggregate")
    states = root.get("tasks")
    if not isinstance(aggregate, dict) or not isinstance(states, list):
        raise ContractError(f"invalid baseline summary: {path}")
    by_task = {state.get("task"): state for state in states if isinstance(state, dict)}
    if set(by_task) != set(tasks):
        raise ContractError("baseline task inventory differs from current evaluation")
    episodes = aggregate.get("episodes", aggregate.get("completed"))
    successes = aggregate.get("successes")
    success_rate = aggregate.get("success_rate")
    if not isinstance(episodes, int) or not isinstance(successes, int) or episodes <= 0:
        raise ContractError(f"invalid baseline aggregate: {aggregate!r}")
    if success_rate is None:
        success_rate = successes / episodes
    if not isinstance(success_rate, (int, float)):
        raise ContractError(f"invalid baseline success rate: {aggregate!r}")
    if any(
        not isinstance(state, dict)
        or state.get("complete") is not True
        or state.get("completed") != 100
        for state in states
    ):
        raise ContractError("baseline does not contain 100 complete episodes for every task")
    return {
        "record_path": str(path.resolve()),
        "record_kind": record_kind,
        "rollouts_complete": True,
        "terminal_status_path": str(terminal_path.resolve()) if terminal is not None else None,
        "terminal_status": terminal.get("status") if terminal is not None else None,
        "terminal_detail": terminal.get("detail") if terminal is not None else None,
        "experiment_id": root.get("experiment_id"),
        "logical_run_id": root.get("logical_run_id"),
        "model_label": root.get("model_label"),
        "episodes": episodes,
        "successes": successes,
        "success_rate": float(success_rate),
        "tasks": by_task,
    }


def check_baseline(args: argparse.Namespace) -> dict[str, Any]:
    tasks = list(args.task)
    if len(set(tasks)) != len(tasks):
        raise ContractError("duplicate baseline task")
    baseline = load_baseline(args.baseline_record, tasks)
    return {
        "status": "passed",
        "record_path": baseline["record_path"],
        "record_kind": baseline["record_kind"],
        "episodes": baseline["episodes"],
        "successes": baseline["successes"],
        "success_rate": baseline["success_rate"],
        "task_count": len(baseline["tasks"]),
    }


def compare_with_baseline(states: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    completed = sum(state["completed"] for state in states)
    successes = sum(state["successes"] for state in states)
    current_rate = successes / completed if completed else None
    task_comparisons = []
    for state in states:
        baseline_state = baseline["tasks"][state["task"]]
        baseline_rate = baseline_state.get("success_rate")
        if not isinstance(baseline_rate, (int, float)):
            raise ContractError(f"invalid baseline task rate: {baseline_state!r}")
        task_comparisons.append(
            {
                "task": state["task"],
                "current_completed": state["completed"],
                "current_successes": state["successes"],
                "current_partial_success_rate": state.get("success_rate", 0.0),
                "baseline_episodes": baseline_state.get("completed"),
                "baseline_successes": baseline_state.get("successes"),
                "baseline_success_rate": baseline_rate,
                "delta_success_rate": state.get("success_rate", 0.0) - baseline_rate
                if state["completed"]
                else None,
            }
        )
    return {
        "comparison_scope": "current_completed_episodes_vs_baseline_final",
        "current_completed": completed,
        "current_successes": successes,
        "current_partial_success_rate": current_rate,
        "baseline_record_path": baseline["record_path"],
        "baseline_record_kind": baseline["record_kind"],
        "baseline_rollouts_complete": baseline["rollouts_complete"],
        "baseline_terminal_status_path": baseline["terminal_status_path"],
        "baseline_terminal_status": baseline["terminal_status"],
        "baseline_terminal_detail": baseline["terminal_detail"],
        "baseline_experiment_id": baseline["experiment_id"],
        "baseline_logical_run_id": baseline["logical_run_id"],
        "baseline_model_label": baseline["model_label"],
        "baseline_episodes": baseline["episodes"],
        "baseline_successes": baseline["successes"],
        "baseline_success_rate": baseline["success_rate"],
        "delta_success_rate": current_rate - baseline["success_rate"] if current_rate is not None else None,
        "tasks": task_comparisons,
    }


def check_task(args: argparse.Namespace) -> dict[str, Any]:
    state = task_state(args.task, args.result)
    if state["completed"] != args.expected_episodes or not state["complete"]:
        raise ContractError(
            f"incomplete task result: expected={args.expected_episodes}, actual={state}"
        )
    return state


def check_partial_task(args: argparse.Namespace) -> dict[str, Any]:
    state = task_state(args.task, args.result)
    if (
        state["completed"] != args.expected_episodes
        or state["successes"] != args.expected_successes
        or state["complete"]
    ):
        raise ContractError(
            "resume source differs from the predeclared partial result: "
            f"expected_episodes={args.expected_episodes}, "
            f"expected_successes={args.expected_successes}, actual={state}"
        )
    return state


def watch_progress(args: argparse.Namespace) -> dict[str, Any]:
    specs = parse_task_specs(args.task_spec)
    baseline = load_baseline(args.baseline_record, [task for task, _ in specs])
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
                    "baseline_comparison": compare_with_baseline(states, baseline),
                }
                append_ndjson(args.output, event)
                write_json(args.comparison_output, event["baseline_comparison"], replace=True)
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


def watch_resume_progress(args: argparse.Namespace) -> dict[str, Any]:
    complete_specs = parse_task_specs(args.task_spec)
    resume_specs = parse_task_specs(args.resume_task_spec)
    fragment_specs = parse_resume_fragment_specs(args.fragment_spec)
    complete_tasks = {task for task, _ in complete_specs}
    resume_tasks = {task for task, _ in resume_specs}
    if complete_tasks & resume_tasks:
        raise ContractError("task appears in both complete and resume task specs")
    fragments_by_task: dict[str, list[tuple[int, int, Path]]] = {
        task: [] for task in resume_tasks
    }
    for task, start, stop, path in fragment_specs:
        if task not in fragments_by_task:
            raise ContractError(f"fragment has no matching resume task: {task}")
        fragments_by_task[task].append((start, stop, path))
    if any(not fragments for fragments in fragments_by_task.values()):
        raise ContractError("every resume task needs at least one fragment")
    tasks = [task for task, _ in complete_specs + resume_specs]
    baseline = load_baseline(args.baseline_record, tasks)
    previous = None
    while True:
        states = None
        try:
            states = [task_state(task, path) for task, path in complete_specs]
            states.extend(
                resume_task_state(
                    task,
                    path,
                    fragments_by_task[task],
                    args.expected_episodes,
                )
                for task, path in resume_specs
            )
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
                    "baseline_comparison": compare_with_baseline(states, baseline),
                }
                append_ndjson(args.output, event)
                write_json(args.comparison_output, event["baseline_comparison"], replace=True)
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


def merge_resume_fragments(args: argparse.Namespace) -> dict[str, Any]:
    fragment_specs = parse_resume_fragment_specs(
        [f"{args.task}:{value}" for value in args.fragment]
    )
    if any(task != args.task for task, _, _, _ in fragment_specs):
        raise ContractError("fragment task mismatch")
    if args.output.exists() or args.receipt.exists():
        raise ContractError("refusing to overwrite merged result or merge receipt")
    root = load_object(args.source)
    source_value = root.get(args.task)
    if not isinstance(source_value, dict):
        raise ContractError(f"missing source task {args.task} in {args.source}")
    source_episodes = source_value.get("episode_results")
    if not isinstance(source_episodes, list) or len(source_episodes) > args.expected_episodes:
        raise ContractError("invalid source episode_results")
    indexed: dict[int, dict[str, Any]] = {}
    for index, episode in enumerate(source_episodes):
        if not isinstance(episode, dict) or type(episode.get("success")) is not bool:
            raise ContractError(f"invalid source episode {args.task}:{index}")
        indexed[index] = dict(episode)
    fragment_receipts = []
    for _, start, stop, path in fragment_specs:
        episodes, _ = fragment_episode_rows(
            args.task,
            path,
            start,
            stop,
            args.expected_episodes,
            require_complete=True,
        )
        for episode in episodes:
            index = episode["episode_index"]
            if index in indexed:
                raise ContractError(f"duplicate absolute episode index for {args.task}: {index}")
            indexed[index] = {key: value for key, value in episode.items() if key != "episode_index"}
        fragment_receipts.append(
            {"start": start, "stop": stop, "episodes": len(episodes), "path": str(path.resolve())}
        )
    expected_indices = list(range(args.expected_episodes))
    if sorted(indexed) != expected_indices:
        missing = sorted(set(expected_indices) - set(indexed))
        extra = sorted(set(indexed) - set(expected_indices))
        raise ContractError(f"merged episode inventory mismatch: missing={missing}, extra={extra}")
    merged_episodes = [indexed[index] for index in expected_indices]
    successes = sum(int(episode["success"]) for episode in merged_episodes)
    place_anytime_count = sum(int(bool(episode.get("place_anytime", False))) for episode in merged_episodes)
    merged_value = dict(source_value)
    for key in ("episode_range_start", "episode_range_stop", "target_episodes"):
        merged_value.pop(key, None)
    merged_value.update(
        {
            "success_rate": successes / args.expected_episodes,
            "successes": successes,
            "place_anytime_rate": place_anytime_count / args.expected_episodes,
            "place_anytime_count": place_anytime_count,
            "episodes": args.expected_episodes,
            "episode_results": merged_episodes,
            "complete": True,
        }
    )
    write_json(args.output, {args.task: merged_value})
    receipt = {
        "schema_version": "metadata_no_hash_v1",
        "kind": "cwam_h_episode_fragment_merge",
        "status": "passed",
        "completed_at_utc": utc_now(),
        "task": args.task,
        "source": str(args.source.resolve()),
        "source_episodes": len(source_episodes),
        "fragments": fragment_receipts,
        "output": str(args.output.resolve()),
        "final_episodes": args.expected_episodes,
        "final_successes": successes,
    }
    write_json(args.receipt, receipt)
    return receipt


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    specs = parse_task_specs(args.task_spec)
    states = [task_state(task, path) for task, path in specs]
    baseline = load_baseline(args.baseline_record, [task for task, _ in specs])
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
        "baseline_comparison": compare_with_baseline(states, baseline),
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
    runtime.add_argument("--expected-gpu-count", type=int, choices=(2, 4, 6, 8), required=True)
    runtime.add_argument("--expected-policy-world-size", type=int, choices=(1, 4), required=True)
    shutdown = commands.add_parser("shutdown-server")
    shutdown.add_argument("--uri", required=True)
    shutdown.add_argument("--output", type=Path, required=True)
    shutdown.add_argument("--timeout-seconds", type=float, default=60)
    shutdown.add_argument("--launch-nonce", required=True)
    shutdown.add_argument("--runtime-receipt", type=Path, required=True)
    baseline = commands.add_parser("check-baseline")
    baseline.add_argument("--baseline-record", type=Path, required=True)
    baseline.add_argument("--task", action="append", required=True)
    progress = commands.add_parser("watch-progress")
    progress.add_argument("--experiment-id", required=True)
    progress.add_argument("--logical-run-id", required=True)
    progress.add_argument("--model-label", required=True)
    progress.add_argument("--task-spec", action="append", required=True)
    progress.add_argument("--pid", type=int, action="append", required=True)
    progress.add_argument("--interval-seconds", type=float, default=15)
    progress.add_argument("--baseline-record", type=Path, required=True)
    progress.add_argument("--comparison-output", type=Path, required=True)
    progress.add_argument("--output", type=Path, required=True)
    resume_progress = commands.add_parser("watch-resume-progress")
    resume_progress.add_argument("--experiment-id", required=True)
    resume_progress.add_argument("--logical-run-id", required=True)
    resume_progress.add_argument("--model-label", required=True)
    resume_progress.add_argument("--task-spec", action="append", required=True)
    resume_progress.add_argument("--resume-task-spec", action="append", required=True)
    resume_progress.add_argument("--fragment-spec", action="append", required=True)
    resume_progress.add_argument("--expected-episodes", type=int, required=True)
    resume_progress.add_argument("--pid", type=int, action="append", required=True)
    resume_progress.add_argument("--interval-seconds", type=float, default=15)
    resume_progress.add_argument("--baseline-record", type=Path, required=True)
    resume_progress.add_argument("--comparison-output", type=Path, required=True)
    resume_progress.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check-task")
    check.add_argument("--task", required=True)
    check.add_argument("--result", type=Path, required=True)
    check.add_argument("--expected-episodes", type=int, required=True)
    partial = commands.add_parser("check-partial-task")
    partial.add_argument("--task", required=True)
    partial.add_argument("--result", type=Path, required=True)
    partial.add_argument("--expected-episodes", type=int, required=True)
    partial.add_argument("--expected-successes", type=int, required=True)
    merge = commands.add_parser("merge-resume-fragments")
    merge.add_argument("--task", required=True)
    merge.add_argument("--source", type=Path, required=True)
    merge.add_argument("--fragment", action="append", required=True)
    merge.add_argument("--expected-episodes", type=int, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--receipt", type=Path, required=True)
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
    summary.add_argument("--baseline-record", type=Path, required=True)
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
        elif args.command == "check-baseline":
            result = check_baseline(args)
        elif args.command == "watch-progress":
            result = watch_progress(args)
        elif args.command == "watch-resume-progress":
            result = watch_resume_progress(args)
        elif args.command == "check-task":
            result = check_task(args)
        elif args.command == "check-partial-task":
            result = check_partial_task(args)
        elif args.command == "merge-resume-fragments":
            result = merge_resume_fragments(args)
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
