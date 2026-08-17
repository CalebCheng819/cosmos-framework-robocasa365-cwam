#!/usr/bin/env python3
"""Validate and summarize the B4 RoboCasa365 DLC evaluation without hashes."""

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


ATOMIC_TASKS = (
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
)
EXPECTED_MODEL_FILES = {
    ".metadata": 1_612_962,
    "__0_0.distcp": 3_828_985_322,
    "__1_0.distcp": 3_828_985_194,
    "__2_0.distcp": 3_828_980_056,
    "__3_0.distcp": 3_828_978_543,
    "__4_0.distcp": 3_828_974_063,
    "__5_0.distcp": 3_828_962_489,
    "__6_0.distcp": 3_828_951_514,
    "__7_0.distcp": 3_828_948_488,
}
EXPECTED_AUX_FILES = {
    "config.yaml": 22_139,
    "ema-bf16-metadata-receipt.json": 2_482,
    "serving-checkpoint-size-manifest.json": 997,
}
EXPECTED_PARALLELISM = {
    "dp_shard_size": 4,
    "dp_replicate_size": 1,
    "cp_size": 1,
    "cfgp_size": 1,
}
EXPECTED_SOURCE_COMMIT = "e7775ffcea0a2d577597fbeb05c8bb92905aba0f"
EXPECTED_RUN_ID = "CWAM-COSMOS3-B4-ADAPTIVE-DSELECT-R1-20260814"


class ContractError(RuntimeError):
    """Raised when a formal evaluation artifact violates the fixed protocol."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any], *, exclusive: bool = True) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and os.path.lexists(path):
        raise ContractError(f"refusing to overwrite JSON artifact: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        if exclusive:
            os.link(temporary, path, follow_symlinks=False)
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return value


def _file_descriptor(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"expected regular non-symlink file: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": metadata.st_size,
        "modified_at_ns": metadata.st_mtime_ns,
    }


def verify_checkpoint(root: Path, output: Path) -> dict[str, Any]:
    root = root.absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ContractError(f"checkpoint root must be a canonical directory: {root}")
    expected = {
        **{f"model/{name}": size for name, size in EXPECTED_MODEL_FILES.items()},
        **EXPECTED_AUX_FILES,
    }
    observed: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"checkpoint contains a symlink: {path}")
        if path.is_file():
            observed[path.relative_to(root).as_posix()] = path.stat().st_size
        elif not path.is_dir():
            raise ContractError(f"checkpoint contains a special entry: {path}")
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        wrong = sorted(name for name in set(expected) & set(observed) if expected[name] != observed[name])
        raise ContractError(f"checkpoint inventory mismatch: missing={missing} extra={extra} wrong_size={wrong}")

    receipt = _load_object(root / "ema-bf16-metadata-receipt.json", "EMA export receipt")
    selection = receipt.get("selection")
    source = receipt.get("source")
    target = receipt.get("target")
    validation = receipt.get("validation")
    if (
        receipt.get("kind") != "cwam_serving_ema_bf16_metadata_receipt"
        or receipt.get("status") != "succeeded"
        or not isinstance(selection, dict)
        or selection.get("key_count") != 986
        or selection.get("chunk_count") != 7_837
        or selection.get("target_dtype") != "torch.bfloat16"
        or selection.get("target_logical_bytes") != 30_619_447_304
        or not isinstance(source, dict)
        or source.get("git_commit") != EXPECTED_SOURCE_COMMIT
        or source.get("run_id") != EXPECTED_RUN_ID
        or not isinstance(target, dict)
        or target.get("file_count") != 9
        or target.get("shard_count") != 8
        or target.get("total_physical_bytes") != sum(EXPECTED_MODEL_FILES.values())
        or not isinstance(validation, dict)
        or validation.get("full_semantic_readback") is not True
        or validation.get("source_file_identities_stable") is not True
    ):
        raise ContractError("EMA export receipt does not bind the validated B4 serving artifact")
    manifest = _load_object(root / "serving-checkpoint-size-manifest.json", "size manifest")
    manifest_files = manifest.get("files")
    if (
        manifest.get("kind") != "cwam_serving_checkpoint_size_manifest"
        or manifest.get("file_count") != 9
        or manifest.get("total_physical_bytes") != sum(EXPECTED_MODEL_FILES.values())
        or not isinstance(manifest_files, list)
        or {item.get("relative_path"): item.get("size_bytes") for item in manifest_files} != EXPECTED_MODEL_FILES
    ):
        raise ContractError("checkpoint size manifest differs from the fixed model inventory")
    result = {
        "schema_version": "metadata_no_hash_v1",
        "kind": "cwam_b4_serving_checkpoint_verification",
        "status": "passed",
        "completed_at_utc": _utc_now(),
        "checkpoint_root": str(root),
        "model_path": str(root / "model"),
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_run_id": EXPECTED_RUN_ID,
        "ema_key_count": 986,
        "ema_chunk_count": 7_837,
        "file_count": len(observed),
        "total_physical_bytes": sum(observed.values()),
        "model_file_count": len(EXPECTED_MODEL_FILES),
        "model_physical_bytes": sum(EXPECTED_MODEL_FILES.values()),
        "validation": {
            "exact_path_and_size_inventory": True,
            "full_semantic_readback_passed_at_export": True,
            "source_file_identities_stable_at_export": True,
        },
    }
    _write_json(output, result)
    return result


def _process_is_descendant(pid: int, ancestor: int) -> bool:
    current = pid
    seen: set[int] = set()
    while current > 1 and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        try:
            fields = Path(f"/proc/{current}/stat").read_text().split()
            current = int(fields[3])
        except (OSError, ValueError, IndexError):
            return False
    return current == ancestor


def _receive_metadata(uri: str, timeout_seconds: float) -> dict[str, Any]:
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
        raise ContractError("policy server metadata must be binary msgpack")
    value = msgpack.unpackb(wire, raw=False)
    if not isinstance(value, dict):
        raise ContractError("policy server metadata must be an object")
    return value


def _gpu_inventory() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    result = []
    for line in output.splitlines():
        index, name, capability = (part.strip() for part in line.split(",", 2))
        result.append({"index": index, "name": name, "compute_capability": capability})
    return result


def _process_environment(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as error:
        raise ContractError(f"cannot read launcher environment: {error}") from error
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode("utf-8", "strict")] = value.decode("utf-8", "strict")
    return result


def capture_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if args.launcher_pid <= 0:
        raise ContractError("launcher PID must be positive")
    deadline = time.monotonic() + args.timeout_seconds
    metadata = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            os.kill(args.launcher_pid, 0)
            metadata = _receive_metadata(args.uri, max(0.1, deadline - time.monotonic()))
            break
        except ProcessLookupError:
            raise ContractError("torchrun exited before server readiness") from None
        except Exception as error:  # Connection failures vary by runtime.
            last_error = error
            time.sleep(2)
    if metadata is None:
        raise ContractError(f"policy server readiness timeout: {last_error}")
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
            raise ContractError(f"policy server metadata mismatch for {key}: {metadata.get(key)!r}")
    weight_receipt = metadata.get("checkpoint_weight_receipt")
    if not isinstance(weight_receipt, dict):
        raise ContractError("policy server omitted checkpoint_weight_receipt")
    server_pid = metadata.get("server_pid")
    if type(server_pid) is not int or not _process_is_descendant(server_pid, args.launcher_pid):
        raise ContractError("rank-0 server is not a descendant of torchrun")
    gpus = _gpu_inventory()
    if len(gpus) != 8 or any("4090" not in gpu["name"] or gpu["compute_capability"] != "8.9" for gpu in gpus):
        raise ContractError(f"expected exactly eight visible RTX 4090 GPUs, got {gpus}")
    cpu_count = len(os.sched_getaffinity(0))
    if cpu_count != 126:
        raise ContractError(f"expected 126 effective CPUs, got {cpu_count}")
    launcher_environment = _process_environment(args.launcher_pid)
    if launcher_environment.get("CUDA_VISIBLE_DEVICES") != args.visible_devices:
        raise ContractError("torchrun CUDA_VISIBLE_DEVICES differs from the declared server group")
    result = {
        "schema_version": "metadata_no_hash_v1",
        "kind": "cwam_b4_dlc_runtime_receipt",
        "status": "passed",
        "captured_at_utc": _utc_now(),
        "platform": {"provider": "Alibaba Cloud", "service": "PAI DLC", "region": "cn-beijing"},
        "hostname": socket.gethostname(),
        "launcher_pid": args.launcher_pid,
        "server_pid": server_pid,
        "uri": args.uri,
        "server_label": args.server_label,
        "visible_devices": args.visible_devices,
        "launch_nonce": args.launch_nonce,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": _file_descriptor(args.config),
        "action_stats": _file_descriptor(args.action_stats),
        "vae": _file_descriptor(args.vae),
        "tokenizer_path": str(args.tokenizer.resolve()),
        "source_commit": args.source_commit,
        "demian_commit": args.demian_commit,
        "resources": {"cpu_affinity_count": cpu_count, "gpus": gpus},
        "policy_server": metadata,
        "validation": {
            "four_rank_fsdp_server": True,
            "rank0_is_torchrun_descendant": True,
            "declared_gpu_group_matches_torchrun_environment": True,
            "exact_input_paths_match_server": True,
            "checkpoint_weight_receipt_present": True,
        },
    }
    _write_json(args.output, result)
    return result


def shutdown_server(args: argparse.Namespace) -> dict[str, Any]:
    import msgpack
    import websockets.sync.client

    runtime = _load_object(args.runtime_receipt, "runtime receipt")
    if runtime.get("launch_nonce") != args.launch_nonce or runtime.get("uri") != args.uri:
        raise ContractError("shutdown inputs differ from runtime receipt")
    with websockets.sync.client.connect(
        args.uri,
        compression=None,
        max_size=None,
        open_timeout=args.timeout_seconds,
        close_timeout=args.timeout_seconds,
        ping_interval=20,
        ping_timeout=args.timeout_seconds,
    ) as connection:
        metadata_wire = connection.recv(timeout=args.timeout_seconds)
        if not isinstance(metadata_wire, bytes):
            raise ContractError("policy metadata must use the binary websocket frame")
        metadata = msgpack.unpackb(metadata_wire, raw=False)
        if metadata != runtime.get("policy_server"):
            raise ContractError("live policy metadata changed before shutdown")
        request = {"__cwam_control__": "shutdown", "launch_nonce": args.launch_nonce}
        connection.send(msgpack.packb(request, use_bin_type=True))
        ack_wire = connection.recv(timeout=args.timeout_seconds)
        if not isinstance(ack_wire, bytes):
            raise ContractError("shutdown ACK must use the binary websocket frame")
        ack = msgpack.unpackb(ack_wire, raw=False)
    if not isinstance(ack, dict):
        raise ContractError("shutdown ACK must be an object")
    expected = {"actions": [], "cwam_control": "shutdown_ack", "launch_nonce": args.launch_nonce}
    if {key: ack.get(key) for key in expected} != expected:
        raise ContractError(f"shutdown ACK mismatch: {ack!r}")
    timing = ack.get("server_timing")
    if not isinstance(timing, dict) or not math.isfinite(float(timing.get("infer_ms", math.nan))):
        raise ContractError("shutdown ACK has invalid server timing")
    result = {
        "schema_version": "metadata_no_hash_v1",
        "kind": "cwam_b4_controlled_shutdown_receipt",
        "status": "passed",
        "completed_at_utc": _utc_now(),
        "uri": args.uri,
        "server_label": runtime.get("server_label"),
        "launch_nonce": args.launch_nonce,
        "request": request,
        "ack": ack,
    }
    _write_json(args.output, result)
    return result


def _validate_episode(episode: Any, task: str, episode_index: int) -> dict[str, Any]:
    if not isinstance(episode, dict):
        raise ContractError(f"{task} episode {episode_index} must be an object")
    success = episode.get("success")
    place_anytime = episode.get("place_anytime")
    steps = episode.get("steps")
    instruction = episode.get("instruction")
    actions = episode.get("action_log")
    if type(success) is not bool or type(place_anytime) is not bool:
        raise ContractError(f"{task} episode {episode_index} has invalid outcomes")
    if type(steps) is not int or not 1 <= steps <= 400:
        raise ContractError(f"{task} episode {episode_index} has invalid steps")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ContractError(f"{task} episode {episode_index} has no instruction")
    if not isinstance(actions, list) or len(actions) != steps:
        raise ContractError(f"{task} episode {episode_index} action count differs from steps")
    for step_index, action in enumerate(actions):
        if not isinstance(action, (list, tuple)) or len(action) != 7:
            raise ContractError(f"{task} episode {episode_index} action {step_index} width is not 7")
        if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in action):
            raise ContractError(f"{task} episode {episode_index} action {step_index} is non-finite")
    return {
        "success": success,
        "place_anytime": place_anytime,
        "steps": steps,
        "instruction": instruction,
    }


def verify_gate(args: argparse.Namespace) -> dict[str, Any]:
    args_json = _load_object(args.results_dir / "args.json", "gate args")
    results = _load_object(args.results_dir / "results_shard0.json", "gate result")
    expected_args = {
        "task_shard": 0,
        "num_shards": 1,
        "episodes": 1,
        "episodes_start": 0,
        "max_steps": 400,
        "open_loop_horizon": 12,
        "seed": 42,
        "checkpoint": str(args.checkpoint.resolve()),
        "run_name": args.run_name,
        "port": args.port,
        "results_dir": str(args.results_dir.parent.resolve()),
        "headless": True,
        "save_video": False,
    }
    for key, value in expected_args.items():
        if args_json.get(key) != value:
            raise ContractError(f"gate args {key} mismatch")
    if set(results) != {"CloseBlenderLid"}:
        raise ContractError("gate must contain only CloseBlenderLid")
    task_result = results["CloseBlenderLid"]
    if not isinstance(task_result, dict) or task_result.get("complete") is not True:
        raise ContractError("gate task did not complete")
    episodes = task_result.get("episode_results")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise ContractError("gate must contain exactly one episode")
    _validate_episode(episodes[0], "CloseBlenderLid", 0)
    result = {
        "schema_version": "metadata_no_hash_v1",
        "kind": "cwam_b4_one_episode_gate",
        "status": "passed",
        "completed_at_utc": _utc_now(),
        "role": "preflight_only_not_part_of_340_primary_metric",
        "server_label": args.server_label,
        "policy_uri": f"ws://127.0.0.1:{args.port}",
        "task": "CloseBlenderLid",
        "episode_count": 1,
    }
    _write_json(args.output, result)
    return result


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    expected_labels = {"server-a", "server-b"}
    collections = {
        "runtime": [_load_object(path, f"runtime receipt {index}") for index, path in enumerate(args.runtime_receipt)],
        "shutdown": [_load_object(path, f"shutdown receipt {index}") for index, path in enumerate(args.shutdown_receipt)],
        "gate": [_load_object(path, f"gate receipt {index}") for index, path in enumerate(args.gate_receipt)],
    }
    for kind, receipts in collections.items():
        if len(receipts) != 2:
            raise ContractError(f"expected two {kind} receipts")
        if {receipt.get("server_label") for receipt in receipts} != expected_labels:
            raise ContractError(f"{kind} receipts do not cover both policy servers")
        if any(receipt.get("status") != "passed" for receipt in receipts):
            raise ContractError(f"all {kind} receipts must pass")
    runtime_by_label = {receipt["server_label"]: receipt for receipt in collections["runtime"]}
    shutdown_by_label = {receipt["server_label"]: receipt for receipt in collections["shutdown"]}
    gate_by_label = {receipt["server_label"]: receipt for receipt in collections["gate"]}
    expected_devices = {"server-a": "0,1,2,3", "server-b": "4,5,6,7"}
    expected_uris = {"server-a": "ws://127.0.0.1:16000", "server-b": "ws://127.0.0.1:16001"}
    for label in sorted(expected_labels):
        runtime = runtime_by_label[label]
        if runtime.get("visible_devices") != expected_devices[label] or runtime.get("uri") != expected_uris[label]:
            raise ContractError(f"{label} runtime topology mismatch")
        if shutdown_by_label[label].get("uri") != runtime.get("uri"):
            raise ContractError(f"{label} shutdown receipt URI mismatch")
        if gate_by_label[label].get("policy_uri") != runtime.get("uri"):
            raise ContractError(f"{label} gate receipt URI mismatch")
    if args.server_exit_code != [0, 0]:
        raise ContractError(f"torchrun exit codes are not both zero: {args.server_exit_code}")
    if len(args.server_log) != 2 or any(not path.is_file() or path.stat().st_size <= 0 for path in args.server_log):
        raise ContractError("both policy server logs must be present and non-empty")
    checkpoint = str(args.checkpoint.resolve())
    results_parent = str(args.results_dir.parent.resolve())
    episode_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    raw_artifacts: list[dict[str, Any]] = []
    for shard in range(8):
        args_path = args.results_dir / f"args_shard{shard}.json"
        result_path = args.results_dir / f"results_shard{shard}.json"
        shard_args = _load_object(args_path, f"shard {shard} args")
        results = _load_object(result_path, f"shard {shard} results")
        expected_args = {
            "task_shard": shard,
            "num_shards": 8,
            "episodes": 20,
            "episodes_start": 0,
            "max_steps": 400,
            "open_loop_horizon": 12,
            "seed": 42,
            "checkpoint": checkpoint,
            "run_name": args.run_name,
            "results_dir": results_parent,
            "eval_dataset": None,
            "headless": True,
            "save_video": False,
        }
        for key, value in expected_args.items():
            if shard_args.get(key) != value:
                raise ContractError(f"shard {shard} args {key} mismatch: {shard_args.get(key)!r}")
        for forbidden in ("dynamic_instructions_file", "gt_reasoning_file", "vlm_captions_file", "fg_captions_path"):
            if shard_args.get(forbidden) not in (None, ""):
                raise ContractError(f"shard {shard} enables forbidden input {forbidden}")
        expected_tasks = [task for index, task in enumerate(ATOMIC_TASKS) if index % 8 == shard]
        if set(results) != set(expected_tasks):
            raise ContractError(f"shard {shard} task coverage mismatch")
        raw_artifacts.append({
            "shard": shard,
            "args": _file_descriptor(args_path),
            "results": _file_descriptor(result_path),
        })
        for within_shard_index, task in enumerate(expected_tasks):
            value = results[task]
            if not isinstance(value, dict) or "error" in value or value.get("complete") is not True:
                raise ContractError(f"task {task} did not complete successfully")
            if value.get("split") != "atomic" or value.get("episodes") != 20:
                raise ContractError(f"task {task} protocol fields mismatch")
            episodes = value.get("episode_results")
            if not isinstance(episodes, list) or len(episodes) != 20:
                raise ContractError(f"task {task} does not contain exactly 20 episodes")
            validated = [_validate_episode(item, task, index) for index, item in enumerate(episodes)]
            successes = sum(int(item["success"]) for item in validated)
            place_count = sum(int(item["place_anytime"]) for item in validated)
            if value.get("successes") != successes or value.get("place_anytime_count") != place_count:
                raise ContractError(f"task {task} aggregate counts mismatch")
            if not math.isclose(float(value.get("success_rate")), successes / 20, abs_tol=1e-12):
                raise ContractError(f"task {task} success rate mismatch")
            if not math.isclose(float(value.get("place_anytime_rate")), place_count / 20, abs_tol=1e-12):
                raise ContractError(f"task {task} place-anytime rate mismatch")
            task_index = ATOMIC_TASKS.index(task)
            for episode_index, episode in enumerate(validated):
                episode_rows.append({
                    "experiment_id": args.experiment_id,
                    "logical_run_id": args.logical_run_id,
                    "task": task,
                    "task_index": task_index,
                    "shard": shard,
                    "episode_index": episode_index,
                    "seed": 42 + within_shard_index * 10_000 + episode_index,
                    "success": str(episode["success"]).lower(),
                    "place_anytime": str(episode["place_anytime"]).lower(),
                    "steps": episode["steps"],
                    "instruction": episode["instruction"],
                })
            task_rows.append({
                "task": task,
                "task_index": task_index,
                "shard": shard,
                "episodes": 20,
                "successes": successes,
                "success_rate": f"{successes / 20:.12f}",
                "place_anytime_count": place_count,
                "place_anytime_rate": f"{place_count / 20:.12f}",
                "execution_complete": "true",
            })
    episode_rows.sort(key=lambda row: (int(row["task_index"]), int(row["episode_index"])))
    task_rows.sort(key=lambda row: int(row["task_index"]))
    if len(episode_rows) != 340 or len(task_rows) != 17:
        raise ContractError("formal result must contain exactly 340 episodes and 17 tasks")
    for path, rows in ((args.summary_csv, episode_rows), (args.task_csv, task_rows)):
        if os.path.lexists(path):
            raise ContractError(f"refusing to overwrite CSV artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.chmod(path, 0o444)
    successes = sum(int(row["success"] == "true") for row in episode_rows)
    evidence = {
        "schema_version": "metadata_no_hash_v1",
        "kind": "cwam_b4_robocasa365_formal_eval_evidence",
        "status": "COMPLETE",
        "completed_at_utc": _utc_now(),
        "experiment_id": args.experiment_id,
        "logical_run_id": args.logical_run_id,
        "source_commit": args.source_commit,
        "demian_commit": args.demian_commit,
        "checkpoint_path": checkpoint,
        "protocol": {
            "split": "atomic",
            "task_count": 17,
            "episodes_per_task": 20,
            "episode_count": 340,
            "logical_shards": 8,
            "base_seed": 42,
            "max_steps": 400,
            "open_loop_horizon": 12,
            "policy_sampler": "unipc",
            "policy_steps": 30,
            "policy_guidance": 1.0,
        },
        "result": {
            "successes": successes,
            "episodes": 340,
            "mean_success_rate": successes / 340,
            "successful_tasks": sum(int(row["successes"] > 0) for row in task_rows),
            "per_task": task_rows,
        },
        "execution": {
            "platform": "Alibaba Cloud PAI DLC cn-beijing",
            "physical_servers": 1,
            "physical_gpus": 8,
            "policy_server_count": 2,
            "distributed_world_size_per_server": 4,
            "aggregate_policy_ranks": 8,
            "evaluator_schedule": "two_parallel_groups_with_serial_shards_a_0_2_4_6_b_1_3_5_7",
            "torchrun_exit_codes": args.server_exit_code,
        },
        "artifacts": {
            "runtime_receipts": [_file_descriptor(path) for path in args.runtime_receipt],
            "gate_receipts": [_file_descriptor(path) for path in args.gate_receipt],
            "shutdown_receipts": [_file_descriptor(path) for path in args.shutdown_receipt],
            "server_logs": [_file_descriptor(path) for path in args.server_log],
            "summary_csv": _file_descriptor(args.summary_csv),
            "task_success_rates_csv": _file_descriptor(args.task_csv),
            "raw_shards": raw_artifacts,
        },
        "validation": {
            "all_eight_shards_present": True,
            "exactly_17_tasks_and_340_episodes": True,
            "all_action_logs_complete_width7_and_finite": True,
            "one_episode_graphics_gate_passed": True,
            "both_policy_servers_passed_runtime_and_graphics_gates": True,
            "both_controlled_server_shutdowns_acknowledged": True,
            "both_torchrun_processes_exited_zero": True,
        },
    }
    _write_json(args.evidence_json, evidence)
    _write_json(args.status_json, {
        "experiment_id": args.experiment_id,
        "logical_run_id": args.logical_run_id,
        "status": "COMPLETE",
        "completed_at_utc": evidence["completed_at_utc"],
        "successes": successes,
        "episodes": 340,
        "mean_success_rate": successes / 340,
        "evidence_json": str(args.evidence_json.resolve()),
    })
    return evidence


def write_status(args: argparse.Namespace) -> dict[str, Any]:
    value = {
        "experiment_id": args.experiment_id,
        "logical_run_id": args.logical_run_id,
        "status": args.status,
        "updated_at_utc": _utc_now(),
    }
    if args.detail:
        value["detail"] = args.detail
    _write_json(args.output, value, exclusive=False)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-checkpoint")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    runtime = commands.add_parser("capture-runtime")
    runtime.add_argument("--uri", default="ws://127.0.0.1:16000")
    runtime.add_argument("--output", type=Path, required=True)
    runtime.add_argument("--timeout-seconds", type=float, default=1800)
    runtime.add_argument("--launcher-pid", type=int, required=True)
    runtime.add_argument("--server-label", choices=("server-a", "server-b"), required=True)
    runtime.add_argument("--visible-devices", required=True)
    runtime.add_argument("--launch-nonce", required=True)
    runtime.add_argument("--checkpoint", type=Path, required=True)
    runtime.add_argument("--config", type=Path, required=True)
    runtime.add_argument("--action-stats", type=Path, required=True)
    runtime.add_argument("--vae", type=Path, required=True)
    runtime.add_argument("--tokenizer", type=Path, required=True)
    runtime.add_argument("--source-commit", required=True)
    runtime.add_argument("--demian-commit", required=True)
    shutdown = commands.add_parser("shutdown-server")
    shutdown.add_argument("--uri", default="ws://127.0.0.1:16000")
    shutdown.add_argument("--output", type=Path, required=True)
    shutdown.add_argument("--timeout-seconds", type=float, default=60)
    shutdown.add_argument("--launch-nonce", required=True)
    shutdown.add_argument("--runtime-receipt", type=Path, required=True)
    gate = commands.add_parser("verify-gate")
    gate.add_argument("--results-dir", type=Path, required=True)
    gate.add_argument("--checkpoint", type=Path, required=True)
    gate.add_argument("--run-name", required=True)
    gate.add_argument("--server-label", choices=("server-a", "server-b"), required=True)
    gate.add_argument("--port", type=int, required=True)
    gate.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--experiment-id", required=True)
    summary.add_argument("--logical-run-id", required=True)
    summary.add_argument("--source-commit", required=True)
    summary.add_argument("--demian-commit", required=True)
    summary.add_argument("--run-name", required=True)
    summary.add_argument("--results-dir", type=Path, required=True)
    summary.add_argument("--checkpoint", type=Path, required=True)
    summary.add_argument("--runtime-receipt", type=Path, action="append", required=True)
    summary.add_argument("--gate-receipt", type=Path, action="append", required=True)
    summary.add_argument("--shutdown-receipt", type=Path, action="append", required=True)
    summary.add_argument("--server-log", type=Path, action="append", required=True)
    summary.add_argument("--server-exit-code", type=int, action="append", required=True)
    summary.add_argument("--summary-csv", type=Path, required=True)
    summary.add_argument("--task-csv", type=Path, required=True)
    summary.add_argument("--evidence-json", type=Path, required=True)
    summary.add_argument("--status-json", type=Path, required=True)
    status = commands.add_parser("write-status")
    status.add_argument("--experiment-id", required=True)
    status.add_argument("--logical-run-id", required=True)
    status.add_argument("--status", required=True)
    status.add_argument("--detail", default="")
    status.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "verify-checkpoint":
            result = verify_checkpoint(args.root, args.output)
        elif args.command == "capture-runtime":
            result = capture_runtime(args)
        elif args.command == "shutdown-server":
            result = shutdown_server(args)
        elif args.command == "verify-gate":
            result = verify_gate(args)
        elif args.command == "summarize":
            result = summarize(args)
        else:
            result = write_status(args)
    except (ContractError, OSError, subprocess.SubprocessError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
