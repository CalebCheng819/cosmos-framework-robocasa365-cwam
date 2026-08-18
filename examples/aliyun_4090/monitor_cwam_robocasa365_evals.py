#!/usr/bin/env python3
"""Monitor transfer-gated DLC evaluations through formal OSS publication."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import runpy
import tempfile
import time
from typing import Any, Callable


DEFAULT_PERSISTENT_RUN_PARENT = Path(
    "/oss-chengjuntao/cpfs-user-chengjuntao/"
    "cwam_cosmos3_robocasa365_eval/runs"
)
SUCCESS_STATUS = "Succeeded"
FAILURE_STATUSES = {"Failed", "Stopped"}
GET_JOB_CONNECT_TIMEOUT_MS = 5_000
GET_JOB_READ_TIMEOUT_MS = 15_000
GET_JOB_MAX_ATTEMPTS = 2


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_object(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must contain a JSON object: {path}")
    return value


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_configs(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        config = load_object(path, "submission config")
        require(config.get("kind") == "cwam_robocasa365_dlc_eval_submission", "invalid config kind")
        experiment_id = str(config.get("experiment_id") or "")
        logical_run_id = str(config.get("logical_run_id") or "")
        display_name = str(config.get("display_name") or "")
        require(experiment_id and logical_run_id and display_name, "config omitted identity")
        require(experiment_id not in result, f"duplicate experiment_id: {experiment_id}")
        result[experiment_id] = config
    return result


def submitted_jobs(
    controller_status: dict[str, Any], configs: dict[str, dict[str, Any]]
) -> dict[str, str] | None:
    if controller_status.get("status") != "submitted":
        return None
    evaluations = controller_status.get("evaluations")
    require(isinstance(evaluations, dict), "submitted controller omitted evaluations")
    require(set(evaluations) == set(configs), "controller/config experiment sets differ")
    result: dict[str, str] = {}
    for experiment_id, config in configs.items():
        value = evaluations[experiment_id]
        require(isinstance(value, dict), f"invalid controller evaluation: {experiment_id}")
        require(value.get("logical_run_id") == config["logical_run_id"], "logical run mismatch")
        job_id = str(value.get("job_id") or "")
        require(job_id.startswith("dlc"), f"invalid DLC job ID for {experiment_id}: {job_id}")
        result[experiment_id] = job_id
    return result


def make_get_job(
    client: Any,
    request_type: Callable[..., Any],
    runtime_options_type: Callable[..., Any],
) -> Callable[[str], dict[str, Any]]:
    """Build a bounded DLC readback call so a transient API stall cannot stop monitoring."""
    runtime = runtime_options_type(
        autoretry=True,
        max_attempts=GET_JOB_MAX_ATTEMPTS,
        connect_timeout=GET_JOB_CONNECT_TIMEOUT_MS,
        read_timeout=GET_JOB_READ_TIMEOUT_MS,
    )

    def get_job(job_id: str) -> dict[str, Any]:
        response = client.get_job_with_options(
            job_id,
            request_type(need_detail=True),
            {},
            runtime,
        )
        return response.body.to_map()

    return get_job


def validate_job_readback(
    readback: dict[str, Any],
    config: dict[str, Any],
    job_id: str,
    control_commit: str,
) -> dict[str, Any]:
    require(readback.get("JobId") == job_id, "DLC readback JobId mismatch")
    require(readback.get("DisplayName") == config["display_name"], "DLC display name mismatch")
    require(str(readback.get("WorkspaceId")) == "270969", "DLC workspace mismatch")
    require(readback.get("ResourceId") == "quotaksvqq2oh2pg", "DLC resource mismatch")
    require(readback.get("Priority") == 7, "DLC priority mismatch")
    specs = readback.get("JobSpecs")
    require(isinstance(specs, list) and len(specs) == 1, "DLC must have exactly one JobSpec")
    spec = specs[0]
    require(spec.get("Type") == "Worker" and spec.get("PodCount") == 1, "DLC topology mismatch")
    require(spec.get("RestartPolicy") == "Never", "DLC restart policy mismatch")
    require(
        spec.get("ResourceConfig")
        == {"CPU": "126", "GPU": "8", "GPUType": "", "Memory": "960Gi", "SharedMemory": "960Gi"},
        "DLC resource contract mismatch",
    )
    envs = readback.get("Envs") or {}
    require(envs.get("EVAL_CONTROL_COMMIT") == control_commit, "DLC control commit mismatch")
    require(envs.get("EVAL_EXPERIMENT_ID") == config["experiment_id"], "DLC experiment mismatch")
    require(envs.get("EVAL_LOGICAL_RUN_ID") == config["logical_run_id"], "DLC run mismatch")
    status = str(readback.get("Status") or "")
    require(status, "DLC readback omitted status")
    pods = readback.get("Pods") or []
    return {
        "captured_at_utc": utc_now(),
        "job_id": job_id,
        "experiment_id": config["experiment_id"],
        "logical_run_id": config["logical_run_id"],
        "display_name": config["display_name"],
        "status": status,
        "reason_code": readback.get("ReasonCode"),
        "reason_message": readback.get("ReasonMessage"),
        "duration_seconds": readback.get("Duration"),
        "created_at_utc": readback.get("GmtCreateTime"),
        "submitted_at_utc": readback.get("GmtSubmittedTime"),
        "running_at_utc": readback.get("GmtRunningTime"),
        "stopped_at_utc": readback.get("GmtStoppedTime"),
        "priority": 7,
        "resource_contract": {
            "worker_pods": 1,
            "gpus_per_worker": 8,
            "cpus_per_worker": 126,
            "memory_per_worker": "960Gi",
            "shared_memory_per_worker": "960Gi",
        },
        "pods": [
            {
                "pod_id": pod.get("PodId"),
                "status": pod.get("Status"),
                "sub_status": pod.get("SubStatus"),
                "node_name": pod.get("NodeName"),
                "started_at_utc": pod.get("GmtStartTime"),
            }
            for pod in pods
        ],
    }


def _csv_rows(path: Path, expected: int, label: str) -> list[dict[str, str]]:
    require(path.is_file() and not path.is_symlink(), f"{label} missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == expected, f"{label} expected {expected} rows, found {len(rows)}")
    return rows


def validate_terminal_output(
    persistent_run_parent: Path,
    config: dict[str, Any],
    job_id: str,
    control_commit: str,
) -> dict[str, Any]:
    experiment_id = config["experiment_id"]
    logical_run_id = config["logical_run_id"]
    root = persistent_run_parent / experiment_id / logical_run_id
    persistence = load_object(root / "persistence-status.json", "persistence status")
    require(persistence.get("experiment_id") == experiment_id, "persistence experiment mismatch")
    require(persistence.get("run_id") == logical_run_id, "persistence run mismatch")
    require(persistence.get("control_commit") == control_commit, "persistence control commit mismatch")
    require(persistence.get("job_id") in {job_id, "unknown-at-runtime"}, "persistence job mismatch")
    require(persistence.get("launcher_exit_code") == 0, "launcher exit code was nonzero")
    require(persistence.get("terminal_status") == "complete", "persistence is not complete")
    require(persistence.get("publication_complete") is True, "OSS publication is incomplete")

    launcher = load_object(root / "launcher-terminal.json", "launcher terminal")
    require(launcher.get("status") == "COMPLETE" and launcher.get("exit_code") == 0, "launcher failed")
    require(launcher.get("experiment_id") == experiment_id, "launcher experiment mismatch")
    require(launcher.get("logical_run_id") == logical_run_id, "launcher run mismatch")
    require(launcher.get("control_commit") == control_commit, "launcher control commit mismatch")
    require(launcher.get("dlc_job_id") in {job_id, None, ""}, "launcher DLC job mismatch")

    evidence = load_object(root / "evaluation_evidence.json", "evaluation evidence")
    status = load_object(root / "status.json", "evaluation status")
    require(evidence.get("status") == "COMPLETE", "formal evidence is not complete")
    require(status.get("status") == "COMPLETE", "formal status is not complete")
    for value, label in ((evidence, "evidence"), (status, "status")):
        require(value.get("experiment_id") == experiment_id, f"{label} experiment mismatch")
        require(value.get("logical_run_id") == logical_run_id, f"{label} run mismatch")
    protocol = evidence.get("protocol") or {}
    result = evidence.get("result") or {}
    validation = evidence.get("validation") or {}
    require(protocol.get("task_count") == 17, "formal task count mismatch")
    require(protocol.get("episodes_per_task") == 20, "episodes-per-task mismatch")
    require(protocol.get("episode_count") == 340, "formal episode count mismatch")
    require(result.get("episodes") == 340 and status.get("episodes") == 340, "result episode count mismatch")
    require(len(result.get("per_task") or []) == 17, "per-task result count mismatch")
    require(validation.get("exactly_17_tasks_and_340_episodes") is True, "formal count validation failed")
    require((evidence.get("execution") or {}).get("physical_gpus") == 8, "formal GPU count mismatch")
    _csv_rows(root / "summary.csv", 340, "summary CSV")
    _csv_rows(root / "task_success_rates.csv", 17, "task success CSV")
    successes = result.get("successes")
    require(type(successes) is int and 0 <= successes <= 340, "invalid success count")
    require(status.get("successes") == successes, "status/evidence success mismatch")
    return {
        "validated_at_utc": utc_now(),
        "status": "complete",
        "job_id": job_id,
        "experiment_id": experiment_id,
        "logical_run_id": logical_run_id,
        "persistent_root": str(root),
        "tasks": 17,
        "episodes": 340,
        "successes": successes,
        "mean_success_rate": result.get("mean_success_rate"),
        "local_regular_file_count": persistence.get("local_regular_file_count"),
        "local_regular_file_bytes": persistence.get("local_regular_file_bytes"),
    }


def monitor_once(
    configs: dict[str, dict[str, Any]],
    jobs: dict[str, str],
    get_job: Callable[[str], dict[str, Any]],
    persistent_run_parent: Path,
    control_commit: str,
    record_dir: Path,
) -> tuple[dict[str, Any], bool, bool]:
    snapshots: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    failed = False
    all_succeeded = True
    for experiment_id, config in configs.items():
        snapshot = validate_job_readback(get_job(jobs[experiment_id]), config, jobs[experiment_id], control_commit)
        snapshots[experiment_id] = snapshot
        atomic_write(record_dir / f"{experiment_id}-job-current.json", snapshot)
        if snapshot["status"] in FAILURE_STATUSES:
            failed = True
        if snapshot["status"] != SUCCESS_STATUS:
            all_succeeded = False
    for experiment_id, config in configs.items():
        if snapshots[experiment_id]["status"] == SUCCESS_STATUS:
            try:
                results[experiment_id] = validate_terminal_output(
                    persistent_run_parent, config, jobs[experiment_id], control_commit
                )
            except RuntimeError as error:
                results[experiment_id] = {"status": "pending_or_invalid", "error": str(error)}
    all_evidence_complete = len(results) == len(configs) and all(
        value.get("status") == "complete" for value in results.values()
    )
    state = "failed" if failed else "complete" if all_succeeded and all_evidence_complete else (
        "succeeded_pending_evidence" if all_succeeded else
        "running" if any(item["status"] == "Running" for item in snapshots.values()) else
        "queued"
    )
    aggregate = {
        "captured_at_utc": utc_now(),
        "status": state,
        "jobs": snapshots,
        "formal_results": results,
    }
    return aggregate, state == "complete", failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-status", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--submitter", type=Path, required=True)
    parser.add_argument("--control-commit", required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--persistent-run-parent", type=Path, default=DEFAULT_PERSISTENT_RUN_PARENT)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    args = parser.parse_args()
    require(args.poll_seconds > 0, "poll-seconds must be positive")
    configs = load_configs(args.config)
    args.record_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = (args.record_dir / "monitor.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another DLC evaluation monitor holds the lock") from error

    client = None
    get_job = None
    while True:
        if not args.controller_status.is_file():
            aggregate = {"captured_at_utc": utc_now(), "status": "waiting_for_submission"}
            atomic_write(args.record_dir / "monitor-status.json", aggregate)
            append_event(args.record_dir / "monitor-history.jsonl", aggregate)
            time.sleep(args.poll_seconds)
            continue
        controller = load_object(args.controller_status, "controller status")
        jobs = submitted_jobs(controller, configs)
        if jobs is None:
            aggregate = {
                "captured_at_utc": utc_now(),
                "status": "waiting_for_submission",
                "controller_status": controller.get("status"),
            }
            atomic_write(args.record_dir / "monitor-status.json", aggregate)
            append_event(args.record_dir / "monitor-history.jsonl", aggregate)
            time.sleep(args.poll_seconds)
            continue
        if client is None:
            namespace = runpy.run_path(str(args.submitter))
            client = namespace["make_client"]()
            from alibabacloud_tea_util.models import RuntimeOptions

            get_job = make_get_job(client, namespace["models"].GetJobRequest, RuntimeOptions)
        try:
            aggregate, complete, failed = monitor_once(
                configs,
                jobs,
                get_job,
                args.persistent_run_parent,
                args.control_commit,
                args.record_dir,
            )
        except Exception as error:
            aggregate = {
                "captured_at_utc": utc_now(),
                "status": "monitor_retrying",
                "error": repr(error),
            }
            complete = failed = False
        atomic_write(args.record_dir / "monitor-status.json", aggregate)
        append_event(args.record_dir / "monitor-history.jsonl", aggregate)
        print(json.dumps(aggregate, sort_keys=True), flush=True)
        if complete:
            return 0
        if failed:
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
