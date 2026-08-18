#!/usr/bin/env python3
"""Wait for checkpoint copies, then audit and submit DLC evaluations once."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


DEFAULT_RECEIPT_ROOT = Path(
    "/oss-chengjuntao/cpfs-user-chengjuntao/"
    "cwam_cosmos3_robocasa365_eval/control/receipts"
)
RECONCILABLE_RECEIPT_STATUSES = {
    "CREATE_REQUEST_FAILED",
    "CREATE_ACCEPTED_READBACK_PENDING",
    "CREATE_ACCEPTED_READBACK_FAILED",
}


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


def transfer_snapshot(state_dir: Path, expected_count: int) -> dict[str, Any]:
    files = sorted(state_dir.glob("*.json"))
    require(len(files) == expected_count, f"expected {expected_count} transfer states, found {len(files)}")
    states = [load_object(path, "transfer state") for path in files]
    destinations = [str(state.get("destination") or "") for state in states]
    require(all(destinations), "transfer state omitted destination")
    require(len(set(destinations)) == len(destinations), "duplicate transfer destination")
    expected = [state.get("expected_bytes") for state in states]
    completed = [state.get("completed_bytes") for state in states]
    require(all(type(value) is int and value > 0 for value in expected), "invalid expected_bytes")
    require(all(type(value) is int and value >= 0 for value in completed), "invalid completed_bytes")
    require(all(done <= total for done, total in zip(completed, expected)), "completed_bytes exceeds expected_bytes")
    failed = [state for state in states if state.get("status") == "failed"]
    require(not failed, f"transfer entered failed state: {failed}")
    complete = all(
        state.get("status") == "complete" and done == total
        for state, done, total in zip(states, completed, expected)
    )
    return {
        "captured_at_utc": utc_now(),
        "state_count": len(states),
        "completed_bytes": sum(completed),
        "expected_bytes": sum(expected),
        "complete": complete,
        "statuses": {
            status: sum(state.get("status") == status for state in states)
            for status in sorted({str(state.get("status")) for state in states})
        },
        "last_errors": [state["last_error"] for state in states if state.get("last_error")],
    }


def run_submitter(
    sdk_python: Path,
    submitter: Path,
    config: Path,
    mode: str,
    control_commit: str,
    record_path: Path,
) -> dict[str, Any]:
    environment = {**os.environ, "EVAL_CONTROL_COMMIT": control_commit}
    command = [str(sdk_python), str(submitter), "--config", str(config), mode]
    result = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
    record = {
        "captured_at_utc": utc_now(),
        "command": command,
        "config": str(config),
        "mode": mode,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    atomic_write(record_path, record)
    require(result.returncode == 0, f"submitter {mode} failed for {config}; see {record_path}")
    payload = json.loads(result.stdout)
    require(isinstance(payload, dict), f"submitter returned non-object JSON for {config}")
    return payload


def receipt_mode(receipt: Path, experiment_id: str, logical_run_id: str) -> str:
    if not receipt.exists():
        return "--submit"
    document = load_object(receipt, "submission receipt")
    require(document.get("experiment_id") == experiment_id, "receipt experiment identity mismatch")
    require(document.get("logical_run_id") == logical_run_id, "receipt run identity mismatch")
    status = document.get("status")
    if status == "SUBMITTED":
        require(bool(document.get("job_id")), "submitted receipt omitted job_id")
        return "already-submitted"
    require(status in RECONCILABLE_RECEIPT_STATUSES, f"receipt is not reconcilable: {status}")
    return "--reconcile"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--expected-state-count", type=int, required=True)
    parser.add_argument("--sdk-python", type=Path, required=True)
    parser.add_argument("--submitter", type=Path, required=True)
    parser.add_argument("--control-commit", required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPT_ROOT)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    args = parser.parse_args()
    require(args.expected_state_count > 0, "expected-state-count must be positive")
    require(args.poll_seconds > 0, "poll-seconds must be positive")
    args.record_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = (args.record_dir / "controller.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another wait-and-submit controller holds the lock") from error

    while True:
        snapshot = transfer_snapshot(args.state_dir, args.expected_state_count)
        atomic_write(args.record_dir / "transfer-snapshot.json", snapshot)
        print(json.dumps(snapshot, sort_keys=True), flush=True)
        if snapshot["complete"]:
            break
        time.sleep(args.poll_seconds)

    summary: dict[str, Any] = {
        "status": "submitting",
        "started_at_utc": utc_now(),
        "control_commit": args.control_commit,
        "evaluations": {},
    }
    atomic_write(args.record_dir / "controller-status.json", summary)
    for config_path in args.config:
        config = load_object(config_path, "submission config")
        experiment_id = str(config.get("experiment_id") or "")
        logical_run_id = str(config.get("logical_run_id") or "")
        require(experiment_id and logical_run_id, "submission config omitted identities")
        receipt = args.receipt_root / f"{experiment_id}-dlc-submit.json"
        mode = receipt_mode(receipt, experiment_id, logical_run_id)
        if mode == "--submit":
            audit = run_submitter(
                args.sdk_python,
                args.submitter,
                config_path,
                "--audit-only",
                args.control_commit,
                args.record_dir / f"{experiment_id}-audit.json",
            )
            require(audit.get("status") == "AUDIT_PASSED", "audit did not report AUDIT_PASSED")
        if mode == "already-submitted":
            payload = load_object(receipt, "submission receipt")
        else:
            payload = run_submitter(
                args.sdk_python,
                args.submitter,
                config_path,
                mode,
                args.control_commit,
                args.record_dir / f"{experiment_id}-submit.json",
            )
        require(payload.get("status") == "SUBMITTED", "submission did not report SUBMITTED")
        summary["evaluations"][experiment_id] = {
            "job_id": payload.get("job_id"),
            "logical_run_id": logical_run_id,
            "submission_mode": mode,
        }
        atomic_write(args.record_dir / "controller-status.json", summary)

    summary["status"] = "submitted"
    summary["completed_at_utc"] = utc_now()
    atomic_write(args.record_dir / "controller-status.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
