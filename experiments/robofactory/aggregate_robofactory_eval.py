#!/usr/bin/env python3
"""Aggregate the Table11 ImageWAM RoboFactory closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


TASKS = (
    "LongPipelineDelivery-rf",
    "StackCube-rf",
    "StrikeCube-rf",
    "TakePhoto-rf",
    "PlaceFood-rf",
    "PickMeat-rf",
    "PassShoe-rf",
    "CameraAlignment-rf",
    "LiftBarrier-rf",
    "ThreeRobotsStackCube-rf",
    "TwoRobotsStackCube-rf",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=8)
    args = parser.parse_args()
    if args.episodes_per_task <= 0:
        parser.error("--episodes-per-task must be positive")
    root = args.results_root.expanduser().resolve(strict=True)

    task_records: list[dict[str, Any]] = []
    infrastructure_ok = True
    for task in TASKS:
        expert_path = root / "expert" / task / "summary.json"
        imagewam_path = root / "imagewam" / task / "summary.json"
        expert = _load(expert_path)
        imagewam = _load(imagewam_path)
        expert_compatible = (
            expert.get("status") == "PASS"
            and int(expert.get("episodes_completed", 0)) == 1
            and int(expert.get("successes", 0)) == 1
        )
        policy_complete = (
            imagewam.get("status") == "PASS"
            and int(imagewam.get("episodes_requested", 0))
            == args.episodes_per_task
            and int(imagewam.get("episodes_completed", 0))
            == args.episodes_per_task
            and int(imagewam.get("infrastructure_errors", 0)) == 0
        )
        infrastructure_ok = infrastructure_ok and policy_complete
        task_records.append(
            {
                "task_name": task,
                "expert_compatibility_gate": expert_compatible,
                "expert_gate_status": expert.get("status"),
                "policy_complete": policy_complete,
                "episodes_completed": int(imagewam.get("episodes_completed", 0)),
                "successes": int(imagewam.get("successes", 0)),
                "strict_success_rate": imagewam.get("strict_success_rate"),
                "action_bound_violations": int(
                    imagewam.get("total_action_bound_violations", 0)
                ),
                "expert_summary": str(expert_path),
                "policy_summary": str(imagewam_path),
            }
        )

    primary = [record for record in task_records if record["expert_compatibility_gate"]]
    all_successes = sum(record["successes"] for record in task_records)
    all_completed = sum(record["episodes_completed"] for record in task_records)
    primary_successes = sum(record["successes"] for record in primary)
    primary_completed = sum(record["episodes_completed"] for record in primary)
    expert_gate_failures = [
        record["task_name"]
        for record in task_records
        if not record["expert_compatibility_gate"]
    ]
    payload = {
        "schema_version": "imagewam-robofactory-table11-aggregate-v1",
        "status": (
            "PASS"
            if infrastructure_ok and not expert_gate_failures
            else "INFRASTRUCTURE_OR_COMPATIBILITY_ERROR"
        ),
        "finished_at": _utc_now(),
        "protocol": {
            "episodes_per_task": args.episodes_per_task,
            "action_horizon_per_agent": 32,
            "exec_horizon": 5,
            "num_inference_steps": 20,
            "action_clipping": False,
            "success_source": "RoboFactory info.success",
        },
        "all_tasks": {
            "task_count": len(task_records),
            "episodes_completed": all_completed,
            "successes": all_successes,
            "success_rate": all_successes / all_completed if all_completed else None,
        },
        "expert_compatible_primary": {
            "task_count": len(primary),
            "tasks": [record["task_name"] for record in primary],
            "episodes_completed": primary_completed,
            "successes": primary_successes,
            "success_rate": (
                primary_successes / primary_completed if primary_completed else None
            ),
        },
        "expert_gate_failures": expert_gate_failures,
        "tasks": task_records,
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
