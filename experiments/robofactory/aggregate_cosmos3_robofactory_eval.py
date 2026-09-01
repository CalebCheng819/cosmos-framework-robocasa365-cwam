#!/usr/bin/env python3
"""Aggregate a bounded subset of Cosmos3 RoboFactory closed-loop tasks."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--exec-horizon", type=int, required=True)
    parser.add_argument("--task", action="append", dest="tasks", required=True)
    args = parser.parse_args()
    if args.episodes_per_task < 1 or args.exec_horizon < 1:
        parser.error("episode and execution horizons must be positive")

    root = args.results_root.expanduser().resolve(strict=True)
    records: list[dict[str, Any]] = []
    complete = True
    for task in args.tasks:
        path = root / "cosmos3" / task / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        task_complete = (
            summary.get("status") == "PASS"
            and int(summary.get("episodes_requested", 0)) == args.episodes_per_task
            and int(summary.get("episodes_completed", 0)) == args.episodes_per_task
            and int(summary.get("infrastructure_errors", 0)) == 0
        )
        complete = complete and task_complete
        records.append(
            {
                "task_name": task,
                "complete": task_complete,
                "episodes_completed": int(summary.get("episodes_completed", 0)),
                "successes": int(summary.get("successes", 0)),
                "success_rate": summary.get("strict_success_rate"),
                "grasped_episodes": int(summary.get("grasped_episodes", 0)),
                "max_object_lift": summary.get("max_object_lift"),
                "action_bound_violations": int(
                    summary.get("total_action_bound_violations", 0)
                ),
                "summary": str(path),
            }
        )

    episodes = sum(record["episodes_completed"] for record in records)
    successes = sum(record["successes"] for record in records)
    payload = {
        "schema_version": "cosmos3-robofactory-subset-aggregate-v1",
        "status": "PASS" if complete else "INCOMPLETE_OR_INFRASTRUCTURE_ERROR",
        "finished_at": _utc_now(),
        "protocol": {
            "tasks": args.tasks,
            "episodes_per_task": args.episodes_per_task,
            "exec_horizon": args.exec_horizon,
            "action_horizon_per_agent": 32,
            "online_gaussian": True,
            "action_clipping": False,
        },
        "episodes_completed": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else None,
        "tasks": records,
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
