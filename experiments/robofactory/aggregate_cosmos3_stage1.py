#!/usr/bin/env python3
"""Aggregate the fixed Cosmos3 RoboFactory Stage-1 causal panel."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ARM_PROTOCOL = {
    "a0-eh1-rolling": (12, 1, "rolling-query", "none"),
    "a1-eh5-rolling": (12, 5, "rolling-query", "none"),
    "a2-eh1-fixed": (12, 1, "fixed-episode", "none"),
    "a3-eh5-fixed": (12, 5, "fixed-episode", "none"),
    "o1-r0-pose": (8, 1, "rolling-query", "robot0-pose"),
    "o2-r0-gripper": (8, 1, "rolling-query", "robot0-gripper"),
    "o3-r1-action": (8, 1, "rolling-query", "robot1-action"),
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _episodes(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _arm_record(
    name: str, records: list[dict[str, Any]], *, source: str
) -> dict[str, Any]:
    expected, exec_horizon, noise_mode, oracle = ARM_PROTOCOL[name]
    completed = [record for record in records if record.get("status") == "completed"]
    complete = len(records) == expected and len(completed) == expected
    successes = sum(bool(record.get("success")) for record in completed)
    grasps = sum(bool(record.get("grasped")) for record in completed)
    lifts = [
        float(record["max_object_lift"])
        for record in completed
        if record.get("max_object_lift") is not None
    ]
    violations_by_agent_dimension: dict[str, dict[str, int]] = {}
    for record in completed:
        for agent, dimensions in record.get(
            "action_bound_violations_by_agent_dimension", {}
        ).items():
            destination = violations_by_agent_dimension.setdefault(agent, {})
            for dimension, count in dimensions.items():
                destination[dimension] = destination.get(dimension, 0) + int(count)
    return {
        "arm": name,
        "source": source,
        "complete": complete,
        "episodes_expected": expected,
        "episodes_completed": len(completed),
        "exec_horizon": exec_horizon,
        "noise_mode": noise_mode,
        "oracle_intervention": oracle,
        "successes": successes,
        "strict_success_rate": successes / expected if complete else None,
        "grasped_episodes": grasps,
        "grasp_rate": grasps / expected if complete else None,
        "maximum_object_lift": max(lifts, default=None),
        "mean_maximum_object_lift": sum(lifts) / len(lifts) if lifts else None,
        "action_bound_violations": sum(
            int(record.get("action_bound_violations", 0)) for record in completed
        ),
        "action_bound_violations_by_agent_dimension": violations_by_agent_dimension,
        "mean_policy_query_seconds": (
            sum(float(record.get("policy_query_seconds_total", 0.0)) for record in completed)
            / sum(int(record.get("policy_queries", 0)) for record in completed)
            if sum(int(record.get("policy_queries", 0)) for record in completed)
            else None
        ),
        "episodes": [
            {
                "task_index": record.get("task_index"),
                "success": record.get("success"),
                "grasped": record.get("grasped"),
                "max_object_lift": record.get("max_object_lift"),
                "minimum_ee_object_distances": record.get(
                    "minimum_ee_object_distances"
                ),
                "first_approach_steps": record.get("first_approach_steps"),
                "first_close_steps": record.get("first_close_steps"),
                "close_before_approach": record.get("close_before_approach"),
                "gripper_flip_counts": record.get("gripper_flip_counts"),
                "maximum_grasp_hold_steps": record.get(
                    "maximum_grasp_hold_steps"
                ),
                "grasp_release_counts": record.get("grasp_release_counts"),
            }
            for record in completed
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--baseline-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = []
    arms.append(
        _arm_record(
            "a0-eh1-rolling",
            _episodes(args.baseline_episodes),
            source=str(args.baseline_episodes),
        )
    )
    for name in tuple(ARM_PROTOCOL)[1:]:
        path = args.results_root / name / "episodes.jsonl"
        arms.append(_arm_record(name, _episodes(path), source=str(path)))
    complete = all(arm["complete"] for arm in arms)
    payload = {
        "schema_version": "cosmos3-robofactory-stage1-causal-panel-v1",
        "status": "PASS" if complete else "INCOMPLETE",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "iter_000060000",
        "task": "PlaceFood-rf",
        "same12": ["a0-eh1-rolling", "a1-eh5-rolling", "a2-eh1-fixed", "a3-eh5-fixed"],
        "same8": ["o1-r0-pose", "o2-r0-gripper", "o3-r1-action"],
        "arms": arms,
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
