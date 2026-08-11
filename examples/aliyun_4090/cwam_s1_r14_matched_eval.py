#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Validate and compare the fixed S1 versus R14 two-task evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASKS = ("CloseBlenderLid", "PickPlaceCounterToStove")
EXPECTED_PROTOCOL = {
    "split": "atomic",
    "episodes": 20,
    "episodes_start": 0,
    "max_steps": 400,
    "open_loop_horizon": 12,
    "camera_height": 256,
    "camera_width": 256,
    "seed": 42,
    "save_video": False,
    "headless": True,
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r} actual={actual!r}")


def _validate_args(value: dict[str, Any], *, candidate: bool) -> None:
    role = "candidate" if candidate else "baseline"
    for key, expected in EXPECTED_PROTOCOL.items():
        _require_equal(f"{role} args.{key}", value.get(key), expected)
    _require_equal(f"{role} args.task_shard", value.get("task_shard"), 0)
    if candidate:
        _require_equal(f"{role} args.num_shards", value.get("num_shards"), 1)
        _require_equal(f"{role} args.tasks", value.get("tasks"), list(TASKS))
    else:
        _require_equal(f"{role} args.num_shards", value.get("num_shards"), 8)
        _require_equal(f"{role} args.tasks", value.get("tasks"), [])


def _validate_task_result(
    task: str,
    value: Any,
    *,
    candidate: bool,
) -> list[dict[str, Any]]:
    role = "candidate" if candidate else "baseline"
    if not isinstance(value, dict):
        raise ValueError(f"{role} result for {task} is not an object")
    rows = value.get("episode_results")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{role} episode_results for {task} is invalid")
    if candidate and len(rows) != 20:
        raise ValueError(f"candidate task {task} must contain exactly 20 episodes")
    if not candidate and not 1 <= len(rows) <= 20:
        raise ValueError(f"baseline task {task} must contain between 1 and 20 episodes")
    _require_equal(f"{role} {task}.episodes", value.get("episodes"), 20)
    successes = sum(bool(row.get("success")) for row in rows)
    _require_equal(f"{role} {task}.successes", value.get("successes"), successes)
    expected_rate = successes / len(rows)
    actual_rate = value.get("success_rate")
    if not isinstance(actual_rate, (int, float)) or not math.isclose(float(actual_rate), expected_rate, abs_tol=1e-12):
        raise ValueError(f"{role} {task}.success_rate mismatch: expected={expected_rate} actual={actual_rate!r}")
    if candidate:
        _require_equal(f"candidate {task}.complete", value.get("complete"), True)
    else:
        _require_equal(f"baseline {task}.complete", value.get("complete"), len(rows) == 20)
    for index, row in enumerate(rows):
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"{role} {task} episode {index} success is not boolean")
        steps = row.get("steps")
        if not isinstance(steps, int) or not 0 <= steps <= 400:
            raise ValueError(f"{role} {task} episode {index} steps is invalid")
    return rows


def _validate_results(value: dict[str, Any], *, candidate: bool) -> dict[str, list[dict[str, Any]]]:
    role = "candidate" if candidate else "baseline"
    _require_equal(f"{role} result tasks", set(value), set(TASKS))
    return {task: _validate_task_result(task, value[task], candidate=candidate) for task in TASKS}


def mcnemar_exact_p(candidate_only: int, baseline_only: int) -> float:
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    lower = min(candidate_only, baseline_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compare(args: argparse.Namespace) -> None:
    candidate_args_path = Path(args.candidate_args).resolve()
    candidate_results_path = Path(args.candidate_results).resolve()
    baseline_args_path = Path(args.baseline_args).resolve()
    baseline_results_path = Path(args.baseline_results).resolve()
    candidate_args = _load_object(candidate_args_path)
    baseline_args = _load_object(baseline_args_path)
    _validate_args(candidate_args, candidate=True)
    _validate_args(baseline_args, candidate=False)
    _require_equal("candidate args.checkpoint", candidate_args.get("checkpoint"), args.checkpoint_path)
    candidate = _validate_results(_load_object(candidate_results_path), candidate=True)
    baseline = _validate_results(_load_object(baseline_results_path), candidate=False)

    episode_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    aggregate = {"both": 0, "candidate_only": 0, "baseline_only": 0, "neither": 0}
    for task in TASKS:
        candidate_rows = candidate[task]
        baseline_rows = baseline[task]
        matched = min(len(candidate_rows), len(baseline_rows))
        counts = {"both": 0, "candidate_only": 0, "baseline_only": 0, "neither": 0}
        for index, candidate_row in enumerate(candidate_rows):
            candidate_success = bool(candidate_row["success"])
            baseline_available = index < matched
            baseline_success = bool(baseline_rows[index]["success"]) if baseline_available else None
            if baseline_available:
                if candidate_success and baseline_success:
                    category = "both"
                elif candidate_success:
                    category = "candidate_only"
                elif baseline_success:
                    category = "baseline_only"
                else:
                    category = "neither"
                counts[category] += 1
                aggregate[category] += 1
            else:
                category = "candidate_only_unpaired"
            episode_rows.append(
                {
                    "task": task,
                    "episode_index": index,
                    "candidate_success": int(candidate_success),
                    "candidate_steps": candidate_row["steps"],
                    "baseline_available": int(baseline_available),
                    "baseline_success": "" if baseline_success is None else int(baseline_success),
                    "baseline_steps": "" if not baseline_available else baseline_rows[index]["steps"],
                    "pair_category": category,
                }
            )

        candidate_full_successes = sum(bool(row["success"]) for row in candidate_rows)
        candidate_matched_successes = sum(bool(row["success"]) for row in candidate_rows[:matched])
        baseline_successes = sum(bool(row["success"]) for row in baseline_rows)
        task_rows.append(
            {
                "task": task,
                "candidate_full_episodes": len(candidate_rows),
                "candidate_full_successes": candidate_full_successes,
                "candidate_full_success_rate": candidate_full_successes / len(candidate_rows),
                "baseline_observed_episodes": len(baseline_rows),
                "baseline_observed_successes": baseline_successes,
                "baseline_observed_success_rate": baseline_successes / len(baseline_rows),
                "matched_episodes": matched,
                "candidate_matched_successes": candidate_matched_successes,
                "candidate_matched_success_rate": candidate_matched_successes / matched,
                "baseline_matched_successes": baseline_successes,
                "baseline_matched_success_rate": baseline_successes / matched,
                "matched_delta_percentage_points": 100.0 * (candidate_matched_successes - baseline_successes) / matched,
                **counts,
                "mcnemar_exact_two_sided_p": mcnemar_exact_p(counts["candidate_only"], counts["baseline_only"]),
            }
        )

    matched_episodes = sum(aggregate.values())
    candidate_matched_successes = aggregate["both"] + aggregate["candidate_only"]
    baseline_matched_successes = aggregate["both"] + aggregate["baseline_only"]
    aggregate_row = {
        "matched_episodes": matched_episodes,
        "candidate_matched_successes": candidate_matched_successes,
        "candidate_matched_success_rate": candidate_matched_successes / matched_episodes,
        "baseline_matched_successes": baseline_matched_successes,
        "baseline_matched_success_rate": baseline_matched_successes / matched_episodes,
        "matched_delta_percentage_points": 100.0
        * (candidate_matched_successes - baseline_matched_successes)
        / matched_episodes,
        **aggregate,
        "mcnemar_exact_two_sided_p": mcnemar_exact_p(aggregate["candidate_only"], aggregate["baseline_only"]),
    }
    comparison = {
        "schema_version": 1,
        "status": "passed",
        "comparison_scope": "selected_two_task_episode_position_matched",
        "candidate_run_id": args.logical_run_id,
        "baseline_run_id": args.baseline_run_id,
        "tasks": task_rows,
        "aggregate": aggregate_row,
        "interpretation_boundary": (
            "This selected two-task comparison is not a full RoboCasa365 ranking and does not "
            "establish general superiority, representation sufficiency, or causal necessity."
        ),
    }
    evidence = {
        "schema_version": 1,
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "logical_run_id": args.logical_run_id,
        "source_commit": args.source_commit,
        "checkpoint_path": args.checkpoint_path,
        "protocol": {
            **EXPECTED_PROTOCOL,
            "tasks": list(TASKS),
            "candidate_num_shards": 1,
            "server_world_size": 8,
        },
        "inputs": {
            "candidate_args": str(candidate_args_path),
            "candidate_results": str(candidate_results_path),
            "baseline_args": str(baseline_args_path),
            "baseline_results": str(baseline_results_path),
        },
        "counts": {
            "tasks": len(TASKS),
            "episodes": len(episode_rows),
            "candidate_full_episodes": len(episode_rows),
            "matched_episodes": matched_episodes,
            "logical_num_shards": 1,
            "physical_gpu_count": 8,
            "physical_server_count": 1,
        },
        "comparison": comparison,
    }

    summary_csv = Path(args.summary_csv)
    task_csv = Path(args.task_success_rates_csv)
    comparison_json = Path(args.comparison_json)
    evidence_json = Path(args.evidence_json)
    verification_receipt = Path(args.verification_receipt)
    _write_csv(summary_csv, list(episode_rows[0]), episode_rows)
    _write_csv(task_csv, list(task_rows[0]), task_rows)
    _write_json(comparison_json, comparison)
    _write_json(evidence_json, evidence)
    _write_json(
        verification_receipt,
        {
            "schema_version": 1,
            "status": "passed",
            "experiment_id": args.experiment_id,
            "logical_run_id": args.logical_run_id,
            "counts": evidence["counts"],
            "artifacts": {
                "summary_csv": str(summary_csv.resolve()),
                "task_success_rates_csv": str(task_csv.resolve()),
                "comparison_json": str(comparison_json.resolve()),
                "evidence_json": str(evidence_json.resolve()),
            },
        },
    )
    print(
        "S1_R14_MATCHED_EVAL_VERIFIED",
        f"candidate_full_episodes={len(episode_rows)}",
        f"matched_episodes={matched_episodes}",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare_parser = subparsers.add_parser("compare")
    for name in (
        "candidate_args",
        "candidate_results",
        "baseline_args",
        "baseline_results",
        "experiment_id",
        "logical_run_id",
        "baseline_run_id",
        "source_commit",
        "checkpoint_path",
        "summary_csv",
        "task_success_rates_csv",
        "comparison_json",
        "evidence_json",
        "verification_receipt",
    ):
        compare_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    compare_parser.set_defaults(func=compare)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
