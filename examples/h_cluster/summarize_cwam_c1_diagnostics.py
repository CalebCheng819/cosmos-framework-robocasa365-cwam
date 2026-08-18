"""Summarize fixed-checkpoint CWAM C1 stage and latent interventions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from cosmos_framework.scripts.robocasa_stage_diagnostics import STAGE_ORDER, SUPPORTED_TASKS


EXPECTED_CONDITIONS = (
    "normal_h1",
    "normal_h4",
    "normal_h12",
    "zero_h12",
    "shuffle_h12",
    "cross_episode_h12",
    "selected_h12",
    "complement_h12",
)


def _load_condition(root: Path, condition: str, episodes_per_task: int) -> dict[str, Any]:
    candidates = sorted((root / condition).glob("results/*/results_shard0.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one result file for {condition}, got {candidates}")
    value = json.loads(candidates[0].read_text(encoding="utf-8"))
    if set(value) != set(SUPPORTED_TASKS):
        raise ValueError(f"{condition} task coverage drifted: {sorted(value)}")
    for task, task_result in value.items():
        episodes = task_result.get("episode_results")
        if task_result.get("complete") is not True or not isinstance(episodes, list):
            raise ValueError(f"{condition}/{task} is not complete")
        if len(episodes) != episodes_per_task:
            raise ValueError(
                f"{condition}/{task} expected {episodes_per_task} episodes, got {len(episodes)}"
            )
    return value


def _stage_summary(episodes: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    reached = [episode.get("stage_reached", {}).get(stage) is True for episode in episodes]
    first_steps = [
        int(episode["stage_first_step"][stage])
        for episode, was_reached in zip(episodes, reached, strict=True)
        if was_reached
    ]
    return {
        "reach_count": sum(reached),
        "reach_rate": sum(reached) / len(episodes),
        "first_step_mean": statistics.fmean(first_steps) if first_steps else None,
        "first_step_median": statistics.median(first_steps) if first_steps else None,
    }


def summarize(root: Path, episodes_per_task: int) -> dict[str, Any]:
    raw = {
        condition: _load_condition(root, condition, episodes_per_task)
        for condition in EXPECTED_CONDITIONS
    }
    conditions: dict[str, Any] = {}
    for condition, task_results in raw.items():
        per_task: dict[str, Any] = {}
        all_episodes: list[dict[str, Any]] = []
        for task in SUPPORTED_TASKS:
            task_result = task_results[task]
            episodes = task_result["episode_results"]
            all_episodes.extend(episodes)
            stage_metrics = {stage: _stage_summary(episodes, stage) for stage in STAGE_ORDER}
            transitions = {}
            for previous, current in zip(STAGE_ORDER, STAGE_ORDER[1:]):
                previous_count = stage_metrics[previous]["reach_count"]
                current_count = stage_metrics[current]["reach_count"]
                transitions[f"{previous}_to_{current}"] = (
                    current_count / previous_count if previous_count else None
                )
            per_task[task] = {
                "episodes": len(episodes),
                "strict_success_count": sum(episode.get("success") is True for episode in episodes),
                "strict_success_rate": sum(episode.get("success") is True for episode in episodes)
                / len(episodes),
                "place_anytime_count": sum(
                    episode.get("place_anytime") is True for episode in episodes
                ),
                "place_anytime_rate": sum(
                    episode.get("place_anytime") is True for episode in episodes
                )
                / len(episodes),
                "stages": stage_metrics,
                "conditional_transitions": transitions,
            }
        conditions[condition] = {
            "tasks": per_task,
            "aggregate": {
                "episodes": len(all_episodes),
                "strict_success_count": sum(
                    episode.get("success") is True for episode in all_episodes
                ),
                "strict_success_rate": sum(
                    episode.get("success") is True for episode in all_episodes
                )
                / len(all_episodes),
                "place_anytime_count": sum(
                    episode.get("place_anytime") is True for episode in all_episodes
                ),
                "place_anytime_rate": sum(
                    episode.get("place_anytime") is True for episode in all_episodes
                )
                / len(all_episodes),
                "stages": {stage: _stage_summary(all_episodes, stage) for stage in STAGE_ORDER},
            },
        }

    normal = conditions["normal_h12"]["aggregate"]
    interventions = {}
    for condition in (
        "zero_h12",
        "shuffle_h12",
        "cross_episode_h12",
        "selected_h12",
        "complement_h12",
    ):
        current = conditions[condition]["aggregate"]
        interventions[condition] = {
            "strict_success_delta_vs_normal": (
                current["strict_success_rate"] - normal["strict_success_rate"]
            ),
            "place_anytime_delta_vs_normal": (
                current["place_anytime_rate"] - normal["place_anytime_rate"]
            ),
            "stage_reach_delta_vs_normal": {
                stage: current["stages"][stage]["reach_rate"]
                - normal["stages"][stage]["reach_rate"]
                for stage in STAGE_ORDER
            },
        }
    return {
        "schema_version": 1,
        "tasks": list(SUPPORTED_TASKS),
        "episodes_per_task": episodes_per_task,
        "stage_order": list(STAGE_ORDER),
        "conditions": conditions,
        "interventions_vs_normal_h12": interventions,
    }


def write_csv(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for condition, condition_result in summary["conditions"].items():
        for task, task_result in condition_result["tasks"].items():
            base = {
                "condition": condition,
                "task": task,
                "episodes": task_result["episodes"],
                "strict_success_rate": task_result["strict_success_rate"],
                "place_anytime_rate": task_result["place_anytime_rate"],
            }
            for stage in STAGE_ORDER:
                row = dict(base)
                row.update({"stage": stage, **task_result["stages"][stage]})
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.results_root, args.episodes_per_task)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(result, args.output_csv)
    print(json.dumps(result["interventions_vs_normal_h12"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
