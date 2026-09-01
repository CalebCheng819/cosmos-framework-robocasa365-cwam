#!/usr/bin/env python3
"""Materialize a deterministic Table11 RoboFactory evaluation panel."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import h5py


SCHEMA_VERSION = "imagewam-robofactory-heldout-panel-v1"
TASK_SOURCES = {
    "LongPipelineDelivery-rf": (
        4,
        "tasks/LongPipelineDelivery-rf/motionplanning/"
        "table-200-longpipelinedelivery-rf.h5",
    ),
    "StackCube-rf": (
        1,
        "tasks/StackCube-rf/StackCube-rf/motionplanning/"
        "table-200-stackcube-rf.h5",
    ),
    "StrikeCube-rf": (
        1,
        "tasks/StrikeCube-rf/StrikeCube-rf/motionplanning/"
        "table-200-strikecube-rf.h5",
    ),
    "TakePhoto-rf": (
        4,
        "tasks/TakePhoto-rf/TakePhoto-rf/motionplanning/"
        "table-200-takephoto-rf.h5",
    ),
    "PlaceFood-rf": (
        2,
        "tasks/PlaceFood-rf/PlaceFood-rf/motionplanning/"
        "table-200-placefood-rf.h5",
    ),
    "PickMeat-rf": (
        1,
        "tasks/PickMeat-rf/PickMeat-rf/motionplanning/"
        "table-200-pickmeat-rf.h5",
    ),
    "PassShoe-rf": (
        2,
        "tasks/PassShoe-rf/PassShoe-rf/motionplanning/"
        "table-200-passshoe-rf.h5",
    ),
    "CameraAlignment-rf": (
        3,
        "tasks/CameraAlignment-rf/CameraAlignment-rf/motionplanning/"
        "table-200-cameraalignment-rf.h5",
    ),
    "LiftBarrier-rf": (
        2,
        "tasks/LiftBarrier-rf/LiftBarrier-rf/motionplanning/"
        "table-200-liftbarrier-rf.h5",
    ),
    "ThreeRobotsStackCube-rf": (
        3,
        "tasks/ThreeRobotsStackCube-rf/ThreeRobotsStackCube-rf/motionplanning/"
        "table-200-threerobotsstackcube-rf.h5",
    ),
    "TwoRobotsStackCube-rf": (
        2,
        "tasks/TwoRobotsStackCube-rf/TwoRobotsStackCube-rf/motionplanning/"
        "table-200-tworobotsstackcube-rf.h5",
    ),
}

SOURCE_EPISODES_PER_TASK = 200
DEFAULT_EPISODES_PER_TASK = 8


def fixed_episode_ids(episodes_per_task: int) -> tuple[int, ...]:
    """Return a deterministic full-range systematic sample of source IDs."""
    if not 1 <= episodes_per_task <= SOURCE_EPISODES_PER_TASK:
        raise ValueError(
            "episodes_per_task must be between 1 and "
            f"{SOURCE_EPISODES_PER_TASK}, got {episodes_per_task}"
        )
    episode_ids = tuple(
        index * SOURCE_EPISODES_PER_TASK // episodes_per_task
        for index in range(episodes_per_task)
    )
    if len(set(episode_ids)) != episodes_per_task:
        raise ValueError("systematic episode selection produced duplicate IDs")
    return episode_ids


# Preserve the validated eight-episode panel as the command-line default.
FIXED_EPISODE_IDS = fixed_episode_ids(DEFAULT_EPISODES_PER_TASK)


def _safe_source(dataset_root: Path, relative: str) -> Path:
    source = (dataset_root / relative).resolve(strict=True)
    source.relative_to(dataset_root)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Source H5 must be a regular file: {source}")
    return source


def _metadata(sidecar: Mapping[str, Any], episode_id: int) -> Mapping[str, Any]:
    matches = [
        record
        for record in sidecar.get("episodes", [])
        if int(record.get("episode_id", -1)) == episode_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one sidecar record for episode_id={episode_id}, got {len(matches)}"
        )
    return matches[0]


def build_panel(
    dataset_root: Path,
    episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    episode_ids = fixed_episode_ids(episodes_per_task)
    episodes: list[dict[str, Any]] = []
    for task, (agent_count, relative) in TASK_SOURCES.items():
        source = _safe_source(dataset_root, relative)
        sidecar_path = source.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        env_info = sidecar.get("env_info")
        if not isinstance(env_info, Mapping) or env_info.get("env_id") != task:
            raise ValueError(f"Sidecar env_id mismatch: {sidecar_path}")
        source_size = int(source.stat().st_size)
        sidecar_size = int(sidecar_path.stat().st_size)
        with h5py.File(source, "r") as handle:
            for task_index, episode_id in enumerate(episode_ids):
                trajectory = f"traj_{episode_id}"
                metadata = _metadata(sidecar, episode_id)
                if not bool(metadata.get("success")):
                    raise ValueError(
                        f"Selected source demonstration is not successful: "
                        f"{source}:{trajectory}"
                    )
                reset_kwargs = metadata.get("reset_kwargs", {})
                if not isinstance(reset_kwargs, Mapping):
                    raise TypeError(
                        f"reset_kwargs is not a mapping: {source}:{trajectory}"
                    )
                group = handle[trajectory]
                if "env_states" not in group or "actions" not in group:
                    raise KeyError(
                        f"Selected trajectory lacks env_states/actions: "
                        f"{source}:{trajectory}"
                    )
                expected_agents = [f"panda-{index}" for index in range(agent_count)]
                actions = group["actions"]
                if isinstance(actions, h5py.Dataset):
                    if agent_count != 1:
                        raise ValueError(
                            f"Only one-agent actions may be a dataset: "
                            f"{source}:{trajectory}:{actions.shape}"
                        )
                    agent_names = expected_agents
                    lengths = {int(actions.shape[0])}
                    dims = {int(actions.shape[1])}
                else:
                    agent_names = sorted(
                        actions, key=lambda name: int(name.rsplit("-", 1)[-1])
                    )
                    if agent_names != expected_agents:
                        raise ValueError(
                            f"Agent contract mismatch: "
                            f"{source}:{trajectory}:{agent_names}"
                        )
                    lengths = {int(actions[name].shape[0]) for name in agent_names}
                    dims = {int(actions[name].shape[1]) for name in agent_names}
                if len(lengths) != 1 or dims != {8}:
                    raise ValueError(
                        f"Action contract mismatch: {source}:{trajectory}"
                    )
                episodes.append(
                    {
                        "panel_index": len(episodes),
                        "task_index": task_index,
                        "task_name": task,
                        "agent_count": agent_count,
                        "agent_names": agent_names,
                        "source_path": relative,
                        "source_h5_bytes": source_size,
                        "source_sidecar_bytes": sidecar_size,
                        "trajectory": trajectory,
                        "episode_id": episode_id,
                        "episode_seed": int(metadata["episode_seed"]),
                        "reset_kwargs": json.loads(json.dumps(reset_kwargs)),
                        "demonstration_steps": lengths.pop(),
                        "source_success": True,
                        "max_episode_steps": int(env_info["max_episode_steps"]),
                    }
                )
    expected_count = len(TASK_SOURCES) * len(episode_ids)
    if len(episodes) != expected_count:
        raise ValueError(
            f"Fixed panel must contain {expected_count} episodes, got {len(episodes)}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_rule": {
            "name": "table11_systematic_successful_episode_ids_without_hash_gates",
            "source_episodes_per_task": SOURCE_EPISODES_PER_TASK,
            "requested_episodes_per_task": episodes_per_task,
            "formula": "floor(index * source_episodes_per_task / requested_episodes_per_task)",
            "episode_ids": list(episode_ids),
        },
        "dataset_root": str(dataset_root),
        "task_agent_counts": {
            task: agent_count for task, (agent_count, _) in TASK_SOURCES.items()
        },
        "trajectories_per_task": len(episode_ids),
        "episodes": episodes,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=DEFAULT_EPISODES_PER_TASK,
        help=(
            "Number of deterministic systematic source episodes per task "
            f"(default: {DEFAULT_EPISODES_PER_TASK})"
        ),
    )
    args = parser.parse_args()
    panel = build_panel(args.dataset_root, args.episodes_per_task)
    _atomic_json(args.output.expanduser(), panel)
    print(
        json.dumps(
            {
                "status": "PASS",
                "episodes": len(panel["episodes"]),
                "tasks": len(TASK_SOURCES),
                "output": str(args.output.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
