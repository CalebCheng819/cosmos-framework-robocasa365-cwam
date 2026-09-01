#!/usr/bin/env python3
"""Run an auditable closed-loop policy evaluation in RoboFactory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

try:
    from .cosmos3_robofactory_policy import Cosmos3RoboFactoryPolicy
    from .imagewam_robofactory_policy import (
        ImageWAMRoboFactoryPolicy,
        raw_agent_qpos,
    )
except ImportError:
    from cosmos3_robofactory_policy import Cosmos3RoboFactoryPolicy
    from imagewam_robofactory_policy import ImageWAMRoboFactoryPolicy, raw_agent_qpos


ROBOFACTORY_COMMIT = "3c4165a17d2af27a95043d9760339833957b2927"
PANEL_SCHEMA = "imagewam-robofactory-heldout-panel-v1"
TASK_CONFIGS = {
    "LongPipelineDelivery-rf": "robofactory/configs/table/long_pipeline_delivery.yaml",
    "StackCube-rf": "robofactory/configs/table/stack_cube.yaml",
    "StrikeCube-rf": "robofactory/configs/table/strike_cube.yaml",
    "TakePhoto-rf": "robofactory/configs/table/take_photo.yaml",
    "PlaceFood-rf": "robofactory/configs/table/place_food.yaml",
    "PickMeat-rf": "robofactory/configs/table/pick_meat.yaml",
    "PassShoe-rf": "robofactory/configs/table/pass_shoe.yaml",
    "CameraAlignment-rf": "robofactory/configs/table/camera_alignment.yaml",
    "LiftBarrier-rf": "robofactory/configs/table/lift_barrier.yaml",
    "ThreeRobotsStackCube-rf": "robofactory/configs/table/three_robots_stack_cube.yaml",
    "TwoRobotsStackCube-rf": "robofactory/configs/table/two_robots_stack_cube.yaml",
}
ACTION_DIMENSION_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "gripper_command",
)
ACTION_TRACE_SCHEMA = "imagewam-robofactory-action-trace-v1"
RESUME_PROGRESS_SCHEMA = "imagewam-robofactory-eval-resume-progress-v1"
RESUME_PROGRESS_FILENAME = "resume_progress.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with temporary.open("wb") as handle:
        for record in records:
            handle.write(
                (
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode()
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(
    path: Path, payload: Mapping[str, Any], *, fsync: bool = True
) -> None:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    try:
        os.write(descriptor, encoded)
        if fsync:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_existing(path: Path) -> None:
    if not path.is_file():
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _verify_robofactory_checkout(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    commit = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain=v1", "-uall")
    if commit != ROBOFACTORY_COMMIT:
        raise ValueError(
            f"RoboFactory commit mismatch: expected={ROBOFACTORY_COMMIT} actual={commit}"
        )
    if dirty:
        raise ValueError(f"RoboFactory checkout is dirty: {root}")
    return {
        "path": str(root),
        "commit": commit,
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "clean": True,
        "remote": _git(root, "remote", "get-url", "origin"),
    }


def _load_panel(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path)
    payload = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    if payload.get("schema_version") != PANEL_SCHEMA:
        raise ValueError(f"Unexpected panel schema: {payload.get('schema_version')!r}")
    if not isinstance(payload.get("episodes"), list):
        raise TypeError("Evaluation panel must contain an episodes list")
    return payload, record


def _selected_episodes(
    panel: Mapping[str, Any], task_name: str, episode_start: int, num_episodes: int
) -> list[dict[str, Any]]:
    records = [
        dict(record)
        for record in panel["episodes"]
        if record.get("task_name") == task_name
    ]
    records.sort(key=lambda record: int(record["task_index"]))
    if episode_start < 0 or num_episodes < 1:
        raise ValueError("episode_start must be >= 0 and num_episodes must be positive")
    selected = records[episode_start : episode_start + num_episodes]
    if len(selected) != num_episodes:
        raise ValueError(
            f"Panel has {len(records)} episodes for {task_name}; requested "
            f"[{episode_start}, {episode_start + num_episodes})"
        )
    return selected


def _episode_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_name": str(record.get("task_name")),
        "task_index": int(record["task_index"]),
        "panel_index": int(record["panel_index"]),
        "source_path": str(record["source_path"]),
        "source_h5_bytes": int(record["source_h5_bytes"]),
        "trajectory": str(record["trajectory"]),
        "episode_id": int(record["episode_id"]),
        "environment_seed": int(
            record.get("environment_seed", record.get("episode_seed"))
        ),
    }


def _resume_signature(
    *,
    args: argparse.Namespace,
    selected: Sequence[Mapping[str, Any]],
    panel_record: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    signature: dict[str, Any] = {
        "mode": args.mode,
        "task_name": args.task,
        "episode_start": int(args.episode_start),
        "num_episodes": int(args.num_episodes),
        "episodes": [_episode_identity(episode) for episode in selected],
        "panel_bytes": int(panel_record["bytes"]),
        "robofactory_commit": str(source_identity["commit"]),
        "eval_code_commit": str(args.eval_code_commit),
        "max_steps_override": args.max_steps,
        "exec_horizon": int(args.exec_horizon),
        "policy_seed_base": int(args.policy_seed),
        "action_horizon": int(args.action_horizon),
        "num_inference_steps": int(args.num_inference_steps),
        "sigma_shift": args.sigma_shift,
        "disable_action_trace": bool(args.disable_action_trace),
        "clip_actions_before_step": bool(args.clip_actions_before_step),
    }
    if args.mode == "imagewam":
        checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        signature["checkpoint"] = {
            "name": checkpoint.name,
            "bytes": int(checkpoint.stat().st_size),
        }
        signature["stats"] = _file_record(args.stats)
        signature["flux2_model"] = _file_record(args.flux2_model)
        signature["ae_model"] = _file_record(args.ae_model)
        signature["flux2_src"] = str(args.flux2_src.expanduser().resolve(strict=True))
        signature["qwen_cache_dir"] = str(
            args.qwen_cache_dir.expanduser().resolve(strict=True)
        )
        signature["policy_lightning_repo"] = str(
            args.policy_lightning_repo.expanduser().resolve(strict=True)
        )
        signature["policy_lightning_commit"] = str(args.policy_lightning_commit)
        signature["noposplat_checkpoint"] = _file_record(
            args.noposplat_checkpoint
        )
        signature["noposplat_checkpoint_bytes"] = int(
            args.noposplat_checkpoint_bytes
        )
        signature["gaussian_input_scale"] = float(args.gaussian_input_scale)
    elif args.mode == "cosmos3":
        signature["policy_server_url"] = str(args.policy_server_url).rstrip("/")
        signature["policy_lightning_repo"] = str(
            args.policy_lightning_repo.expanduser().resolve(strict=True)
        )
        signature["policy_lightning_commit"] = str(args.policy_lightning_commit)
        signature["noposplat_checkpoint"] = _file_record(args.noposplat_checkpoint)
        signature["noposplat_checkpoint_bytes"] = int(
            args.noposplat_checkpoint_bytes
        )
        signature["request_timeout"] = float(args.request_timeout)
    return signature


def _load_resume_results(
    *,
    output_dir: Path,
    signature: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    progress_path = output_dir / RESUME_PROGRESS_FILENAME
    if not progress_path.is_file():
        _atomic_jsonl(output_dir / "episodes.jsonl", [])
        return []
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if not isinstance(progress, dict):
        raise TypeError(f"Resume progress must be a JSON object: {progress_path}")
    if progress.get("schema_version") != RESUME_PROGRESS_SCHEMA:
        raise ValueError(
            f"Unexpected resume progress schema: {progress.get('schema_version')!r}"
        )
    if progress.get("signature") != signature:
        raise ValueError("Resume progress signature does not match this evaluation")
    records = progress.get("completed_results")
    if not isinstance(records, list):
        raise TypeError("Resume progress completed_results must be a list")
    if len(records) > len(selected):
        raise ValueError("Resume progress contains more episodes than requested")
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("status") != "completed":
            raise ValueError(f"Resume result {index} is not a completed episode")
        if _episode_identity(record) != _episode_identity(selected[index]):
            raise ValueError(f"Resume result {index} is not the expected panel prefix")
        if not isinstance(record.get("success"), bool):
            raise ValueError(f"Resume result {index} has a non-boolean success value")
        validated.append(dict(record))
    if int(progress.get("episodes_completed", -1)) != len(validated):
        raise ValueError("Resume progress episode count does not match its records")
    _atomic_jsonl(output_dir / "episodes.jsonl", validated)
    return validated


def _write_resume_progress(
    *,
    output_dir: Path,
    signature: Mapping[str, Any],
    completed_results: Sequence[Mapping[str, Any]],
) -> None:
    _atomic_json(
        output_dir / RESUME_PROGRESS_FILENAME,
        {
            "schema_version": RESUME_PROGRESS_SCHEMA,
            "status": (
                "complete"
                if len(completed_results) == signature["num_episodes"]
                else "partial"
            ),
            "episodes_completed": len(completed_results),
            "signature": signature,
            "completed_results": list(completed_results),
            "updated_at": _utc_now(),
        },
    )


def _source_path(dataset_root: Path, relative: str) -> Path:
    source = (dataset_root / relative).resolve(strict=True)
    try:
        source.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"Source path escapes dataset root: {relative!r}") from error
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Source H5 must be a regular non-symlink file: {source}")
    return source


def _as_bool(value: Any, *, label: str) -> bool:
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{label} must contain one value, got {tuple(tensor.shape)}")
    return bool(tensor.item())


def _as_float(value: Any, *, label: str) -> float:
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{label} must contain one value, got {tuple(tensor.shape)}")
    result = float(tensor.item())
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result}")
    return result


def _object_diagnostics(env: Any, task_name: str) -> dict[str, Any]:
    """Read task-native grasp state and tracked-object height without mutation."""
    base = env.unwrapped
    if task_name == "PlaceFood-rf":
        agent_grasps = {
            f"panda-{index}": _as_bool(
                agent.is_grasping(base.meat),
                label=f"panda-{index}.is_grasping(meat)",
            )
            for index, agent in enumerate(base.agent.agents)
        }
        heights = {
            "meat": _as_float(base.meat.pose.p[..., 2], label="meat.pose.z")
        }
    elif task_name == "TwoRobotsStackCube-rf":
        agent_grasps = {
            "panda-0": _as_bool(
                base.left_agent.is_grasping(base.cubeA),
                label="panda-0.is_grasping(cubeA)",
            ),
            "panda-1": _as_bool(
                base.right_agent.is_grasping(base.cubeB),
                label="panda-1.is_grasping(cubeB)",
            ),
        }
        heights = {
            "cubeA": _as_float(base.cubeA.pose.p[..., 2], label="cubeA.pose.z"),
            "cubeB": _as_float(base.cubeB.pose.p[..., 2], label="cubeB.pose.z"),
        }
    else:
        return {"agent_grasps": {}, "object_heights": {}, "grasped": False}
    return {
        "agent_grasps": agent_grasps,
        "object_heights": heights,
        "grasped": any(agent_grasps.values()),
    }


def _restore_initial_state(env: Any, source: Path, trajectory: str) -> None:
    from mani_skill.trajectory import utils as trajectory_utils

    with h5py.File(source, "r") as handle:
        states = trajectory_utils.dict_to_list_of_dicts(handle[trajectory]["env_states"])
    if not states:
        raise ValueError(f"No env states in {source}:{trajectory}")
    env.unwrapped.set_state_dict(states[0])


def _load_expert_actions(
    source: Path, trajectory: str, agent_names: Sequence[str]
) -> dict[str, np.ndarray]:
    with h5py.File(source, "r") as handle:
        stored = handle[trajectory]["actions"]
        if isinstance(stored, h5py.Dataset):
            if tuple(agent_names) != ("panda-0",):
                raise ValueError(
                    f"Flat expert actions require one panda-0 agent: "
                    f"{source}:{trajectory}:{tuple(agent_names)}"
                )
            actions = {"panda-0": np.asarray(stored[:], dtype=np.float32)}
        else:
            actions = {
                name: np.asarray(stored[name][:], dtype=np.float32)
                for name in agent_names
            }
    if len({value.shape[0] for value in actions.values()}) != 1:
        raise ValueError(f"Expert action horizons differ in {source}:{trajectory}")
    if {value.shape[1:] for value in actions.values()} != {(8,)}:
        raise ValueError(f"Expert actions are not 8D in {source}:{trajectory}")
    return actions


def _flat_action_to_dict(
    action: np.ndarray, agent_names: Sequence[str]
) -> dict[str, np.ndarray]:
    flat = np.asarray(action, dtype=np.float32).reshape(-1)
    expected = len(agent_names) * 8
    if flat.shape != (expected,):
        raise ValueError(f"Flat action must have shape ({expected},), got {flat.shape}")
    if not np.isfinite(flat).all():
        raise FloatingPointError("Policy action contains non-finite values")
    return {
        name: np.ascontiguousarray(flat[index * 8 : (index + 1) * 8])
        for index, name in enumerate(agent_names)
    }


def _action_diagnostics(
    action: Mapping[str, np.ndarray], action_space: Any
) -> tuple[int, float, float]:
    violations = 0
    minimum = float("inf")
    maximum = float("-inf")
    for name, value in action.items():
        space = _agent_action_space(action_space, name, len(action))
        violations += int(np.count_nonzero((value < space.low) | (value > space.high)))
        minimum = min(minimum, float(np.min(value)))
        maximum = max(maximum, float(np.max(value)))
    return violations, minimum, maximum


def _prepare_action_step(
    *,
    target: Mapping[str, np.ndarray],
    normalized: Mapping[str, np.ndarray] | None,
    action_space: Any,
    clip_actions_before_step: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray | None]]]:
    """Project an action for diagnostics and optionally apply that projection."""

    applied: dict[str, np.ndarray] = {}
    diagnostic: dict[str, dict[str, np.ndarray | None]] = {}
    for name, raw_value in target.items():
        value = np.asarray(raw_value, dtype=np.float32).reshape(-1)
        space = _agent_action_space(action_space, name, len(target))
        lower = np.asarray(space.low, dtype=np.float32).reshape(-1)
        upper = np.asarray(space.high, dtype=np.float32).reshape(-1)
        if value.shape != (8,) or lower.shape != (8,) or upper.shape != (8,):
            raise ValueError(
                f"{name} action/space must be 8D, got "
                f"target={value.shape} low={lower.shape} high={upper.shape}"
            )
        normalized_value = None
        if normalized is not None:
            normalized_value = np.asarray(normalized[name], dtype=np.float32).reshape(-1)
            if normalized_value.shape != (8,):
                raise ValueError(
                    f"{name} normalized action must be 8D, got {normalized_value.shape}"
                )
        projected = np.clip(value, lower, upper)
        applied_value = projected if clip_actions_before_step else value
        applied[name] = np.ascontiguousarray(applied_value, dtype=np.float32)
        diagnostic[name] = {
            "normalized": normalized_value,
            "target": value,
            "lower": lower,
            "upper": upper,
            "projected": projected,
            "applied": applied_value,
        }
    return applied, diagnostic


def _qpos_dimension_value(qpos: np.ndarray, dimension: int) -> float | list[float]:
    if dimension < 7:
        return float(qpos[dimension])
    return [float(qpos[7]), float(qpos[8])]


def _action_trace_record(
    *,
    args: argparse.Namespace,
    episode: Mapping[str, Any],
    agent_names: Sequence[str],
    diagnostic: Mapping[str, Mapping[str, np.ndarray | None]],
    qpos_before: Mapping[str, np.ndarray],
    qpos_after: Mapping[str, np.ndarray] | None,
    env_step_index: int,
    policy_query_index: int | None,
    policy_query_seed: int | None,
    chunk_index: int,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    agents = []
    for name in agent_names:
        before = np.asarray(qpos_before[name], dtype=np.float32).reshape(9)
        after = (
            None
            if qpos_after is None
            else np.asarray(qpos_after[name], dtype=np.float32).reshape(9)
        )
        values = diagnostic[name]
        dimensions = []
        for dimension, dimension_name in enumerate(ACTION_DIMENSION_NAMES):
            normalized = values["normalized"]
            target = float(values["target"][dimension])
            lower = float(values["lower"][dimension])
            upper = float(values["upper"][dimension])
            projected = float(values["projected"][dimension])
            applied = float(values["applied"][dimension])
            dimensions.append(
                {
                    "dimension": dimension,
                    "name": dimension_name,
                    "normalized_output": (
                        None if normalized is None else float(normalized[dimension])
                    ),
                    "denormalized_target": target,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "pre_clip": target,
                    "post_clip": projected,
                    "applied": applied,
                    "out_of_bounds": bool(target < lower or target > upper),
                    "lower_excess": max(lower - target, 0.0),
                    "upper_excess": max(target - upper, 0.0),
                    "projection_changed": bool(projected != target),
                    "qpos_before": _qpos_dimension_value(before, dimension),
                    "qpos_after": (
                        None
                        if after is None
                        else _qpos_dimension_value(after, dimension)
                    ),
                }
            )
        agents.append(
            {
                "agent": name,
                "qpos_before": before.tolist(),
                "qpos_after": None if after is None else after.tolist(),
                "gripper_opening_before": float(before[7] + before[8]),
                "gripper_opening_after": (
                    None if after is None else float(after[7] + after[8])
                ),
                "dimensions": dimensions,
            }
        )
    return {
        "schema_version": ACTION_TRACE_SCHEMA,
        "timestamp": _utc_now(),
        "status": status,
        "mode": args.mode,
        "task_name": args.task,
        "task_index": int(episode["task_index"]),
        "panel_index": int(episode["panel_index"]),
        "trajectory": str(episode["trajectory"]),
        "episode_id": int(episode["episode_id"]),
        "env_step_index": int(env_step_index),
        "policy_query_index": policy_query_index,
        "policy_query_seed": policy_query_seed,
        "chunk_index": int(chunk_index),
        "clip_actions_before_step": bool(args.clip_actions_before_step),
        "error": error,
        "agents": agents,
    }


def _configure_robofactory_runtime_assets(
    robofactory: Any, robofactory_runtime_root: Path
) -> Path:
    runtime_assets = robofactory_runtime_root / "assets"
    if not runtime_assets.is_dir():
        raise FileNotFoundError(f"RoboFactory runtime assets are missing: {runtime_assets}")
    robofactory.ASSET_DIR = str(runtime_assets)
    robofactory.DIR_MAP["${ASSET_DIR}"] = str(runtime_assets)
    return runtime_assets


def _build_environment(
    robofactory_root: Path, robofactory_runtime_root: Path, task_name: str
):
    if str(robofactory_root) not in sys.path:
        sys.path.insert(0, str(robofactory_root))
    os.chdir(robofactory_runtime_root)
    robofactory = __import__("robofactory")
    _configure_robofactory_runtime_assets(robofactory, robofactory_runtime_root)
    __import__("robofactory.tasks")
    import gymnasium as gym

    config = robofactory_root / TASK_CONFIGS[task_name]
    if not config.is_file():
        raise FileNotFoundError(f"Task config is missing: {config}")
    return gym.make(
        task_name,
        config=str(config),
        obs_mode="rgb",
        reward_mode="dense",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        sensor_configs={"shader_pack": "default"},
        human_render_camera_configs={"shader_pack": "default"},
        viewer_camera_configs={"shader_pack": "default"},
        num_envs=1,
        sim_backend="auto",
        render_backend="gpu",
        enable_shadow=True,
        parallel_in_single_scene=False,
    )


def _reset_environment(env: Any, episode: Mapping[str, Any]) -> None:
    reset_kwargs = dict(episode["reset_kwargs"])
    reset_seed = reset_kwargs.pop("seed", episode["episode_seed"])
    env.reset(seed=reset_seed, **reset_kwargs)


def _gpu_memory() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _environment_agent_names(env: Any) -> tuple[str, ...]:
    """Return the canonical panel names for single- and multi-agent envs."""
    agent = env.unwrapped.agent
    component_agents = getattr(agent, "agents", None)
    agent_count = len(component_agents) if component_agents is not None else 1
    return tuple(f"panda-{index}" for index in range(agent_count))


def _canonical_observation(
    observation: Mapping[str, Any], agent_names: Sequence[str]
) -> Mapping[str, Any]:
    """Wrap ManiSkill's flat single-agent state in the panel agent namespace."""
    agents = observation.get("agent")
    if not isinstance(agents, Mapping):
        raise TypeError("RoboFactory observation agent field is not a mapping")
    if all(name in agents for name in agent_names):
        return observation
    if tuple(agent_names) == ("panda-0",) and "qpos" in agents and "qvel" in agents:
        canonical = dict(observation)
        canonical["agent"] = {"panda-0": agents}
        return canonical
    raise KeyError(
        f"RoboFactory observation agents do not match expected names {tuple(agent_names)}"
    )


def _named_action_spaces(action_space: Any) -> Mapping[str, Any] | None:
    spaces = getattr(action_space, "spaces", None)
    if isinstance(spaces, Mapping):
        return spaces
    return action_space if isinstance(action_space, Mapping) else None


def _agent_action_space(action_space: Any, name: str, agent_count: int) -> Any:
    named_spaces = _named_action_spaces(action_space)
    if named_spaces is not None and name in named_spaces:
        return named_spaces[name]
    if agent_count == 1 and name == "panda-0":
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            return action_space
    raise KeyError(f"Action space does not provide agent {name!r}")


def _environment_action(
    action: Mapping[str, np.ndarray], action_space: Any
) -> Mapping[str, np.ndarray] | np.ndarray:
    """Use a flat action for ManiSkill single-agent Box environments."""
    if _named_action_spaces(action_space) is not None:
        return action
    if tuple(action) != ("panda-0",):
        raise ValueError(f"Flat action space requires only panda-0, got {tuple(action)}")
    return action["panda-0"]


def _run_episode(
    *,
    args: argparse.Namespace,
    episode: Mapping[str, Any],
    dataset_root: Path,
    robofactory_root: Path,
    robofactory_runtime_root: Path,
    policy: ImageWAMRoboFactoryPolicy | Cosmos3RoboFactoryPolicy | None,
) -> dict[str, Any]:
    started_at = _utc_now()
    start_time = time.monotonic()
    source = _source_path(dataset_root, str(episode["source_path"]))
    if source.stat().st_size != int(episode["source_h5_bytes"]):
        raise ValueError(f"Source H5 byte-size mismatch: {source}")

    episode_policy_seed = args.policy_seed + int(episode["task_index"])
    np.random.seed(episode_policy_seed)
    torch.manual_seed(episode_policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(episode_policy_seed)
        torch.cuda.reset_peak_memory_stats()

    env = None
    steps = 0
    policy_queries = 0
    query_seconds: list[float] = []
    bound_violations = 0
    action_minimum = float("inf")
    action_maximum = float("-inf")
    success = False
    termination_reason = "max_steps"
    grasped = False
    initial_object_heights: dict[str, float] = {}
    maximum_object_heights: dict[str, float] = {}
    final_object_diagnostics: dict[str, Any] = {}
    trace_path = args.output_dir / "action_trace.jsonl"
    try:
        env = _build_environment(
            robofactory_root, robofactory_runtime_root, args.task
        )
        _reset_environment(env, episode)
        _restore_initial_state(env, source, str(episode["trajectory"]))
        agent_names = tuple(episode["agent_names"])
        actual_agents = _environment_agent_names(env)
        if actual_agents != agent_names:
            raise ValueError(
                f"Environment agents {actual_agents} do not match panel {agent_names}"
            )
        initial_object_diagnostics = _object_diagnostics(env, args.task)
        initial_object_heights = dict(initial_object_diagnostics["object_heights"])
        maximum_object_heights = dict(initial_object_heights)
        final_object_diagnostics = initial_object_diagnostics
        grasped = bool(initial_object_diagnostics["grasped"])

        def update_object_diagnostics() -> None:
            nonlocal final_object_diagnostics, grasped
            final_object_diagnostics = _object_diagnostics(env, args.task)
            grasped = grasped or bool(final_object_diagnostics["grasped"])
            for name, height in final_object_diagnostics["object_heights"].items():
                maximum_object_heights[name] = max(
                    maximum_object_heights.get(name, height), height
                )
        max_steps = int(episode["max_episode_steps"])
        if args.max_steps is not None:
            max_steps = min(max_steps, int(args.max_steps))

        if args.mode == "expert-replay":
            expert = _load_expert_actions(source, str(episode["trajectory"]), agent_names)
            horizon = min(len(next(iter(expert.values()))), max_steps)
            for index in range(horizon):
                current_observation = _canonical_observation(
                    env.unwrapped.get_obs(), agent_names
                )
                qpos_before = raw_agent_qpos(current_observation, agent_names)
                target_action = {name: expert[name][index] for name in agent_names}
                violations, minimum, maximum = _action_diagnostics(
                    target_action, env.action_space
                )
                bound_violations += violations
                action_minimum = min(action_minimum, minimum)
                action_maximum = max(action_maximum, maximum)
                env_action, diagnostic = _prepare_action_step(
                    target=target_action,
                    normalized=None,
                    action_space=env.action_space,
                    clip_actions_before_step=args.clip_actions_before_step,
                )
                try:
                    _, _, terminated, truncated, info = env.step(
                        _environment_action(env_action, env.action_space)
                    )
                except Exception as error:
                    if not args.disable_action_trace:
                        _append_jsonl(
                            trace_path,
                            _action_trace_record(
                                args=args,
                                episode=episode,
                                agent_names=agent_names,
                                diagnostic=diagnostic,
                                qpos_before=qpos_before,
                                qpos_after=None,
                                env_step_index=steps,
                                policy_query_index=None,
                                policy_query_seed=None,
                                chunk_index=index,
                                status="env_step_error",
                                error=f"{type(error).__name__}: {error}",
                            ),
                            fsync=False,
                        )
                    raise
                qpos_after = raw_agent_qpos(
                    _canonical_observation(env.unwrapped.get_obs(), agent_names),
                    agent_names,
                )
                if not args.disable_action_trace:
                    _append_jsonl(
                        trace_path,
                        _action_trace_record(
                            args=args,
                            episode=episode,
                            agent_names=agent_names,
                            diagnostic=diagnostic,
                            qpos_before=qpos_before,
                            qpos_after=qpos_after,
                            env_step_index=steps,
                            policy_query_index=None,
                            policy_query_seed=None,
                            chunk_index=index,
                            status="executed",
                        ),
                        fsync=False,
                    )
                steps += 1
                update_object_diagnostics()
                success = _as_bool(info["success"], label="info.success")
                if success:
                    termination_reason = "success"
                    break
                if _as_bool(terminated, label="terminated"):
                    termination_reason = "terminated"
                    break
                if _as_bool(truncated, label="truncated"):
                    termination_reason = "truncated"
                    break
        else:
            if policy is None:
                raise RuntimeError("Policy mode requires an initialized policy")
            policy.start_episode(episode_policy_seed)
            current_observation = _canonical_observation(
                env.unwrapped.get_obs(), agent_names
            )
            policy.update_obs(current_observation, env.unwrapped.get_state_dict())
            while steps < max_steps:
                query_started = time.monotonic()
                action_views = policy.get_action_with_trace()
                query_seconds.append(time.monotonic() - query_started)
                policy_queries += 1
                normalized_chunk = np.asarray(
                    action_views["normalized"], dtype=np.float32
                )
                action_chunk = np.asarray(
                    action_views["denormalized"], dtype=np.float32
                )
                policy_query_index = int(action_views["query_index"])
                policy_query_seed = int(action_views["query_seed"])
                if normalized_chunk.shape != action_chunk.shape:
                    raise ValueError(
                        "Normalized and denormalized action chunks differ in shape: "
                        f"{normalized_chunk.shape} vs {action_chunk.shape}"
                    )
                execute = min(args.exec_horizon, len(action_chunk), max_steps - steps)
                if execute < 1:
                    raise ValueError("Policy returned an empty executable action chunk")
                stop_chunk = False
                for index in range(execute):
                    qpos_before = raw_agent_qpos(current_observation, agent_names)
                    target_action = _flat_action_to_dict(
                        action_chunk[index], agent_names
                    )
                    normalized_action = _flat_action_to_dict(
                        normalized_chunk[index], agent_names
                    )
                    violations, minimum, maximum = _action_diagnostics(
                        target_action, env.action_space
                    )
                    bound_violations += violations
                    action_minimum = min(action_minimum, minimum)
                    action_maximum = max(action_maximum, maximum)
                    env_action, diagnostic = _prepare_action_step(
                        target=target_action,
                        normalized=normalized_action,
                        action_space=env.action_space,
                        clip_actions_before_step=args.clip_actions_before_step,
                    )
                    try:
                        _, _, terminated, truncated, info = env.step(
                            _environment_action(env_action, env.action_space)
                        )
                    except Exception as error:
                        if not args.disable_action_trace:
                            _append_jsonl(
                                trace_path,
                                _action_trace_record(
                                    args=args,
                                    episode=episode,
                                    agent_names=agent_names,
                                    diagnostic=diagnostic,
                                    qpos_before=qpos_before,
                                    qpos_after=None,
                                    env_step_index=steps,
                                    policy_query_index=policy_query_index,
                                    policy_query_seed=policy_query_seed,
                                    chunk_index=index,
                                    status="env_step_error",
                                    error=f"{type(error).__name__}: {error}",
                                ),
                                fsync=False,
                            )
                        raise
                    current_observation = _canonical_observation(
                        env.unwrapped.get_obs(), agent_names
                    )
                    qpos_after = raw_agent_qpos(current_observation, agent_names)
                    if not args.disable_action_trace:
                        _append_jsonl(
                            trace_path,
                            _action_trace_record(
                                args=args,
                                episode=episode,
                                agent_names=agent_names,
                                diagnostic=diagnostic,
                                qpos_before=qpos_before,
                                qpos_after=qpos_after,
                                env_step_index=steps,
                                policy_query_index=policy_query_index,
                                policy_query_seed=policy_query_seed,
                                chunk_index=index,
                                status="executed",
                            ),
                            fsync=False,
                        )
                    steps += 1
                    update_object_diagnostics()
                    success = _as_bool(info["success"], label="info.success")
                    policy.update_obs(
                        current_observation, env.unwrapped.get_state_dict()
                    )
                    if success:
                        termination_reason = "success"
                        stop_chunk = True
                        break
                    if _as_bool(terminated, label="terminated"):
                        termination_reason = "terminated"
                        stop_chunk = True
                        break
                    if _as_bool(truncated, label="truncated"):
                        termination_reason = "truncated"
                        stop_chunk = True
                        break
                if stop_chunk:
                    break

        return {
            "status": "completed",
            "mode": args.mode,
            "task_name": args.task,
            "task_index": int(episode["task_index"]),
            "panel_index": int(episode["panel_index"]),
            "source_path": str(episode["source_path"]),
            "source_h5_bytes": int(episode["source_h5_bytes"]),
            "trajectory": episode["trajectory"],
            "episode_id": int(episode["episode_id"]),
            "environment_seed": int(episode["episode_seed"]),
            "policy_seed": episode_policy_seed,
            "success": success,
            "steps": steps,
            "policy_queries": policy_queries,
            "policy_query_seconds_total": float(sum(query_seconds)),
            "policy_query_seconds_mean": (
                float(sum(query_seconds) / len(query_seconds)) if query_seconds else None
            ),
            "action_bound_violations": bound_violations,
            "action_min": None if steps == 0 else action_minimum,
            "action_max": None if steps == 0 else action_maximum,
            "grasped": grasped,
            "final_agent_grasps": final_object_diagnostics.get("agent_grasps", {}),
            "initial_object_heights": initial_object_heights,
            "maximum_object_heights": maximum_object_heights,
            "maximum_object_lifts": {
                name: maximum_object_heights[name] - initial_object_heights[name]
                for name in initial_object_heights
            },
            "max_object_lift": max(
                (
                    maximum_object_heights[name] - initial_object_heights[name]
                    for name in initial_object_heights
                ),
                default=None,
            ),
            "termination_reason": termination_reason,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": time.monotonic() - start_time,
            "gpu_memory": _gpu_memory(),
        }
    finally:
        _fsync_existing(trace_path)
        if env is not None:
            env.close()


def _require_imagewam_arguments(args: argparse.Namespace) -> None:
    required = (
        "checkpoint",
        "stats",
        "flux2_src",
        "flux2_model",
        "ae_model",
        "qwen_cache_dir",
        "policy_lightning_repo",
        "noposplat_checkpoint",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"ImageWAM mode is missing required arguments: {missing}")


def _require_cosmos3_arguments(args: argparse.Namespace) -> None:
    required = (
        "policy_server_url",
        "policy_lightning_repo",
        "noposplat_checkpoint",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Cosmos3 mode is missing required arguments: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("expert-replay", "imagewam", "cosmos3"), required=True
    )
    parser.add_argument("--task", choices=tuple(TASK_CONFIGS), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--robofactory-root", type=Path, required=True)
    parser.add_argument("--robofactory-runtime-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-code-commit", required=True)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--exec-horizon", type=int, default=5)
    parser.add_argument("--policy-seed", type=int, default=10000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--flux2-src", type=Path)
    parser.add_argument("--flux2-model", type=Path)
    parser.add_argument("--ae-model", type=Path)
    parser.add_argument("--qwen-cache-dir", type=Path)
    parser.add_argument("--policy-lightning-repo", type=Path)
    parser.add_argument("--noposplat-checkpoint", type=Path)
    parser.add_argument("--policy-server-url")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument(
        "--policy-lightning-commit",
        default="c944b4989a89c99c69d2572ea870f6a04680f5e7",
    )
    parser.add_argument("--noposplat-checkpoint-bytes", type=int, default=2448469295)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float)
    parser.add_argument(
        "--gaussian-input-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the online NoPoSplat condition before ImageWAM inference. "
            "The formal protocol uses 1.0; 0.0 is a diagnostic no-Gaussian ablation."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only an exact, validated prefix of completed fixed-panel "
            "episodes from resume_progress.json."
        ),
    )
    parser.add_argument(
        "--disable-action-trace",
        action="store_true",
        help="Disable the default per-step/per-agent/per-dimension JSONL trace.",
    )
    parser.add_argument(
        "--clip-actions-before-step",
        action="store_true",
        help=(
            "Apply the diagnostic action-space projection before env.step. "
            "Default behavior records pre/post projection but executes the raw target."
        ),
    )
    args = parser.parse_args()

    if args.exec_horizon < 1:
        raise ValueError("exec_horizon must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    args.output_dir = output_dir
    robofactory_root = args.robofactory_root.expanduser().resolve(strict=True)
    robofactory_runtime_root = (
        robofactory_root
        if args.robofactory_runtime_root is None
        else args.robofactory_runtime_root.expanduser().resolve(strict=True)
    )
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    source_identity = _verify_robofactory_checkout(robofactory_root)
    panel, panel_record = _load_panel(args.panel)
    selected = _selected_episodes(
        panel, args.task, args.episode_start, args.num_episodes
    )

    if args.mode == "imagewam":
        _require_imagewam_arguments(args)
    elif args.mode == "cosmos3":
        _require_cosmos3_arguments(args)
    resume_signature = _resume_signature(
        args=args,
        selected=selected,
        panel_record=panel_record,
        source_identity=source_identity,
    )
    resumed_results = (
        _load_resume_results(
            output_dir=output_dir,
            signature=resume_signature,
            selected=selected,
        )
        if args.resume
        else []
    )

    policy = None
    policy_init_seconds = None
    if args.mode in ("imagewam", "cosmos3") and len(resumed_results) < len(selected):
        print(
            json.dumps(
                {
                    "event": "policy_initializing",
                    "task_name": args.task,
                    "timestamp": _utc_now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        init_started = time.monotonic()
        if args.mode == "imagewam":
            policy = ImageWAMRoboFactoryPolicy(
                checkpoint_path=args.checkpoint,
                stats_path=args.stats,
                flux2_src_path=args.flux2_src,
                flux2_model_path=args.flux2_model,
                ae_model_path=args.ae_model,
                qwen_cache_dir=args.qwen_cache_dir,
                policy_lightning_repo=args.policy_lightning_repo,
                noposplat_checkpoint=args.noposplat_checkpoint,
                policy_lightning_commit=args.policy_lightning_commit,
                noposplat_checkpoint_bytes=args.noposplat_checkpoint_bytes,
                task_name=args.task,
                device=args.device,
                action_horizon=args.action_horizon,
                num_inference_steps=args.num_inference_steps,
                sigma_shift=args.sigma_shift,
                seed=args.policy_seed,
                gaussian_input_scale=args.gaussian_input_scale,
            )
        else:
            policy = Cosmos3RoboFactoryPolicy(
                server_url=args.policy_server_url,
                policy_lightning_repo=args.policy_lightning_repo,
                noposplat_checkpoint=args.noposplat_checkpoint,
                policy_lightning_commit=args.policy_lightning_commit,
                noposplat_checkpoint_bytes=args.noposplat_checkpoint_bytes,
                task_name=args.task,
                device=args.device,
                seed=args.policy_seed,
                request_timeout=args.request_timeout,
            )
        policy_init_seconds = time.monotonic() - init_started
        print(
            json.dumps(
                {
                    "event": "policy_initialized",
                    "task_name": args.task,
                    "seconds": policy_init_seconds,
                    "timestamp": _utc_now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    run_manifest = {
        "schema_version": "robofactory-policy-eval-run-v2",
        "status": "running",
        "started_at": _utc_now(),
        "mode": args.mode,
        "task_name": args.task,
        "episode_start": args.episode_start,
        "num_episodes": args.num_episodes,
        "max_steps_override": args.max_steps,
        "exec_horizon": args.exec_horizon,
        "policy_seed_base": args.policy_seed,
        "policy_seed_schedule": "base_plus_task_index_then_plus_query_index_v1",
        "eval_code_commit": args.eval_code_commit,
        "panel": panel_record,
        "dataset_root": str(dataset_root),
        "robofactory": source_identity,
        "robofactory_runtime_root": str(robofactory_runtime_root),
        "policy_init_seconds": policy_init_seconds,
        "policy": None if policy is None else policy.provenance(),
        "action_trace": {
            "enabled": not args.disable_action_trace,
            "schema_version": ACTION_TRACE_SCHEMA,
            "path": str(output_dir / "action_trace.jsonl"),
            "clip_actions_before_step": bool(args.clip_actions_before_step),
            "default_semantics": (
                "post_clip_is_diagnostic_only"
                if not args.clip_actions_before_step
                else "post_clip_is_applied"
            ),
        },
        "resume": {
            "enabled": bool(args.resume),
            "episodes_restored": len(resumed_results),
            "progress_path": str(output_dir / RESUME_PROGRESS_FILENAME),
        },
        "argv": sys.argv,
    }
    _atomic_json(output_dir / "run_manifest.json", run_manifest)

    results: list[dict[str, Any]] = list(resumed_results)
    infrastructure_errors = 0
    for episode in selected[len(resumed_results) :]:
        print(
            json.dumps(
                {
                    "event": "episode_started",
                    "task_name": args.task,
                    "task_index": int(episode["task_index"]),
                    "trajectory": episode["trajectory"],
                    "timestamp": _utc_now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            result = _run_episode(
                args=args,
                episode=episode,
                dataset_root=dataset_root,
                robofactory_root=robofactory_root,
                robofactory_runtime_root=robofactory_runtime_root,
                policy=policy,
            )
        except Exception as error:  # noqa: BLE001 - terminal record is persisted
            infrastructure_errors += 1
            result = {
                "status": "infrastructure_error",
                "mode": args.mode,
                "task_name": args.task,
                "task_index": int(episode["task_index"]),
                "panel_index": int(episode["panel_index"]),
                "source_path": episode["source_path"],
                "trajectory": episode["trajectory"],
                "episode_id": int(episode["episode_id"]),
                "environment_seed": int(episode["episode_seed"]),
                "policy_seed": args.policy_seed + int(episode["task_index"]),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "finished_at": _utc_now(),
            }
        results.append(result)
        _append_jsonl(output_dir / "episodes.jsonl", result)
        if result["status"] == "completed" and all(
            previous["status"] == "completed" for previous in results
        ):
            _write_resume_progress(
                output_dir=output_dir,
                signature=resume_signature,
                completed_results=results,
            )
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "completed" and not args.continue_on_error:
            break

    completed = [result for result in results if result["status"] == "completed"]
    successes = sum(bool(result["success"]) for result in completed)
    summary = {
        "schema_version": "robofactory-policy-eval-summary-v2",
        "status": (
            "PASS"
            if infrastructure_errors == 0 and len(results) == len(selected)
            else "INFRASTRUCTURE_ERROR"
        ),
        "mode": args.mode,
        "task_name": args.task,
        "episodes_requested": len(selected),
        "episodes_recorded": len(results),
        "episodes_completed": len(completed),
        "infrastructure_errors": infrastructure_errors,
        "successes": successes,
        "strict_success_rate": (
            successes / len(selected)
            if infrastructure_errors == 0 and len(completed) == len(selected)
            else None
        ),
        "diagnostic_success_rate_completed": (
            successes / len(completed) if completed else None
        ),
        "total_action_bound_violations": sum(
            int(result.get("action_bound_violations", 0)) for result in completed
        ),
        "grasped_episodes": sum(bool(result.get("grasped")) for result in completed),
        "grasp_rate": (
            sum(bool(result.get("grasped")) for result in completed) / len(completed)
            if completed
            else None
        ),
        "max_object_lift": max(
            (
                float(result["max_object_lift"])
                for result in completed
                if result.get("max_object_lift") is not None
            ),
            default=None,
        ),
        "finished_at": _utc_now(),
        "episodes_record": _file_record(output_dir / "episodes.jsonl"),
        "action_trace": (
            _file_record(output_dir / "action_trace.jsonl")
            if (output_dir / "action_trace.jsonl").is_file()
            else None
        ),
    }
    if (
        args.mode == "expert-replay"
        and summary["status"] == "PASS"
        and successes != len(selected)
    ):
        summary["status"] = "EXPERT_REPLAY_FAILURE"
    _atomic_json(output_dir / "summary.json", summary)
    run_manifest["status"] = "terminal"
    run_manifest["finished_at"] = _utc_now()
    run_manifest["summary"] = _file_record(output_dir / "summary.json")
    _atomic_json(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if summary["status"] == "INFRASTRUCTURE_ERROR":
        raise SystemExit(2)
    if summary["status"] == "EXPERT_REPLAY_FAILURE":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
