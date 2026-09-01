# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native ImageWAM-compatible RoboFactory multi-robot HDF5 dataset.

Each raw window contains 33 synchronized observations and 32 actions.  Cosmos
receives the 17 observation frames at offsets ``0,2,...,32`` and agent-major
action tokens ``[N*32,8]``.  State and Gaussian conditions retain the native
agent axis and undergo the same optional permutation as the action tokens.
"""

from __future__ import annotations

import bisect
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_processing import ActionAffineNormalization
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.datasets.robofactory_gaussian_cache import GaussianCache


DEFAULT_INSTRUCTIONS = {
    "LongPipelineDelivery-rf": "Deliver the objects through the long pipeline with four robots.",
    "StackCube-rf": "Stack the cubes with one robot.",
    "StrikeCube-rf": "Strike the cube to the target with one robot.",
    "TakePhoto-rf": "Coordinate four robots to take the photo.",
    "PlaceFood-rf": "Place the food at the target location with two robots.",
    "PickMeat-rf": "Pick up the meat with one robot.",
    "PassShoe-rf": "Pass the shoe between two robots.",
    "CameraAlignment-rf": "Align the camera with three robots.",
    "LiftBarrier-rf": "Lift the barrier with two robots.",
    "ThreeRobotsStackCube-rf": "Stack the cubes with three robots.",
    "TwoRobotsStackCube-rf": "Stack the cubes with two robots.",
}


@dataclass(frozen=True)
class Episode:
    path: Path
    source_path: str
    trajectory: str
    task: str
    transitions: int
    windows: int
    agent_count: int


def _agent_number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"Agent name must end with an integer: {name}")
    return int(match.group(1))


def _task_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.endswith("-rf"):
            return part
    return path.stem


def load_imagewam_zscore_normalizer(stats_path: str | Path) -> ActionAffineNormalization:
    """Load either compact Cosmos stats or ImageWAM dataset-stat JSON."""
    path = Path(stats_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"RoboFactory action statistics do not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    block: Any = raw
    if isinstance(block, dict) and "action" in block:
        block = block["action"]
    if isinstance(block, dict) and "default" in block:
        block = block["default"]
    if isinstance(block, dict) and "global" in block:
        block = block["global"]
    if not isinstance(block, dict):
        raise TypeError(f"Unsupported RoboFactory action-stat structure in {path}")

    mean = block.get("mean", block.get("global_mean"))
    std = block.get("std", block.get("global_std"))
    if mean is None or std is None:
        raise KeyError(f"RoboFactory z-score stats require mean/std or global_mean/global_std: {path}")
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
    std_tensor = torch.as_tensor(std, dtype=torch.float32)
    if mean_tensor.shape != (8,) or std_tensor.shape != (8,):
        raise ValueError(
            f"RoboFactory action stats must have eight channels, got {tuple(mean_tensor.shape)} "
            f"and {tuple(std_tensor.shape)}"
        )
    if not bool(torch.isfinite(mean_tensor).all()) or not bool(torch.isfinite(std_tensor).all()):
        raise ValueError(f"RoboFactory action stats contain non-finite values: {path}")
    if bool((std_tensor <= 0).any()):
        raise ValueError(f"RoboFactory action std must be strictly positive: {path}")
    return ActionAffineNormalization(offset=mean_tensor, scale=std_tensor)


class RoboFactoryMultiRobotDataset(Dataset):
    """Read variable-cardinality RoboFactory HDF5 windows without empty agents."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        val_ratio: float = 0.1,
        split_seed: int = 42,
        chunk_length: int = 32,
        image_stride: int = 2,
        sample_stride: int = 16,
        conditioning_fps: float = 10.0,
        accepted_agent_counts: Iterable[int] = (1, 2, 3, 4),
        camera_path: str = "obs/sensor_data/head_camera_global/rgb",
        single_agent_camera_path: str = "obs/sensor_data/head_camera/rgb",
        gaussian_cache_root: str | Path | None = None,
        gaussian_preflight: bool = True,
        gaussian_source_path_strip_prefix: str | None = "tasks",
        permute_agents: bool = True,
        permutation_seed: int = 42,
        task_instructions: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"RoboFactory dataset directory does not exist: {self.root}")
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be train, val, or all; got {split!r}")
        if not 0.0 <= float(val_ratio) < 1.0:
            raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
        if int(chunk_length) != 32:
            raise ValueError(f"ImageWAM RoboFactory action horizon is fixed at 32, got {chunk_length}")
        if int(image_stride) != 2:
            raise ValueError(f"Cosmos RoboFactory video stride is fixed at 2, got {image_stride}")
        if int(sample_stride) <= 0:
            raise ValueError(f"sample_stride must be positive, got {sample_stride}")

        self.split = split
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)
        self.chunk_length = int(chunk_length)
        self.raw_observation_length = self.chunk_length + 1
        self.image_stride = int(image_stride)
        self.image_offsets = tuple(range(0, self.raw_observation_length, self.image_stride))
        if self.image_offsets != tuple(range(0, 33, 2)):
            raise AssertionError(f"Expected 17 offsets 0,2,...,32; got {self.image_offsets}")
        self.sample_stride = int(sample_stride)
        self.conditioning_fps = float(conditioning_fps)
        self.accepted_agent_counts = tuple(sorted({int(value) for value in accepted_agent_counts}))
        if not self.accepted_agent_counts or any(value < 1 or value > 4 for value in self.accepted_agent_counts):
            raise ValueError(f"accepted_agent_counts must be a subset of 1..4, got {accepted_agent_counts}")
        self.camera_path = str(camera_path).strip("/")
        self.single_agent_camera_path = str(single_agent_camera_path).strip("/")
        self.permute_agents = bool(permute_agents)
        self.permutation_seed = int(permutation_seed)
        self.task_instructions = dict(DEFAULT_INSTRUCTIONS)
        if task_instructions:
            self.task_instructions.update(task_instructions)
        self._domain_id = get_domain_id("robofactory")

        if gaussian_cache_root is None:
            raise ValueError("RoboFactory multi-robot training requires gaussian_cache_root")
        self._gaussian_cache = GaussianCache(
            gaussian_cache_root,
            source_path_strip_prefix=gaussian_source_path_strip_prefix,
        )

        files = sorted(path.resolve() for path in self.root.rglob("*.h5") if path.is_file())
        if not files:
            raise FileNotFoundError(f"No RoboFactory .h5 files found under {self.root}")
        all_episodes = self._inspect_episodes(files)
        if not all_episodes:
            raise ValueError(f"No complete 32-action RoboFactory windows found under {self.root}")
        self.episodes = tuple(self._split_episodes(all_episodes))
        if not self.episodes:
            raise ValueError(f"RoboFactory split {split!r} selected no episodes")

        self._cumulative_windows: list[int] = []
        total = 0
        for episode in self.episodes:
            total += episode.windows
            self._cumulative_windows.append(total)

        if gaussian_preflight:
            self._gaussian_cache.preflight(
                (
                    episode.source_path,
                    episode.trajectory,
                    start,
                    f"panda-{agent_index}",
                )
                for episode in self.episodes
                for start in range(0, episode.windows * self.sample_stride, self.sample_stride)
                for agent_index in range(episode.agent_count)
            )

    @staticmethod
    def _agent_names(group: h5py.Group) -> list[str]:
        actions = group["actions"]
        if isinstance(actions, h5py.Dataset):
            return ["panda-0"]
        return sorted(actions.keys(), key=_agent_number)

    @staticmethod
    def _action_dataset(group: h5py.Group, agent: str) -> h5py.Dataset:
        actions = group["actions"]
        if isinstance(actions, h5py.Dataset):
            if agent != "panda-0":
                raise ValueError(f"Flat single-agent action layout cannot serve {agent}")
            return actions
        return actions[agent]

    @staticmethod
    def _state_path(group: h5py.Group, agent: str, field: str) -> str:
        nested = f"obs/agent/{agent}/{field}"
        if nested in group:
            return nested
        flat = f"obs/agent/{field}"
        if agent == "panda-0" and flat in group:
            return flat
        raise ValueError(f"Missing {field} for {agent}; tried {nested} and {flat}")

    def _camera_path_for_group(self, group: h5py.Group) -> str:
        if self.camera_path in group:
            return self.camera_path
        if self._agent_names(group) == ["panda-0"] and self.single_agent_camera_path in group:
            return self.single_agent_camera_path
        raise ValueError(
            f"Missing camera dataset; tried {self.camera_path} and "
            f"single-agent fallback {self.single_agent_camera_path}"
        )

    def _inspect_episodes(self, files: list[Path]) -> list[Episode]:
        episodes: list[Episode] = []
        for path in files:
            source_path = path.relative_to(self.root).as_posix()
            task = _task_from_path(path)
            with h5py.File(path, "r") as handle:
                for trajectory in sorted(handle.keys()):
                    group = handle[trajectory]
                    if not isinstance(group, h5py.Group) or "actions" not in group:
                        continue
                    agents = tuple(self._agent_names(group))
                    agent_count = len(agents)
                    expected_agents = tuple(f"panda-{index}" for index in range(agent_count))
                    if agent_count not in self.accepted_agent_counts or agents != expected_agents:
                        raise ValueError(
                            f"{path}:{trajectory} agents={agents}; expected contiguous panda-0..N-1 "
                            f"with N in {self.accepted_agent_counts}"
                        )
                    camera = group[self._camera_path_for_group(group)]
                    if camera.ndim != 4 or camera.shape[-1] != 3:
                        raise ValueError(
                            f"{path}:{trajectory} camera shape={camera.shape}; expected [frames,height,width,3]"
                        )
                    for agent in agents:
                        action = self._action_dataset(group, agent)
                        if action.ndim != 2 or action.shape[-1] != 8:
                            raise ValueError(
                                f"{path}:{trajectory}/actions/{agent} shape={action.shape}; expected [T,8]"
                            )
                        for field in ("qpos", "qvel"):
                            state = group[self._state_path(group, agent, field)]
                            if state.ndim != 2 or state.shape[-1] != 9:
                                raise ValueError(
                                    f"{path}:{trajectory}/{field}/{agent} shape={state.shape}; expected [T,9]"
                                )
                    transitions = min(int(self._action_dataset(group, agent).shape[0]) for agent in agents)
                    transitions = min(
                        transitions,
                        min(
                            int(group[self._state_path(group, agent, field)].shape[0]) - 1
                            for agent in agents
                            for field in ("qpos", "qvel")
                        ),
                        int(camera.shape[0]) - 1,
                    )
                    raw_windows = transitions - self.chunk_length + 1
                    windows = 0 if raw_windows <= 0 else (raw_windows - 1) // self.sample_stride + 1
                    if windows:
                        episodes.append(
                            Episode(path, source_path, trajectory, task, transitions, windows, agent_count)
                        )
        return episodes

    def _split_episodes(self, episodes: list[Episode]) -> list[Episode]:
        if self.split == "all" or self.val_ratio == 0.0:
            return episodes
        grouped: dict[tuple[str, int], list[Episode]] = defaultdict(list)
        for episode in episodes:
            grouped[(episode.task, episode.agent_count)].append(episode)

        selected: list[Episode] = []
        for bucket_index, ((task, agent_count), group) in enumerate(sorted(grouped.items())):
            ordered = sorted(group, key=lambda item: (item.source_path, item.trajectory))
            if len(ordered) < 2:
                raise ValueError(
                    f"A non-zero validation split requires at least two episodes for task={task}, N={agent_count}"
                )
            random.Random(self.split_seed + bucket_index).shuffle(ordered)
            validation_count = max(1, min(len(ordered) - 1, round(len(ordered) * self.val_ratio)))
            validation = set(ordered[:validation_count])
            selected.extend(
                episode
                for episode in group
                if (episode in validation) == (self.split == "val")
            )
        return selected

    def __len__(self) -> int:
        return self._cumulative_windows[-1]

    def _resolve_index(self, index: int) -> tuple[Episode, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self._cumulative_windows, index)
        previous = 0 if episode_index == 0 else self._cumulative_windows[episode_index - 1]
        start = (index - previous) * self.sample_stride
        return self.episodes[episode_index], start

    def agent_count_for_index(self, index: int) -> int:
        episode, _ = self._resolve_index(index)
        return int(episode.agent_count)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        previous = 0
        for cumulative in self._cumulative_windows:
            blocks.append((previous, cumulative - previous))
            previous = cumulative
        return blocks

    def _permutation(self, index: int, agent_count: int) -> torch.Tensor:
        if not self.permute_agents:
            return torch.arange(agent_count)
        generator = torch.Generator()
        generator.manual_seed(self.permutation_seed + int(index))
        return torch.randperm(agent_count, generator=generator)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, start = self._resolve_index(index)
        with h5py.File(episode.path, "r") as handle:
            group = handle[episode.trajectory]
            agent_names = self._agent_names(group)
            action = np.stack(
                [
                    self._action_dataset(group, agent)[start : start + self.chunk_length]
                    for agent in agent_names
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            state = np.stack(
                [
                    np.concatenate(
                        [
                            group[self._state_path(group, agent, "qpos")][start],
                            group[self._state_path(group, agent, "qvel")][start],
                        ],
                        axis=-1,
                    )
                    for agent in agent_names
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            image_indices = [start + offset for offset in self.image_offsets]
            images = np.asarray(group[self._camera_path_for_group(group)][image_indices], dtype=np.uint8)

        action_tensor = torch.from_numpy(np.array(action, copy=True))
        state_tensor = torch.from_numpy(np.array(state, copy=True))
        gaussian = self._gaussian_cache.get_agents(
            episode.source_path,
            episode.trajectory,
            start,
            agent_names,
        )
        if not bool(torch.isfinite(action_tensor).all()):
            raise ValueError(f"Non-finite action in {episode.source_path}:{episode.trajectory}:{start}")
        if not bool(torch.isfinite(state_tensor).all()):
            raise ValueError(f"Non-finite state in {episode.source_path}:{episode.trajectory}:{start}")
        if not bool(torch.isfinite(gaussian).all()):
            raise ValueError(f"Non-finite Gaussian in {episode.source_path}:{episode.trajectory}:{start}")

        permutation = self._permutation(index, episode.agent_count)
        action_tensor = action_tensor[:, permutation]
        state_tensor = state_tensor[permutation]
        gaussian = gaussian[permutation]
        packed_action = action_tensor.permute(1, 0, 2).reshape(episode.agent_count * self.chunk_length, 8)
        video = torch.from_numpy(np.array(images.transpose(3, 0, 1, 2), copy=True))

        return {
            "ai_caption": self.task_instructions.get(episode.task, episode.task),
            "video": video,
            "action": packed_action,
            "conditioning_fps": torch.tensor(round(self.conditioning_fps), dtype=torch.long),
            "mode": "policy",
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": "third_person_view",
            "idle_frames": torch.tensor(0, dtype=torch.long),
            "agent_count": torch.tensor(episode.agent_count, dtype=torch.long),
            "action_horizon_per_agent": torch.tensor(self.chunk_length, dtype=torch.long),
            "agent_state": state_tensor,
            "agent_gaussian": gaussian,
            "action_token_mask": torch.ones(episode.agent_count * self.chunk_length, dtype=torch.bool),
        }

    def __getstate__(self):
        return dict(self.__dict__)

    def __del__(self):
        cache = getattr(self, "_gaussian_cache", None)
        if cache is not None:
            cache.close()
