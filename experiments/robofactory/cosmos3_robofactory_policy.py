#!/usr/bin/env python3
"""RoboFactory-side client for the Cosmos3 multi-robot action server."""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from experiments.robofactory.imagewam_robofactory_policy import (
    TASK_INSTRUCTIONS,
    global_rgb_tensor,
    ordered_agent_names,
    raw_agent_state,
)
from experiments.robofactory.online_gaussian import (
    NOPOSPLAT_CHECKPOINT_BYTES,
    POLICY_LIGHTNING_COMMIT,
    OnlineNoPoSplatTeacher,
    encode_compact_agent_gaussian,
)


def _base64_tensor(tensor: torch.Tensor, dtype: np.dtype) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=dtype)
    return base64.b64encode(array.tobytes(order="C")).decode("ascii")


def _base64_global_png(observation: Mapping[str, Any]) -> str:
    image = global_rgb_tensor(observation)[0]
    image = image.add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image.permute(1, 2, 0).numpy()).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _request_json(url: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"Cosmos3 server returned {type(result).__name__}, expected object")
    if result.get("error"):
        raise RuntimeError(f"Cosmos3 server error: {result['error']}")
    return result


class Cosmos3RoboFactoryPolicy:
    """Expose remote Cosmos3 predictions through the evaluator policy protocol."""

    def __init__(
        self,
        *,
        server_url: str,
        policy_lightning_repo: str | Path,
        noposplat_checkpoint: str | Path,
        task_name: str,
        policy_lightning_commit: str = POLICY_LIGHTNING_COMMIT,
        noposplat_checkpoint_bytes: int = NOPOSPLAT_CHECKPOINT_BYTES,
        device: str = "cuda:0",
        seed: int = 10000,
        request_timeout: float = 900.0,
    ) -> None:
        if task_name not in TASK_INSTRUCTIONS:
            raise KeyError(f"Unsupported task {task_name!r}")
        self.server_url = server_url.rstrip("/")
        self.task_name = task_name
        self.seed = int(seed)
        self.request_timeout = float(request_timeout)
        self.gaussian_teacher = OnlineNoPoSplatTeacher(
            repo_path=policy_lightning_repo,
            checkpoint_path=noposplat_checkpoint,
            expected_commit=policy_lightning_commit,
            expected_checkpoint_bytes=noposplat_checkpoint_bytes,
            device=device,
        )
        with urllib.request.urlopen(f"{self.server_url}/info", timeout=60.0) as response:
            self.server_info = json.loads(response.read().decode("utf-8"))
        if self.server_info.get("adapter") != "RoboFactoryActionModelService":
            raise ValueError(f"Unexpected Cosmos3 server: {self.server_info}")
        self._observation: Mapping[str, Any] | None = None
        self._episode_seed = self.seed
        self._query_index = 0

    def start_episode(self, policy_seed: int | None = None) -> None:
        self._observation = None
        self._episode_seed = self.seed if policy_seed is None else int(policy_seed)
        self._query_index = 0

    def update_obs(
        self,
        observation: Mapping[str, Any],
        env_state: Mapping[str, Any] | None = None,
    ) -> None:
        del env_state
        names = ordered_agent_names(observation)
        if len(names) not in (1, 2, 3, 4):
            raise ValueError(f"Checkpoint supports N=1/2/3/4, got N={len(names)}")
        raw_agent_state(observation, names)
        global_rgb_tensor(observation)
        self._observation = observation

    @torch.inference_mode()
    def get_action_with_trace(self) -> dict[str, Any]:
        if self._observation is None:
            raise RuntimeError("No observation is recorded; call update_obs first")
        names = ordered_agent_names(self._observation)
        state = raw_agent_state(self._observation, names)
        gaussian = encode_compact_agent_gaussian(
            self.gaussian_teacher, self._observation, names
        )
        query_index = self._query_index
        query_seed = self._episode_seed + query_index
        result = _request_json(
            f"{self.server_url}/predict",
            {
                "image": _base64_global_png(self._observation),
                "prompt": TASK_INSTRUCTIONS[self.task_name],
                "agent_count": len(names),
                "agent_state": _base64_tensor(state, np.dtype("<f4")),
                "agent_gaussian": _base64_tensor(gaussian, np.dtype("<f2")),
                "query_seed": query_seed,
            },
            self.request_timeout,
        )
        width = len(names) * 8
        normalized = np.asarray(result["normalized"], dtype=np.float32)
        denormalized = np.asarray(result["denormalized"], dtype=np.float32)
        expected = (32, width)
        if normalized.shape != expected or denormalized.shape != expected:
            raise ValueError(
                f"Cosmos3 action views must both be {expected}, got "
                f"{normalized.shape} and {denormalized.shape}"
            )
        if not np.isfinite(normalized).all() or not np.isfinite(denormalized).all():
            raise FloatingPointError("Cosmos3 server returned non-finite actions")
        self._query_index += 1
        return {
            "normalized": normalized,
            "denormalized": denormalized,
            "query_index": query_index,
            "query_seed": query_seed,
        }

    def get_action(self) -> np.ndarray:
        return self.get_action_with_trace()["denormalized"]

    def provenance(self) -> dict[str, Any]:
        return {
            "adapter": "Cosmos3RoboFactoryPolicy",
            "task_name": self.task_name,
            "server_url": self.server_url,
            "server_info": self.server_info,
            "online_gaussian": self.gaussian_teacher.provenance(),
            "action_layout": "agent_major_model_to_chronological_environment",
            "state_normalization": "none_raw_qpos9_qvel9",
        }
