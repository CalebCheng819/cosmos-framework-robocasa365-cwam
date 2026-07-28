# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed WebSocket policy adapter for CWAM/Cosmos3 RoboCasa365 evaluation.

The DeMiAn RoboCasa evaluator sends OpenPI msgpack observations with three
camera images, a 16-D state vector, and a prompt.  The CWAM training recipe,
however, conditions only on the left agent view concatenated with the wrist
view and predicts normalized relative 10-D actions:

``[dpos(3), rot6d-columns(6), gripper(1)]``.

This adapter preserves that exact training contract and returns the evaluator's
required relative 7-D actions:

``[dpos(3), axis-angle(3), gripper(1)]``.

The right agent view and state are validated because they are part of the
wire protocol, but they are deliberately not introduced as new model
conditioning signals.
"""

from cosmos_framework.inference.common.init import init_script

init_script()

import base64
import hashlib
import io
import json
import os
import re
import socket
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic
import torch
import tyro
from PIL import Image

from cosmos_framework.data.generator.action.datasets.robocasa365_segment_dataset import (
    RoboCasa365SegmentDataset,
)
from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.pose_utils import convert_rotation
from cosmos_framework.inference.common.args import tyro_cli
from cosmos_framework.scripts.action_policy_server_libero import ActionModelService, ActionServerArgs
from cosmos_framework.scripts.action_policy_server_robolab import _load_openpi_websocket_policy_server
from cosmos_framework.utils import log

_CAMERA_KEYS = (
    "observation/agentview_left",
    "observation/eye_in_hand",
    "observation/agentview_right",
)
_STATE_KEY = "observation/state"
_RAW_ACTION_DIM = 10
_ENV_ACTION_DIM = 7
_LAUNCH_NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
_TRAINING_CAMERA_DESCRIPTION = (
    "The left half shows the left third-person agent view; "
    "the right half shows the wrist-mounted camera."
)


def _require_rgb_uint8_camera(
    value: Any,
    *,
    key: str,
    height: int,
    width: int,
) -> np.ndarray:
    """Validate one wire camera without silently changing dtype or geometry."""
    image = np.asarray(value)
    expected_shape = (height, width, 3)
    if image.shape != expected_shape:
        raise ValueError(f"{key!r} must have exact shape {expected_shape}, got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"{key!r} must have dtype uint8, got {image.dtype}")
    return np.ascontiguousarray(image)


def _validate_state16(value: Any) -> np.ndarray:
    """Validate the DeMiAn 16-D proprio field, which training did not consume."""
    state = np.asarray(value)
    if state.shape != (16,):
        raise ValueError(f"{_STATE_KEY!r} must have exact shape (16,), got {state.shape}")
    if not np.issubdtype(state.dtype, np.floating):
        raise TypeError(f"{_STATE_KEY!r} must have a floating dtype, got {state.dtype}")
    if not np.isfinite(state).all():
        raise ValueError(f"{_STATE_KEY!r} contains non-finite values")
    return np.ascontiguousarray(state, dtype=np.float64)


def _extract_training_observation(
    obs: dict[str, Any],
    *,
    camera_height: int,
    camera_width: int,
) -> tuple[np.ndarray, str]:
    """Return the exact left+wrist training image and validated prompt."""
    missing = [key for key in (*_CAMERA_KEYS, _STATE_KEY, "prompt") if key not in obs]
    if missing:
        raise KeyError(f"RoboCasa365 observation is missing required keys: {missing}")

    prompt = obs["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' must be a non-empty string")
    reasoning_text = obs.get("reasoning_text")
    if reasoning_text not in (None, ""):
        raise ValueError(
            "'reasoning_text' is not part of the causal CWAM training contract; "
            "formal evaluation must not inject it"
        )

    left = _require_rgb_uint8_camera(
        obs[_CAMERA_KEYS[0]],
        key=_CAMERA_KEYS[0],
        height=camera_height,
        width=camera_width,
    )
    wrist = _require_rgb_uint8_camera(
        obs[_CAMERA_KEYS[1]],
        key=_CAMERA_KEYS[1],
        height=camera_height,
        width=camera_width,
    )
    # Validate the evaluator's complete observation contract, but do not add a
    # third view that the model never saw during training.
    _require_rgb_uint8_camera(
        obs[_CAMERA_KEYS[2]],
        key=_CAMERA_KEYS[2],
        height=camera_height,
        width=camera_width,
    )
    _validate_state16(obs[_STATE_KEY])

    concat = np.concatenate((left, wrist), axis=1)
    return np.ascontiguousarray(concat), prompt.strip()


def _encode_png_base64(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG", optimize=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _rot6d10_to_axisangle7(action: Any) -> np.ndarray:
    """Convert denormalized CWAM actions to the DeMiAn environment contract."""
    raw = np.asarray(action, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != _RAW_ACTION_DIM:
        raise ValueError(f"Expected denormalized action [T,{_RAW_ACTION_DIM}], got {raw.shape}")
    if raw.shape[0] < 1:
        raise ValueError("Predicted action chunk must contain at least one step")
    if not np.isfinite(raw).all():
        raise ValueError("Predicted action chunk contains non-finite values")

    rot6d = raw[:, 3:9]
    col0 = rot6d[:, :3]
    col1 = rot6d[:, 3:]
    col0_norm = np.linalg.norm(col0, axis=-1)
    col0_unit = col0 / np.maximum(col0_norm[:, None], 1e-12)
    residual = col1 - np.sum(col0_unit * col1, axis=-1, keepdims=True) * col0_unit
    residual_norm = np.linalg.norm(residual, axis=-1)
    if np.any(col0_norm < 1e-6) or np.any(residual_norm < 1e-6):
        bad = np.flatnonzero((col0_norm < 1e-6) | (residual_norm < 1e-6))
        raise ValueError(f"Predicted rot6d is degenerate at action rows {bad[:16].tolist()}")

    axis_angle = np.asarray(
        convert_rotation(rot6d, input_format="rot6d", output_format="axisangle", normalize_matrix=True),
        dtype=np.float32,
    )
    converted = np.concatenate((raw[:, :3], axis_angle, raw[:, 9:10]), axis=-1)
    if converted.shape != (raw.shape[0], _ENV_ACTION_DIM) or not np.isfinite(converted).all():
        raise RuntimeError(f"Invalid converted RoboCasa action shape/content: {converted.shape}")
    # The converter that built the training view copied the raw gripper channel
    # unchanged.  Preserve it here: unlike the DROID RoboLab server, no 1-x or
    # sign inversion is part of this dataset contract.
    return np.ascontiguousarray(converted, dtype=np.float32)


def _load_required_quantile_rot_normalizer(stats_path: Path) -> Any:
    """Load normalization through the same resolver used by training."""
    path = Path(stats_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RoboCasa365 action stats file not found: {path}")
    normalizer = RoboCasa365SegmentDataset._resolve_action_normalizer("quantile_rot", path)
    if normalizer is None:
        raise RuntimeError(f"quantile_rot unexpectedly resolved to no normalizer: {path}")
    return normalizer


class RoboCasa365ServerArgs(ActionServerArgs):
    """Cosmos inference options plus the immutable RoboCasa365 wire contract."""

    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    fps: int = 20
    """Training FPS; formal RoboCasa365 checkpoints require exactly 20."""
    action_chunk_size: int | None = 16
    """Training action horizon; formal RoboCasa365 checkpoints require exactly 16."""
    raw_action_dim: int | None = 10
    """Denormalized training action width; formal RoboCasa365 checkpoints require exactly 10."""
    action_normalization: Literal["quantile_rot"] = "quantile_rot"
    """Must remain quantile_rot to match the formal training recipe."""
    format_prompt_as_json: bool | None = True
    """Must remain true to reproduce the training prompt formatter."""
    camera_height: int = 256
    """Height of each DeMiAn camera image after its client-side resize."""
    camera_width: int = 256
    """Width of each DeMiAn camera image after its client-side resize."""
    domain_name: str = "robocasa365"
    """Action domain ID used by the formal training recipe."""
    launch_nonce: str
    """Per-process 256-bit nonce used to bind receipts to this exact launch."""


class RoboCasa365PolicyService(ActionModelService):
    """OpenPI-compatible service with strict RoboCasa365 pre/post processing."""

    def __init__(self, args: RoboCasa365ServerArgs) -> None:
        if _LAUNCH_NONCE_RE.fullmatch(args.launch_nonce) is None:
            raise ValueError("RoboCasa365 formal serving requires a 64-character lowercase hex launch nonce")
        if args.fps != 20:
            raise ValueError(f"RoboCasa365 formal serving requires fps=20, got {args.fps}")
        if args.action_chunk_size != 16:
            raise ValueError(
                f"RoboCasa365 formal serving requires action_chunk_size=16, got {args.action_chunk_size}"
            )
        if args.raw_action_dim != _RAW_ACTION_DIM:
            raise ValueError(f"RoboCasa365 formal serving requires raw_action_dim=10, got {args.raw_action_dim}")
        if args.action_normalization != "quantile_rot":
            raise ValueError(
                "RoboCasa365 formal serving requires action_normalization='quantile_rot', "
                f"got {args.action_normalization!r}"
            )
        if args.format_prompt_as_json is not True:
            raise ValueError("RoboCasa365 formal serving requires format_prompt_as_json=true")
        if args.camera_height != 256 or args.camera_width != 256:
            raise ValueError(
                "RoboCasa365 formal serving requires each client camera to be exactly 256x256, "
                f"got {args.camera_height}x{args.camera_width}"
            )
        if args.domain_name != "robocasa365":
            raise ValueError(f"RoboCasa365 formal serving requires domain_name='robocasa365', got {args.domain_name!r}")
        if args.action_stats_path is None:
            raise ValueError("--action-stats-path is required for fail-closed RoboCasa365 serving")

        self.camera_height = int(args.camera_height)
        self.camera_width = int(args.camera_width)
        self.domain_name = args.domain_name
        self.launch_nonce = args.launch_nonce
        super().__init__(args)

        if self.raw_action_dim != _RAW_ACTION_DIM:
            raise RuntimeError(f"Resolved action stats width must be 10, got {self.raw_action_dim}")
        if self.cfg.action_chunk_size != 16 or self.cfg.fps != 20:
            raise RuntimeError(
                "Loaded inference config drifted from the RoboCasa365 training contract: "
                f"chunk={self.cfg.action_chunk_size}, fps={self.cfg.fps}"
            )
        self._prompt_json_formatter = ActionPromptJsonFormatter(
            caption_key="ai_caption",
            include_idle_frames=False,
        )

    def _load_action_normalization_stats(self) -> None:
        """Replace the generic server's duplicated math with the training resolver."""
        stats_path = self.cfg.action_stats_path
        if stats_path is None:
            raise ValueError("--action-stats-path is required")
        self._action_normalizer = _load_required_quantile_rot_normalizer(Path(stats_path))
        self.action_normalization = "quantile_rot"
        self.raw_action_dim = _RAW_ACTION_DIM

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        model_action = action[..., :_RAW_ACTION_DIM]
        denormalized = self._action_normalizer.denormalize_action(model_action)
        if denormalized.shape[-1] != _RAW_ACTION_DIM:
            raise RuntimeError(
                f"RoboCasa365 action normalizer changed width to {denormalized.shape[-1]}"
            )
        return denormalized

    def _build_json_prompt(self, prompt: str, *, video: torch.Tensor, image_size: torch.Tensor) -> str:
        data_dict: dict[str, Any] = {
            "ai_caption": prompt,
            "viewpoint": "concat_view",
            "additional_view_description": _TRAINING_CAMERA_DESCRIPTION,
            "video": video,
            "image_size": image_size,
            "conditioning_fps": torch.tensor(self.cfg.fps, dtype=torch.long),
            "mode": "policy",
            "action": torch.zeros(
                (self.cfg.action_chunk_size, self.cfg.max_action_dim),
                dtype=torch.float32,
            ),
        }
        formatted = self._prompt_json_formatter(data_dict)["ai_caption"]
        if not isinstance(formatted, dict):
            raise RuntimeError("RoboCasa365 JSON prompt formatter did not return a dictionary")
        return json.dumps(formatted)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        image, prompt = _extract_training_observation(
            obs,
            camera_height=self.camera_height,
            camera_width=self.camera_width,
        )
        request = {
            "image": _encode_png_base64(image),
            "prompt": prompt,
            "domain_name": self.domain_name,
            "image_size": self.camera_height,
        }
        model_response = self.predict_policy_batch([request])
        batches = model_response.get("actions")
        if not isinstance(batches, list) or len(batches) != 1:
            raise RuntimeError("Cosmos policy batch response must contain exactly one action chunk")
        actions = _rot6d10_to_axisangle7(batches[0])
        if actions.shape[0] != self.cfg.action_chunk_size:
            raise RuntimeError(
                f"Cosmos policy returned {actions.shape[0]} actions; expected {self.cfg.action_chunk_size}"
            )
        return {"actions": actions}

    def reset(self, _reset_info: dict[str, Any] | None = None) -> None:
        """The formal service is stateless across DeMiAn episodes."""
        return None

    def metadata(self) -> dict[str, Any]:
        stats_path = Path(self.cfg.action_stats_path).expanduser().resolve()
        source_module_path = Path(__file__).resolve()
        return {
            "schema_version": 1,
            "service": "cwam_cosmos3_robocasa365",
            "checkpoint": self.cfg.checkpoint_dir,
            "action_stats_path": str(stats_path),
            "source_module_path": str(source_module_path),
            "source_module_sha256": hashlib.sha256(source_module_path.read_bytes()).hexdigest(),
            "server_pid": os.getpid(),
            "launch_nonce": self.launch_nonce,
            "wire_inputs": [*_CAMERA_KEYS, _STATE_KEY, "prompt"],
            "conditioning_views": ["observation/agentview_left", "observation/eye_in_hand"],
            "state_conditioning": False,
            "model_action": "[dpos3,rot6d_columns6,gripper1]",
            "output_action": "[dpos3,axisangle3,gripper1]",
            "gripper_transform": "identity",
            "fps": self.cfg.fps,
            "action_chunk_size": self.cfg.action_chunk_size,
            "format_prompt_as_json": True,
            "sampler": str(self.setup_args.sampler),
            "policy_seed": self.cfg.seed,
            "guidance": self.cfg.guidance,
            "num_steps": self.cfg.num_steps,
            "reuse_policy_seed_per_request": True,
            "use_ema_weights": bool(self.setup_args.use_ema_weights),
        }


def serve(args: RoboCasa365ServerArgs) -> None:
    hostname = socket.gethostname()
    log.info(f"[robocasa365-policy-server] starting host={hostname} bind={args.host}:{int(args.port)}")
    service = RoboCasa365PolicyService(args)
    server_cls = _load_openpi_websocket_policy_server()
    server_cls(
        policy=service,
        host=args.host,
        port=int(args.port),
        metadata=service.metadata(),
    ).serve_forever()


def main() -> None:
    cascade_subcommand_args = getattr(
        tyro.conf,
        "CascadeSubcommandArgs",
        tyro.conf.ConsolidateSubcommandArgs,
    )
    args = tyro_cli(
        RoboCasa365ServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            cascade_subcommand_args,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    serve(args)


if __name__ == "__main__":
    main()
