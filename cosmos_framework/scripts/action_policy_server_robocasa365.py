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
import inspect
import io
import json
import os
import re
import socket
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pydantic
import torch
import tyro
from PIL import Image
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from cosmos_framework.data.generator.action.datasets.robocasa365_segment_dataset import (
    RoboCasa365SegmentDataset,
)
from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.pose_utils import convert_rotation
from cosmos_framework.inference.common.args import tyro_cli
from cosmos_framework.model.generator.mot.action_relevance import (
    set_action_relevance_fixed_routed_c,
)
from cosmos_framework.model.generator.mot.action_relevance_replay import (
    FixedInterventionReplayBank,
)
from cosmos_framework.scripts.action_policy_server_libero import ActionModelService, ActionServerArgs
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
_ONLINE_WAN2PT2_TARGET = "cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface"
_TRAINING_CAMERA_DESCRIPTION = (
    "The left half shows the left third-person agent view; the right half shows the wrist-mounted camera."
)
_DISTRIBUTED_COMMAND_SCHEMA_VERSION = 2
_DISTRIBUTED_COMMAND_OPS = frozenset(("infer", "reset", "shutdown"))
_CANONICAL_REQUEST_KEYS = frozenset(("image", "prompt", "domain_name", "image_size"))
_CONTROL_REQUEST_KEYS = frozenset(("__cwam_control__", "launch_nonce"))
_WIRE_EVIDENCE_KEYS = frozenset(
    ("cwam_episode_key", "cwam_task_id", "cwam_control_step", "cwam_post_terminal")
)


def _load_openpi_websocket_policy_server() -> type[Any]:
    """Load the WebSocket implementation vendored by the pinned DeMiAn tree.

    The generic RoboLab loader prefers the separately distributed
    ``openpi_server`` package when it happens to be installed.  Formal
    RoboCasa365 evaluation instead imports the ``openpi`` module placed on
    ``PYTHONPATH`` from the checksum-bound DeMiAn checkout, so an unrelated
    environment package cannot silently change the wire implementation.
    """

    try:
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RoboCasa365 formal serving requires the pinned DeMiAn "
            "openpi.serving.websocket_policy_server module on PYTHONPATH"
        ) from exc
    return WebsocketPolicyServer


def _canonical_non_symlink_path(path: Path, *, kind: Literal["file", "directory"]) -> Path:
    """Require an already-canonical absolute regular file or directory."""
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise ValueError(f"{kind} path must be absolute and canonical: {supplied}")
    try:
        canonical = supplied.resolve(strict=True)
        metadata = supplied.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"{kind} path does not exist: {supplied}") from None
    if supplied != canonical or stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{kind} path must be canonical and non-symlink: {supplied}")
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"file path must name a regular file: {supplied}")
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"directory path must name a directory: {supplied}")
    return canonical


def _sha256_regular_file(path: Path) -> str:
    """Hash a canonical regular file while rejecting an in-flight replacement."""
    canonical = _canonical_non_symlink_path(path, kind="file")
    before = canonical.stat()
    digest = hashlib.sha256()
    with canonical.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    after = canonical.stat()
    identity = lambda value: (  # noqa: E731 - compact stable-file identity.
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError(f"File changed while hashing: {canonical}")
    return digest.hexdigest()


def _websocket_server_receipt(server_cls: type[Any]) -> dict[str, Any]:
    """Bind the selected WebSocket class to its exact source file bytes."""

    source_file = inspect.getsourcefile(server_cls)
    if source_file is None:
        raise RuntimeError("RoboCasa365 WebSocket server class has no inspectable Python source")
    source_path = _canonical_non_symlink_path(Path(source_file), kind="file")
    expected_module = "openpi.serving.websocket_policy_server"
    expected_qualname = "WebsocketPolicyServer"
    if server_cls.__module__ != expected_module or server_cls.__qualname__ != expected_qualname:
        raise RuntimeError(
            "RoboCasa365 formal serving selected an unexpected WebSocket class: "
            f"{server_cls.__module__}.{server_cls.__qualname__}"
        )
    return {
        "module": expected_module,
        "qualname": expected_qualname,
        "source_path": str(source_path),
        "source_sha256": _sha256_regular_file(source_path),
    }


def _dcp_selected_weight_receipt(
    checkpoint_path: Path,
    *,
    use_ema_weights: bool,
    require_serving_only_ema_bf16: bool,
    expected_ema_key_count: int,
) -> dict[str, Any]:
    """Describe the exact DCP namespace selected by the inference load planner.

    Formal serving uses a derived checkpoint that intentionally contains only
    ``net_ema.*`` tensors, converted from FP32 to BF16. Reading the live DCP
    metadata after model construction proves that the successfully loaded
    service was pointed at the manifest-bound namespace instead of merely
    echoing the requested ``use_ema_weights`` boolean.
    """
    if type(expected_ema_key_count) is not int or expected_ema_key_count <= 0:
        raise ValueError("expected_ema_key_count must be a positive integer")
    root = _canonical_non_symlink_path(checkpoint_path, kind="directory")
    metadata_path = _canonical_non_symlink_path(root / ".metadata", kind="file")
    metadata = FileSystemReader(str(root)).read_metadata()
    state_metadata = metadata.state_dict_metadata
    if not isinstance(state_metadata, dict) or not state_metadata:
        raise RuntimeError("RoboCasa365 serving checkpoint has no DCP state metadata")

    selected_prefix = "net_ema." if use_ema_weights else "net."
    selected_keys = sorted(key for key in state_metadata if key.startswith(selected_prefix))
    regular_keys = sorted(key for key in state_metadata if key.startswith("net."))
    ema_keys = sorted(key for key in state_metadata if key.startswith("net_ema."))
    other_keys = sorted(set(state_metadata) - set(regular_keys) - set(ema_keys))
    if not selected_keys:
        raise RuntimeError(f"RoboCasa365 checkpoint has no selected {selected_prefix} tensors")

    signature: list[dict[str, Any]] = []
    dtype_census: dict[str, int] = {}
    logical_bytes = 0
    for key in selected_keys:
        value = state_metadata[key]
        if not isinstance(value, TensorStorageMetadata):
            raise RuntimeError(f"Selected checkpoint entry is not tensor metadata: {key}")
        dtype = str(value.properties.dtype)
        dtype_census[dtype] = dtype_census.get(dtype, 0) + 1
        shape = [int(size) for size in value.size]
        logical_bytes += int(value.size.numel()) * value.properties.dtype.itemsize
        signature.append({"key": key, "shape": shape, "dtype": dtype})

    signature_sha256 = hashlib.sha256(
        json.dumps(signature, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    serving_only_selected_tree = (
        not other_keys
        and (not regular_keys if selected_prefix == "net_ema." else not ema_keys)
        and len(selected_keys) == len(state_metadata)
    )
    if require_serving_only_ema_bf16:
        if not use_ema_weights:
            raise RuntimeError("formal serving requires use_ema_weights=true")
        expected_dtype_census = {"torch.bfloat16": expected_ema_key_count}
        if len(ema_keys) != expected_ema_key_count or regular_keys or other_keys:
            raise RuntimeError(
                "formal checkpoint must contain exactly the manifest-bound number "
                f"of net_ema.* tensors ({expected_ema_key_count}) "
                f"and no other trees, got ema={len(ema_keys)} regular={len(regular_keys)} "
                f"other={len(other_keys)}"
            )
        if dtype_census != expected_dtype_census:
            raise RuntimeError(
                "formal net_ema.* source tensors must all be BF16: "
                f"expected={expected_dtype_census}, actual={dtype_census}"
            )

    return {
        "schema_version": 1,
        "checkpoint_path": str(root),
        "checkpoint_metadata_sha256": _sha256_regular_file(metadata_path),
        "selected_weight_tree": selected_prefix.removesuffix("."),
        "selected_source_prefix": selected_prefix,
        "selected_key_count": len(selected_keys),
        "selected_source_dtype_census": dtype_census,
        "selected_logical_bytes": logical_bytes,
        "selected_fqn_shape_dtype_sha256": signature_sha256,
        "regular_key_count": len(regular_keys),
        "ema_key_count": len(ema_keys),
        "other_key_count": len(other_keys),
        "serving_only_selected_tree": serving_only_selected_tree,
        "load_planner": (
            "CustomLoadPlanner(load_ema_to_reg=True)" if use_ema_weights else "CustomLoadPlanner(load_ema_to_reg=False)"
        ),
    }


def _distributed_rank_and_world_size() -> tuple[int, int]:
    """Return the initialized process-group identity, or the single-rank identity."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0, 1
    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    if world_size < 1 or not 0 <= rank < world_size:
        raise RuntimeError(f"Invalid distributed identity: rank={rank}, world_size={world_size}")
    return rank, world_size


def _declared_world_size() -> int:
    """Resolve torchrun world size before the model setup has been built."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return _distributed_rank_and_world_size()[1]
    raw_world_size = os.environ.get("WORLD_SIZE")
    if raw_world_size in (None, ""):
        return 1
    try:
        world_size = int(raw_world_size)
    except ValueError as exc:
        raise ValueError(f"WORLD_SIZE must be a positive integer, got {raw_world_size!r}") from exc
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be a positive integer, got {world_size}")
    return world_size


def _validate_fsdp_topology(setup_args: Any, *, world_size: int) -> None:
    """Require one pure FSDP shard group spanning the complete worker group."""
    if world_size not in (1, 4):
        raise RuntimeError(
            "RoboCasa365 serving supports either the existing single-rank path "
            f"or the DSW four-rank FSDP path, got world_size={world_size}"
        )
    expected = {
        "dp_shard_size": world_size,
        "dp_replicate_size": 1,
        "cp_size": 1,
        "cfgp_size": 1,
    }
    actual = {key: int(getattr(setup_args, key)) for key in expected}
    if actual != expected:
        raise RuntimeError(f"RoboCasa365 distributed serving requires pure FSDP topology {expected}, got {actual}")


def _validate_canonical_policy_request(request: Any) -> dict[str, Any]:
    """Validate the compact rank-0 request before any rank enters inference."""
    if not isinstance(request, dict):
        raise TypeError(f"Canonical RoboCasa365 request must be a dict, got {type(request).__name__}")
    if set(request) != _CANONICAL_REQUEST_KEYS:
        raise ValueError(
            "Canonical RoboCasa365 request keys drifted: "
            f"expected={sorted(_CANONICAL_REQUEST_KEYS)}, got={sorted(request)}"
        )
    if not isinstance(request["image"], str) or not request["image"]:
        raise ValueError("Canonical RoboCasa365 request 'image' must be a non-empty base64 string")
    if not isinstance(request["prompt"], str) or not request["prompt"]:
        raise ValueError("Canonical RoboCasa365 request 'prompt' must be a non-empty string")
    if request["domain_name"] != "robocasa365":
        raise ValueError(
            f"Canonical RoboCasa365 request requires domain_name='robocasa365', got {request['domain_name']!r}"
        )
    if request["image_size"] != 256:
        raise ValueError(f"Canonical RoboCasa365 request requires image_size=256, got {request['image_size']!r}")
    return request


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
            "'reasoning_text' is not part of the causal CWAM training contract; formal evaluation must not inject it"
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
    expected_ema_key_count: int = 809
    """Manifest-bound EMA tensor count; 809 preserves the published R14 contract."""
    wan2pt2_vae_path: Path | None = None
    """Optional canonical Wan2.2 VAE file override for portable serving."""
    qwen_tokenizer_path: Path | None = None
    """Optional canonical local Qwen tokenizer directory override."""
    action_relevance_intervention_bank: Path | None = None
    """Strict fixed routed-C bank required by shuffle and cross_episode replay."""

    def resolve_external_model_paths(self) -> tuple[Path | None, Path | None]:
        """Resolve optional portable model assets without changing H defaults."""
        vae_path = (
            _canonical_non_symlink_path(self.wan2pt2_vae_path, kind="file")
            if self.wan2pt2_vae_path is not None
            else None
        )
        qwen_path = (
            _canonical_non_symlink_path(self.qwen_tokenizer_path, kind="directory")
            if self.qwen_tokenizer_path is not None
            else None
        )
        return vae_path, qwen_path

    def build_setup_overrides(self) -> Any:
        """Build the fail-closed online action-serving configuration."""
        overrides = super().build_setup_overrides()
        overrides.guardrails = False
        # Never inherit the Omni latency preset here.  Every torchrun rank must
        # participate in one FSDP shard group for every policy request; context,
        # CFG, or replicated data parallelism would require a different request
        # dispatch contract.
        world_size = _declared_world_size()
        overrides.parallelism_preset = "throughput"
        overrides.dp_shard_size = world_size
        overrides.dp_replicate_size = 1
        overrides.cp_size = 1
        overrides.cfgp_size = 1
        _validate_fsdp_topology(overrides, world_size=world_size)
        # R14 was trained from checksum-bound, precomputed Wan latents and its
        # serialized config therefore names the metadata-only tokenizer.  A
        # closed-loop evaluator supplies fresh pixels, so serving must restore
        # the weight-compatible online Wan2.2 encoder.  Keep this override
        # local to action serving; training continues to use the strict cache.
        experiment_overrides = [
            *overrides.experiment_overrides,
            f"model.config.tokenizer._target_={_ONLINE_WAN2PT2_TARGET}",
            "~model.config.tokenizer.latent_ch",
        ]
        vae_path, qwen_path = self.resolve_external_model_paths()
        if vae_path is not None:
            experiment_overrides.append(f"model.config.tokenizer.vae_path={vae_path}")
        if qwen_path is not None:
            experiment_overrides.append(f"model.config.vlm_config.tokenizer.pretrained_model_name={qwen_path}")
        overrides.experiment_overrides = experiment_overrides
        return overrides


class RoboCasa365PolicyService(ActionModelService):
    """OpenPI-compatible service with strict RoboCasa365 pre/post processing."""

    def __init__(self, args: RoboCasa365ServerArgs) -> None:
        if _LAUNCH_NONCE_RE.fullmatch(args.launch_nonce) is None:
            raise ValueError("RoboCasa365 formal serving requires a 64-character lowercase hex launch nonce")
        if type(args.expected_ema_key_count) is not int or args.expected_ema_key_count <= 0:
            raise ValueError("RoboCasa365 formal serving requires a positive expected_ema_key_count")
        if args.fps != 20:
            raise ValueError(f"RoboCasa365 formal serving requires fps=20, got {args.fps}")
        if args.action_chunk_size != 16:
            raise ValueError(f"RoboCasa365 formal serving requires action_chunk_size=16, got {args.action_chunk_size}")
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
        replay_conditions = {"shuffle", "cross_episode"}
        if args.action_relevance_intervention in replay_conditions and args.action_relevance_intervention_bank is None:
            raise ValueError(
                f"RoboCasa365 {args.action_relevance_intervention} requires "
                "--action-relevance-intervention-bank"
            )
        if args.action_relevance_intervention not in replay_conditions and args.action_relevance_intervention_bank is not None:
            raise ValueError(
                "An action-relevance intervention bank is valid only for shuffle or cross_episode"
            )

        self.camera_height = int(args.camera_height)
        self.camera_width = int(args.camera_width)
        self.domain_name = args.domain_name
        self.launch_nonce = args.launch_nonce
        self.wan2pt2_vae_path, self.qwen_tokenizer_path = args.resolve_external_model_paths()
        self._evidence_episode = -1
        self._evidence_control_step = 0
        self._pending_evidence: dict[str, Any] | None = None
        self._action_relevance_intervention = args.action_relevance_intervention
        super().__init__(args)

        if args.action_relevance_intervention_bank is None:
            self.action_relevance_intervention_bank_path = None
            self._action_relevance_replay_bank = None
        else:
            bank_path = _canonical_non_symlink_path(
                args.action_relevance_intervention_bank,
                kind="file",
            )
            self.action_relevance_intervention_bank_path = bank_path
            self._action_relevance_replay_bank = FixedInterventionReplayBank.load(bank_path)

        self.distributed_rank, self.distributed_world_size = _distributed_rank_and_world_size()
        if self.distributed_rank != 0:
            # Every FSDP rank participates in routing, but only rank 0 owns the
            # evaluator observation and may persist evidence.
            self.action_relevance_trace_dir = None
        _validate_fsdp_topology(
            self.setup_args,
            world_size=self.distributed_world_size,
        )
        self.checkpoint_path = _canonical_non_symlink_path(Path(self.cfg.checkpoint_dir), kind="directory")
        self.training_config_path = _canonical_non_symlink_path(Path(self.setup_args.config_file), kind="file")
        self.action_stats_path = _canonical_non_symlink_path(Path(self.cfg.action_stats_path), kind="file")
        self._checkpoint_weight_receipt = _dcp_selected_weight_receipt(
            self.checkpoint_path,
            use_ema_weights=bool(self.setup_args.use_ema_weights),
            require_serving_only_ema_bf16=self.distributed_world_size > 1,
            expected_ema_key_count=args.expected_ema_key_count,
        )
        self._dispatch_lock = threading.Lock()
        self._next_distributed_sequence = 0
        self._shutdown_requested = False

        if self.setup_args.guardrails:
            raise RuntimeError("RoboCasa365 action serving must keep generative-media guardrails disabled")
        tokenizer_cls = type(self.model.tokenizer_vision_gen)
        tokenizer_type = f"{tokenizer_cls.__module__}.{tokenizer_cls.__qualname__}"
        if tokenizer_type != _ONLINE_WAN2PT2_TARGET:
            raise RuntimeError(
                f"RoboCasa365 online serving requires the pixel-encoding Wan2.2 tokenizer, got {tokenizer_type!r}"
            )
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

    def _make_observation_evidence(self, obs: dict[str, Any]) -> dict[str, Any]:
        state = _validate_state16(obs[_STATE_KEY])
        present = _WIRE_EVIDENCE_KEYS.intersection(obs)
        if present and present != _WIRE_EVIDENCE_KEYS:
            missing = sorted(_WIRE_EVIDENCE_KEYS - present)
            raise ValueError(f"RoboCasa365 CWAM evidence metadata is partial; missing={missing}")
        explicit = present == _WIRE_EVIDENCE_KEYS
        requires_explicit = (
            getattr(self, "action_relevance_trace_dir", None) is not None
            or getattr(self, "_action_relevance_intervention", "normal") in {"shuffle", "cross_episode"}
        )
        if requires_explicit and not explicit:
            raise ValueError(
                "RoboCasa365 trace/intervention requests require explicit episode, task, "
                "control-step, and terminal metadata"
            )
        if explicit:
            episode_key = obs["cwam_episode_key"]
            task_id = obs["cwam_task_id"]
            control_step = obs["cwam_control_step"]
            post_terminal = obs["cwam_post_terminal"]
            if not isinstance(episode_key, str) or not episode_key.strip():
                raise ValueError("cwam_episode_key must be a non-empty string")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("cwam_task_id must be a non-empty string")
            if type(control_step) is not int or control_step < 0:
                raise ValueError("cwam_control_step must be a non-negative integer")
            if type(post_terminal) is not bool:
                raise ValueError("cwam_post_terminal must be a bool")
            episode_key = episode_key.strip()
            task_id = task_id.strip()
        else:
            episode_key = self._current_evidence_episode_key()
            task_id = str(obs["prompt"]).strip()
            control_step = int(getattr(self, "_evidence_control_step", 0))
            post_terminal = False
        return {
            "episode_key": episode_key,
            "task_id": task_id,
            "control_step": control_step,
            "post_terminal": post_terminal,
            "evidence_source": "explicit_wire_v1" if explicit else "legacy_server_counter",
            "relevant_factors": state.tolist(),
            "relevant_fields": {"wire_state16": state.tolist()},
            "label_scope": "wire_state16_proprioceptive_proxy_no_object_ground_truth",
            "state_conditioning": False,
        }

    def _current_evidence_episode_key(self) -> str:
        episode = int(getattr(self, "_evidence_episode", -1))
        if episode < 0:
            episode = 0
            self._evidence_episode = episode
        return f"robocasa365:{episode:06d}"

    def _configure_replay_for_current_step(self, *, evidence: dict[str, Any]) -> tuple[str, str] | None:
        condition = getattr(self, "_action_relevance_intervention", "normal")
        if condition not in {"shuffle", "cross_episode"}:
            return None
        bank = getattr(self, "_action_relevance_replay_bank", None)
        if bank is None:
            raise RuntimeError(f"{condition} was selected without a loaded routed-C replay bank")
        layers, donor_episode_key, donor_task_id = bank.routed_c(
            condition=condition,
            episode_key=evidence["episode_key"],
            task_id=evidence["task_id"],
            control_step=evidence["control_step"],
        )
        configured = set_action_relevance_fixed_routed_c(self.model, layers)
        if configured == 0:
            raise RuntimeError("The loaded model has no ActionRelevanceAdapter for fixed replay")
        return donor_episode_key, donor_task_id

    def _advance_evidence_control_step(self) -> None:
        self._evidence_control_step = int(getattr(self, "_evidence_control_step", 0)) + 1

    def _write_action_relevance_trace(
        self,
        *,
        requests: list[dict[str, Any]],
        actions: list[list[list[float]]],
        records: list[dict[str, Any]],
    ) -> None:
        if self.distributed_rank != 0 or self.action_relevance_trace_dir is None:
            return
        if self._pending_evidence is None:
            raise RuntimeError("RoboCasa365 trace capture has no rank-0 observation evidence")
        traced_requests = []
        for request in requests:
            traced = dict(request)
            traced["evidence"] = dict(self._pending_evidence)
            traced_requests.append(traced)
        super()._write_action_relevance_trace(
            requests=traced_requests,
            actions=actions,
            records=records,
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
            raise RuntimeError(f"RoboCasa365 action normalizer changed width to {denormalized.shape[-1]}")
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

    def _canonicalize_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Validate the wire observation on rank 0 and remove unused fields."""
        if not isinstance(obs, dict):
            raise TypeError(f"RoboCasa365 observation must be a dict, got {type(obs).__name__}")
        image, prompt = _extract_training_observation(
            obs,
            camera_height=self.camera_height,
            camera_width=self.camera_width,
        )
        return _validate_canonical_policy_request(
            {
                "image": _encode_png_base64(image),
                "prompt": prompt,
                "domain_name": self.domain_name,
                "image_size": self.camera_height,
            }
        )

    def _convert_model_response(self, model_response: Any) -> dict[str, Any]:
        if not isinstance(model_response, dict):
            raise RuntimeError(f"Cosmos policy batch response must be a dict, got {type(model_response).__name__}")
        batches = model_response.get("actions")
        if not isinstance(batches, list) or len(batches) != 1:
            raise RuntimeError("Cosmos policy batch response must contain exactly one action chunk")
        actions = _rot6d10_to_axisangle7(batches[0])
        if actions.shape[0] != self.cfg.action_chunk_size:
            raise RuntimeError(
                f"Cosmos policy returned {actions.shape[0]} actions; expected {self.cfg.action_chunk_size}"
            )
        return {"actions": actions}

    def _run_synchronized_stage(self, stage: str, operation: Callable[[], Any]) -> Any:
        """Run one stage and make every rank fail when any participant fails."""
        try:
            result = operation()
            local_exception: Exception | None = None
            local_error: str | None = None
        except Exception as exc:
            result = None
            local_exception = exc
            local_error = f"{type(exc).__name__}: {exc}"

        if self.distributed_world_size > 1:
            local_report = {
                "rank": self.distributed_rank,
                "stage": stage,
                "error": local_error,
            }
            reports: list[Any] = [None] * self.distributed_world_size
            torch.distributed.all_gather_object(reports, local_report)
            observed_ranks = {
                report.get("rank")
                for report in reports
                if isinstance(report, dict) and report.get("stage") == stage and isinstance(report.get("rank"), int)
            }
            if observed_ranks != set(range(self.distributed_world_size)):
                raise RuntimeError(
                    f"RoboCasa365 distributed stage {stage!r} returned invalid rank reports: {reports!r}"
                )
            errors = [f"rank {report['rank']}: {report['error']}" for report in reports if report["error"] is not None]
            if errors:
                message = f"RoboCasa365 distributed stage {stage!r} failed closed: " + "; ".join(errors)
                if local_exception is not None:
                    raise RuntimeError(message) from local_exception
                raise RuntimeError(message)
        elif local_exception is not None:
            raise local_exception
        return result

    def _execute_canonical_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute the same validated request on every FSDP rank."""
        request = _validate_canonical_policy_request(request)
        if self.distributed_world_size == 1:
            return self._convert_model_response(self.predict_policy_batch([request]))

        # Preflight input preparation before any rank can enter FSDP collectives.
        # ``predict_policy_batch`` repeats this deterministic preparation, but
        # the explicit stage prevents one rank from entering model collectives
        # while another rank has already rejected the request.
        self._run_synchronized_stage(
            "prepare",
            lambda: self._prep_policy_item(request),
        )
        # Do not attempt a second collective after an exception raised from an
        # FSDP/NCCL model collective.  A rank that already failed while a peer
        # is still inside that collective cannot safely join all_gather_object;
        # propagating the exception lets torchrun terminate the complete worker
        # group instead of deadlocking in a mismatched recovery collective.
        try:
            model_response = self.predict_policy_batch([request])
        except BaseException as error:
            if self.distributed_world_size > 1:
                log.error(
                    "[robocasa365-policy-server] "
                    f"fatal distributed model failure on rank={self.distributed_rank}: "
                    f"{type(error).__name__}: {error}"
                )
                # The OpenPI WebSocket handler catches ordinary exceptions in a
                # connection task, which would otherwise leave rank 0 alive
                # while peer ranks are blocked or have exited.  Exiting this
                # worker forces torchrun to terminate the complete local group.
                os._exit(70)
            raise
        return self._run_synchronized_stage(
            "postprocess",
            lambda: self._convert_model_response(model_response),
        )

    def _validate_distributed_command(
        self,
        command: Any,
        *,
        expected_sequence: int,
    ) -> tuple[str, Any]:
        if not isinstance(command, dict):
            raise RuntimeError(f"RoboCasa365 distributed command must be a dict, got {type(command).__name__}")
        expected_keys = {"schema_version", "sequence", "op", "payload"}
        if set(command) != expected_keys:
            raise RuntimeError(
                f"RoboCasa365 distributed command keys drifted: expected={sorted(expected_keys)}, got={sorted(command)}"
            )
        if command["schema_version"] != _DISTRIBUTED_COMMAND_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported RoboCasa365 distributed command schema: {command['schema_version']!r}")
        if command["sequence"] != expected_sequence:
            raise RuntimeError(
                "RoboCasa365 distributed command sequence drifted: "
                f"expected={expected_sequence}, got={command['sequence']!r}"
            )
        op = command["op"]
        if op not in _DISTRIBUTED_COMMAND_OPS:
            raise RuntimeError(f"Unsupported RoboCasa365 distributed command op: {op!r}")
        payload = command["payload"]
        if op == "infer":
            if not isinstance(payload, dict) or set(payload) != {"request", "evidence"}:
                raise RuntimeError("RoboCasa365 distributed infer payload requires request and evidence")
            payload = {
                "request": _validate_canonical_policy_request(payload["request"]),
                "evidence": self._validate_distributed_evidence(payload["evidence"]),
            }
        elif payload is not None:
            raise RuntimeError(f"RoboCasa365 distributed {op!r} command payload must be null")
        return op, payload

    @staticmethod
    def _validate_distributed_evidence(evidence: Any) -> dict[str, Any]:
        if not isinstance(evidence, dict):
            raise RuntimeError("RoboCasa365 distributed evidence must be a dict")
        required = {"episode_key", "task_id", "control_step", "post_terminal"}
        if not required.issubset(evidence):
            raise RuntimeError(
                f"RoboCasa365 distributed evidence is missing {sorted(required - set(evidence))}"
            )
        if not isinstance(evidence["episode_key"], str) or not evidence["episode_key"]:
            raise RuntimeError("Distributed episode_key must be a non-empty string")
        if not isinstance(evidence["task_id"], str) or not evidence["task_id"]:
            raise RuntimeError("Distributed task_id must be a non-empty string")
        if type(evidence["control_step"]) is not int or evidence["control_step"] < 0:
            raise RuntimeError("Distributed control_step must be a non-negative integer")
        if type(evidence["post_terminal"]) is not bool:
            raise RuntimeError("Distributed post_terminal must be a bool")
        return evidence

    def _broadcast_distributed_command(self, op: str, payload: Any = None) -> None:
        if self.distributed_world_size <= 1 or self.distributed_rank != 0:
            raise RuntimeError("Only rank 0 of a multi-rank RoboCasa365 service may broadcast commands")
        command = {
            "schema_version": _DISTRIBUTED_COMMAND_SCHEMA_VERSION,
            "sequence": self._next_distributed_sequence,
            "op": op,
            "payload": payload,
        }
        self._validate_distributed_command(
            command,
            expected_sequence=self._next_distributed_sequence,
        )
        objects = [command]
        torch.distributed.broadcast_object_list(objects, src=0)
        self._next_distributed_sequence += 1

    def _receive_distributed_command(self) -> tuple[str, Any]:
        if self.distributed_world_size <= 1 or self.distributed_rank == 0:
            raise RuntimeError("Only nonzero RoboCasa365 ranks may receive worker commands")
        objects: list[Any] = [None]
        torch.distributed.broadcast_object_list(objects, src=0)
        op, payload = self._validate_distributed_command(
            objects[0],
            expected_sequence=self._next_distributed_sequence,
        )
        self._next_distributed_sequence += 1
        return op, payload

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if isinstance(obs, dict) and "__cwam_control__" in obs:
            return self._handle_shutdown_control(obs)
        request = self._canonicalize_observation(obs)
        if self.distributed_rank != 0:
            raise RuntimeError("WebSocket inference is only valid on RoboCasa365 rank 0")
        evidence = self._make_observation_evidence(obs)
        with self._dispatch_lock:
            donor = self._configure_replay_for_current_step(evidence=evidence)
            if donor is not None:
                evidence["donor_episode_key"], evidence["donor_task_id"] = donor
            self._pending_evidence = evidence
            try:
                if self.distributed_world_size > 1:
                    self._broadcast_distributed_command(
                        "infer",
                        {"request": request, "evidence": evidence},
                    )
                response = self._execute_canonical_request(request)
                self._advance_evidence_control_step()
                return response
            finally:
                self._pending_evidence = None

    def _handle_shutdown_control(self, request: dict[str, Any]) -> dict[str, Any]:
        """Acknowledge a nonce-bound rank-0 shutdown and exit cleanly."""
        if self.distributed_rank != 0:
            raise RuntimeError("Controlled shutdown is valid only on rank 0")
        if set(request) != _CONTROL_REQUEST_KEYS:
            raise ValueError(
                f"Shutdown control keys drifted: expected={sorted(_CONTROL_REQUEST_KEYS)}, got={sorted(request)}"
            )
        if request["__cwam_control__"] != "shutdown":
            raise ValueError(f"Unsupported CWAM control operation: {request['__cwam_control__']!r}")
        if request["launch_nonce"] != self.launch_nonce:
            raise ValueError("Shutdown control launch nonce mismatch")
        with self._dispatch_lock:
            if self._shutdown_requested:
                raise RuntimeError("Distributed shutdown was already requested")
            self._shutdown_requested = True
            if self.distributed_world_size > 1:
                self._shutdown_distributed_workers_locked()
            threading.Thread(
                target=self._exit_rank0_after_shutdown_ack,
                name="cwam-rank0-clean-shutdown",
                daemon=True,
            ).start()
        return {
            "actions": [],
            "cwam_control": "shutdown_ack",
            "launch_nonce": self.launch_nonce,
        }

    @staticmethod
    def _exit_rank0_after_shutdown_ack() -> None:
        """Allow the WebSocket response to flush, then finish rank 0 successfully."""
        time.sleep(2.0)
        os._exit(0)

    def _reset_local_state(self) -> None:
        """Reset evidence bookkeeping without changing model state."""
        self._evidence_episode = int(getattr(self, "_evidence_episode", -1)) + 1
        self._evidence_control_step = 0
        if self.distributed_rank == 0:
            self._pending_evidence = None

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        if reset_info is not None and not isinstance(reset_info, dict):
            raise TypeError(f"RoboCasa365 reset control must be a dict or null, got {type(reset_info).__name__}")
        if self.distributed_world_size == 1:
            return self._reset_local_state()
        if self.distributed_rank != 0:
            raise RuntimeError("WebSocket reset is only valid on RoboCasa365 rank 0")
        with self._dispatch_lock:
            self._broadcast_distributed_command("reset")
            self._run_synchronized_stage("reset", self._reset_local_state)

    def run_distributed_worker(self) -> None:
        """Block a nonzero rank until rank 0 sends a fail-closed shutdown."""
        if self.distributed_world_size <= 1 or self.distributed_rank == 0:
            raise RuntimeError("run_distributed_worker requires a nonzero rank in a multi-rank service")
        while True:
            op, payload = self._receive_distributed_command()
            if op == "infer":
                # All ranks execute the identical model call; only rank 0 owns
                # the client response, so worker outputs are deliberately dropped.
                self._configure_replay_for_current_step(evidence=payload["evidence"])
                self._execute_canonical_request(payload["request"])
                self._advance_evidence_control_step()
            elif op == "reset":
                self._run_synchronized_stage("reset", self._reset_local_state)
            elif op == "shutdown":
                self._run_synchronized_stage("shutdown", lambda: None)
                return
            else:  # pragma: no cover - command validation rejects this first.
                raise AssertionError(f"Unreachable distributed command op: {op!r}")

    def shutdown_distributed_workers(self) -> None:
        """Stop workers after a normal rank-0 server return and require their ack."""
        if self.distributed_world_size <= 1:
            return
        if self.distributed_rank != 0:
            raise RuntimeError("Only RoboCasa365 rank 0 may shut down distributed workers")
        with self._dispatch_lock:
            self._shutdown_distributed_workers_locked()

    def _shutdown_distributed_workers_locked(self) -> None:
        """Broadcast shutdown while the caller owns the rank-0 dispatch lock."""
        self._broadcast_distributed_command("shutdown")
        self._run_synchronized_stage("shutdown", lambda: None)

    def metadata(self) -> dict[str, Any]:
        source_module_path = Path(__file__).resolve()
        return {
            "schema_version": 1,
            "service": "cwam_cosmos3_robocasa365",
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_metadata_sha256": self._checkpoint_weight_receipt["checkpoint_metadata_sha256"],
            "checkpoint_weight_tree": self._checkpoint_weight_receipt["selected_weight_tree"],
            "checkpoint_weight_receipt": self._checkpoint_weight_receipt,
            "config_file": str(self.training_config_path),
            "config_file_sha256": _sha256_regular_file(self.training_config_path),
            "action_stats_path": str(self.action_stats_path),
            "action_stats_sha256": _sha256_regular_file(self.action_stats_path),
            "source_module_path": str(source_module_path),
            "source_module_sha256": hashlib.sha256(source_module_path.read_bytes()).hexdigest(),
            "server_pid": os.getpid(),
            "launch_nonce": self.launch_nonce,
            "wire_inputs": [*_CAMERA_KEYS, _STATE_KEY, "prompt"],
            "conditioning_views": ["observation/agentview_left", "observation/eye_in_hand"],
            "state_conditioning": False,
            "action_relevance_intervention": getattr(self, "_action_relevance_intervention", "normal"),
            "action_relevance_seed": self._action_relevance_seed,
            "action_relevance_jacobian_projections": self._action_relevance_jacobian_projections,
            "action_relevance_jacobian_endpoint": (
                "final_denoise_action_velocity_wrt_routed_c"
                if self._action_relevance_jacobian_projections > 0
                else None
            ),
            "action_relevance_intervention_bank": (
                str(getattr(self, "action_relevance_intervention_bank_path", None))
                if getattr(self, "action_relevance_intervention_bank_path", None) is not None
                else None
            ),
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
            "guardrails": bool(self.setup_args.guardrails),
            "vision_tokenizer": _ONLINE_WAN2PT2_TARGET,
            "online_pixel_encoding": True,
            "wan2pt2_vae_path": (str(self.wan2pt2_vae_path) if self.wan2pt2_vae_path is not None else None),
            "qwen_tokenizer_path": (str(self.qwen_tokenizer_path) if self.qwen_tokenizer_path is not None else None),
            "distributed_world_size": self.distributed_world_size,
            "parallelism": {
                "dp_shard_size": int(self.setup_args.dp_shard_size),
                "dp_replicate_size": int(self.setup_args.dp_replicate_size),
                "cp_size": int(self.setup_args.cp_size),
                "cfgp_size": int(self.setup_args.cfgp_size),
            },
            "control_protocol": {
                "shutdown": "nonce_bound_websocket_control_v1",
                "requires_loopback_endpoint": True,
            },
        }


def serve(args: RoboCasa365ServerArgs) -> None:
    hostname = socket.gethostname()
    service = RoboCasa365PolicyService(args)
    if service.distributed_world_size > 1 and service.distributed_rank != 0:
        log.info(
            "[robocasa365-policy-server] "
            f"worker rank={service.distributed_rank}/{service.distributed_world_size} "
            f"host={hostname} waiting for rank-0 commands"
        )
        service.run_distributed_worker()
        return

    log.info(
        "[robocasa365-policy-server] "
        f"rank=0/{service.distributed_world_size} host={hostname} "
        f"bind={args.host}:{int(args.port)}"
    )
    server_cls = _load_openpi_websocket_policy_server()
    metadata = service.metadata()
    metadata["websocket_server"] = _websocket_server_receipt(server_cls)
    websocket_server = server_cls(
        policy=service,
        host=args.host,
        port=int(args.port),
        metadata=metadata,
    )
    try:
        websocket_server.serve_forever()
    except BaseException:
        # Do not attempt another collective after an exceptional server exit:
        # a peer may already have failed inside FSDP.  Propagation to torchrun
        # is the fail-closed path that terminates the complete worker group.
        log.error("[robocasa365-policy-server] rank-0 server failed; terminating worker group")
        raise
    else:
        service.shutdown_distributed_workers()


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
