#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Capture and bind the live CWAM/Cosmos3 RoboCasa365 WebSocket metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import msgpack
import websockets.sync.client

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_WIRE_INPUTS = [
    "observation/agentview_left",
    "observation/eye_in_hand",
    "observation/agentview_right",
    "observation/state",
    "prompt",
]
_CONDITIONING_VIEWS = [
    "observation/agentview_left",
    "observation/eye_in_hand",
]


def _require_digest(value: str, label: str, pattern: re.Pattern[str]) -> str:
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{label} has an invalid digest/commit format: {value!r}")
    return value


def _connect_once(uri: str, timeout_seconds: float) -> dict[str, Any]:
    with websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        open_timeout=timeout_seconds,
        close_timeout=timeout_seconds,
        ping_interval=20,
        ping_timeout=timeout_seconds,
    ) as connection:
        response = connection.recv(timeout=timeout_seconds)
    if not isinstance(response, bytes):
        raise TypeError(f"Expected binary server metadata, got {type(response).__name__}")
    metadata = msgpack.unpackb(response, raw=False)
    if not isinstance(metadata, dict):
        raise TypeError(f"Server metadata must be a map, got {type(metadata).__name__}")
    return metadata


def _validate_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "service": "cwam_cosmos3_robocasa365",
        "wire_inputs": _WIRE_INPUTS,
        "conditioning_views": _CONDITIONING_VIEWS,
        "state_conditioning": False,
        "model_action": "[dpos3,rot6d_columns6,gripper1]",
        "output_action": "[dpos3,axisangle3,gripper1]",
        "gripper_transform": "identity",
        "fps": 20,
        "action_chunk_size": 16,
        "format_prompt_as_json": True,
        "reuse_policy_seed_per_request": True,
        "use_ema_weights": True,
        "guardrails": False,
        "vision_tokenizer": (
            "cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16."
            "Wan2pt2VAEInterface"
        ),
        "online_pixel_encoding": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Server metadata {key} mismatch: expected={value!r}, actual={metadata.get(key)!r}")
    for key in ("checkpoint", "action_stats_path", "source_module_path"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Server metadata {key} must be a non-empty string")
    _require_digest(
        metadata.get("source_module_sha256"),
        "Server metadata source_module_sha256",
        _SHA256_RE,
    )
    _require_digest(metadata.get("launch_nonce"), "Server metadata launch_nonce", _SHA256_RE)
    if type(metadata.get("server_pid")) is not int or metadata["server_pid"] <= 0:
        raise ValueError("Server metadata server_pid must be a positive integer")
    if metadata.get("sampler") not in {"unipc", "edm"}:
        raise ValueError("Server metadata sampler must be unipc or edm")
    for key in ("policy_seed", "num_steps"):
        if type(metadata.get(key)) is not int:
            raise TypeError(f"Server metadata {key} must be an integer")
    if metadata["policy_seed"] < 0 or metadata["num_steps"] < 1:
        raise ValueError("Server metadata policy_seed/num_steps are out of range")
    guidance = metadata.get("guidance")
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(float(guidance))
    ):
        raise TypeError("Server metadata guidance must be finite and numeric")


def capture(
    *,
    uri: str,
    output: Path,
    timeout_seconds: float,
    checkpoint_manifest_sha256: str,
    training_config_sha256: str,
    action_stats_sha256: str,
    source_manifest_sha256: str,
    demian_commit: str,
    shard_index: int,
    expected_launch_nonce: str,
    expected_server_pid: int,
    watch_pid: int | None = None,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    for label, value in (
        ("checkpoint_manifest_sha256", checkpoint_manifest_sha256),
        ("training_config_sha256", training_config_sha256),
        ("action_stats_sha256", action_stats_sha256),
        ("source_manifest_sha256", source_manifest_sha256),
    ):
        _require_digest(value, label, _SHA256_RE)
    _require_digest(demian_commit, "demian_commit", _COMMIT_RE)
    _require_digest(expected_launch_nonce, "expected_launch_nonce", _SHA256_RE)
    if expected_server_pid <= 0:
        raise ValueError("expected_server_pid must be positive")
    if watch_pid is not None and watch_pid != expected_server_pid:
        raise ValueError("watch_pid and expected_server_pid must identify the same process")
    if shard_index < 0:
        raise ValueError("shard_index must be non-negative")

    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    metadata: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if watch_pid is not None:
            try:
                os.kill(watch_pid, 0)
            except ProcessLookupError:
                raise RuntimeError(f"Policy server process {watch_pid} exited before becoming ready") from None
            try:
                process_state = Path(f"/proc/{watch_pid}/stat").read_text(encoding="utf-8").split()[2]
            except FileNotFoundError:
                raise RuntimeError(f"Policy server process {watch_pid} exited before becoming ready") from None
            if process_state == "Z":
                raise RuntimeError(f"Policy server process {watch_pid} became a zombie before becoming ready")
        remaining = deadline - time.monotonic()
        try:
            metadata = _connect_once(uri, min(10.0, max(0.1, remaining)))
            break
        except Exception as exc:  # noqa: BLE001 - connection failures vary by websockets version
            last_error = exc
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    if metadata is None:
        raise TimeoutError(f"Timed out waiting for RoboCasa365 policy metadata at {uri}: {last_error}")
    _validate_metadata(metadata)
    if metadata["launch_nonce"] != expected_launch_nonce:
        raise ValueError("Connected policy server launch nonce does not match this launch")
    if metadata["server_pid"] != expected_server_pid:
        raise ValueError("Connected policy server PID does not match this launch")

    receipt = {
        **metadata,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "server_uri": uri,
        "shard_index": shard_index,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "training_config_sha256": training_config_sha256,
        "action_stats_sha256": action_stats_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "demian_commit": demian_commit,
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing server receipt: {output}")
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--training-config-sha256", required=True)
    parser.add_argument("--action-stats-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--demian-commit", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--expected-launch-nonce", required=True)
    parser.add_argument("--expected-server-pid", type=int, required=True)
    parser.add_argument("--watch-pid", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = capture(
        uri=args.uri,
        output=args.output,
        timeout_seconds=args.timeout_seconds,
        checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
        training_config_sha256=args.training_config_sha256,
        action_stats_sha256=args.action_stats_sha256,
        source_manifest_sha256=args.source_manifest_sha256,
        demian_commit=args.demian_commit,
        shard_index=args.shard_index,
        expected_launch_nonce=args.expected_launch_nonce,
        expected_server_pid=args.expected_server_pid,
        watch_pid=args.watch_pid,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
