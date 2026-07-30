# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the live RoboCasa365 server receipt helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.h_cluster import capture_robocasa365_server_receipt as receipt


def _metadata() -> dict:
    return {
        "schema_version": 1,
        "service": "cwam_cosmos3_robocasa365",
        "checkpoint": "/checkpoint/model",
        "action_stats_path": "/data/stats.json",
        "source_module_path": "/source/cosmos_framework/scripts/action_policy_server_robocasa365.py",
        "source_module_sha256": "a" * 64,
        "server_pid": 12345,
        "launch_nonce": "b" * 64,
        "wire_inputs": [
            "observation/agentview_left",
            "observation/eye_in_hand",
            "observation/agentview_right",
            "observation/state",
            "prompt",
        ],
        "conditioning_views": [
            "observation/agentview_left",
            "observation/eye_in_hand",
        ],
        "state_conditioning": False,
        "model_action": "[dpos3,rot6d_columns6,gripper1]",
        "output_action": "[dpos3,axisangle3,gripper1]",
        "gripper_transform": "identity",
        "fps": 20,
        "action_chunk_size": 16,
        "format_prompt_as_json": True,
        "sampler": "unipc",
        "policy_seed": 42,
        "guidance": 1.0,
        "num_steps": 30,
        "reuse_policy_seed_per_request": True,
        "use_ema_weights": True,
        "guardrails": False,
        "vision_tokenizer": (
            "cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16."
            "Wan2pt2VAEInterface"
        ),
        "online_pixel_encoding": True,
    }


def test_capture_persists_validated_runtime_binding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(receipt, "_connect_once", lambda _uri, _timeout: _metadata())
    output = tmp_path / "server-runtime-receipt.json"

    value = receipt.capture(
        uri="ws://127.0.0.1:6000",
        output=output,
        timeout_seconds=1.0,
        checkpoint_manifest_sha256="1" * 64,
        training_config_sha256="2" * 64,
        action_stats_sha256="3" * 64,
        source_manifest_sha256="4" * 64,
        demian_commit="5" * 40,
        shard_index=0,
        expected_launch_nonce="b" * 64,
        expected_server_pid=12345,
    )

    assert value["service"] == "cwam_cosmos3_robocasa365"
    assert value["checkpoint_manifest_sha256"] == "1" * 64
    assert json.loads(output.read_text(encoding="utf-8")) == value


def test_capture_rejects_metadata_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["conditioning_views"] = ["observation/agentview_right"]
    monkeypatch.setattr(receipt, "_connect_once", lambda _uri, _timeout: metadata)

    with pytest.raises(ValueError, match="conditioning_views mismatch"):
        receipt.capture(
            uri="ws://127.0.0.1:6000",
            output=tmp_path / "receipt.json",
            timeout_seconds=1.0,
            checkpoint_manifest_sha256="1" * 64,
            training_config_sha256="2" * 64,
            action_stats_sha256="3" * 64,
            source_manifest_sha256="4" * 64,
            demian_commit="5" * 40,
            shard_index=0,
            expected_launch_nonce="b" * 64,
            expected_server_pid=12345,
        )


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("launch_nonce", "c" * 64, "launch nonce"),
        ("server_pid", 54321, "PID"),
    ],
)
def test_capture_rejects_a_different_server_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    key: str,
    value: object,
    match: str,
) -> None:
    metadata = _metadata()
    metadata[key] = value
    monkeypatch.setattr(receipt, "_connect_once", lambda _uri, _timeout: metadata)

    with pytest.raises(ValueError, match=match):
        receipt.capture(
            uri="ws://127.0.0.1:6000",
            output=tmp_path / "receipt.json",
            timeout_seconds=1.0,
            checkpoint_manifest_sha256="1" * 64,
            training_config_sha256="2" * 64,
            action_stats_sha256="3" * 64,
            source_manifest_sha256="4" * 64,
            demian_commit="5" * 40,
            shard_index=0,
            expected_launch_nonce="b" * 64,
            expected_server_pid=12345,
        )


def test_capture_never_overwrites_existing_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(receipt, "_connect_once", lambda _uri, _timeout: _metadata())
    output = tmp_path / "receipt.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        receipt.capture(
            uri="ws://127.0.0.1:6000",
            output=output,
            timeout_seconds=1.0,
            checkpoint_manifest_sha256="1" * 64,
            training_config_sha256="2" * 64,
            action_stats_sha256="3" * 64,
            source_manifest_sha256="4" * 64,
            demian_commit="5" * 40,
            shard_index=0,
            expected_launch_nonce="b" * 64,
            expected_server_pid=12345,
        )
