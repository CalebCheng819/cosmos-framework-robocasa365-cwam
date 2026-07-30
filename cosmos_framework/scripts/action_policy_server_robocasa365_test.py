# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU contract tests for the CWAM/Cosmos3 RoboCasa365 policy adapter."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

with patch("cosmos_framework.inference.common.init._init_script", lambda **kwargs: None):
    for module_name in (
        "cosmos_framework.scripts.action_policy_server_robocasa365",
        "cosmos_framework.scripts.action_policy_server_robolab",
        "cosmos_framework.scripts.action_policy_server_libero",
    ):
        if module_name in sys.modules:
            del sys.modules[module_name]
    from cosmos_framework.scripts import action_policy_server_robocasa365 as server  # noqa: E402

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _observation() -> dict:
    return {
        "observation/agentview_left": np.full((256, 256, 3), 11, dtype=np.uint8),
        "observation/eye_in_hand": np.full((256, 256, 3), 22, dtype=np.uint8),
        "observation/agentview_right": np.full((256, 256, 3), 33, dtype=np.uint8),
        "observation/state": np.arange(16, dtype=np.float64),
        "prompt": "open the drawer",
    }


def _stats_file(path: Path) -> Path:
    stats = {
        "global_raw": {
            "mean": [0.0] * 10,
            "std": [1.0] * 10,
            "min": [-4.0] * 10,
            "max": [4.0] * 10,
            "q01": [-2.0] * 10,
            "q99": [2.0] * 10,
        }
    }
    path.write_text(json.dumps(stats), encoding="utf-8")
    return path


def test_server_args_disable_generative_media_guardrails() -> None:
    args = server.RoboCasa365ServerArgs.model_validate(
        {
            "checkpoint": {"checkpoint_path": "/checkpoint/model"},
            "launch_nonce": "a" * 64,
        }
    )

    overrides = args.build_setup_overrides()

    assert overrides.guardrails is False


def test_extract_training_observation_uses_left_then_wrist_only() -> None:
    image, prompt = server._extract_training_observation(
        _observation(),
        camera_height=256,
        camera_width=256,
    )

    assert image.shape == (256, 512, 3)
    assert image.dtype == np.uint8
    assert np.all(image[:, :256] == 11)
    assert np.all(image[:, 256:] == 22)
    assert not np.any(image == 33)
    assert prompt == "open the drawer"


def test_extract_training_observation_rejects_noncausal_reasoning() -> None:
    obs = _observation()
    obs["reasoning_text"] = "oracle future action"

    with pytest.raises(ValueError, match="not part of the causal"):
        server._extract_training_observation(obs, camera_height=256, camera_width=256)


def test_extract_training_observation_rejects_dtype_and_state_drift() -> None:
    obs = _observation()
    obs["observation/agentview_left"] = obs["observation/agentview_left"].astype(np.float32)
    with pytest.raises(TypeError, match="dtype uint8"):
        server._extract_training_observation(obs, camera_height=256, camera_width=256)

    obs = _observation()
    obs["observation/state"] = np.zeros(15, dtype=np.float64)
    with pytest.raises(ValueError, match="shape \\(16,\\)"):
        server._extract_training_observation(obs, camera_height=256, camera_width=256)


def test_quantile_rot_normalizer_round_trip_uses_training_resolver(tmp_path: Path) -> None:
    normalizer = server._load_required_quantile_rot_normalizer(_stats_file(tmp_path / "stats.json"))
    raw = torch.tensor(
        [[-1.5, -1.0, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75]],
        dtype=torch.float32,
    )

    recovered = normalizer.denormalize_action(normalizer.normalize_action(raw))

    torch.testing.assert_close(recovered, raw)


def test_rot6d10_to_axisangle7_preserves_translation_and_gripper() -> None:
    raw = np.array(
        [
            [0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            [-0.4, 0.5, -0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25],
        ],
        dtype=np.float32,
    )

    env_action = server._rot6d10_to_axisangle7(raw)

    np.testing.assert_allclose(env_action[:, :3], raw[:, :3])
    np.testing.assert_allclose(env_action[:, 6], raw[:, 9])
    np.testing.assert_allclose(env_action[0, 3:6], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(env_action[1, 3:6], np.array([np.pi / 2, 0.0, 0.0]), atol=1e-6)


def test_rot6d10_to_axisangle7_rejects_degenerate_rotation() -> None:
    raw = np.zeros((1, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="degenerate"):
        server._rot6d10_to_axisangle7(raw)


def test_json_prompt_includes_exact_training_camera_description() -> None:
    service = object.__new__(server.RoboCasa365PolicyService)
    service.cfg = SimpleNamespace(fps=20, action_chunk_size=16, max_action_dim=64)
    service._prompt_json_formatter = server.ActionPromptJsonFormatter(
        caption_key="ai_caption",
        include_idle_frames=False,
    )
    video = torch.zeros((3, 17, 256, 512), dtype=torch.uint8)
    image_size = torch.tensor([256, 512, 256, 512], dtype=torch.float32)

    encoded = service._build_json_prompt("open the drawer", video=video, image_size=image_size)
    prompt = json.loads(encoded)

    framing = prompt["cinematography"]["framing"]
    assert server._TRAINING_CAMERA_DESCRIPTION in framing
    assert prompt["actions"][0]["description"] == "open the drawer."
    assert "idle_frame" not in prompt["actions"][0]
    assert prompt["fps"] == 20.0


def test_infer_returns_demian_actions_contract_and_png_camera_order() -> None:
    service = object.__new__(server.RoboCasa365PolicyService)
    service.camera_height = 256
    service.camera_width = 256
    service.domain_name = "robocasa365"
    service.cfg = SimpleNamespace(action_chunk_size=16)
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    raw = np.array([[0.1, 0.2, 0.3, *identity, -0.75]] * 16, dtype=np.float32)
    captured: list[dict] = []

    def fake_predict(reqs: list[dict]) -> dict:
        captured.extend(reqs)
        return {"actions": [raw.tolist()]}

    service.predict_policy_batch = fake_predict
    response = service.infer(_observation())

    assert set(response) == {"actions"}
    assert response["actions"].shape == (16, 7)
    np.testing.assert_allclose(response["actions"][:, 6], -0.75)
    assert captured[0]["domain_name"] == "robocasa365"
    assert captured[0]["image_size"] == 256
    image_bytes = base64.b64decode(captured[0]["image"])
    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    assert image.shape == (256, 512, 3)
    assert np.all(image[:, :256] == 11)
    assert np.all(image[:, 256:] == 22)
