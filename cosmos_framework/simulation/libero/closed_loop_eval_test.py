# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from typing import Any

import numpy as np

from cosmos_framework.simulation.libero.closed_loop_eval import (
    _run_episode,
    _validate_server_intervention,
)


def _observation(value: int) -> dict[str, Any]:
    return {
        "agentview_image": np.full((8, 8, 3), value, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((8, 8, 3), value + 1, dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_joint_vel": np.zeros(7, dtype=np.float32),
        "robot0_gripper_qvel": np.zeros(2, dtype=np.float32),
        "object-state": np.zeros(4, dtype=np.float32),
    }


class _TerminalEnv:
    def __init__(self) -> None:
        self.step_calls = 0
        self.obs = _observation(10)

    def reset(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any]:
        return self.obs

    def step(self, _action: list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.step_calls += 1
        self.obs = _observation(20)
        return self.obs, 1.0, True, {"success": True}


class _RecordingClient:
    def __init__(self) -> None:
        self.evidence: list[dict[str, Any]] = []

    def predict(
        self,
        _observation: np.ndarray | list[np.ndarray],
        *,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert evidence is not None
        self.evidence.append(evidence)
        return {"action": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]}


def test_fixed_horizon_freezes_terminal_observation_without_stepping_env() -> None:
    env = _TerminalEnv()
    client = _RecordingClient()

    result = _run_episode(
        env,
        client,  # type: ignore[arg-type]
        cameras=["agentview"],
        flip_images=False,
        rotate_180=False,
        action_horizon=1,
        action_dim=7,
        action_space="frame_wise_relative",
        rotation_space="3d",
        gripper_mode="pm_one",
        max_steps=3,
        warmup_steps=0,
        initial_state=None,
        task_id=4,
        protocol_task_id="libero_spatial:004",
        episode_idx=7,
        gif_path=None,
        gif_fps=20,
        trace_fixed_horizon=True,
    )

    assert result.success is True
    assert result.steps == 1
    assert result.trace_steps == 3
    assert env.step_calls == 1
    assert [item["control_step"] for item in client.evidence] == [0, 1, 2]
    assert [item["post_terminal"] for item in client.evidence] == [False, True, True]
    assert {item["episode_key"] for item in client.evidence} == {"libero_spatial:004:007"}
    assert {item["task_id"] for item in client.evidence} == {"libero_spatial:004"}
    assert {len(item["relevant_factors"]) for item in client.evidence} == {137}
    assert all("individual_state" not in item for item in client.evidence)


def test_server_intervention_validation_fails_closed() -> None:
    _validate_server_intervention(
        {
            "action_relevance_intervention": "normal",
            "action_relevance_intervention_bank": None,
        },
        "normal",
    )

    for server_info, expected in (
        ({}, "normal"),
        ({"action_relevance_intervention": "normal"}, "zero"),
        ({"action_relevance_intervention": "shuffle"}, "shuffle"),
        (
            {
                "action_relevance_intervention": "normal",
                "action_relevance_intervention_bank": "/unexpected/bank.npz",
            },
            "normal",
        ),
    ):
        try:
            _validate_server_intervention(server_info, expected)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Expected validation failure for {server_info=}, {expected=}")
