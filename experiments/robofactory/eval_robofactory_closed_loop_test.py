"""Unit coverage for Stage-1 causal evaluation helpers."""

from __future__ import annotations

import numpy as np

from experiments.robofactory.eval_robofactory_closed_loop import (
    _apply_oracle_intervention,
    _chunk_overlap_error,
)


def _actions(offset: float) -> dict[str, np.ndarray]:
    return {
        "panda-0": np.arange(8, dtype=np.float32) + offset,
        "panda-1": np.arange(8, dtype=np.float32) + offset + 10,
    }


def test_robot0_pose_oracle_replaces_only_seven_joint_dimensions() -> None:
    learned = _actions(0)
    expert = _actions(100)
    result, replaced = _apply_oracle_intervention(
        learned, expert, "robot0-pose"
    )
    np.testing.assert_array_equal(result["panda-0"][:7], expert["panda-0"][:7])
    assert result["panda-0"][7] == learned["panda-0"][7]
    np.testing.assert_array_equal(result["panda-1"], learned["panda-1"])
    assert replaced["panda-0"] == [True] * 7 + [False]


def test_robot0_gripper_and_robot1_oracles_are_isolated() -> None:
    learned = _actions(0)
    expert = _actions(100)
    gripper, _ = _apply_oracle_intervention(
        learned, expert, "robot0-gripper"
    )
    np.testing.assert_array_equal(gripper["panda-0"][:7], learned["panda-0"][:7])
    assert gripper["panda-0"][7] == expert["panda-0"][7]
    robot1, _ = _apply_oracle_intervention(
        learned, expert, "robot1-action"
    )
    np.testing.assert_array_equal(robot1["panda-0"], learned["panda-0"])
    np.testing.assert_array_equal(robot1["panda-1"], expert["panda-1"])


def test_chunk_overlap_compares_the_same_future_actions() -> None:
    previous = np.arange(12, dtype=np.float32).reshape(6, 2)
    current = previous[2:].copy()
    current = np.concatenate([current, np.zeros((2, 2), dtype=np.float32)])
    result = _chunk_overlap_error(previous, current, 2)
    assert result is not None
    assert result["overlap_actions"] == 4
    assert result["mean_absolute_error"] == 0.0
    assert result["maximum_absolute_error"] == 0.0
