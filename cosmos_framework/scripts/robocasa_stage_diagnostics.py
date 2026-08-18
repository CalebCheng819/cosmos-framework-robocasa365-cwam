"""Task-aware ordered manipulation-stage diagnostics for RoboCasa365."""

from __future__ import annotations

from typing import Any

import numpy as np


STAGE_ORDER = ("approach", "contact", "grasp", "transport", "place", "release")
SUPPORTED_TASKS = ("CloseBlenderLid", "PickPlaceCounterToStove")


class OrderedStageTracker:
    """Record raw predicates and first passage through ordered control stages."""

    def __init__(self, raw_env: Any, *, object_utils: Any | None = None) -> None:
        self.raw_env = raw_env
        self.task_name = type(raw_env).__name__
        if self.task_name not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported diagnostic task: {self.task_name}")
        if object_utils is None:
            from robocasa.utils import object_utils

        self.object_utils = object_utils
        self.first_step: dict[str, int | None] = {stage: None for stage in STAGE_ORDER}
        self.trace: list[dict[str, Any]] = []
        self._initial_target_position = self._target_position()
        self._initial_destination_distance = self._destination_distance(self._initial_target_position)

    def _eef_position(self) -> np.ndarray:
        site_id = self.raw_env.robots[0].eef_site_id["right"]
        return np.asarray(self.raw_env.sim.data.site_xpos[site_id], dtype=np.float64).copy()

    def _target(self) -> Any:
        if self.task_name == "PickPlaceCounterToStove":
            return self.raw_env.objects["obj"]
        return self.raw_env.blender.blender_lid

    def _target_position(self) -> np.ndarray:
        if self.task_name == "PickPlaceCounterToStove":
            body_id = self.raw_env.obj_body_id["obj"]
            value = self.raw_env.sim.data.body_xpos[body_id]
        else:
            value = self.raw_env.sim.data.get_body_xpos(f"{self.raw_env.blender.blender_lid.name}_main")
        return np.asarray(value, dtype=np.float64).copy()

    def _destination_distance(self, target_position: np.ndarray) -> float:
        if self.task_name == "PickPlaceCounterToStove":
            destination = np.asarray(
                self.raw_env.sim.data.body_xpos[self.raw_env.obj_body_id["container"]],
                dtype=np.float64,
            )
            return float(np.linalg.norm(target_position[:2] - destination[:2]))
        destination = np.asarray(self.raw_env.blender.get_lid_closed_pos(self.raw_env), dtype=np.float64)
        return float(np.linalg.norm(target_position - destination))

    def _gripper_contact(self) -> bool:
        robot = self.raw_env.robots[0]
        if "right" not in robot.gripper:
            raise AttributeError("RoboCasa diagnostic requires robot.gripper['right']")
        return bool(self.raw_env.check_contact(robot.gripper["right"], self._target()))

    def _gripper_closed(self, aperture_threshold: float = 0.025) -> bool:
        gripper = self.raw_env.robots[0].gripper["right"]
        joints = tuple(gripper.joints)
        if len(joints) < 2:
            raise ValueError(f"Expected at least two gripper joints, got {joints!r}")
        model = self.raw_env.sim.model
        qpos = self.raw_env.sim.data.qpos
        aperture = sum(abs(float(qpos[model.get_joint_qpos_addr(joint)])) for joint in joints)
        return aperture <= aperture_threshold

    def _raw_predicates(self, metrics: dict[str, float]) -> dict[str, bool]:
        contact = self._gripper_contact()
        if self.task_name == "PickPlaceCounterToStove":
            grasp = bool(self.object_utils.check_obj_grasped(self.raw_env, "obj"))
            place = bool(self.object_utils.check_obj_in_receptacle(self.raw_env, "obj", "container", th=0.07))
        else:
            grasp = contact and self._gripper_closed()
            place = bool(self.raw_env.blender.get_state()["lid_on_blender"])

        displacement = metrics["target_displacement"]
        destination_progress = metrics["destination_progress"]
        transport = displacement >= 0.04 and (
            metrics["target_height_delta"] >= 0.025 or destination_progress >= 0.03
        )
        release_distance = 0.15 if self.task_name == "CloseBlenderLid" else 0.10
        return {
            "approach": metrics["eef_target_distance"] <= 0.12,
            "contact": contact,
            "grasp": grasp,
            "transport": transport,
            "place": place,
            "release": place and not contact and metrics["eef_target_distance"] >= release_distance,
        }

    def update(self, step: int) -> list[str]:
        target_position = self._target_position()
        destination_distance = self._destination_distance(target_position)
        metrics = {
            "eef_target_distance": float(np.linalg.norm(self._eef_position() - target_position)),
            "target_displacement": float(np.linalg.norm(target_position - self._initial_target_position)),
            "target_height_delta": float(target_position[2] - self._initial_target_position[2]),
            "destination_distance": destination_distance,
            "destination_progress": float(self._initial_destination_distance - destination_distance),
        }
        predicates = self._raw_predicates(metrics)
        newly_reached: list[str] = []
        for stage_index, stage in enumerate(STAGE_ORDER):
            prerequisite_met = stage_index == 0 or self.first_step[STAGE_ORDER[stage_index - 1]] is not None
            if self.first_step[stage] is None and prerequisite_met and predicates[stage]:
                self.first_step[stage] = int(step)
                newly_reached.append(stage)

        reached = {stage: self.first_step[stage] is not None for stage in STAGE_ORDER}
        self.trace.append(
            {
                "step": int(step),
                "predicates": predicates,
                "ordered_reached": reached,
                "metrics": metrics,
            }
        )
        return newly_reached

    def summary(self) -> dict[str, Any]:
        reached = {stage: self.first_step[stage] is not None for stage in STAGE_ORDER}
        terminal_stage = "none"
        for stage in STAGE_ORDER:
            if reached[stage]:
                terminal_stage = stage
            else:
                break
        return {
            "stage_order": list(STAGE_ORDER),
            "stage_reached": reached,
            "stage_first_step": dict(self.first_step),
            "terminal_stage": terminal_stage,
            "stage_trace": self.trace,
        }
