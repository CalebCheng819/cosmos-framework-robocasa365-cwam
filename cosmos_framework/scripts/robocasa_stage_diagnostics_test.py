from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cosmos_framework.scripts.robocasa_stage_diagnostics import OrderedStageTracker, STAGE_ORDER


class PickPlaceCounterToStove:
    def __init__(self) -> None:
        data = SimpleNamespace(
            body_xpos={0: np.array([0.0, 0.0, 0.10]), 1: np.array([0.5, 0.0, 0.10])},
            site_xpos={0: np.array([0.5, 0.5, 0.5])},
            qpos=np.array([0.0, 0.0]),
        )
        model = SimpleNamespace(get_joint_qpos_addr=lambda name: 0 if name.endswith("1") else 1)
        self.sim = SimpleNamespace(data=data, model=model)
        self.robots = [SimpleNamespace(eef_site_id={"right": 0}, gripper={"right": object()})]
        self.objects = {"obj": object(), "container": object()}
        self.obj_body_id = {"obj": 0, "container": 1}
        self.contact = False
        self.grasp = False
        self.place = False

    def check_contact(self, left, right) -> bool:
        del left, right
        return self.contact


class _ObjectUtils:
    @staticmethod
    def check_obj_grasped(env, name) -> bool:
        assert name == "obj"
        return env.grasp

    @staticmethod
    def check_obj_in_receptacle(env, obj, container, th) -> bool:
        assert (obj, container, th) == ("obj", "container", 0.07)
        return env.place


class _BlenderData:
    def __init__(self) -> None:
        self.site_xpos = {0: np.array([0.5, 0.5, 0.5])}
        self.qpos = np.array([0.020833, -0.020833])
        self.body_position = np.array([0.0, 0.0, 0.10])

    def get_body_xpos(self, name: str) -> np.ndarray:
        assert name == "lid_main"
        return self.body_position


class CloseBlenderLid:
    def __init__(self) -> None:
        self.sim = SimpleNamespace(
            data=_BlenderData(),
            model=SimpleNamespace(get_joint_qpos_addr=lambda name: 0 if name.endswith("1") else 1),
        )
        gripper = SimpleNamespace(joints=("gripper0_right_finger_joint1", "gripper0_right_finger_joint2"))
        self.robots = [SimpleNamespace(eef_site_id={"right": 0}, gripper={"right": gripper})]
        self.contact = False
        self.lid_on_blender = False
        self.blender = SimpleNamespace(
            blender_lid=SimpleNamespace(name="lid"),
            get_lid_closed_pos=lambda env: np.array([0.5, 0.0, 0.10]),
            get_state=lambda: {"lid_on_blender": self.lid_on_blender},
        )

    def check_contact(self, left, right) -> bool:
        del left, right
        return self.contact


def test_ordered_pick_place_stages_require_real_progress() -> None:
    env = PickPlaceCounterToStove()
    tracker = OrderedStageTracker(env, object_utils=_ObjectUtils)

    env.sim.data.site_xpos[0] = np.array([0.05, 0.0, 0.10])
    assert tracker.update(0) == ["approach"]
    env.contact = True
    env.grasp = True
    assert tracker.update(1) == ["contact", "grasp"]
    env.sim.data.body_xpos[0] = np.array([0.30, 0.0, 0.16])
    env.sim.data.site_xpos[0] = np.array([0.30, 0.0, 0.16])
    assert tracker.update(2) == ["transport"]
    env.sim.data.body_xpos[0] = np.array([0.50, 0.0, 0.10])
    env.place = True
    assert tracker.update(3) == ["place"]
    env.contact = False
    env.grasp = False
    env.sim.data.site_xpos[0] = np.array([0.50, 0.20, 0.10])
    assert tracker.update(4) == ["release"]

    summary = tracker.summary()
    assert summary["stage_order"] == list(STAGE_ORDER)
    assert summary["terminal_stage"] == "release"
    assert summary["stage_first_step"] == {
        "approach": 0,
        "contact": 1,
        "grasp": 1,
        "transport": 2,
        "place": 3,
        "release": 4,
    }


def test_late_place_does_not_skip_missing_contact_and_grasp() -> None:
    env = PickPlaceCounterToStove()
    tracker = OrderedStageTracker(env, object_utils=_ObjectUtils)
    env.place = True
    env.sim.data.body_xpos[0] = np.array([0.50, 0.0, 0.10])
    tracker.update(0)
    assert tracker.summary()["terminal_stage"] == "none"


def test_blender_grasp_requires_contact_and_closed_gripper() -> None:
    env = CloseBlenderLid()
    tracker = OrderedStageTracker(env, object_utils=_ObjectUtils)

    env.sim.data.site_xpos[0] = np.array([0.05, 0.0, 0.10])
    env.contact = True
    assert tracker.update(0) == ["approach", "contact"]

    env.sim.data.qpos[:] = np.array([0.005, -0.005])
    assert tracker.update(1) == ["grasp"]
    env.sim.data.body_position = np.array([0.25, 0.0, 0.16])
    env.sim.data.site_xpos[0] = env.sim.data.body_position.copy()
    assert tracker.update(2) == ["transport"]
    env.sim.data.body_position = np.array([0.5, 0.0, 0.10])
    env.lid_on_blender = True
    assert tracker.update(3) == ["place"]
    env.contact = False
    env.sim.data.site_xpos[0] = np.array([0.5, 0.3, 0.10])
    assert tracker.update(4) == ["release"]
