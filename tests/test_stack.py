from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vla_sim.envs.ur5e_stack import UR5eStackConfig, UR5eStackEnv


def _raw() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "agentview_depth": np.ones((16, 16, 1), dtype=np.float32),
        "robot0_joint_pos": np.zeros(6, dtype=np.float32),
        "robot0_eef_pos": np.asarray([0.0, 0.0, 0.95], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([-0.024, -0.024], dtype=np.float32),
        "cubeA_pos": np.asarray([0.0, 0.0, 0.835], dtype=np.float32),
        "cubeB_pos": np.asarray([0.08, 0.0, 0.835], dtype=np.float32),
    }


class _Data:
    linear = np.zeros(3)
    angular = np.zeros(3)

    def get_body_xvelp(self, _name: str) -> np.ndarray:
        return self.linear

    def get_body_xvelr(self, _name: str) -> np.ndarray:
        return self.angular


class StackBackend:
    action_spec = (-np.ones(7), np.ones(7))

    def __init__(self) -> None:
        self.raw = _raw()
        self.cubeA = SimpleNamespace(root_body="cubeA")
        self.cubeB = SimpleNamespace(root_body="cubeB")
        self.robots = [SimpleNamespace(gripper=object())]
        self.sim = SimpleNamespace(data=_Data())
        self.touching = True
        self.grasped = False

    def reset(self):
        return self.raw

    def step(self, _action):
        return self.raw, 0.0, False, {}

    def check_contact(self, _a, _b) -> bool:
        return self.touching

    def _check_grasp(self, _gripper, _cube) -> bool:
        return self.grasped


def test_stack_observation_has_cartesian_eef_state() -> None:
    env = UR5eStackEnv(StackBackend())
    observation, _ = env.reset()
    assert observation["observation.state"].shape == (10,)
    np.testing.assert_allclose(observation["observation.state"][6:9], [0.0, 0.0, 0.95])


def test_strict_stack_requires_alignment_release_stability_and_hold() -> None:
    backend = StackBackend()
    env = UR5eStackEnv(backend, UR5eStackConfig(success_hold_steps=3))
    env.reset()
    backend.raw["cubeA_pos"] = np.asarray([0.08, 0.0, 0.885], dtype=np.float32)
    for expected in (False, False, True):
        _, _, terminated, _, info = env.step(np.zeros(7))
        assert info["success"] is expected
        assert terminated is expected

    backend.grasped = True
    _, _, _, _, info = env.step(np.zeros(7))
    assert not info["success"]
    assert not info["stack_conditions"]["gripper_released"]

    backend.grasped = False
    backend.sim.data.linear = np.asarray([0.03, 0.0, 0.0])
    _, _, _, _, info = env.step(np.zeros(7))
    assert not info["stack_conditions"]["objects_stable"]


def test_side_contact_is_not_a_stack() -> None:
    backend = StackBackend()
    env = UR5eStackEnv(backend)
    env.reset()
    backend.raw["cubeA_pos"] = np.asarray([0.08, 0.0, 0.835], dtype=np.float32)
    _, _, _, _, info = env.step(np.zeros(7))
    assert not info["success"]
    assert not info["stack_conditions"]["above_target"]
