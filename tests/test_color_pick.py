from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vla_sim.color_pick_contract import (
    build_color_pick_contract,
    color_pick_prompt,
    color_pick_prompts,
)
from vla_sim.contracts import IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY
from vla_sim.domain_randomization import DomainRandomizationSample
from vla_sim.envs import UR5eColorPickConfig, UR5eColorPickEnv
from vla_sim.scenes import (
    COLOR_PICK_COLORS,
    COLOR_PICK_GRIPPER_CLEARANCE_M,
    COLOR_PICK_OBJECT_XY_M,
    generate_color_pick_scenes,
    oriented_rectangles_overlap,
)
from vla_sim.sim import HeuristicColorPickExpert


def _raw() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((16, 16, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(6, dtype=np.float32),
        "robot0_eef_pos": np.asarray([0.0, 0.0, 0.92], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([-0.024, -0.024], dtype=np.float32),
        "cube_pos": np.asarray([-0.06, -0.07, 0.825], dtype=np.float64),
        "red_cube_pos": np.asarray([-0.06, -0.07, 0.825], dtype=np.float64),
        "green_cube_pos": np.asarray([0.06, -0.07, 0.825], dtype=np.float64),
        "blue_cube_pos": np.asarray([0.0, 0.08, 0.825], dtype=np.float64),
    }


class _Data:
    def __init__(self, backend: _Backend) -> None:
        self.backend = backend

    def set_joint_qpos(self, joint: str, qpos: np.ndarray) -> None:
        color = joint.removesuffix("_joint")
        key = f"{color}_cube_pos"
        self.backend.raw[key] = np.asarray(qpos[:3], dtype=np.float64)
        if color == "red":
            self.backend.raw["cube_pos"] = self.backend.raw[key]

    def set_joint_qvel(self, _joint: str, _qvel: np.ndarray) -> None:
        return None


class _Robot:
    gripper = object()

    def reset(self, *, deterministic: bool) -> None:
        assert deterministic


class _Backend:
    action_spec = (-np.ones(7), np.ones(7))
    table_offset = np.asarray([0.0, 0.0, 0.8])

    def __init__(self) -> None:
        self.raw = _raw()
        self.cube = SimpleNamespace(root_body="red_cube", joints=["red_joint"])
        self.green_cube = SimpleNamespace(root_body="green_cube", joints=["green_joint"])
        self.blue_cube = SimpleNamespace(root_body="blue_cube", joints=["blue_joint"])
        self.color_cubes = {
            "red": self.cube,
            "green": self.green_cube,
            "blue": self.blue_cube,
        }
        self.robots = [_Robot()]
        self.grasped_color: str | None = None
        self.sim = SimpleNamespace(data=_Data(self), forward=lambda: None)

    def reset(self):
        return self.raw

    def _get_observations(self, *, force_update: bool):
        assert force_update
        return self.raw

    def step(self, _action):
        return self.raw, 0.0, False, {}

    def _check_grasp(self, _gripper, cube) -> bool:
        return cube is self.color_cubes.get(self.grasped_color)


def test_color_pick_scenes_are_balanced_deterministic_and_separated() -> None:
    scenes = generate_color_pick_scenes("color_pick", 60, 71000)
    assert scenes == generate_color_pick_scenes("color_pick", 60, 71000)
    targets = [scene.overrides["target_color"] for scene in scenes]
    assert {color: targets.count(color) for color in COLOR_PICK_COLORS} == {
        "red": 20,
        "green": 20,
        "blue": 20,
    }
    for scene in scenes:
        positions = {
            "red": (scene.x_m, scene.y_m),
            "green": (
                float(scene.overrides["green_x_m"]),
                float(scene.overrides["green_y_m"]),
            ),
            "blue": (
                float(scene.overrides["blue_x_m"]),
                float(scene.overrides["blue_y_m"]),
            ),
        }
        for left, right in (("red", "green"), ("red", "blue"), ("green", "blue")):
            assert not oriented_rectangles_overlap(
                positions[left],
                0.0,
                COLOR_PICK_OBJECT_XY_M,
                positions[right],
                0.0,
                COLOR_PICK_OBJECT_XY_M,
                clearance_m=COLOR_PICK_GRIPPER_CLEARANCE_M,
            )


def test_color_pick_policy_observation_has_no_privileged_target_signal() -> None:
    scene = generate_color_pick_scenes("color_pick", 3, 72000)[0]
    env = UR5eColorPickEnv(_Backend())
    observation, info = env.reset(scene=scene)
    assert set(observation) == {IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY}
    assert observation[STATE_KEY].shape == (10,)
    assert "target_color" not in observation
    assert "target_cube_pos" not in observation
    assert info["target_color"] == scene.overrides["target_color"]


def test_color_pick_applies_manifest_photometric_randomization_deterministically() -> None:
    scene = generate_color_pick_scenes("color_pick", 3, 72001)[0]
    sample = DomainRandomizationSample(
        tier="medium",
        seed=91,
        brightness=0.75,
        contrast=0.8,
        saturation=0.9,
        gaussian_noise_std=0.02,
    )
    scene = type(scene)(
        **{
            **scene.__dict__,
            "overrides": {
                **scene.overrides,
                "domain_randomization": sample.as_overrides(),
            },
        }
    )
    backend = _Backend()
    backend.raw["agentview_image"].fill(128)
    env = UR5eColorPickEnv(backend)
    first, _ = env.reset(scene=scene)
    second, _ = env.reset(scene=scene)
    assert np.array_equal(first[IMAGE_KEY], second[IMAGE_KEY])
    assert not np.all(first[IMAGE_KEY] == 128)


def test_color_pick_success_requires_requested_cube_grasp_and_lift_hold() -> None:
    backend = _Backend()
    env = UR5eColorPickEnv(backend, UR5eColorPickConfig(success_hold_steps=2))
    scene = generate_color_pick_scenes("color_pick", 3, 73000)[0]
    scene = type(scene)(
        **{**scene.__dict__, "overrides": {**scene.overrides, "target_color": "green"}}
    )
    env.reset(scene=scene)

    backend.grasped_color = "green"
    backend.raw["green_cube_pos"] = backend.raw["green_cube_pos"].copy()
    backend.raw["green_cube_pos"][2] += 0.09
    for expected in (False, True):
        _, _, terminated, _, info = env.step(np.zeros(7))
        assert info["success"] is expected
        assert terminated is expected
    assert info["target_color"] == "green"
    assert info["target_lift_m"] >= 0.08


def test_color_pick_wrong_color_grasp_terminates_as_failure() -> None:
    backend = _Backend()
    env = UR5eColorPickEnv(backend)
    scene = generate_color_pick_scenes("color_pick", 3, 74000)[0]
    target = str(scene.overrides["target_color"])
    wrong = next(color for color in COLOR_PICK_COLORS if color != target)
    env.reset(scene=scene)
    backend.grasped_color = wrong
    _, reward, terminated, truncated, info = env.step(np.zeros(7))
    assert terminated and not truncated
    assert reward == -1.0
    assert not info["success"]
    assert info["ever_wrong_object_grasped"]
    assert info["wrong_colors_grasped"] == [wrong]


def test_color_pick_expert_and_contract_use_requested_color_prompt() -> None:
    expert = HeuristicColorPickExpert("blue")
    observation = _raw()
    observation["target_cube_pos"] = np.asarray([0.08, 0.0, 0.825])
    observation["robot0_eef_pos"] = np.asarray([0.0, 0.0, 0.925])
    action = expert.act(observation)
    assert expert.prompt == "pick up the blue cube"
    assert action[0] > 0
    assert color_pick_prompt("red") == "pick up the red cube"
    assert color_pick_prompts() == (
        "pick up the red cube",
        "pick up the green cube",
        "pick up the blue cube",
    )
    contract = build_color_pick_contract(UR5eColorPickConfig())
    assert contract["policy_visible_target_signal"] == "language_only"
    assert contract["prompts"] == list(color_pick_prompts())
