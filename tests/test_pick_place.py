from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vla_sim.contracts import PickPlaceObservationAdapter, validate_pick_place_observation
from vla_sim.envs.ur5e_pick_place import UR5ePickPlaceConfig, UR5ePickPlaceEnv
from vla_sim.pick_place_control import (
    VLA_ONLY_CONTROL_MODES,
    VLAOnlyActionCalibration,
    VLAOnlyActionCalibrator,
    VLAOnlySafetyConfig,
    calibrate_vla_only_action,
    filter_vla_only_action,
    scene_policy_seed,
    uses_vla_only_action_calibration,
)
from vla_sim.scenes import PICK_PLACE_DISTANCE_BINS_M, generate_pick_place_scenes


def _raw() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((16, 16, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(6, dtype=np.float32),
        "robot0_eef_pos": np.asarray([0.0, 0.0, 0.90], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([-0.024, -0.024], dtype=np.float32),
        "cube_pos": np.asarray([0.06, 0.0, 0.827], dtype=np.float32),
    }


def test_vla_only_control_mode_declares_action_calibration() -> None:
    assert VLA_ONLY_CONTROL_MODES == ("vla_raw_safety", "vla_action_calibrated")
    assert not uses_vla_only_action_calibration("vla_raw_safety")
    assert uses_vla_only_action_calibration("vla_action_calibrated")
    with pytest.raises(ValueError, match="Unknown VLA-only control mode"):
        uses_vla_only_action_calibration("vla_rgbd_supervised")


def test_dual_camera_contract() -> None:
    observation = PickPlaceObservationAdapter().convert(_raw())
    assert observation["observation.images.front"].shape == (16, 16, 3)
    assert observation["observation.images.wrist"].shape == (16, 16, 3)
    assert validate_pick_place_observation(observation)["observation.state"].shape == (10,)


def test_pick_place_scene_generation_is_balanced_and_deterministic() -> None:
    scenes = generate_pick_place_scenes("pick_place", 100, 42)
    assert scenes == generate_pick_place_scenes("pick_place", 100, 42)
    assert [int(scene.overrides["distance_bin"]) for scene in scenes].count(0) == 50
    assert [int(scene.overrides["distance_bin"]) for scene in scenes].count(1) == 50
    for scene in scenes:
        target = scene.overrides
        distance = np.hypot(float(target["target_x_m"]) - scene.x_m, float(target["target_y_m"]) - scene.y_m)
        lower, upper = PICK_PLACE_DISTANCE_BINS_M[int(target["distance_bin"])]
        assert lower <= distance <= upper


def test_pick_place_scene_generation_can_target_negative_y_without_eval_leakage() -> None:
    scenes = generate_pick_place_scenes(
        "pick_place_hard_negative_y",
        40,
        66000,
        target_y_bounds_m=(-0.160, -0.080),
    )
    assert len(scenes) == 40
    assert {scene.overrides["distance_bin"] for scene in scenes} == {0, 1}
    assert all(-0.160 <= scene.overrides["target_y_m"] <= -0.080 for scene in scenes)
    assert all(not scene.scene_id.startswith("pick_place_screen_v1") for scene in scenes)


def test_pick_place_scene_generation_can_target_the_hard_distance_bin() -> None:
    scenes = generate_pick_place_scenes(
        "pick_place_correction", 12, 68000, target_y_bounds_m=(-0.160, -0.060), distance_bins=(1,)
    )
    assert {int(scene.overrides["distance_bin"]) for scene in scenes} == {1}
    assert all(-0.160 <= float(scene.overrides["target_y_m"]) <= -0.060 for scene in scenes)


def test_pick_place_scene_generation_can_focus_on_positive_y_grasp_entry() -> None:
    scenes = generate_pick_place_scenes(
        "pick_place_grasp_recovery", 12, 69000, source_y_bounds_m=(0.020, 0.040), distance_bins=(1,)
    )
    assert all(0.020 <= scene.y_m <= 0.040 for scene in scenes)


@pytest.mark.parametrize(
    "bounds",
    [
        (-0.17, -0.08),
        (-0.08, -0.08),
        (-0.07, -0.09),
        (0.08, 0.17),
    ],
)
def test_pick_place_scene_generation_rejects_invalid_target_y_bounds(
    bounds: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="target_y_bounds_m"):
        generate_pick_place_scenes("invalid", 2, 1, target_y_bounds_m=bounds)


class _Data:
    def get_body_xvelp(self, _name: str) -> np.ndarray:
        return np.zeros(3)

    def get_body_xvelr(self, _name: str) -> np.ndarray:
        return np.zeros(3)


class _Backend:
    action_spec = (-np.ones(7), np.ones(7))

    def __init__(self) -> None:
        self.raw = _raw()
        self.cube = SimpleNamespace(root_body="cube")
        self.robots = [SimpleNamespace(gripper=object())]
        self.sim = SimpleNamespace(data=_Data())
        self.grasped = False

    def step(self, _action):
        return self.raw, 0.0, False, {}

    def _check_grasp(self, _gripper, _cube) -> bool:
        return self.grasped


def test_success_requires_prior_grasp_lift_release_and_hold() -> None:
    backend = _Backend()
    env = UR5ePickPlaceEnv(backend, UR5ePickPlaceConfig(success_hold_steps=2))
    env._raw_observation = dict(backend.raw, target_zone_pos=np.asarray([0.06, 0.0, 0.802]))
    env._target_xyz = np.asarray([0.06, 0.0, 0.802])
    env._initial_cube_z = 0.827
    _, _, _, _, info = env.step(np.zeros(7))
    assert not info["success"]  # A push into the zone cannot pass.
    env._ever_grasped = True
    env._ever_lifted = True
    for expected in (False, True):
        _, _, terminated, _, info = env.step(np.zeros(7))
        assert info["success"] is expected
        assert terminated is expected


def test_vla_only_safety_preserves_xyz_and_gripper_inside_workspace() -> None:
    action = np.asarray([0.2, -0.3, 0.4, 0.7, -0.8, 0.9, 0.6], dtype=np.float32)
    filtered = filter_vla_only_action(action, eef_xyz=np.asarray([0.0, 0.0, 0.9]))
    np.testing.assert_allclose(filtered[:3], action[:3])
    np.testing.assert_allclose(filtered[3:6], 0.0)
    assert filtered[6] == action[6]


def test_vla_only_calibration_scales_only_closed_negative_y_model_output() -> None:
    config = VLAOnlyActionCalibration(
        closed_negative_y_gain=1.25,
        transport_direction_lock_steps=1,
    )
    closed_negative = calibrate_vla_only_action(
        np.asarray([0.1, -0.4, 0.2, 0.0, 0.0, 0.0, 0.3], dtype=np.float32),
        config,
    )
    assert np.isclose(closed_negative[1], -0.5)
    open_negative = calibrate_vla_only_action(
        np.asarray([0.1, -0.4, 0.2, 0.0, 0.0, 0.0, -0.3], dtype=np.float32),
        config,
    )
    assert np.isclose(open_negative[1], -0.4)
    closed_positive = calibrate_vla_only_action(
        np.asarray([0.1, 0.4, 0.2, 0.0, 0.0, 0.0, 0.3], dtype=np.float32),
        config,
    )
    assert np.isclose(closed_positive[1], 0.4)


def test_vla_only_calibrator_locks_transport_direction_and_ignores_lift() -> None:
    config = VLAOnlyActionCalibration(
        closed_negative_y_gain=1.3,
        transport_positive_x_gain=0.95,
    )
    calibrator = VLAOnlyActionCalibrator(config)
    lift = calibrator.calibrate(
        np.asarray([0.2, -0.4, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(lift[:3], [0.2, -0.4, 0.8])
    assert calibrator.transport_y_direction == 0

    transport_action = np.asarray(
        [0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    calibrator.calibrate(transport_action)
    calibrator.calibrate(transport_action)
    transport = calibrator.calibrate(transport_action)
    np.testing.assert_allclose(transport[:2], [0.76, -1.04])
    assert calibrator.transport_y_direction == -1

    positive_correction = calibrator.calibrate(
        np.asarray([0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(positive_correction[:2], [0.38, 0.2])


def test_vla_only_calibrator_does_not_amplify_negative_correction_after_positive_lock() -> None:
    calibrator = VLAOnlyActionCalibrator(
        VLAOnlyActionCalibration(closed_negative_y_gain=1.3)
    )
    positive_transport = np.asarray(
        [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    calibrator.calibrate(positive_transport)
    calibrator.calibrate(positive_transport)
    calibrator.calibrate(positive_transport)
    correction = calibrator.calibrate(
        np.asarray([0.4, -0.4, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    assert np.isclose(correction[1], -0.4)
    assert calibrator.transport_y_direction == 1


def test_vla_only_calibrator_uses_multiple_actions_to_reject_direction_outlier() -> None:
    calibrator = VLAOnlyActionCalibrator(
        VLAOnlyActionCalibration(
            closed_negative_y_gain=1.5,
            transport_direction_lock_steps=3,
        )
    )
    outlier = calibrator.calibrate(
        np.asarray([0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    assert np.isclose(outlier[1], -0.8)
    for _ in range(2):
        calibrated = calibrator.calibrate(
            np.asarray([0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
    assert np.isclose(calibrated[1], 0.8)
    assert calibrator.transport_y_direction == 1


def test_vla_only_safety_clamps_workspace_without_task_pose() -> None:
    config = VLAOnlySafetyConfig(workspace_high=(0.25, 0.25, 1.0))
    filtered = filter_vla_only_action(
        np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0]),
        eef_xyz=np.asarray([0.24, 0.24, 0.99]),
        config=config,
    )
    np.testing.assert_allclose(filtered[:3], 0.2, atol=1e-6)
    assert filtered[6] == -1.0


def test_vla_only_policy_seed_is_scene_stable() -> None:
    assert scene_policy_seed(2000, 61003) == 63003
    assert scene_policy_seed(2000, 61003) == scene_policy_seed(2000, 61003)
    assert scene_policy_seed(2000, 61004) != scene_policy_seed(2000, 61003)
