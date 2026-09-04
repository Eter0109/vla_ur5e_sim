from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vla_sim.domain_randomization import (
    Sim2RealEpisodeRuntime,
    apply_domain_randomization,
    capture_render_baseline,
    euler_xyz_quaternion,
    look_at_quaternion,
    photometric_randomize,
    points_visible,
    quaternion_to_matrix,
    sample_domain_randomization,
    sample_sim2real_v2,
)
from vla_sim.scenes import (
    attach_domain_randomization,
    attach_sim2real_v2_randomization,
    generate_push_scenes,
)


def test_randomization_is_reproducible_and_tiers_are_distinct() -> None:
    assert sample_domain_randomization("medium", 17) == sample_domain_randomization("medium", 17)
    assert sample_domain_randomization("nominal", 17) != sample_domain_randomization("light", 17)


def test_scene_assignment_keeps_requested_tier_mix() -> None:
    scenes = generate_push_scenes("test", 10, 3)
    assigned = attach_domain_randomization(scenes, tier_counts={"nominal": 2, "light": 4, "medium": 4}, seed=4)
    tiers = [scene.overrides["domain_randomization"]["tier"] for scene in assigned]
    assert tiers.count("nominal") == 2
    assert tiers.count("light") == 4
    assert tiers.count("medium") == 4


def test_photometric_randomization_is_deterministic_and_shape_safe() -> None:
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    sample = sample_domain_randomization("medium", 6)
    first = photometric_randomize(image, sample, stream=1)
    assert np.array_equal(first, photometric_randomize(image, sample, stream=1))
    assert first.shape == image.shape
    assert first.dtype == np.uint8


def test_camera_math_and_visibility_contract() -> None:
    position = np.asarray([0.5, 0.0, 1.35])
    target = np.asarray([0.0, 0.0, 0.8])
    rotation = quaternion_to_matrix(look_at_quaternion(position, target))
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    assert points_visible([(0.0, 0.0, 0.8)], position, target, 45.0)
    assert not points_visible([(5.0, 0.0, 0.8)], position, target, 45.0)
    assert np.allclose(quaternion_to_matrix(euler_xyz_quaternion((0.0, 0.0, 0.0))), np.eye(3))


def test_sim2real_v2_sampling_and_stratified_mix_are_exact() -> None:
    assert sample_sim2real_v2("medium", 31) == sample_sim2real_v2("medium", 31)
    assert sample_sim2real_v2("medium", 31).schema_version == 2
    scenes = attach_sim2real_v2_randomization(
        generate_push_scenes("v2", 100, 40),
        group_fields=("angle_bin", "distance_bin"),
        seed=50,
    )
    for angle_bin in range(5):
        for distance_bin in range(2):
            tiers = [
                scene.overrides["domain_randomization"]["tier"]
                for scene in scenes
                if scene.overrides["angle_bin"] == angle_bin
                and scene.overrides["distance_bin"] == distance_bin
            ]
            assert tiers.count("nominal") == 2
            assert tiers.count("light") == 5
            assert tiers.count("medium") == 3


def test_sim2real_runtime_delay_gain_noise_and_reset_are_deterministic() -> None:
    sample = sample_sim2real_v2("medium", 7)
    sample = type(sample)(
        **{
            **sample.__dict__,
            "temporal_mode": "action_delay",
            "translation_action_gain": 0.95,
            "rotation_action_gain": 1.05,
        }
    )
    runtime = Sim2RealEpisodeRuntime()
    runtime.reset(sample)
    first = np.asarray([0.5, 0.2, -0.1, 0.4, 0.1, -0.2, -1.0], dtype=np.float32)
    second = -first
    assert np.allclose(runtime.execution_action(first)[:3], first[:3] * 0.95)
    assert np.allclose(runtime.execution_action(second)[:3], first[:3] * 0.95)
    observation = {
        "observation.images.front": np.full((8, 8, 3), 128, dtype=np.uint8),
        "observation.state": np.zeros(10, dtype=np.float32),
    }
    transformed = runtime.policy_observation(observation, 3)
    runtime.reset(sample)
    repeated = runtime.policy_observation(observation, 3)
    assert np.array_equal(transformed["observation.images.front"], repeated["observation.images.front"])
    assert np.array_equal(transformed["observation.state"], repeated["observation.state"])


def test_color_sensitive_sim2real_transform_preserves_dominant_channel() -> None:
    colors = np.asarray(
        [[220, 30, 20], [20, 200, 40], [20, 50, 220]], dtype=np.uint8
    )
    for seed in range(20):
        sample = sample_sim2real_v2("medium", seed, color_sensitive=True)
        dominant_channels = []
        for color in colors:
            patch = np.broadcast_to(color, (16, 16, 3)).copy()
            transformed = photometric_randomize(patch, sample, stream=0)
            dominant_channels.append(int(np.argmax(transformed.mean(axis=(0, 1)))))
        assert dominant_channels == [0, 1, 2]


def test_nominal_v2_policy_io_is_byte_identical() -> None:
    sample = sample_sim2real_v2("nominal", 19)
    runtime = Sim2RealEpisodeRuntime()
    runtime.reset(sample)
    action = np.linspace(-1.0, 1.0, 7, dtype=np.float32)
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    state = np.linspace(-0.5, 0.5, 10, dtype=np.float32)
    transformed = runtime.policy_observation(
        {"observation.images.front": image, "observation.state": state},
        0,
    )
    assert runtime.execution_action(action).tobytes() == action.tobytes()
    assert transformed["observation.images.front"].tobytes() == image.tobytes()
    assert transformed["observation.state"].tobytes() == state.tobytes()


def test_image_delay_is_exactly_one_step_and_reset_clears_it() -> None:
    sample = sample_sim2real_v2("medium", 29)
    sample = type(sample)(**{**sample.__dict__, "temporal_mode": "image_delay"})
    runtime = Sim2RealEpisodeRuntime()
    runtime.reset(sample)
    first = np.full((8, 8, 3), 10, dtype=np.uint8)
    second = np.full((8, 8, 3), 240, dtype=np.uint8)
    first_output = runtime.policy_observation({"front": first}, 0)["front"]
    second_output = runtime.policy_observation({"front": second}, 1)["front"]
    expected_first = photometric_randomize(first, sample, stream=0)
    assert np.array_equal(first_output, expected_first)
    assert np.array_equal(second_output, expected_first)
    runtime.reset(sample)
    reset_output = runtime.policy_observation({"front": second}, 1)["front"]
    assert np.array_equal(reset_output, photometric_randomize(second, sample, stream=17))


class _FakeModel:
    def __init__(self) -> None:
        self.cam_pos = np.asarray([[0.5, 0.0, 1.35], [0.0, 0.0, 0.0]], dtype=float)
        self.cam_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=float)
        self.cam_fovy = np.asarray([45.0, 60.0], dtype=float)
        self.light_pos = np.asarray([[0.0, 0.0, 2.0]], dtype=float)
        self.light_diffuse = np.ones((1, 3), dtype=float)
        self.light_ambient = np.zeros((1, 3), dtype=float)
        self.light_specular = np.ones((1, 3), dtype=float)
        self.geom_rgba = np.ones((4, 4), dtype=float)
        self.geom_friction = np.asarray([[1.0, 0.1, 0.01]] * 4, dtype=float)
        self.body_mass = np.asarray([0.0, 0.2], dtype=float)
        self.body_inertia = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]], dtype=float)
        self.ngeom = 4
        self.nlight = 1
        self._geom_names = ["table", "floor", "cube_g0", "right_fingerpad_collision"]

    def geom_id2name(self, index: int) -> str:
        return self._geom_names[index]

    def geom_name2id(self, name: str) -> int:
        return self._geom_names.index(name)

    def camera_name2id(self, name: str) -> int:
        return {"frontview": 0, "robot0_eye_in_hand": 1}[name]

    def body_name2id(self, name: str) -> int:
        return {"cube_main": 1}[name]


def test_v2_physical_parameters_enter_model_and_nominal_reset_restores_baseline() -> None:
    model = _FakeModel()
    simulation = SimpleNamespace(model=model, forward=lambda: None)
    backend = SimpleNamespace(sim=simulation)
    baseline = capture_render_baseline(backend)
    assert baseline is not None
    sample = sample_sim2real_v2("medium", 37)
    apply_domain_randomization(
        backend,
        baseline,
        sample,
        front_camera_name="frontview",
        front_look_at_m=(0.0, 0.0, 0.8),
        object_body_names=("cube_main",),
        object_geom_names=("cube_g0",),
    )
    assert np.isclose(model.body_mass[1], baseline.body_mass[1] * sample.object_mass_scale)
    assert np.allclose(
        model.geom_friction[2], baseline.geom_friction[2] * sample.object_friction_scale
    )
    assert np.allclose(
        model.geom_friction[3], baseline.geom_friction[3] * sample.gripper_friction_scale
    )
    assert not np.array_equal(model.cam_pos, baseline.camera_positions)
    apply_domain_randomization(
        backend,
        baseline,
        sample_sim2real_v2("nominal", 38),
        front_camera_name="frontview",
        front_look_at_m=(0.0, 0.0, 0.8),
        object_body_names=("cube_main",),
        object_geom_names=("cube_g0",),
    )
    assert np.array_equal(model.cam_pos, baseline.camera_positions)
    assert np.array_equal(model.body_mass, baseline.body_mass)
    assert np.array_equal(model.geom_friction, baseline.geom_friction)
