from __future__ import annotations

import numpy as np

from vla_sim.domain_randomization import (
    euler_xyz_quaternion,
    look_at_quaternion,
    photometric_randomize,
    points_visible,
    quaternion_to_matrix,
    sample_domain_randomization,
)
from vla_sim.scenes import attach_domain_randomization, generate_push_scenes


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
