from __future__ import annotations

import math

from vla_sim.scenes import (
    STACK_DISTANCE_BINS_M,
    SceneSpec,
    generate_stack_scenes,
    load_manifest,
    load_manifest_metadata,
    oriented_rectangles_overlap,
    save_manifest,
)


def test_v2_manifest_round_trip_preserves_metadata_and_environment_seed(tmp_path) -> None:
    destination = tmp_path / "validation_v2.json"
    save_manifest(
        destination,
        [SceneSpec("scene-0000", 100, 0.01, -0.02, 0.3, env_seed=200)],
        benchmark_id="validation_v2",
        role="development",
        generator_seed=45000,
        environment_preset="legacy_20260719",
    )
    scene = load_manifest(destination)[0]
    metadata = load_manifest_metadata(destination)
    assert scene.effective_env_seed == 200
    assert metadata["role"] == "development"
    assert metadata["benchmark_id"] == "validation_v2"


def test_legacy_manifest_keeps_seed_as_environment_seed(tmp_path) -> None:
    destination = tmp_path / "legacy.json"
    save_manifest(destination, [SceneSpec("scene-0000", 100, 0.0, 0.0, 0.0)])
    scene = load_manifest(destination)[0]
    assert scene.effective_env_seed == 100
    assert load_manifest_metadata(destination)["role"] == "legacy"


def test_oriented_rectangle_overlap_detects_collision_and_clearance() -> None:
    assert oriented_rectangles_overlap((0.0, 0.0), 0.0, (0.05, 0.05), (0.04, 0.0), 0.0, (0.05, 0.05))
    assert not oriented_rectangles_overlap(
        (0.0, 0.0), 0.0, (0.05, 0.05), (0.08, 0.0), math.pi / 4, (0.05, 0.05), clearance_m=0.005
    )


def test_stack_scene_generation_is_balanced_stratified_and_non_overlapping() -> None:
    scenes = generate_stack_scenes("stack", 120, 1234)
    assert scenes == generate_stack_scenes("stack", 120, 1234)
    tasks = [scene.overrides["task"] for scene in scenes]
    assert tasks.count("red_on_blue") == tasks.count("blue_on_red") == 60
    bins = [int(scene.overrides["distance_bin"]) for scene in scenes]
    assert [bins.count(index) for index in range(3)] == [40, 40, 40]
    for scene in scenes:
        b_x = float(scene.overrides["cubeB_x_m"])
        b_y = float(scene.overrides["cubeB_y_m"])
        distance = math.hypot(scene.x_m - b_x, scene.y_m - b_y)
        lower, upper = STACK_DISTANCE_BINS_M[int(scene.overrides["distance_bin"])]
        assert lower <= distance <= upper
        if distance < 0.087:
            assert abs((b_x - scene.x_m) / distance) >= 0.75
        assert not oriented_rectangles_overlap(
            (scene.x_m, scene.y_m),
            scene.yaw_rad,
            (0.05, 0.05),
            (b_x, b_y),
            float(scene.overrides["cubeB_yaw_rad"]),
            (0.05, 0.05),
            clearance_m=0.005,
        )


def test_blind_split_is_exactly_task_balanced() -> None:
    scenes = generate_stack_scenes("stack_blind_v1", 100, 57000)
    tasks = [scene.overrides["task"] for scene in scenes]
    assert tasks.count("red_on_blue") == tasks.count("blue_on_red") == 50


def test_collection_first_round_has_400_scenes_per_cell() -> None:
    scenes = generate_stack_scenes("stack_collect_v1", 3200, 58000)
    counts = {
        (task, distance_bin): sum(
            scene.overrides["task"] == task
            and scene.overrides["distance_bin"] == distance_bin
            for scene in scenes[:2400]
        )
        for task in ("red_on_blue", "blue_on_red")
        for distance_bin in range(3)
    }
    assert set(counts.values()) == {400}
