from __future__ import annotations

import math

from vla_sim.scenes import (
    PUSH_ANGLE_BINS_RAD,
    PUSH_DISTANCE_BINS_M,
    STACK_DISTANCE_BINS_M,
    SceneSpec,
    attach_domain_randomization,
    generate_push_scenes,
    generate_stack_scenes,
    load_manifest,
    load_manifest_metadata,
    oriented_rectangles_overlap,
    save_manifest,
    select_stratified_scenes,
    select_targeted_push_recovery_scenes,
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


def test_targeted_push_recovery_selection_balances_fresh_hard_cells() -> None:
    pool = attach_domain_randomization(
        generate_push_scenes("targeted_pool", 6000, 410_000),
        tier_counts={"light": 1200, "medium": 4800},
        seed=460_000,
    )
    selected = select_targeted_push_recovery_scenes(pool, count=900)

    assert len(selected) == 900
    assert [scene.overrides["angle_bin"] for scene in selected[::2]] == [1] * 450
    assert [scene.overrides["angle_bin"] for scene in selected[1::2]] == [4] * 450
    assert {scene.overrides["distance_bin"] for scene in selected} == {1}
    assert {scene.overrides["domain_randomization"]["tier"] for scene in selected} == {
        "medium"
    }
    assert all(
        PUSH_ANGLE_BINS_RAD[int(scene.overrides["angle_bin"])][0]
        <= float(scene.overrides["target_angle_rad"])
        <= PUSH_ANGLE_BINS_RAD[int(scene.overrides["angle_bin"])][1]
        for scene in selected
    )
    assert all(
        PUSH_DISTANCE_BINS_M[1][0]
        <= float(scene.overrides["target_distance_m"])
        <= PUSH_DISTANCE_BINS_M[1][1]
        for scene in selected
    )


def test_stratified_scene_selection_round_robins_groups_deterministically() -> None:
    scenes = [
        SceneSpec(f"scene-{index}", index, 0.0, 0.0, 0.0, overrides={"group": index % 3})
        for index in range(12)
    ]

    selected = select_stratified_scenes(
        scenes, count=7, stratum=lambda scene: scene.overrides["group"]
    )

    assert [scene.scene_id for scene in selected] == [
        "scene-0",
        "scene-1",
        "scene-2",
        "scene-3",
        "scene-4",
        "scene-5",
        "scene-6",
    ]
