from __future__ import annotations

import math

from vla_sim.simulation.scenes import (
    SceneSpec,
    load_manifest,
    load_manifest_metadata,
    oriented_rectangles_overlap,
    save_manifest,
    select_stratified_scenes,
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
