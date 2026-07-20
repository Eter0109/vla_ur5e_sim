from __future__ import annotations

from vla_sim.scenes import SceneSpec, load_manifest, load_manifest_metadata, save_manifest


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
