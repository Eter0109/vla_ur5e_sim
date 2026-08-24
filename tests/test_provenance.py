import pytest

from vla_sim.provenance import (
    collection_resume_source_index,
    immutable_mismatches,
    scene_seed_overlap,
    targeted_push_distribution_errors,
)


def test_provenance_rejects_dataset_or_inference_argument_changes():
    previous = {"dataset_sha256": "a", "arguments_sha256": "b", "checkpoint": "c"}
    current = {"dataset_sha256": "changed", "arguments_sha256": "b", "checkpoint": "c"}
    assert immutable_mismatches(
        previous, current, ("dataset_sha256", "arguments_sha256", "checkpoint")
    ) == ["dataset_sha256"]


def _push_scene(seed: int, randomization_seed: int, angle_bin: int) -> dict:
    return {
        "seed": seed,
        "env_seed": seed,
        "overrides": {
            "angle_bin": angle_bin,
            "distance_bin": 1,
            "domain_randomization": {"tier": "medium", "seed": randomization_seed},
        },
    }


def test_scene_seed_overlap_covers_environment_and_randomization_seeds() -> None:
    candidates = [_push_scene(100, 200, 1), _push_scene(101, 201, 4)]
    references = [_push_scene(101, 999, 1), _push_scene(999, 200, 4)]

    assert scene_seed_overlap(candidates, references) == {
        "environment_seed": [101],
        "domain_randomization_seed": [200],
    }


def test_targeted_push_distribution_requires_balanced_medium_far_cells() -> None:
    valid = [_push_scene(100, 200, 1), _push_scene(101, 201, 4)]
    assert targeted_push_distribution_errors(valid, expected_count=2) == []

    invalid = [_push_scene(100, 200, 1), _push_scene(101, 201, 1)]
    invalid[0]["overrides"]["distance_bin"] = 0
    invalid[1]["overrides"]["domain_randomization"]["tier"] = "light"
    assert targeted_push_distribution_errors(invalid, expected_count=2) == [
        "angle_bin_imbalance=2 maximum=0 (angle1=2,angle4=0)",
        "non_far_distance_scene",
        "non_medium_randomization_scene",
    ]

    almost_balanced = [
        _push_scene(100, 200, 1),
        _push_scene(101, 201, 1),
        _push_scene(102, 202, 4),
        _push_scene(103, 203, 1),
    ]
    assert targeted_push_distribution_errors(
        almost_balanced, expected_count=4, max_angle_imbalance=2
    ) == []

    assert targeted_push_distribution_errors(
        almost_balanced, expected_count=4, max_angle_imbalance=1
    ) == ["angle_bin_imbalance=2 maximum=1 (angle1=3,angle4=1)"]


def test_collection_resume_uses_last_processed_source_after_skipped_failure() -> None:
    progress = [
        {"source_index": 0, "success": True},
        {"source_index": 1, "success": False},
        {"source_index": 2, "success": True},
    ]

    assert collection_resume_source_index(
        progress, dataset_episodes=2, scene_count=10
    ) == 3


def test_collection_resume_rejects_progress_dataset_mismatch_and_duplicates() -> None:
    with pytest.raises(ValueError, match="progress has 1 successes"):
        collection_resume_source_index(
            [{"source_index": 0, "success": True}],
            dataset_episodes=2,
            scene_count=10,
        )
    with pytest.raises(ValueError, match="duplicate source indices"):
        collection_resume_source_index(
            [
                {"source_index": 0, "success": True},
                {"source_index": 0, "success": True},
            ],
            dataset_episodes=2,
            scene_count=10,
        )
