"""Helpers for rejecting mixed experiment provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def immutable_mismatches(
    previous: Mapping[str, Any], current: Mapping[str, Any], fields: Sequence[str]
) -> list[str]:
    """Return locked fields whose values differ between two metadata records."""

    return [field for field in fields if previous.get(field) != current.get(field)]


def scene_seed_overlap(
    candidates: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]
) -> dict[str, list[int]]:
    """Return environment and domain-randomization seeds shared by two splits."""

    def environment_seed(scene: Mapping[str, Any]) -> int:
        value = scene.get("env_seed")
        return int(scene.get("seed") if value is None else value)

    def randomization_seed(scene: Mapping[str, Any]) -> int | None:
        value = scene.get("overrides", {}).get("domain_randomization", {}).get("seed")
        return None if value is None else int(value)

    candidate_environment = {environment_seed(scene) for scene in candidates}
    reference_environment = {environment_seed(scene) for scene in references}
    candidate_randomization = {
        value for scene in candidates if (value := randomization_seed(scene)) is not None
    }
    reference_randomization = {
        value for scene in references if (value := randomization_seed(scene)) is not None
    }
    return {
        "environment_seed": sorted(candidate_environment & reference_environment),
        "domain_randomization_seed": sorted(candidate_randomization & reference_randomization),
    }


def targeted_push_distribution_errors(
    scenes: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    angle_bins: tuple[int, ...] = (1, 4),
    max_angle_imbalance: int = 0,
) -> list[str]:
    """Return violations of the target-80 hard-cell Push collection contract."""

    errors: list[str] = []
    if len(scenes) != expected_count:
        errors.append(f"scene_count={len(scenes)} expected={expected_count}")
    counts = {angle_bin: 0 for angle_bin in angle_bins}
    for scene in scenes:
        overrides = scene.get("overrides", {})
        angle_bin = int(overrides.get("angle_bin", -1))
        if angle_bin not in counts:
            errors.append(f"unexpected_angle_bin={angle_bin}")
        else:
            counts[angle_bin] += 1
        if int(overrides.get("distance_bin", -1)) != 1:
            errors.append("non_far_distance_scene")
        if overrides.get("domain_randomization", {}).get("tier") != "medium":
            errors.append("non_medium_randomization_scene")
    if max_angle_imbalance < 0:
        raise ValueError("max_angle_imbalance must be non-negative")
    imbalance = max(counts.values()) - min(counts.values())
    if imbalance > max_angle_imbalance:
        detail = ",".join(f"angle{key}={value}" for key, value in counts.items())
        errors.append(
            f"angle_bin_imbalance={imbalance} maximum={max_angle_imbalance} ({detail})"
        )
    return sorted(set(errors))


def collection_resume_source_index(
    progress: Sequence[Mapping[str, Any]], *, dataset_episodes: int, scene_count: int
) -> int:
    """Return the next source scene while validating persisted collection progress."""

    successful = [entry for entry in progress if bool(entry.get("success"))]
    if len(successful) != dataset_episodes:
        raise ValueError(
            f"collection progress has {len(successful)} successes but dataset has "
            f"{dataset_episodes} episodes"
        )
    source_indices = [int(entry["source_index"]) for entry in progress]
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("collection progress contains duplicate source indices")
    if source_indices != sorted(source_indices):
        raise ValueError("collection progress source indices are not monotonic")
    next_index = source_indices[-1] + 1 if source_indices else 0
    if not 0 <= next_index <= scene_count:
        raise ValueError("collection resume source index is outside the manifest")
    return next_index
