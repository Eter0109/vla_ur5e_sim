"""Generate deterministic three-task Sim2Real-v2 collection and preflight manifests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import (
    SceneSpec,
    attach_sim2real_v2_randomization,
    generate_color_pick_scenes,
    generate_pick_place_scenes,
    generate_push_scenes,
    save_manifest,
)

PRIMARY_COUNT = 1_500
RESERVE_COUNT = 300
RANDOMIZATION_SEED_OFFSET = 500_000


def _partition(scenes: list[SceneSpec], name: str) -> list[SceneSpec]:
    return [
        SceneSpec(
            scene_id=scene.scene_id,
            seed=scene.seed,
            env_seed=scene.env_seed,
            x_m=scene.x_m,
            y_m=scene.y_m,
            yaw_rad=scene.yaw_rad,
            overrides={**scene.overrides, "candidate_partition": name},
        )
        for scene in scenes
    ]


def _preflight(
    scenes: list[SceneSpec],
    group_fields: tuple[str, ...],
) -> list[SceneSpec]:
    selected: list[SceneSpec] = []
    groups = sorted(
        {
            tuple(scene.overrides[field] for field in group_fields)
            for scene in scenes[:PRIMARY_COUNT]
        },
        key=str,
    )
    for tier_index, tier in enumerate(("nominal", "light", "medium")):
        base, remainder = divmod(10, len(groups))
        for group_index, group in enumerate(groups):
            rotated_index = (group_index - tier_index) % len(groups)
            count = base + int(rotated_index < remainder)
            matches = [
                scene
                for scene in scenes[:PRIMARY_COUNT]
                if scene.overrides["domain_randomization"]["tier"] == tier
                and tuple(scene.overrides[field] for field in group_fields) == group
            ]
            selected.extend(matches[:count])
    if len(selected) != 30:
        raise AssertionError("preflight selection must contain exactly 30 scenes")
    return selected


def _write_task(
    task: str,
    seed: int,
    group_fields: tuple[str, ...],
    *,
    color_sensitive: bool = False,
) -> None:
    generators = {
        "push": generate_push_scenes,
        "pick_place": generate_pick_place_scenes,
        "color_pick": generate_color_pick_scenes,
    }
    generator = generators[task]
    benchmark_id = f"{task}_sim2real_v2_collection"
    primary = _partition(
        generator(f"{benchmark_id}-primary", PRIMARY_COUNT, seed),
        "primary",
    )
    reserve = _partition(
        generator(f"{benchmark_id}-reserve", RESERVE_COUNT, seed + 10_000),
        "reserve",
    )
    primary = attach_sim2real_v2_randomization(
        primary,
        group_fields=group_fields,
        seed=seed + RANDOMIZATION_SEED_OFFSET,
        color_sensitive=color_sensitive,
    )
    reserve = attach_sim2real_v2_randomization(
        reserve,
        group_fields=group_fields,
        seed=seed + RANDOMIZATION_SEED_OFFSET + PRIMARY_COUNT,
        color_sensitive=color_sensitive,
    )
    scenes = primary + reserve
    output = ROOT / "configs" / "benchmarks"
    collection_path = save_manifest(
        output / f"{benchmark_id}.json",
        scenes,
        benchmark_id=benchmark_id,
        role="collection",
        generator_seed=seed,
        environment_preset=f"{task}_sim2real_v2",
        extra_metadata={
            "randomization_schema_version": 2,
            "primary_candidates": PRIMARY_COUNT,
            "reserve_candidates": RESERVE_COUNT,
            "accepted_episode_target": PRIMARY_COUNT,
            "tier_mix": {"nominal": 0.20, "light": 0.50, "medium": 0.30},
            "group_fields": list(group_fields),
        },
    )
    preflight_id = f"{task}_sim2real_v2_preflight30"
    preflight_path = save_manifest(
        output / f"{preflight_id}.json",
        _preflight(scenes, group_fields),
        benchmark_id=preflight_id,
        role="preflight",
        generator_seed=seed,
        environment_preset=f"{task}_sim2real_v2",
        extra_metadata={
            "randomization_schema_version": 2,
            "source_manifest": str(collection_path.relative_to(ROOT)),
            "tier_counts": {"nominal": 10, "light": 10, "medium": 10},
        },
    )
    print(f"manifest_ok task={task} collection={collection_path} preflight={preflight_path}")


def main() -> int:
    _write_task("push", 310_000, ("angle_bin", "distance_bin"))
    _write_task("pick_place", 320_000, ("distance_bin",))
    _write_task("color_pick", 330_000, ("target_color",), color_sensitive=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
