"""Generate frozen manifests for the three-color target-selection task."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import (
    attach_domain_randomization,
    generate_color_pick_scenes,
    save_manifest,
)


def _write(
    name: str,
    count: int,
    seed: int,
    role: str,
    tier_counts: dict[str, int],
) -> Path:
    scenes = generate_color_pick_scenes(name, count, seed)
    scenes = attach_domain_randomization(
        scenes,
        tier_counts=tier_counts,
        seed=seed + 50_000,
    )
    return save_manifest(
        ROOT / "configs" / "benchmarks" / f"{name}.json",
        scenes,
        benchmark_id=name,
        role=role,
        generator_seed=seed,
        environment_preset="color_pick_v1",
    )


def main() -> int:
    # Match the robust two-task protocol: training sees nominal/light/medium,
    # while standard, randomized, blind, and stress evaluation remain isolated.
    for name, count, seed, role, tier_counts in (
        ("color_pick_smoke_v1", 3, 73_000, "development", {"nominal": 3}),
        (
            "color_pick_collection_v1",
            1_500,
            74_000,
            "collection",
            {"nominal": 300, "light": 600, "medium": 600},
        ),
        (
            "color_pick_development_v1",
            60,
            75_000,
            "development",
            {"nominal": 60},
        ),
        (
            "color_pick_development_randomized_v1",
            60,
            76_000,
            "development",
            {"light": 30, "medium": 30},
        ),
        ("color_pick_blind_v1", 60, 77_000, "blind", {"blind": 60}),
        ("color_pick_stress_v1", 30, 78_000, "test", {"stress": 30}),
    ):
        path = _write(name, count, seed, role, tier_counts)
        print(f"{path} tiers={tier_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
