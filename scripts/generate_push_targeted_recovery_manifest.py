"""Generate fresh hard-cell Push scenes for targeted recovery collection."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import (  # noqa: E402
    attach_domain_randomization,
    generate_push_scenes,
    save_manifest,
    select_targeted_push_recovery_scenes,
)

BENCHMARK_ID = "push_robust_targeted_recovery_collection_v2"
GEOMETRY_SEED = 410_000
RANDOMIZATION_SEED = 460_000
POOL_SIZE = 6000
CANDIDATE_COUNT = 900


def main() -> int:
    pool = generate_push_scenes(f"{BENCHMARK_ID}_pool", POOL_SIZE, GEOMETRY_SEED)
    pool = attach_domain_randomization(
        pool,
        tier_counts={"light": 1200, "medium": 4800},
        seed=RANDOMIZATION_SEED,
    )
    selected = select_targeted_push_recovery_scenes(pool, count=CANDIDATE_COUNT)
    scenes = [
        replace(scene, scene_id=f"{BENCHMARK_ID}-{index:04d}")
        for index, scene in enumerate(selected)
    ]
    path = save_manifest(
        ROOT / "configs" / "benchmarks" / f"{BENCHMARK_ID}.json",
        scenes,
        benchmark_id=BENCHMARK_ID,
        role="collection",
        generator_seed=GEOMETRY_SEED,
        environment_preset="push_robust_v1",
    )
    print(
        f"manifest_ok path={path} candidates={len(scenes)} "
        "tier=medium distance_bin=1 angle_bins=1,4"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
