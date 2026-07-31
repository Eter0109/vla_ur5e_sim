"""Generate frozen native-bin PickPlace collection and evaluation manifests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import generate_pick_place_scenes, save_manifest  # noqa: E402


def main() -> int:
    output = ROOT / "configs" / "benchmarks"
    for name, count, seed, role in (
        ("pick_place_screen_v1", 24, 61000, "development"),
        ("pick_place_dev_v1", 100, 62000, "development"),
        ("pick_place_blind_v1", 100, 63000, "blind"),
        ("pick_place_collect_v1", 1200, 64000, "collection"),
        ("pick_place_test_v2_50", 50, 65000, "test"),
        ("pick_place_holdout_v4_50", 50, 67000, "test"),
    ):
        path = save_manifest(
            output / f"{name}.json",
            generate_pick_place_scenes(name, count, seed),
            benchmark_id=name,
            role=role,
            generator_seed=seed,
            environment_preset="pick_place_v1",
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
