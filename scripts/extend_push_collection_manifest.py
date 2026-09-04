"""Append same-distribution reserve Push scenes to a fixed collection manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import attach_domain_randomization, generate_push_scenes, load_manifest, save_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserve", type=int, default=200)
    parser.add_argument("--seed", type=int, default=211_000)
    args = parser.parse_args()
    if args.reserve < 1:
        raise ValueError("reserve must be positive")
    base = load_manifest(args.base)
    reserve = generate_push_scenes("push_robust_collection_v1_reserve", args.reserve, args.seed)
    nominal = args.reserve // 5
    light = args.reserve * 2 // 5
    reserve = attach_domain_randomization(
        reserve,
        tier_counts={"nominal": nominal, "light": light, "medium": args.reserve - nominal - light},
        seed=args.seed + 50_000,
    )
    save_manifest(
        args.output,
        [*base, *reserve],
        benchmark_id="push_robust_collection_v1_with_reserve",
        role="collection",
        generator_seed=args.seed,
        environment_preset="push_robust_v1",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
