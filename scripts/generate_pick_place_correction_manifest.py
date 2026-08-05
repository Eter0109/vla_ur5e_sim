"""Generate a non-benchmark collection split for hard PickPlace transport."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import generate_pick_place_scenes, save_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=68000)
    parser.add_argument(
        "--target-y-bounds-m",
        type=float,
        nargs=2,
        default=(-0.160, -0.060),
        metavar=("MIN", "MAX"),
        help="Collection-only target lateral range; never reuse an evaluation manifest.",
    )
    parser.add_argument("--split", default="pick_place_correction_negative_y_v1")
    parser.add_argument(
        "--source-y-bounds-m",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional collection-only source lateral range for grasp recovery.",
    )
    args = parser.parse_args()
    scenes = generate_pick_place_scenes(
        args.split,
        args.episodes,
        args.seed,
        target_y_bounds_m=tuple(args.target_y_bounds_m),
        source_y_bounds_m=(tuple(args.source_y_bounds_m) if args.source_y_bounds_m else (-0.040, 0.040)),
        distance_bins=(1,),
    )
    save_manifest(
        args.output,
        scenes,
        benchmark_id=args.split,
        role="collection",
        generator_seed=args.seed,
        environment_preset="pick_place_v1",
    )
    print(f"manifest_ok output={args.output} episodes={len(scenes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
