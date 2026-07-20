"""Generate legacy splits or schema-v2 development/blind benchmark manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import generate_scenes, save_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "manifests")
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--validation", type=int, default=30)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument(
        "--v2", action="store_true", help="Generate validation_v2, test_v2, and ood_v1 manifests."
    )
    args = parser.parse_args()
    if args.v2:
        for split, count, seed, role in (
            ("validation_v2", 40, 45000, "development"),
            ("test_v2", 100, 46000, "blind"),
            ("ood_v1", 40, 47000, "diagnostic"),
        ):
            path = save_manifest(
                args.output / f"{split}.json",
                generate_scenes(split.replace("_", "-"), count, seed),
                benchmark_id=split,
                role=role,
                generator_seed=seed,
                environment_preset="legacy_20260719",
            )
            print(f"{split}: {count} scenes ({role}) -> {path}")
        return 0
    for split, count, seed in (
        ("train", args.train, 42000),
        ("validation", args.validation, 43000),
        ("test", args.test, 44000),
    ):
        path = save_manifest(args.output / f"{split}.json", generate_scenes(split, count, seed))
        print(f"{split}: {count} scenes -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
