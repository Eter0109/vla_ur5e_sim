"""Generate non-overlapping deterministic train/validation/test scene files."""

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
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--validation", type=int, default=30)
    parser.add_argument("--test", type=int, default=50)
    args = parser.parse_args()
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
