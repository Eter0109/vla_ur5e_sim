"""Apply the fixed three-seed P4 promotion gate to validation rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.evaluation import seed_promotion_gate, summarize_results  # noqa: E402


def _load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed1000", type=Path, required=True)
    parser.add_argument("--seed1001", type=Path, required=True)
    parser.add_argument("--seed1002", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = {
        seed: summarize_results(_load(getattr(args, seed)))
        for seed in ("seed1000", "seed1001", "seed1002")
    }
    report = {"benchmark": "validation_v2", "seeds": summaries, "promotion": seed_promotion_gate(summaries.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
