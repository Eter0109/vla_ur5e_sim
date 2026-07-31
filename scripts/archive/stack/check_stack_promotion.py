"""Archived gate for Stack checkpoints before the one-time blind evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.evaluation import stack_seed_promotion_gate, summarize_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs=3, help="Three 120-scene development results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for path in args.results:
        values = json.loads(path.read_text(encoding="utf-8"))
        if len(values) != 120:
            raise ValueError(f"Development result must contain 120 scenes: {path}")
        summaries.append(summarize_results(values))
    result = stack_seed_promotion_gate(summaries)
    result["development_results"] = [str(path.resolve()) for path in args.results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
