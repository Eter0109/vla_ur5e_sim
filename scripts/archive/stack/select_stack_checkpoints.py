"""Archived Stack checkpoint selector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.evaluation import summarize_results  # noqa: E402


def score(path: Path) -> tuple[tuple[float, float, float, int], dict]:
    results = json.loads(path.read_text(encoding="utf-8"))
    summary = summarize_results(results)
    task_rates = [value["success_rate"] for value in summary["by_task"].values()]
    task_gap = abs(task_rates[0] - task_rates[1]) if len(task_rates) == 2 else 1.0
    step = int(path.stem.split("step")[-1].split("_")[0])
    ranking = (
        float(summary["success_rate"]),
        float(summary["stage_funnel"]["target_reached"] or 0.0),
        -task_gap,
        -step,
    )
    return ranking, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranked = sorted(
        ((ranking, path, summary) for path in args.results for ranking, summary in [score(path)]),
        reverse=True,
        key=lambda item: item[0],
    )
    selection = [
        {"path": str(path.resolve()), "ranking": ranking, "summary": summary}
        for ranking, path, summary in ranked[: args.top]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
