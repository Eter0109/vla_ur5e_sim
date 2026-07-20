"""Summarize rollout JSON results and optionally compare a frozen baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.evaluation import paired_comparison, summarize_results  # noqa: E402


def _load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON result list: {path}")
    return value


def _markdown(report: dict) -> str:
    candidate = report["candidate"]
    interval = candidate["wilson_95"]
    lines = [
        "# Rollout summary",
        "",
        f"Success: **{candidate['successes']}/{candidate['episodes']}** "
        f"({candidate['success_rate']:.1%}); Wilson 95%: "
        f"{interval['lower']:.1%}–{interval['upper']:.1%}.",
        "",
        "## Failure stages",
        "",
    ]
    lines.extend(f"- {stage}: {count}" for stage, count in candidate["failure_stages"].items())
    if "paired" in report:
        paired = report["paired"]
        lines.extend(
            [
                "",
                "## Paired baseline comparison",
                "",
                f"- Candidate-only successes: {paired['candidate_only_success']}",
                f"- Baseline-only successes: {paired['baseline_only_success']}",
                f"- Exact McNemar p-value: {paired['mcnemar_exact_pvalue']:.4g}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {"candidate": summarize_results(_load(args.candidate))}
    if args.baseline:
        report["baseline"] = summarize_results(_load(args.baseline))
        report["paired"] = paired_comparison(_load(args.candidate), _load(args.baseline))
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
