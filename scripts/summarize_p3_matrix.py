"""Create the P3 ablation decision record from completed validation rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.evaluation import paired_comparison, select_ablation_winner, summarize_results  # noqa: E402

SPECS = {
    "A": {"peak_lr": 2e-5, "transition_oversample_factor": 1, "window": 0},
    "B": {"peak_lr": 2e-5, "transition_oversample_factor": 3, "window": 5},
    "C": {"peak_lr": 5e-5, "transition_oversample_factor": 1, "window": 0},
    "D": {"peak_lr": 5e-5, "transition_oversample_factor": 3, "window": 5},
}


def _load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list: {path}")
    return value


def _markdown(report: dict) -> str:
    lines = [
        "# P3 ablation matrix summary",
        "",
        "| Group | Peak LR | Transition sampling | Success | Wilson 95% |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for group, value in report["groups"].items():
        spec = value["spec"]
        summary = value["summary"]
        interval = summary["wilson_95"]
        lines.append(
            f"| {group} | {spec['peak_lr']:.0e} | "
            f"{spec['transition_oversample_factor']}x/window={spec['window']} | "
            f"{summary['successes']}/{summary['episodes']} | "
            f"{interval['lower']:.1%}–{interval['upper']:.1%} |"
        )
    selection = report["selection"]
    lines.extend(
        [
            "",
            f"Selected candidate: **{selection['winner']}**.",
            f"Groups within one success of the high-water mark: {', '.join(selection['tied_within_one'])}.",
            "",
            "Paired comparisons are recorded in the adjacent JSON artifact.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rollouts = {group: _load(getattr(args, group.lower())) for group in SPECS}
    summaries = {group: summarize_results(results) for group, results in rollouts.items()}
    report = {
        "benchmark": "validation_v2",
        "groups": {group: {"spec": SPECS[group], "summary": summaries[group]} for group in SPECS},
        "selection": select_ablation_winner(summaries, SPECS),
        "paired_vs_d": {
            group: paired_comparison(rollouts[group], rollouts["D"])
            for group in ("A", "B", "C")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
