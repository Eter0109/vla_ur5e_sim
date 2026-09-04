"""Evaluate one multitask checkpoint against the two fixed nominal 50-scene suites."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError(f"evaluation failed with exit code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--push-dataset", type=Path, required=True)
    parser.add_argument("--pick-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {args.output}")
    args.output.mkdir(parents=True)
    _run([
        sys.executable, "scripts/run_push_vla_only_benchmark.py",
        "--checkpoint", str(args.checkpoint),
        "--dataset-root", str(args.push_dataset),
        "--repo-id", "local/multitask_robust_push_1500",
        "--manifest", "configs/benchmarks/push_robust_development_nominal_v1.json",
        "--episodes", "50", "--horizon", "250", "--replan-steps", "4",
        "--temporal-decay", "0.75", "--policy-seed", "1000", "--samples-per-plan", "1",
        "--output", str(args.output / "push_nominal.json"),
    ])
    _run([
        sys.executable, "scripts/run_pick_place_vla_only.py",
        "--checkpoint", str(args.checkpoint),
        "--dataset-root", str(args.pick_dataset),
        "--repo-id", "local/multitask_robust_pick_place_1500",
        "--manifest", "configs/benchmarks/pick_place_robust_development_nominal_v1.json",
        "--episodes", "50", "--horizon", "250", "--replan-steps", "4",
        "--temporal-ensemble-decay", "0.75", "--samples-per-plan", "2",
        "--control-mode", "vla_action_calibrated", "--closed-negative-y-gain", "1.8",
        "--policy-seed", "1000", "--output", str(args.output / "pick_nominal.json"),
    ])
    _run([
        sys.executable, "scripts/verify_multitask_nominal90.py",
        "--checkpoint", str(args.checkpoint), "--push", str(args.output / "push_nominal.json"),
        "--pick", str(args.output / "pick_nominal.json"),
        "--output", str(args.output / "nominal90_gate.json"),
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
