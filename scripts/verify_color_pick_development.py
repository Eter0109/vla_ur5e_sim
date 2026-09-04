"""Verify >=90% overall and per-color success on the frozen 60-scene suite."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.artifact_identity import sha256_directory, sha256_file
from vla_sim.color_pick_contract import color_pick_prompt
from vla_sim.scenes import COLOR_PICK_COLORS

MANIFEST = ROOT / "configs" / "benchmarks" / "color_pick_development_v1.json"
EPISODES_PER_COLOR = 20
REQUIRED_PER_COLOR = 18


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _json(args.results)
    metadata = _json(args.results.with_suffix(args.results.suffix + ".meta.json"))
    expected_checkpoint = sha256_directory(args.checkpoint)
    errors: list[str] = []
    if len(rows) != EPISODES_PER_COLOR * len(COLOR_PICK_COLORS):
        errors.append(f"expected 60 episodes, got {len(rows)}")
    if metadata.get("manifest_sha256") != sha256_file(MANIFEST):
        errors.append("manifest sha256 mismatch")
    if str(metadata.get("checkpoint_sha256", "")).lower() != expected_checkpoint.lower():
        errors.append("checkpoint sha256 mismatch")
    if metadata.get("control_mode") != "vla_raw_safety":
        errors.append("control mode mismatch")
    if metadata.get("target_signal") != "language_only":
        errors.append("target signal must be language_only")
    expected_inference = {
        "samples_per_plan": 1,
        "replan_steps": 4,
        "temporal_ensemble_decay": 0.75,
        "policy_seed": 1000,
    }
    if metadata.get("inference") != expected_inference:
        errors.append("inference contract mismatch")

    totals = Counter(str(row.get("target_color")) for row in rows)
    successes = Counter(
        str(row.get("target_color")) for row in rows if bool(row.get("success"))
    )
    wrong_grasps = Counter(
        str(row.get("target_color"))
        for row in rows
        if bool(row.get("ever_wrong_object_grasped"))
    )
    for color in COLOR_PICK_COLORS:
        if totals[color] != EPISODES_PER_COLOR:
            errors.append(f"{color}: expected 20 episodes, got {totals[color]}")
        if successes[color] < REQUIRED_PER_COLOR:
            errors.append(
                f"{color}: required {REQUIRED_PER_COLOR}/20, got {successes[color]}/20"
            )
    for row in rows:
        color = str(row.get("target_color"))
        if color in COLOR_PICK_COLORS and row.get("prompt") != color_pick_prompt(color):
            errors.append(f"{row.get('scene_id')}: prompt/target mismatch")

    report = {
        "schema_version": 1,
        "threshold": 0.9,
        "required_per_color": REQUIRED_PER_COLOR,
        "checkpoint_sha256": expected_checkpoint,
        "episodes": len(rows),
        "successes": sum(successes.values()),
        "wrong_color_grasps": sum(wrong_grasps.values()),
        "by_target_color": {
            color: {
                "episodes": totals[color],
                "successes": successes[color],
                "wrong_color_grasps": wrong_grasps[color],
            }
            for color in COLOR_PICK_COLORS
        },
        "errors": errors,
        "passed": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
