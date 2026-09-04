"""Verify one checkpoint reaches >=45/50 on both fixed nominal development manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.artifact_identity import sha256_directory, sha256_file

PUSH_MANIFEST = ROOT / "configs" / "benchmarks" / "push_robust_development_nominal_v1.json"
PICK_MANIFEST = ROOT / "configs" / "benchmarks" / "pick_place_robust_development_nominal_v1.json"
REQUIRED_SUCCESSES = 45


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _push(path: Path) -> dict[str, Any]:
    payload = _json(path)
    if payload.get("inference") != {
        "replan_steps": 4,
        "temporal_decay": 0.75,
        "policy_seed": 1000,
        "samples_per_plan": 1,
    }:
        raise ValueError("Push inference contract mismatch")
    if payload.get("control") != "vla_raw_safety_fixed_rotation_workspace_clamp":
        raise ValueError("Push control contract mismatch")
    summary = payload["summary"]
    return {
        "episodes": int(summary["episodes"]),
        "successes": int(summary["successes"]),
        "checkpoint_sha256": str(payload["checkpoint_sha256"]),
        "manifest_sha256": str(payload["manifest_sha256"]),
    }


def _pick(path: Path) -> dict[str, Any]:
    rows = _json(path)
    metadata = _json(path.with_suffix(path.suffix + ".meta.json"))
    if metadata.get("inference") != {
        "samples_per_plan": 2,
        "replan_steps": 4,
        "temporal_ensemble_decay": 0.75,
        "policy_seed": 1000,
    }:
        raise ValueError("PickPlace inference contract mismatch")
    if metadata.get("control_mode") != "vla_action_calibrated":
        raise ValueError("PickPlace control contract mismatch")
    if any(float(row.get("closed_negative_y_gain", 0.0)) != 1.8 for row in rows):
        raise ValueError("PickPlace gain contract mismatch")
    return {
        "episodes": len(rows),
        "successes": sum(bool(row.get("success")) for row in rows),
        "checkpoint_sha256": str(metadata["checkpoint_sha256"]),
        "manifest_sha256": str(metadata["manifest_sha256"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--push", type=Path, required=True)
    parser.add_argument("--pick", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_checkpoint = sha256_directory(args.checkpoint)
    expected_manifests = {
        "push": sha256_file(PUSH_MANIFEST),
        "pick": sha256_file(PICK_MANIFEST),
    }
    entries = {"push": _push(args.push), "pick": _pick(args.pick)}
    errors: list[str] = []
    for task, entry in entries.items():
        if entry["episodes"] != 50:
            errors.append(f"{task}: expected 50 episodes, got {entry['episodes']}")
        if entry["successes"] < REQUIRED_SUCCESSES:
            errors.append(f"{task}: required {REQUIRED_SUCCESSES}/50, got {entry['successes']}/50")
        if entry["checkpoint_sha256"].lower() != expected_checkpoint.lower():
            errors.append(f"{task}: checkpoint sha256 mismatch")
        if entry["manifest_sha256"].lower() != expected_manifests[task].lower():
            errors.append(f"{task}: manifest sha256 mismatch")
    report = {
        "schema_version": 1,
        "threshold": REQUIRED_SUCCESSES / 50,
        "required_successes": REQUIRED_SUCCESSES,
        "checkpoint_sha256": expected_checkpoint,
        "entries": entries,
        "errors": errors,
        "passed": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
