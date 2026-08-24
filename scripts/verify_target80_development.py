"""Verify four original development results from one checkpoint at the 80% gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.artifact_identity import sha256_directory  # noqa: E402
from vla_sim.target80 import target80_gate_report  # noqa: E402

MANIFESTS = {
    "push_nominal": ROOT / "configs/benchmarks/push_robust_development_nominal_v1.json",
    "push_randomized": ROOT / "configs/benchmarks/push_robust_development_randomized_v1.json",
    "pick_nominal": ROOT / "configs/benchmarks/pick_place_robust_development_nominal_v1.json",
    "pick_randomized": ROOT / "configs/benchmarks/pick_place_robust_development_randomized_v1.json",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _push_entry(path: Path) -> dict[str, Any]:
    payload = _json(path)
    inference = payload.get("inference", {})
    expected_inference = {
        "replan_steps": 4,
        "temporal_decay": 0.75,
        "policy_seed": 1000,
        "samples_per_plan": 1,
    }
    if inference != expected_inference:
        raise ValueError(f"Push inference contract mismatch in {path}: {inference}")
    if payload.get("control") != "vla_raw_safety_fixed_rotation_workspace_clamp":
        raise ValueError(f"Push control contract mismatch in {path}")
    summary = payload["summary"]
    return {
        "episodes": summary["episodes"],
        "successes": summary["successes"],
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "manifest_sha256": payload.get("manifest_sha256"),
    }


def _pick_entry(path: Path) -> dict[str, Any]:
    rows = _json(path)
    meta = _json(path.with_suffix(path.suffix + ".meta.json"))
    expected_inference = {
        "samples_per_plan": 2,
        "replan_steps": 4,
        "temporal_ensemble_decay": 0.75,
        "policy_seed": 1000,
    }
    if meta.get("inference") != expected_inference:
        raise ValueError(f"PickPlace inference contract mismatch in {path}")
    if meta.get("control_mode") != "vla_action_calibrated":
        raise ValueError(f"PickPlace control contract mismatch in {path}")
    if any(float(row.get("closed_negative_y_gain", 0.0)) != 1.8 for row in rows):
        raise ValueError(f"PickPlace gain contract mismatch in {path}")
    return {
        "episodes": len(rows),
        "successes": sum(bool(row.get("success")) for row in rows),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "manifest_sha256": meta.get("manifest_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--push-nominal", type=Path, required=True)
    parser.add_argument("--push-randomized", type=Path, required=True)
    parser.add_argument("--pick-nominal", type=Path, required=True)
    parser.add_argument("--pick-randomized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = {
        "push_nominal": _push_entry(args.push_nominal),
        "push_randomized": _push_entry(args.push_randomized),
        "pick_nominal": _pick_entry(args.pick_nominal),
        "pick_randomized": _pick_entry(args.pick_randomized),
    }
    report = target80_gate_report(
        entries,
        checkpoint_sha256=sha256_directory(args.checkpoint),
        manifest_sha256={key: _sha256(path) for key, path in MANIFESTS.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
