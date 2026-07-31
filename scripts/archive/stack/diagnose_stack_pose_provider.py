"""Archived RGB-D Stack pose diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if os.name == "nt":
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from vla_sim.envs import PrimitiveObjectConfig, UR5eStackConfig, make_ur5e_stack  # noqa: E402
from vla_sim.scenes import load_manifest  # noqa: E402
from vla_sim.stack_control import ColorDepthObjectPoseProvider  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-p95-m", type=float, default=0.008)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scenes = load_manifest(args.manifest)[: args.episodes]
    red = PrimitiveObjectConfig(rgba=(0.85, 0.12, 0.08, 1.0))
    blue = PrimitiveObjectConfig(rgba=(0.12, 0.2, 0.85, 1.0))
    environments = {
        "red_on_blue": make_ur5e_stack(
            UR5eStackConfig(horizon=1, objectA=red, objectB=blue)
        ),
        "blue_on_red": make_ur5e_stack(
            UR5eStackConfig(horizon=1, objectA=blue, objectB=red)
        ),
    }
    provider = ColorDepthObjectPoseProvider()
    records = []
    try:
        for scene in scenes:
            task = str(scene.overrides["task"])
            env = environments[task]
            env.reset(seed=scene.effective_env_seed, scene=scene)
            estimate = provider.estimate(
                env.raw_observation,
                task=task,
                simulator=env.backend.sim,
            )
            if estimate is None or estimate.pick_xyz is None or estimate.target_xyz is None:
                records.append({"scene_id": scene.scene_id, "detected": False})
                continue
            pick_truth = np.asarray(env.raw_observation["cubeA_pos"], dtype=np.float64)
            target_truth = np.asarray(env.raw_observation["cubeB_pos"], dtype=np.float64)
            records.append(
                {
                    "scene_id": scene.scene_id,
                    "detected": True,
                    "pick_error_m": float(np.linalg.norm(estimate.pick_xyz - pick_truth)),
                    "target_error_m": float(np.linalg.norm(estimate.target_xyz - target_truth)),
                    "pick_xyz_error_m": (estimate.pick_xyz - pick_truth).tolist(),
                    "target_xyz_error_m": (estimate.target_xyz - target_truth).tolist(),
                }
            )
    finally:
        for env in environments.values():
            env.close()

    errors = [
        float(record[key])
        for record in records
        if record["detected"]
        for key in ("pick_error_m", "target_error_m")
    ]
    detected = sum(bool(record["detected"]) for record in records)
    summary = {
        "episodes": len(records),
        "detected": detected,
        "detection_rate": detected / len(records) if records else 0.0,
        "error_m": {
            "mean": float(np.mean(errors)) if errors else None,
            "p50": float(np.quantile(errors, 0.50)) if errors else None,
            "p95": float(np.quantile(errors, 0.95)) if errors else None,
            "max": float(np.max(errors)) if errors else None,
        },
        "threshold_m": args.max_p95_m,
        "passed": detected == len(records)
        and bool(errors)
        and float(np.quantile(errors, 0.95)) <= args.max_p95_m,
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
