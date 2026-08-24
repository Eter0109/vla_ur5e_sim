"""Evaluate the privileged Push expert on a frozen manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.envs import UR5ePushConfig, make_ur5e_push  # noqa: E402
from vla_sim.evaluation import summarize_results  # noqa: E402
from vla_sim.pick_place_control import filter_vla_only_action  # noqa: E402
from vla_sim.scenes import load_manifest, load_manifest_metadata  # noqa: E402
from vla_sim.sim import HeuristicPushExpert  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = load_manifest_metadata(args.manifest)
    if metadata.get("environment_preset") not in {"push_forward_v1", "push_robust_v1"}:
        raise ValueError("Push expert evaluation requires a supported Push manifest.")
    scenes = load_manifest(args.manifest)
    episodes = len(scenes) if args.episodes is None else args.episodes
    if not 1 <= episodes <= len(scenes):
        raise ValueError(f"episodes must be in [1, {len(scenes)}].")
    env = make_ur5e_push(UR5ePushConfig(horizon=args.horizon))
    results = []
    try:
        for index, scene in enumerate(scenes[:episodes], start=1):
            observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
            initial = np.asarray(env.raw_observation["cube_pos"], dtype=float)[:2].copy()
            target = env.target_pos[:2].copy()
            direction = target - initial
            initial_distance = float(np.linalg.norm(direction))
            direction /= initial_distance
            positions = [initial.copy()]
            expert = HeuristicPushExpert()
            success = False
            for step in range(args.horizon):
                action = filter_vla_only_action(
                    expert.act(env.raw_observation),
                    eef_xyz=np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float),
                )
                observation, _, terminated, truncated, info = env.step(action)
                positions.append(np.asarray(env.raw_observation["cube_pos"], dtype=float)[:2].copy())
                success = bool(info["success"])
                if terminated or truncated:
                    break
            final = np.asarray(env.raw_observation["cube_pos"], dtype=float)[:2]
            trajectory = np.asarray(positions)
            displacement = trajectory - initial
            forward_progress = displacement @ direction
            distances = np.linalg.norm(trajectory - target, axis=1)
            results.append(
                {
                    "scene_id": scene.scene_id,
                    "success": success,
                    "failure_stage": "success" if success else "expert_failed",
                    "angle_bin": int(scene.overrides["angle_bin"]),
                    "distance_bin": int(scene.overrides["distance_bin"]),
                    "initial_distance_m": initial_distance,
                    "final_distance_m": float(np.linalg.norm(target - final)),
                    "min_distance_m": float(np.min(distances)),
                    "max_forward_progress_m": float(np.max(forward_progress)),
                    "final_cube_xy_m": final.tolist(),
                    "target_xy_m": target.tolist(),
                    "steps": step + 1,
                }
            )
            successes = sum(bool(value["success"]) for value in results)
            print(f"episode={index}/{episodes} successes={successes} scene={scene.scene_id}", flush=True)
    finally:
        env.close()
    payload = {"manifest": str(args.manifest.resolve()), "summary": summarize_results(results), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"expert_summary {payload['summary']['successes']}/{episodes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
