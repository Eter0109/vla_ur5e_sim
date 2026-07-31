"""Archived evaluator for the privileged Stack collection expert."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if os.name == "nt":
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from vla_sim.envs import PrimitiveObjectConfig, UR5eStackConfig, make_ur5e_stack  # noqa: E402
from vla_sim.scenes import load_manifest  # noqa: E402
from vla_sim.sim import HeuristicStackExpert  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--task", choices=("red_on_blue", "blue_on_red"))
    parser.add_argument("--distance-bin", type=int, choices=(0, 1, 2))
    parser.add_argument("--render", action="store_true", help="show the MuJoCo viewer")
    args = parser.parse_args()
    scenes = load_manifest(args.manifest)[args.start :]
    if args.task:
        scenes = [scene for scene in scenes if scene.overrides["task"] == args.task]
    if args.distance_bin is not None:
        scenes = [
            scene
            for scene in scenes
            if int(scene.overrides["distance_bin"]) == args.distance_bin
        ]
    if args.episodes:
        scenes = scenes[: args.episodes]
    red = PrimitiveObjectConfig(rgba=(0.85, 0.12, 0.08, 1.0))
    blue = PrimitiveObjectConfig(rgba=(0.12, 0.2, 0.85, 1.0))
    environments = {
        "red_on_blue": make_ur5e_stack(
            UR5eStackConfig(horizon=args.horizon, objectA=red, objectB=blue, has_renderer=args.render)
        ),
        "blue_on_red": make_ur5e_stack(
            UR5eStackConfig(horizon=args.horizon, objectA=blue, objectB=red, has_renderer=args.render)
        ),
    }
    results = []
    try:
        for scene in scenes:
            task = str(scene.overrides["task"])
            env = environments[task]
            env.reset(seed=scene.effective_env_seed, scene=scene)
            expert = HeuristicStackExpert()
            success = False
            for step in range(1, env.config.horizon + 1):
                _, _, terminated, truncated, info = env.step(expert.act(env.raw_observation))
                if args.render:
                    env.render()
                success = bool(info["success"])
                if terminated or truncated:
                    break
            result = {
                "scene_id": scene.scene_id,
                "task": task,
                "distance_bin": int(scene.overrides["distance_bin"]),
                "success": success,
                "steps": step,
                "final_phase": expert.phase.name.lower(),
                "retries": expert.retries,
                "stack_conditions": info.get("stack_conditions"),
                "cubeA_pos": list(map(float, env.raw_observation["cubeA_pos"])),
                "cubeB_pos": list(map(float, env.raw_observation["cubeB_pos"])),
                "eef_pos": list(map(float, env.raw_observation["robot0_eef_pos"])),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        for env in environments.values():
            env.close()
    cells = Counter((result["task"], result["distance_bin"]) for result in results)
    successes = Counter(
        (result["task"], result["distance_bin"]) for result in results if result["success"]
    )
    rates = {f"{task}/distance_{distance}": successes[(task, distance)] / total for (task, distance), total in cells.items()}
    summary = {
        "episodes": len(results),
        "successes": sum(bool(result["success"]) for result in results),
        "success_rate": sum(bool(result["success"]) for result in results) / len(results),
        "cell_rates": rates,
        "passed": sum(bool(result["success"]) for result in results) / len(results) >= 0.98
        and min(rates.values()) >= 0.95,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
