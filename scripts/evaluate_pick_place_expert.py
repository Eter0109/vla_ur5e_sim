"""Evaluate the privileged PickPlace expert on a frozen manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.envs import UR5ePickPlaceConfig, make_ur5e_pick_place  # noqa: E402
from vla_sim.scenes import load_manifest  # noqa: E402
from vla_sim.sim import HeuristicExpertConfig, HeuristicPickPlaceExpert  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--place-release-tolerance-m", type=float, default=0.020)
    parser.add_argument("--place-release-xy-tolerance-m", type=float)
    args = parser.parse_args()
    scenes = load_manifest(args.manifest)
    if args.episodes is not None:
        scenes = scenes[: args.episodes]
    env = make_ur5e_pick_place(UR5ePickPlaceConfig(has_renderer=args.render))
    results: list[dict] = []
    try:
        for scene in scenes:
            observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
            expert = HeuristicPickPlaceExpert(
                HeuristicExpertConfig(
                    place_release_tolerance_m=args.place_release_tolerance_m,
                    place_release_xy_tolerance_m=args.place_release_xy_tolerance_m,
                )
            )
            success = False
            for step in range(1, env.config.horizon + 1):
                action = expert.act(env.raw_observation)
                observation, _, terminated, truncated, info = env.step(action)
                if args.render:
                    env.render()
                success = bool(info["success"])
                if terminated or truncated:
                    break
            results.append({"scene_id": scene.scene_id, "distance_bin": scene.overrides["distance_bin"], "success": success, "steps": step})
    finally:
        env.close()
    bins = Counter(int(value["distance_bin"]) for value in results if value["success"])
    total_bins = Counter(int(value["distance_bin"]) for value in results)
    summary = {"episodes": len(results), "successes": sum(bool(value["success"]) for value in results), "by_distance_bin": {str(index): {"successes": bins[index], "episodes": total_bins[index]} for index in sorted(total_bins)}, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
