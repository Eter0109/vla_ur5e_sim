"""Evaluate the privileged ColorPick expert on a frozen manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.envs import UR5eColorPickConfig, make_ur5e_color_pick
from vla_sim.scenes import (
    COLOR_PICK_COLORS,
    load_manifest,
    load_manifest_metadata,
)
from vla_sim.sim import HeuristicColorPickExpert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    metadata = load_manifest_metadata(args.manifest)
    if metadata.get("environment_preset") not in {
        "color_pick_v1",
        "color_pick_sim2real_v2",
    }:
        parser.error("manifest must use a supported ColorPick environment preset")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    scenes = load_manifest(args.manifest)
    if args.episodes is not None:
        if not 1 <= args.episodes <= len(scenes):
            parser.error("episodes must be within the manifest length")
        scenes = scenes[: args.episodes]

    env = make_ur5e_color_pick(UR5eColorPickConfig(has_renderer=args.render))
    results: list[dict] = []
    try:
        for scene in scenes:
            target_color = str(scene.overrides["target_color"])
            env.reset(seed=scene.effective_env_seed, scene=scene)
            expert = HeuristicColorPickExpert(target_color)
            success = False
            info: dict = {}
            for step in range(1, env.config.horizon + 1):
                action = expert.act(env.raw_observation)
                _, _, terminated, truncated, info = env.step(action)
                if args.render:
                    env.render()
                success = bool(info["success"])
                if terminated or truncated:
                    break
            result = {
                "scene_id": scene.scene_id,
                "target_color": target_color,
                "prompt": expert.prompt,
                "success": success,
                "steps": step,
                "ever_target_grasped": bool(info.get("ever_target_grasped", False)),
                "ever_wrong_object_grasped": bool(
                    info.get("ever_wrong_object_grasped", False)
                ),
                "target_lift_m": float(info.get("target_lift_m", 0.0)),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        env.close()

    successes = Counter(
        result["target_color"] for result in results if result["success"]
    )
    totals = Counter(result["target_color"] for result in results)
    summary = {
        "schema_version": 1,
        "episodes": len(results),
        "successes": sum(bool(result["success"]) for result in results),
        "wrong_color_grasps": sum(
            bool(result["ever_wrong_object_grasped"]) for result in results
        ),
        "by_target_color": {
            color: {"successes": successes[color], "episodes": totals[color]}
            for color in COLOR_PICK_COLORS
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["successes"] == summary["episodes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
