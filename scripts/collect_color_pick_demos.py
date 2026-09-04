"""Collect balanced dual-camera demonstrations for three-color target selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla_sim.color_pick_contract import (
    COLOR_PICK_FPS,
    color_pick_prompt,
    write_color_pick_contract,
)
from vla_sim.contracts import (
    ACTION_KEY,
    IMAGE_KEY,
    STATE_KEY,
    TASK_KEY,
    WRIST_IMAGE_KEY,
)
from vla_sim.envs import UR5eColorPickConfig, make_ur5e_color_pick
from vla_sim.scenes import COLOR_PICK_COLORS, load_manifest
from vla_sim.sim import HeuristicColorPickExpert


def features() -> dict:
    image = {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        IMAGE_KEY: image,
        WRIST_IMAGE_KEY: image.copy(),
        STATE_KEY: {
            "dtype": "float32",
            "shape": (10,),
            "names": [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "eef_x",
                "eef_y",
                "eef_z",
                "gripper",
            ],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/color_pick_1500")
    parser.add_argument("--episodes", type=int, default=1_500)
    args = parser.parse_args()
    if args.root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset root: {args.root}")
    if args.episodes < 3 or args.episodes % len(COLOR_PICK_COLORS):
        parser.error("episodes must be positive and divisible by three target colors")
    scenes = load_manifest(args.manifest)
    if args.episodes > len(scenes):
        parser.error("manifest does not contain enough scenes")

    per_color = args.episodes // len(COLOR_PICK_COLORS)
    config = UR5eColorPickConfig()
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=COLOR_PICK_FPS,
        root=args.root,
        robot_type="UR5e",
        features=features(),
        use_videos=False,
        image_writer_threads=2,
    )
    env = make_ur5e_color_pick(config)
    accepted = {color: 0 for color in COLOR_PICK_COLORS}
    try:
        for scene in scenes:
            target_color = str(scene.overrides["target_color"])
            if accepted[target_color] >= per_color:
                continue
            observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
            expert = HeuristicColorPickExpert(target_color)
            success = False
            info: dict = {}
            for _step in range(config.horizon):
                action = expert.act(env.raw_observation)
                dataset.add_frame(
                    {
                        IMAGE_KEY: observation[IMAGE_KEY],
                        WRIST_IMAGE_KEY: observation[WRIST_IMAGE_KEY],
                        STATE_KEY: observation[STATE_KEY],
                        ACTION_KEY: action,
                        TASK_KEY: color_pick_prompt(target_color),
                    }
                )
                observation, _, terminated, truncated, info = env.step(action)
                success = bool(info["success"])
                if terminated or truncated:
                    break
            if success and not info.get("ever_wrong_object_grasped", False):
                dataset.save_episode(parallel_encoding=True)
                accepted[target_color] += 1
                print(
                    f"accepted={sum(accepted.values())}/{args.episodes} "
                    f"target={target_color} counts={accepted}",
                    flush=True,
                )
            else:
                dataset.clear_episode_buffer()
            if set(accepted.values()) == {per_color}:
                break
    finally:
        env.close()
    if set(accepted.values()) != {per_color}:
        raise RuntimeError(f"Collected unbalanced ColorPick data: {accepted}")

    dataset.finalize()
    write_color_pick_contract(args.root, config)
    manifest_copy = args.root / "meta" / "collection_manifest.json"
    shutil.copyfile(args.manifest, manifest_copy)
    provenance = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "accepted_episodes": args.episodes,
        "accepted_per_target_color": accepted,
        "prompts": {color: color_pick_prompt(color) for color in COLOR_PICK_COLORS},
        "selection_signal": "language_only",
    }
    (args.root / "meta" / "collection_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    (args.root / "collection.complete").write_text(
        f"episodes={args.episodes} colors={accepted}\n", encoding="utf-8"
    )
    print(f"dataset_ok root={args.root} episodes={args.episodes} colors={accepted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
