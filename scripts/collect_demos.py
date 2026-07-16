"""Collect successful privileged-expert episodes into a local LeRobotDataset."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNTIME = ROOT / ".runtime"
for directory in (RUNTIME / "hf", RUNTIME / "hf_datasets"):
    directory.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(RUNTIME / "hf")
os.environ["HF_DATASETS_CACHE"] = str(RUNTIME / "hf_datasets")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from vla_sim.contracts import ACTION_KEY, IMAGE_KEY, STATE_KEY, TASK_KEY  # noqa: E402
from vla_sim.envs import CameraConfig, UR5eLiftConfig, make_ur5e_lift  # noqa: E402
from vla_sim.scenes import load_manifest  # noqa: E402
from vla_sim.sim import HeuristicLiftExpert  # noqa: E402


TASK = "Grasp the red object and lift it at least ten centimeters"


def features(height: int, width: int) -> dict:
    return {
        IMAGE_KEY: {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
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
    parser.add_argument("--repo-id", default="local/ur5e_custom_lift")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scenes = load_manifest(args.manifest)
    if args.limit is not None:
        scenes = scenes[: args.limit]
    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset root exists: {args.root}; use --overwrite")
        shutil.rmtree(args.root)

    height = width = 256
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=10,
        root=args.root,
        robot_type="UR5e",
        features=features(height, width),
        use_videos=False,
        image_writer_threads=2,
    )
    env = make_ur5e_lift(
        UR5eLiftConfig(camera=CameraConfig(width=width, height=height), horizon=200)
    )
    successes = 0
    try:
        for index, scene in enumerate(scenes):
            observation, _ = env.reset(seed=scene.seed, scene=scene)
            expert = HeuristicLiftExpert()
            successful = False
            steps = 0
            for steps in range(1, env.config.horizon + 1):
                action = expert.act(env.raw_observation)
                dataset.add_frame(
                    {
                        IMAGE_KEY: observation[IMAGE_KEY],
                        STATE_KEY: observation[STATE_KEY],
                        ACTION_KEY: np.asarray(action, dtype=np.float32),
                        TASK_KEY: TASK,
                    }
                )
                observation, _, terminated, truncated, info = env.step(action)
                successful = bool(info["success"])
                if terminated or truncated:
                    break
            if successful:
                dataset.save_episode(parallel_encoding=False)
                successes += 1
            else:
                dataset.clear_episode_buffer()
            print(
                f"scene={scene.scene_id} success={successful} steps={steps} "
                f"collected={successes}/{index + 1}",
                flush=True,
            )
    finally:
        env.close()
    dataset.finalize()

    reopened = LeRobotDataset(args.repo_id, root=args.root)
    print(
        f"dataset_ok root={args.root} episodes={reopened.num_episodes} "
        f"frames={len(reopened)} success_rate={successes / len(scenes):.3f}"
    )
    return 0 if successes == len(scenes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
