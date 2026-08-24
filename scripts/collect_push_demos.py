"""Collect successful dual-camera Push expert demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from vla_sim.contracts import ACTION_KEY, IMAGE_KEY, STATE_KEY, TASK_KEY, WRIST_IMAGE_KEY  # noqa: E402
from vla_sim.envs import UR5ePushConfig, make_ur5e_push  # noqa: E402
from vla_sim.pick_place_contract import PICK_PLACE_FPS  # noqa: E402
from vla_sim.sim import HeuristicPushExpert  # noqa: E402

def features() -> dict:
    image = {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
    return {
        IMAGE_KEY: image,
        WRIST_IMAGE_KEY: image.copy(),
        STATE_KEY: {"dtype": "float32", "shape": (10,), "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "eef_x", "eef_y", "eef_z", "gripper"]},
        ACTION_KEY: {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"]},
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/ur5e_push_1000")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    
    if args.root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset root: {args.root}")
        
    config = UR5ePushConfig(horizon=250)
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=PICK_PLACE_FPS, root=args.root, robot_type="UR5e", features=features(), use_videos=False, image_writer_threads=4)
    env = make_ur5e_push(config)
    
    accepted = 0
    try:
        # Loop until we get enough accepted episodes
        # We try random seeds starting from 0. If it succeeds, we keep it.
        seed = 0
        while accepted < args.episodes:
            if seed > 0 and seed % 20 == 0:
                env.close()
                import gc
                gc.collect()
                env = make_ur5e_push(config)
                
            observation, _ = env.reset(seed=seed)
            seed += 1
            
            expert = HeuristicPushExpert()
            success = False
            
            for _step in range(env.config.horizon):
                action = expert.act(env.raw_observation)
                
                # We do not have a prompt property on HeuristicPushExpert currently, so we'll just hardcode it
                prompt = "push the block into the red target circle"
                
                dataset.add_frame({
                    IMAGE_KEY: observation[IMAGE_KEY], 
                    WRIST_IMAGE_KEY: observation[WRIST_IMAGE_KEY], 
                    STATE_KEY: observation[STATE_KEY], 
                    ACTION_KEY: action, 
                    TASK_KEY: prompt
                })
                
                observation, _, terminated, truncated, info = env.step(action)
                success = bool(info["success"])
                
                if terminated or truncated:
                    break
                    
            if success:
                dataset.save_episode(parallel_encoding=True)
                accepted += 1
                print(f"accepted={accepted}/{args.episodes} seed={seed-1}", flush=True)
            else:
                dataset.clear_episode_buffer()
                print(f"failed seed={seed-1}", flush=True)
                
    finally:
        env.close()
        
    dataset.finalize()
    
    # Save a generic manifest info
    meta_dir = args.root / "meta"
    meta_dir.mkdir(exist_ok=True, parents=True)
    
    provenance = {
        "schema_version": 1,
        "accepted_episodes": args.episodes,
    }
    (meta_dir / "collection_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    (args.root / "collection.complete").write_text(
        f"episodes={args.episodes}\n",
        encoding="utf-8",
    )
    print(f"dataset_ok root={args.root} episodes={args.episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
