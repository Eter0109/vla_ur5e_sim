"""Collect successful dual-camera PickPlace expert demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from vla_sim.contracts import ACTION_KEY, IMAGE_KEY, STATE_KEY, TASK_KEY, WRIST_IMAGE_KEY  # noqa: E402
from vla_sim.envs import UR5ePickPlaceConfig, make_ur5e_pick_place  # noqa: E402
from vla_sim.pick_place_contract import PICK_PLACE_FPS, write_pick_place_contract  # noqa: E402
from vla_sim.scenes import load_manifest  # noqa: E402
from vla_sim.sim import HeuristicExpertConfig, HeuristicPickPlaceExpert  # noqa: E402


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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/ur5e_pick_place_v2_native_bin")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--distance-bins", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--place-release-tolerance-m", type=float, default=0.020)
    parser.add_argument("--place-release-xy-tolerance-m", type=float)
    parser.add_argument(
        "--task-prompt",
        default="place the red cube in the blue storage bin",
        help="Single task-level language label written for every demonstration frame.",
    )
    args = parser.parse_args()
    scenes = load_manifest(args.manifest)
    if args.root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset root: {args.root}")
    bins = sorted(set(args.distance_bins))
    if args.episodes < len(bins) or args.episodes % len(bins):
        parser.error("episodes must be positive and divisible by selected distance bins")
    if any(index not in (0, 1) for index in bins):
        parser.error("distance-bins must be a subset of 0 1")
    if not 0.0 < args.place_release_tolerance_m <= 0.020:
        parser.error("--place-release-tolerance-m must be in (0, 0.020]")
    if (
        args.place_release_xy_tolerance_m is not None
        and not 0.0 < args.place_release_xy_tolerance_m <= 0.020
    ):
        parser.error("--place-release-xy-tolerance-m must be in (0, 0.020]")
    per_bin = args.episodes // len(bins)
    config = UR5ePickPlaceConfig(horizon=250)
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=PICK_PLACE_FPS, root=args.root, robot_type="UR5e", features=features(), use_videos=False, image_writer_threads=2)
    env = make_ur5e_pick_place(config)
    accepted = {index: 0 for index in bins}
    try:
        for scene in scenes:
            bin_index = int(scene.overrides["distance_bin"])
            if bin_index not in accepted or accepted[bin_index] >= per_bin:
                continue
            observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
            expert = HeuristicPickPlaceExpert(
                HeuristicExpertConfig(
                    place_release_tolerance_m=args.place_release_tolerance_m,
                    place_release_xy_tolerance_m=args.place_release_xy_tolerance_m,
                )
            )
            success = False
            for _step in range(env.config.horizon):
                action = expert.act(env.raw_observation)
                dataset.add_frame(
                    {
                        IMAGE_KEY: observation[IMAGE_KEY],
                        WRIST_IMAGE_KEY: observation[WRIST_IMAGE_KEY],
                        STATE_KEY: observation[STATE_KEY],
                        ACTION_KEY: action,
                        TASK_KEY: args.task_prompt,
                    }
                )
                observation, _, terminated, truncated, info = env.step(action)
                success = bool(info["success"])
                if terminated or truncated:
                    break
            if success:
                dataset.save_episode(parallel_encoding=True)
                accepted[bin_index] += 1
                print(f"accepted={sum(accepted.values())}/{args.episodes} bin={bin_index}", flush=True)
            else:
                dataset.clear_episode_buffer()
            if sum(accepted.values()) == args.episodes:
                break
    finally:
        env.close()
    if set(accepted.values()) != {per_bin}:
        raise RuntimeError(f"Collected unbalanced PickPlace data: {accepted}")
    dataset.finalize()
    write_pick_place_contract(args.root, config)
    manifest_copy = args.root / "meta" / "collection_manifest.json"
    shutil.copyfile(args.manifest, manifest_copy)
    provenance = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "accepted_episodes": args.episodes,
        "accepted_per_distance_bin": accepted,
        "place_release_tolerance_m": args.place_release_tolerance_m,
        "place_release_xy_tolerance_m": args.place_release_xy_tolerance_m,
    }
    (args.root / "meta" / "collection_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    (args.root / "collection.complete").write_text(
        (
            f"episodes={args.episodes} bins={accepted} "
            f"place_release_tolerance_m={args.place_release_tolerance_m} "
            f"place_release_xy_tolerance_m={args.place_release_xy_tolerance_m}\n"
        ),
        encoding="utf-8",
    )
    print(f"dataset_ok root={args.root} episodes={args.episodes} bins={accepted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
