"""Collect balanced Push demonstrations with the deployment safety contract."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
RUNTIME = ROOT / ".runtime"
os.environ.update(
    {
        "HF_HOME": str(RUNTIME / "hf"),
        "HF_DATASETS_CACHE": str(RUNTIME / "hf_datasets"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "USE_TF": "0",
        "NUMBA_CACHE_DIR": str(Path(tempfile.gettempdir()) / "vla_sim_numba"),
        "NUMBA_DISABLE_JIT": "1" if os.name == "nt" else "0",
    }
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from vla_sim.contracts import ACTION_KEY, IMAGE_KEY, STATE_KEY, TASK_KEY, WRIST_IMAGE_KEY  # noqa: E402
from vla_sim.envs import UR5ePushConfig, make_ur5e_push  # noqa: E402
from vla_sim.pick_place_control import filter_vla_only_action  # noqa: E402
from vla_sim.pick_place_contract import PICK_PLACE_FPS  # noqa: E402
from vla_sim.provenance import collection_resume_source_index  # noqa: E402
from vla_sim.scenes import load_manifest, load_manifest_metadata  # noqa: E402
from vla_sim.sim import HeuristicPushExpert  # noqa: E402

PROMPT = "push the block into the red target circle"


def _features() -> dict:
    image = {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
    return {
        IMAGE_KEY: image,
        WRIST_IMAGE_KEY: image.copy(),
        STATE_KEY: {"dtype": "float32", "shape": (10,), "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "eef_x", "eef_y", "eef_z", "gripper"]},
        ACTION_KEY: {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"]},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/ur5e_push_1000_v2")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--reset-env-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Record irrecoverable expert failures and continue through supplemental scenes.",
    )
    args = parser.parse_args()
    metadata = load_manifest_metadata(args.manifest)
    if metadata.get("role") != "collection" or metadata.get("environment_preset") not in {
        "push_forward_v1",
        "push_robust_v1",
    }:
        raise ValueError("Push collection requires a collection Push manifest.")
    scenes = load_manifest(args.manifest)
    episodes = len(scenes) if args.episodes is None else args.episodes
    if not 1 <= episodes <= len(scenes):
        raise ValueError(f"episodes must be in [1, {len(scenes)}].")
    if args.reset_env_every < 1:
        raise ValueError("reset-env-every must be positive.")
    start_source_index = 0
    resumed_from_episodes = 0
    results: list[dict[str, object]] = []
    if args.root.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite dataset root: {args.root}")
    if args.resume:
        dataset = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
        resumed_from_episodes = dataset.num_episodes
        progress_path = args.root / "collection_progress.json"
        if not progress_path.exists():
            raise ValueError("Resume requires collection_progress.json")
        results = json.loads(progress_path.read_text(encoding="utf-8"))
        start_source_index = collection_resume_source_index(
            results,
            dataset_episodes=dataset.num_episodes,
            scene_count=len(scenes),
        )
        if dataset.num_episodes >= episodes:
            raise ValueError(
                f"Dataset already has {dataset.num_episodes} episodes; requested {episodes}."
            )
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=PICK_PLACE_FPS,
            root=args.root,
            robot_type="UR5e",
            features=_features(),
            use_videos=False,
            image_writer_threads=4,
        )
    env = make_ur5e_push(UR5ePushConfig(horizon=args.horizon))
    accepted = dataset.num_episodes if args.resume else 0
    try:
        for source_index, scene in enumerate(
            scenes[start_source_index:], start=start_source_index
        ):
            if accepted >= episodes:
                break
            success = False
            for attempt in range(1, 3):
                observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
                expert = HeuristicPushExpert()
                for step in range(args.horizon):
                    action = filter_vla_only_action(
                        expert.act(env.raw_observation),
                        eef_xyz=np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float),
                    )
                    dataset.add_frame(
                        {
                            IMAGE_KEY: observation[IMAGE_KEY],
                            WRIST_IMAGE_KEY: observation[WRIST_IMAGE_KEY],
                            STATE_KEY: observation[STATE_KEY],
                            ACTION_KEY: action,
                            TASK_KEY: PROMPT,
                        }
                    )
                    observation, _, terminated, truncated, info = env.step(action)
                    success = bool(info["success"])
                    if terminated or truncated:
                        break
                if success:
                    break
                dataset.clear_episode_buffer()
                if attempt == 1:
                    # Rebuild only the simulator, then retry the identical
                    # manifest scene. This recovers native-state drift without
                    # changing the scene distribution or hiding an expert bug.
                    env.close()
                    env = make_ur5e_push(UR5ePushConfig(horizon=args.horizon))
            result = {
                "scene_id": scene.scene_id,
                "source_index": source_index,
                "success": success,
                "frames": step + 1,
                "attempts": attempt,
                "angle_bin": int(scene.overrides["angle_bin"]),
                "distance_bin": int(scene.overrides["distance_bin"]),
            }
            if not success:
                dataset.clear_episode_buffer()
                results.append(result)
                if args.allow_failures:
                    print(f"skipped_failed_scene={scene.scene_id}", flush=True)
                    continue
                raise RuntimeError(f"Expert failed collection scene {scene.scene_id}; dataset was not finalized.")
            dataset.save_episode(parallel_encoding=True)
            results.append(result)
            accepted += 1
            if accepted % args.reset_env_every == 0 and accepted < episodes:
                # MuJoCo's Windows binding retains native allocations across
                # many resets. Bound the lifetime without changing scenes.
                env.close()
                del env
                gc.collect()
                env = make_ur5e_push(UR5ePushConfig(horizon=args.horizon))
            if accepted % 10 == 0 or accepted == episodes:
                args.root.mkdir(parents=True, exist_ok=True)
                (args.root / "collection_progress.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(f"accepted={accepted}/{episodes} scene={scene.scene_id}", flush=True)
    finally:
        env.close()
    if accepted != episodes:
        raise RuntimeError(f"Collected {accepted}/{episodes} successful Push episodes; manifest is exhausted.")
    dataset.finalize()
    provenance = {
        "schema_version": 2,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "accepted_episodes": episodes,
        "resumed_from_episodes": resumed_from_episodes,
        "prompt": PROMPT,
        "control": "fixed_rotation_workspace_safety",
        "results": results,
    }
    meta = args.root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "collection_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (args.root / "collection.complete").write_text(f"episodes={episodes}\n", encoding="utf-8")
    print(f"dataset_ok root={args.root} episodes={episodes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
