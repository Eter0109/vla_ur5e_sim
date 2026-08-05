"""Collect successful static-prompt rollouts from a non-oracle calibrated VLA teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from vla_sim.contracts import ACTION_KEY, IMAGE_KEY, STATE_KEY, TASK_KEY, WRIST_IMAGE_KEY  # noqa: E402
from vla_sim.envs import UR5ePickPlaceConfig, make_ur5e_pick_place  # noqa: E402
from vla_sim.pick_place_contract import PICK_PLACE_FPS, PICK_PLACE_GLOBAL_PROMPT, write_pick_place_contract  # noqa: E402
from vla_sim.pick_place_control import (  # noqa: E402
    VLAOnlyActionCalibration,
    VLAOnlyActionCalibrator,
    filter_vla_only_action,
    scene_policy_seed,
)
from vla_sim.pick_place_phases import pick_place_phase_prompt  # noqa: E402
from vla_sim.policy_runtime import load_policy, predict_ensemble_chunk  # noqa: E402
from vla_sim.scenes import load_manifest  # noqa: E402
from vla_sim.temporal import TemporalEnsemble  # noqa: E402


def features() -> dict:
    image = {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
    return {
        IMAGE_KEY: image,
        WRIST_IMAGE_KEY: image.copy(),
        STATE_KEY: {"dtype": "float32", "shape": (10,), "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "eef_x", "eef_y", "eef_z", "gripper"]},
        ACTION_KEY: {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"]},
    }


def _cpu_numpy(value: object) -> np.ndarray:
    """Detach policy-mutated CUDA observations before LeRobot serializes them."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _inference_copy(observation: dict) -> dict:
    """Keep LeRobot preprocessing from mutating the frames queued for recording."""

    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in observation.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/ur5e_pick_place_v2_native_bin")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--samples-per-plan", type=int, default=2)
    parser.add_argument("--policy-seed", type=int, default=1000)
    parser.add_argument("--negative-y-gain", type=float, default=1.3)
    parser.add_argument("--positive-x-gain", type=float, default=0.95)
    args = parser.parse_args()
    if args.root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset root: {args.root}")
    if args.episodes < 2 or args.episodes % 2:
        parser.error("episodes must be positive and even for balanced distance bins")

    scenes = load_manifest(args.manifest)
    required_per_bin = args.episodes // 2
    config = UR5ePickPlaceConfig(horizon=250)
    source = LeRobotDataset(args.repo_id, root=args.dataset_root)
    policy_config, policy, preprocessor, postprocessor = load_policy(args.checkpoint, source)
    if not 1 <= args.replan_steps <= policy_config.n_action_steps:
        parser.error(f"replan-steps must be in [1, {policy_config.n_action_steps}]")
    dataset = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        fps=PICK_PLACE_FPS,
        root=args.root,
        robot_type="UR5e",
        features=features(),
        use_videos=False,
        image_writer_threads=2,
    )
    env = make_ur5e_pick_place(config)
    accepted = {0: 0, 1: 0}
    device = torch.device("cuda")
    calibration = VLAOnlyActionCalibration(
        closed_negative_y_gain=args.negative_y_gain,
        transport_positive_x_gain=args.positive_x_gain,
    )
    try:
        for scene in scenes:
            bin_index = int(scene.overrides["distance_bin"])
            if accepted[bin_index] >= required_per_bin:
                continue
            observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
            seed = scene_policy_seed(args.policy_seed, scene.effective_env_seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            temporal = TemporalEnsemble(policy_config.n_action_steps, 7, gripper_mode="latest")
            calibrator = VLAOnlyActionCalibrator(calibration)
            success = False
            phase_info: dict[str, object] = {
                "grasped": False, "ever_grasped": False, "ever_lifted": False,
            }
            for step in range(1, env.config.horizon + 1):
                if (step - 1) % args.replan_steps == 0:
                    policy.reset()
                    chunk = predict_ensemble_chunk(
                        _inference_copy(observation), policy, preprocessor, postprocessor, device,
                        policy_config.use_amp, args.samples_per_plan, PICK_PLACE_GLOBAL_PROMPT,
                    )
                    temporal.add_chunk(step, chunk)
                model_action = np.asarray(temporal.get_action(step)[:7], dtype=np.float32)
                action = filter_vla_only_action(
                    calibrator.calibrate(model_action),
                    eef_xyz=np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float),
                )
                phase_prompt = pick_place_phase_prompt(
                    env.raw_observation, action, phase_info,
                )
                dataset.add_frame({
                    IMAGE_KEY: _cpu_numpy(observation[IMAGE_KEY]),
                    WRIST_IMAGE_KEY: _cpu_numpy(observation[WRIST_IMAGE_KEY]),
                    STATE_KEY: _cpu_numpy(observation[STATE_KEY]),
                    ACTION_KEY: action,
                    TASK_KEY: phase_prompt,
                })
                observation, _, terminated, truncated, info = env.step(action)
                phase_info = info
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
    if set(accepted.values()) != {required_per_bin}:
        raise RuntimeError(f"Collected unbalanced calibrated rollouts: {accepted}")
    dataset.finalize()
    write_pick_place_contract(args.root, config)
    (args.root / "meta" / "collection_provenance.json").write_text(json.dumps({
        "schema_version": 2, "teacher_checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "accepted_episodes": args.episodes, "accepted_per_distance_bin": accepted,
        "control_mode": "vla_action_calibrated", "oracle_control": False,
        "negative_y_gain": args.negative_y_gain, "positive_x_gain": args.positive_x_gain,
        "samples_per_plan": args.samples_per_plan, "replan_steps": args.replan_steps,
        "phase_labels": "privileged_sim_state_training_only",
        "deployment_prompt": PICK_PLACE_GLOBAL_PROMPT,
    }, indent=2), encoding="utf-8")
    (args.root / "collection.complete").write_text(f"episodes={args.episodes} bins={accepted}\n", encoding="utf-8")
    print(f"dataset_ok root={args.root} episodes={args.episodes} bins={accepted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
