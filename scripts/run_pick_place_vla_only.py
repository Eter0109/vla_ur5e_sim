"""Evaluate static-prompt PickPlace with no object-pose or phase-supervisor control."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
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
from vla_sim.contracts import IMAGE_KEY, WRIST_IMAGE_KEY  # noqa: E402
from vla_sim.envs import UR5ePickPlaceConfig, make_ur5e_pick_place  # noqa: E402
from vla_sim.pick_place_contract import (  # noqa: E402
    PICK_PLACE_GLOBAL_PROMPT,
    validate_pick_place_contract,
)
from vla_sim.pick_place_control import (  # noqa: E402
    VLAOnlyActionCalibration,
    VLAOnlyActionCalibrator,
    filter_vla_only_action,
    scene_policy_seed,
)
from vla_sim.policy_runtime import load_policy, predict_ensemble_chunk  # noqa: E402
from vla_sim.scenes import load_manifest, load_manifest_metadata  # noqa: E402
from vla_sim.temporal import TemporalEnsemble  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/ur5e_pick_place_v2_native_bin")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--rgb-window", action="store_true")
    parser.add_argument("--rgb-delay-ms", type=int, default=1)
    parser.add_argument("--rgb-initial-hold-ms", type=int, default=1)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.5)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--samples-per-plan", type=int, default=1)
    parser.add_argument("--closed-negative-y-gain", type=float, default=1.0)
    parser.add_argument("--transport-positive-x-gain", type=float, default=1.0)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=1000,
        help="Base seed combined with each environment seed for reproducible flow sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "pick_place_vla_only_rollouts.json",
    )
    args = parser.parse_args()
    try:
        action_calibration = VLAOnlyActionCalibration(
            closed_negative_y_gain=args.closed_negative_y_gain,
            transport_positive_x_gain=args.transport_positive_x_gain,
        )
    except ValueError as error:
        parser.error(str(error))

    metadata = load_manifest_metadata(args.manifest)
    if metadata.get("environment_preset") != "pick_place_v1":
        parser.error("VLA-only evaluation requires a compatible PickPlace manifest")
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.samples_per_plan < 1:
        parser.error("samples-per-plan must be positive")
    if metadata.get("role") == "blind" and args.episodes != 100:
        parser.error("PickPlace blind evaluation must run all 100 scenes")
    if metadata.get("role") == "blind" and args.output.exists():
        parser.error("Blind output already exists and cannot be overwritten")
    scenes = load_manifest(args.manifest)
    if args.episodes > len(scenes):
        parser.error(f"manifest contains only {len(scenes)} scenes")
    scenes = scenes[: args.episodes]

    env_config = UR5ePickPlaceConfig(horizon=args.horizon, has_renderer=args.render)
    validate_pick_place_contract(args.dataset_root, env_config)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    config, policy, preprocessor, postprocessor = load_policy(
        args.checkpoint,
        dataset,
        None,
    )
    device = torch.device("cuda")
    env = make_ur5e_pick_place(env_config)
    results: list[dict] = []
    try:
        for scene in scenes:
            observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
            policy_seed = scene_policy_seed(args.policy_seed, scene.effective_env_seed)
            torch.manual_seed(policy_seed)
            torch.cuda.manual_seed_all(policy_seed)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            temporal = TemporalEnsemble(
                config.n_action_steps,
                7,
                decay=args.temporal_ensemble_decay,
                gripper_mode="latest",
            )
            action_calibrator = VLAOnlyActionCalibrator(action_calibration)
            actions: list[np.ndarray] = []
            initial = np.asarray(env.raw_observation["cube_pos"], dtype=float)
            max_z = float(initial[2])
            ever_grasped = False
            success = False
            info: dict = {}
            if args.rgb_window:
                _show_rgb_window(
                    observation,
                    step=0,
                    delay_ms=args.rgb_initial_hold_ms,
                )
            for step in range(1, args.horizon + 1):
                if (step - 1) % args.replan_steps == 0:
                    chunk = predict_ensemble_chunk(
                        observation,
                        policy,
                        preprocessor,
                        postprocessor,
                        device,
                        config.use_amp,
                        args.samples_per_plan,
                        task_prompt=PICK_PLACE_GLOBAL_PROMPT,
                    )
                    temporal.add_chunk(step, chunk)
                raw_action = temporal.get_action(step)
                calibrated_action = action_calibrator.calibrate(
                    np.asarray(raw_action[:7], dtype=np.float32)
                )
                action = filter_vla_only_action(
                    calibrated_action,
                    eef_xyz=np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float),
                )
                actions.append(action)
                observation, _, terminated, truncated, info = env.step(action)
                if args.rgb_window:
                    _show_rgb_window(
                        observation,
                        step=step,
                        delay_ms=args.rgb_delay_ms,
                    )
                if args.render:
                    _render_native_bin(env)
                max_z = max(max_z, float(np.asarray(env.raw_observation["cube_pos"])[2]))
                ever_grasped = ever_grasped or bool(info["grasped"])
                success = bool(info["success"])
                if terminated or truncated:
                    break
            action_array = np.asarray(actions)
            final_cube = np.asarray(env.raw_observation["cube_pos"], dtype=float)
            target = np.asarray(env.raw_observation["target_zone_pos"], dtype=float)
            result = {
                "scene_id": scene.scene_id,
                "distance_bin": scene.overrides["distance_bin"],
                "success": success,
                "steps": step,
                "ever_grasped": ever_grasped,
                "max_lift_m": max_z - float(initial[2]),
                "failure_stage": (
                    "success"
                    if success
                    else ("no_grasp" if not ever_grasped else "placement_failed")
                ),
                "place_conditions": info.get("place_conditions", {}),
                "mean_action": action_array.mean(axis=0).tolist(),
                "gripper_close_fraction": float((action_array[:, 6] > 0.0).mean()),
                "control_mode": (
                    "vla_only_xyz_gripper"
                    if (
                        args.closed_negative_y_gain == 1.0
                        and args.transport_positive_x_gain == 1.0
                    )
                    else "vla_only_xyz_gripper_action_calibrated"
                ),
                "static_prompt": PICK_PLACE_GLOBAL_PROMPT,
                "oracle_control": False,
                "fixed_rotation_safety": True,
                "workspace_clamp": True,
                "closed_negative_y_gain": args.closed_negative_y_gain,
                "transport_positive_x_gain": args.transport_positive_x_gain,
                "transport_direction_lock": True,
                "samples_per_plan": args.samples_per_plan,
                "policy_seed": policy_seed,
                "metrics_only": {
                    "final_cube_xyz_m": final_cube.tolist(),
                    "target_xyz_m": target.tolist(),
                    "xy_error_vector_m": (final_cube[:2] - target[:2]).tolist(),
                },
            }
            results.append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
    finally:
        env.close()
        if args.rgb_window:
            import cv2

            cv2.destroyAllWindows()
    successes = sum(bool(result["success"]) for result in results)
    grasps = sum(bool(result["ever_grasped"]) for result in results)
    print(
        f"vla_only_summary successes={successes}/{len(results)} "
        f"ever_grasped={grasps}/{len(results)} output={args.output}",
        flush=True,
    )
    return 0


def _show_rgb_window(observation: dict, *, step: int, delay_ms: int) -> None:
    """Show only the dual-RGB tensors supplied to the policy."""

    import cv2

    front = cv2.cvtColor(np.asarray(observation[IMAGE_KEY]), cv2.COLOR_RGB2BGR)
    wrist = cv2.cvtColor(np.asarray(observation[WRIST_IMAGE_KEY]), cv2.COLOR_RGB2BGR)
    panel = np.concatenate((front, wrist), axis=0)
    cv2.putText(
        panel,
        f"VLA-only agentview | step={step}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        panel,
        "wrist RGB",
        (8, front.shape[0] + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        panel,
        PICK_PLACE_GLOBAL_PROMPT,
        (8, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
    )
    cv2.imshow("PickPlace VLA-only input: agentview + wrist", panel)
    cv2.waitKey(max(1, delay_ms))


def _render_native_bin(env) -> None:
    env.render()
    native_viewer = getattr(getattr(env.backend, "viewer", None), "viewer", None)
    if native_viewer is not None:
        native_viewer.opt.geomgroup[:] = 1
        native_viewer.cam.type = 2
        native_viewer.cam.fixedcamid = env.backend.sim.model.camera_name2id(
            env.config.camera.third_person.name
        )


if __name__ == "__main__":
    raise SystemExit(main())
