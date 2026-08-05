"""Run dual-camera PickPlace policy rollouts on a versioned manifest."""

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
from lerobot.utils.control_utils import predict_action  # noqa: E402
from vla_sim.contracts import IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY  # noqa: E402
from vla_sim.envs import UR5ePickPlaceConfig, make_ur5e_pick_place  # noqa: E402
from vla_sim.pick_place_contract import validate_pick_place_contract  # noqa: E402
from vla_sim.pick_place_control import PickPlaceSupervisor  # noqa: E402
from vla_sim.policy_runtime import load_policy, predict_ensemble_chunk  # noqa: E402
from vla_sim.scenes import load_manifest, load_manifest_metadata  # noqa: E402
from vla_sim.stack_control import ColorDepthObjectPoseProvider  # noqa: E402
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
    parser.add_argument("--temporal-ensemble", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.5)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--samples-per-plan", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pick_place_rollouts.json")
    args = parser.parse_args()
    metadata = load_manifest_metadata(args.manifest)
    if metadata.get("environment_preset") != "pick_place_v1":
        parser.error("PickPlace rollouts require a compatible PickPlace manifest")
    if metadata.get("role") == "blind" and args.episodes != 100:
        parser.error("PickPlace blind evaluation must run all 100 scenes")
    if metadata.get("role") == "blind" and args.output.exists():
        parser.error("Blind output already exists and cannot be overwritten")
    scenes = load_manifest(args.manifest)[: args.episodes]
    # This legacy supervisor estimates object poses from RGB-D. The pure VLA
    # runner deliberately leaves depth disabled to reduce rollout memory use.
    env_config = UR5ePickPlaceConfig(
        horizon=args.horizon,
        has_renderer=args.render,
        use_camera_depths=True,
    )
    validate_pick_place_contract(args.dataset_root, env_config)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    config, policy, preprocessor, postprocessor = load_policy(args.checkpoint, dataset, None)
    env = make_ur5e_pick_place(env_config)
    provider = ColorDepthObjectPoseProvider()
    results: list[dict] = []
    try:
        for scene in scenes:
            observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            supervisor = PickPlaceSupervisor()
            temporal = TemporalEnsemble(config.n_action_steps, 7, decay=args.temporal_ensemble_decay, gripper_mode="latest") if args.temporal_ensemble else None
            actions: list[np.ndarray] = []
            phase_trace = [supervisor.phase.value]
            force_replan = False
            initial = np.asarray(env.raw_observation["cube_pos"], dtype=float)
            max_z = float(initial[2])
            ever_grasped = False
            success = False
            if args.rgb_window:
                _show_rgb_window(
                    observation,
                    step=0,
                    prompt=supervisor.prompt,
                    delay_ms=args.rgb_initial_hold_ms,
                )
            for step in range(1, args.horizon + 1):
                if temporal is not None:
                    if force_replan or (step - 1) % args.replan_steps == 0:
                        chunk = predict_ensemble_chunk(observation, policy, preprocessor, postprocessor, torch.device("cuda"), config.use_amp, args.samples_per_plan, task_prompt=supervisor.prompt)
                        temporal.add_chunk(step, chunk)
                        force_replan = False
                    raw_action = temporal.get_action(step)
                else:
                    tensor = predict_action(observation=observation, policy=policy, device=torch.device("cuda"), preprocessor=preprocessor, postprocessor=postprocessor, use_amp=config.use_amp, task=supervisor.prompt, robot_type="UR5e")
                    raw_action = tensor.detach().float().cpu().numpy().reshape(-1)
                state = observation[STATE_KEY]
                state = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
                state = np.asarray(state, dtype=np.float32).reshape(-1)
                estimate = provider.estimate(env.raw_observation, task="red_to_storage_bin", simulator=env.backend.sim)
                action = supervisor.filter_action(np.asarray(raw_action[:7], dtype=np.float32), eef_xyz=np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float), estimate=estimate, gripper_opening=float(state[-1]))
                if phase_trace[-1] != supervisor.phase.value:
                    phase_trace.append(supervisor.phase.value)
                    if temporal is not None:
                        temporal.reset()
                        force_replan = True
                actions.append(action)
                observation, _, terminated, truncated, info = env.step(action)
                if args.rgb_window:
                    _show_rgb_window(
                        observation,
                        step=step,
                        prompt=supervisor.prompt,
                        delay_ms=args.rgb_delay_ms,
                    )
                if args.render:
                    env.render()
                    # robosuite's interactive MuJoCo viewer hides geom group 0
                    # by default. The native storage bin uses that group, so
                    # re-enable it for faithful on-screen task visualization.
                    native_viewer = getattr(getattr(env.backend, "viewer", None), "viewer", None)
                    if native_viewer is not None:
                        native_viewer.opt.geomgroup[:] = 1
                        native_viewer.cam.type = 2
                        native_viewer.cam.fixedcamid = env.backend.sim.model.camera_name2id(
                            env.config.camera.third_person.name
                        )
                max_z = max(max_z, float(np.asarray(env.raw_observation["cube_pos"])[2]))
                ever_grasped = ever_grasped or bool(info["grasped"])
                success = bool(info["success"])
                if terminated or truncated:
                    break
            result = {"scene_id": scene.scene_id, "distance_bin": scene.overrides["distance_bin"], "success": success, "steps": step, "ever_grasped": ever_grasped, "max_lift_m": max_z - float(initial[2]), "phase_trace": phase_trace, "final_supervisor_phase": supervisor.phase.value, "place_conditions": info.get("place_conditions", {}), "failure_stage": "success" if success else ("no_grasp" if not ever_grasped else "placement_failed"), "mean_action": np.asarray(actions).mean(axis=0).tolist()}
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
    print(f"rollout_summary successes={successes}/{len(results)} output={args.output}")
    return 0


def _show_rgb_window(observation: dict, *, step: int, prompt: str, delay_ms: int) -> None:
    """Show the exact dual-RGB tensors consumed by SmolVLA."""
    import cv2

    front = cv2.cvtColor(np.asarray(observation[IMAGE_KEY]), cv2.COLOR_RGB2BGR)
    wrist = cv2.cvtColor(np.asarray(observation[WRIST_IMAGE_KEY]), cv2.COLOR_RGB2BGR)
    panel = np.concatenate((front, wrist), axis=0)
    cv2.putText(panel, f"agentview | step={step}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(panel, "wrist RGB", (8, front.shape[0] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(panel, prompt, (8, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.imshow("PickPlace VLA input: agentview + wrist", panel)
    cv2.waitKey(max(1, delay_ms))


if __name__ == "__main__":
    raise SystemExit(main())
