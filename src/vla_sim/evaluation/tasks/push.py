"""Run pure-VLA Push evaluation on a versioned scene manifest."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_sim.paths import load_catalog, project_root, resolve_asset

ROOT = project_root()
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

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla_sim.evaluation.metrics import summarize_results
from vla_sim.policy.runtime import load_policy, predict_ensemble_chunk
from vla_sim.policy.temporal import TemporalEnsemble
from vla_sim.simulation.artifacts import sha256_directory
from vla_sim.simulation.pick_place_control import filter_vla_only_action
from vla_sim.simulation.scenes import load_manifest, load_manifest_metadata
from vla_sim.simulation.tasks import UR5ePushConfig, make_ur5e_push

PROMPT = "push the block into the red target circle"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    default_dataset = resolve_asset(load_catalog("simulation")["tasks"]["push"]["dataset"])
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--repo-id", default="local/ur5e_push_1000")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--temporal-decay", type=float, default=0.5)
    parser.add_argument("--policy-seed", type=int, default=1000)
    parser.add_argument("--samples-per-plan", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-trace", action="store_true")
    parser.add_argument("--overwrite-development", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = load_manifest_metadata(args.manifest)
    role = str(metadata.get("role", ""))
    if metadata.get("environment_preset") not in {"push_forward_v1", "push_robust_v1"}:
        raise ValueError("Push evaluation requires a compatible Push manifest.")
    scenes = load_manifest(args.manifest)
    if args.episodes < 1 or args.episodes > len(scenes):
        raise ValueError(f"episodes must be in [1, {len(scenes)}].")
    if not 1 <= args.replan_steps <= 16:
        raise ValueError("replan_steps must be in [1, 16].")
    if not 0 < args.temporal_decay <= 1:
        raise ValueError("temporal_decay must be in (0, 1].")
    if role in {"blind", "test"} and args.episodes != len(scenes):
        raise ValueError(f"{role} evaluation must run all {len(scenes)} scenes.")
    if role in {"blind", "test"} and args.development_trace:
        raise ValueError("development traces are forbidden for blind/test manifests.")
    if args.output.exists() and (role in {"blind", "test"} or not args.overwrite_development):
        raise FileExistsError("Output already exists; blind/test outputs are immutable.")

    np.random.seed(args.policy_seed)
    torch.manual_seed(args.policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.policy_seed)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    config, policy, preprocessor, postprocessor = load_policy(args.checkpoint, dataset, None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_ur5e_push(UR5ePushConfig(horizon=args.horizon))
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index, scene in enumerate(scenes[: args.episodes], start=1):
            observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
            initial_cube = np.asarray(env.raw_observation["cube_pos"], dtype=float)[:2].copy()
            target = env.target_pos[:2].copy()
            direction = target - initial_cube
            initial_distance = float(np.linalg.norm(direction))
            direction /= initial_distance
            temporal = TemporalEnsemble(16, 7, decay=args.temporal_decay)
            positions = [initial_cube.copy()]
            trace: list[dict[str, Any]] = []
            plan_latencies_ms: list[float] = []
            max_hold = 0
            success = False
            for step in range(args.horizon):
                if step % args.replan_steps == 0:
                    policy.reset()
                    plan_started = time.perf_counter()
                    chunk = predict_ensemble_chunk(
                        observation,
                        policy,
                        preprocessor,
                        postprocessor,
                        device,
                        config.use_amp,
                        args.samples_per_plan,
                        task_prompt=PROMPT,
                    )
                    plan_latencies_ms.append((time.perf_counter() - plan_started) * 1_000)
                    temporal.add_chunk(step, chunk)
                model_action = temporal.get_action(step)
                action = filter_vla_only_action(
                    model_action,
                    eef_xyz=np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float),
                )
                observation, _, terminated, truncated, info = env.step(action)
                cube_xy = np.asarray(env.raw_observation["cube_pos"], dtype=float)[:2].copy()
                positions.append(cube_xy)
                max_hold = max(max_hold, int(info["success_hold_count"]))
                if args.development_trace:
                    trace.append(
                        {
                            "step": step,
                            "model_action": model_action.tolist(),
                            "executed_action": action.tolist(),
                            "cube_xy_m": cube_xy.tolist(),
                            "eef_xyz_m": np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float).tolist(),
                            "distance_m": float(info["dist_to_target"]),
                        }
                    )
                success = bool(info["success"])
                if terminated or truncated:
                    break
            trajectory = np.asarray(positions)
            result = _result(
                scene.overrides,
                scene.scene_id,
                success,
                step + 1,
                initial_cube,
                target,
                trajectory,
                max_hold,
                plan_latencies_ms,
                args,
            )
            if args.development_trace:
                result["development_trace"] = trace
            results.append(result)
            _partial_path(args.output).parent.mkdir(parents=True, exist_ok=True)
            _partial_path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
            elapsed = time.perf_counter() - started
            eta = elapsed / index * (args.episodes - index)
            successes = sum(bool(value["success"]) for value in results)
            print(f"episode={index}/{args.episodes} successes={successes} eta_s={eta:.1f} scene={scene.scene_id}", flush=True)
            del observation, temporal, trajectory
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        env.close()
    summary = summarize_results(results)
    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256_file(args.manifest),
        "manifest_role": role,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_directory(args.checkpoint),
        "checkpoint_config_sha256": _sha256_file(
            (args.checkpoint / "config.json")
            if (args.checkpoint / "config.json").is_file()
            else (args.checkpoint / "adapter_config.json")
        ),
        "dataset_root": str(args.dataset_root.resolve()),
        "prompt": PROMPT,
        "control": "vla_raw_safety_fixed_rotation_workspace_clamp",
        "inference": {
            "replan_steps": args.replan_steps,
            "temporal_decay": args.temporal_decay,
            "policy_seed": args.policy_seed,
            "samples_per_plan": args.samples_per_plan,
        },
        "summary": summary,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _partial_path(args.output).unlink(missing_ok=True)
    print(f"push_summary successes={summary['successes']}/{summary['episodes']} rate={summary['success_rate']:.3f}", flush=True)
    return 0


def _result(
    overrides: dict[str, Any], scene_id: str, success: bool, steps: int, initial_cube: np.ndarray,
    target: np.ndarray, trajectory: np.ndarray, max_hold: int, plan_latencies_ms: list[float], args: argparse.Namespace,
) -> dict[str, Any]:
    direction = target - initial_cube
    initial_distance = float(np.linalg.norm(direction))
    direction /= initial_distance
    displacement = trajectory - initial_cube
    forward = displacement @ direction
    lateral = np.abs(displacement[:, 0] * direction[1] - displacement[:, 1] * direction[0])
    distances = np.linalg.norm(trajectory - target, axis=1)
    failure_stage = _failure_stage(success, float(np.max(np.linalg.norm(displacement, axis=1))), float(np.max(forward)), float(np.min(distances)), max_hold)
    return {
        "scene_id": scene_id,
        "success": success,
        "steps": steps,
        "failure_stage": failure_stage,
        "angle_bin": int(overrides["angle_bin"]),
        "distance_bin": int(overrides["distance_bin"]),
        "initial_distance_m": initial_distance,
        "final_distance_m": float(distances[-1]),
        "min_distance_m": float(np.min(distances)),
        "max_forward_progress_m": float(np.max(forward)),
        "final_lateral_error_m": float(lateral[-1]),
        "max_hold_steps": max_hold,
        "policy_inference_p50_s": float(np.median(plan_latencies_ms) / 1_000),
        "policy_inference_p95_s": float(np.percentile(plan_latencies_ms, 95) / 1_000),
        "episode_wall_time_s": float(steps),
        "control_mode": "vla_raw_safety",
        "replan_steps": args.replan_steps,
        "temporal_decay": args.temporal_decay,
    }


def _failure_stage(success: bool, max_displacement: float, max_forward: float, min_distance: float, max_hold: int) -> str:
    if success:
        return "success"
    if max_displacement < 0.005:
        return "no_contact"
    if max_forward <= 0.005:
        return "wrong_direction"
    if min_distance <= 0.05 or max_hold > 0:
        return "overshoot_or_unstable"
    if max_forward < 0.05:
        return "stopped_short"
    return "lateral_miss"


def _partial_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".partial")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
