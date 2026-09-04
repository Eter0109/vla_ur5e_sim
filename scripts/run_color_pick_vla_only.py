"""Evaluate a VLA policy on language-conditioned three-color cube selection."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
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

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla_sim.artifact_identity import sha256_directory, sha256_file
from vla_sim.color_pick_contract import color_pick_prompt, validate_color_pick_contract
from vla_sim.envs import UR5eColorPickConfig, make_ur5e_color_pick
from vla_sim.pick_place_control import filter_vla_only_action, scene_policy_seed
from vla_sim.policy_runtime import load_policy, predict_ensemble_chunk
from vla_sim.scenes import load_manifest, load_manifest_metadata
from vla_sim.temporal import TemporalEnsemble


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/color_pick_1500")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.75)
    parser.add_argument("--samples-per-plan", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--overwrite-development", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = load_manifest_metadata(args.manifest)
    if metadata.get("environment_preset") != "color_pick_v1":
        parser.error("manifest must use the color_pick_v1 environment preset")
    if args.episodes < 1 or args.samples_per_plan < 1:
        parser.error("episodes and samples-per-plan must be positive")
    scenes = load_manifest(args.manifest)
    role = str(metadata.get("role", ""))
    if args.episodes > len(scenes):
        parser.error(f"manifest contains only {len(scenes)} scenes")
    if role in {"test", "blind"} and args.episodes != len(scenes):
        parser.error(f"ColorPick {role} evaluation must run all {len(scenes)} scenes")
    if args.output.exists() and not (role == "development" and args.overwrite_development):
        parser.error("output already exists and cannot be overwritten")
    scenes = scenes[: args.episodes]

    env_config = UR5eColorPickConfig(horizon=args.horizon, has_renderer=args.render)
    validate_color_pick_contract(args.dataset_root, env_config)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    config, policy, preprocessor, postprocessor = load_policy(
        args.checkpoint, dataset, None
    )
    if not 1 <= args.replan_steps <= config.n_action_steps:
        parser.error(f"replan-steps must be in [1, {config.n_action_steps}]")

    device = torch.device("cuda")
    checkpoint_sha256 = sha256_directory(args.checkpoint)
    env = make_ur5e_color_pick(env_config)
    results: list[dict] = []
    try:
        for scene in scenes:
            target_color = str(scene.overrides["target_color"])
            prompt = color_pick_prompt(target_color)
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
            plan_latencies_ms: list[float] = []
            initial_z = float(np.asarray(env.raw_observation["target_cube_pos"])[2])
            max_lift_m = 0.0
            success = False
            info: dict = {}
            for step in range(1, args.horizon + 1):
                if (step - 1) % args.replan_steps == 0:
                    policy.reset()
                    started = time.perf_counter()
                    chunk = predict_ensemble_chunk(
                        observation,
                        policy,
                        preprocessor,
                        postprocessor,
                        device,
                        config.use_amp,
                        args.samples_per_plan,
                        task_prompt=prompt,
                    )
                    plan_latencies_ms.append((time.perf_counter() - started) * 1000.0)
                    temporal.add_chunk(step, chunk)
                    del chunk
                    gc.collect()
                    torch.cuda.empty_cache()
                model_action = np.asarray(temporal.get_action(step)[:7], dtype=np.float32)
                action = filter_vla_only_action(
                    model_action,
                    eef_xyz=np.asarray(
                        env.raw_observation["robot0_eef_pos"], dtype=np.float64
                    ),
                )
                observation, _, terminated, truncated, info = env.step(action)
                target_z = float(np.asarray(env.raw_observation["target_cube_pos"])[2])
                max_lift_m = max(max_lift_m, target_z - initial_z)
                success = bool(info["success"])
                if args.render:
                    env.render()
                if terminated or truncated:
                    break
            result = {
                "scene_id": scene.scene_id,
                "target_color": target_color,
                "prompt": prompt,
                "success": success,
                "steps": step,
                "ever_target_grasped": bool(info.get("ever_target_grasped", False)),
                "ever_wrong_object_grasped": bool(
                    info.get("ever_wrong_object_grasped", False)
                ),
                "wrong_colors_grasped": info.get("wrong_colors_grasped", []),
                "max_target_lift_m": max_lift_m,
                "failure_stage": _failure_stage(success, info, max_lift_m),
                "static_prompt": prompt,
                "oracle_control": False,
                "fixed_rotation_safety": True,
                "workspace_clamp": True,
                "samples_per_plan": args.samples_per_plan,
                "policy_seed": policy_seed,
                "plan_latency_ms": {
                    "mean": float(np.mean(plan_latencies_ms)),
                    "p95": float(np.percentile(plan_latencies_ms, 95)),
                    "count": len(plan_latencies_ms),
                },
            }
            results.append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _partial_path(args.output).write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
            print(json.dumps(result), flush=True)
            del observation, temporal
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        env.close()

    _partial_path(args.output).replace(args.output)
    _metadata_path(args.output).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluation_fingerprint": _evaluation_fingerprint(
                    args, metadata, checkpoint_sha256
                ),
                "manifest_role": role,
                "manifest_sha256": sha256_file(args.manifest),
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "dataset_root": str(args.dataset_root.resolve()),
                "control_mode": "vla_raw_safety",
                "target_signal": "language_only",
                "inference": {
                    "samples_per_plan": args.samples_per_plan,
                    "replan_steps": args.replan_steps,
                    "temporal_ensemble_decay": args.temporal_ensemble_decay,
                    "policy_seed": args.policy_seed,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    successes = sum(bool(result["success"]) for result in results)
    wrong = sum(bool(result["ever_wrong_object_grasped"]) for result in results)
    print(
        f"color_pick_summary successes={successes}/{len(results)} "
        f"wrong_color_grasps={wrong}/{len(results)} output={args.output}",
        flush=True,
    )
    return 0


def _failure_stage(success: bool, info: dict, max_lift_m: float) -> str:
    if success:
        return "success"
    if info.get("ever_wrong_object_grasped", False):
        return "wrong_color"
    if not info.get("ever_target_grasped", False):
        return "no_target_grasp"
    if max_lift_m < 0.080:
        return "no_target_lift"
    return "hold_timeout"


def _partial_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".partial")


def _metadata_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".meta.json")


def _evaluation_fingerprint(
    args: argparse.Namespace, metadata: dict, checkpoint_sha256: str
) -> str:
    payload = {
        "checkpoint_sha256": checkpoint_sha256,
        "dataset": str(args.dataset_root.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_role": metadata.get("role"),
        "control_mode": "vla_raw_safety",
        "target_signal": "language_only",
        "samples": args.samples_per_plan,
        "replan": args.replan_steps,
        "decay": args.temporal_ensemble_decay,
        "policy_seed": args.policy_seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
