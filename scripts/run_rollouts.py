"""Run a saved SmolVLA/LoRA checkpoint in the UR5e simulation loop."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUNTIME = ROOT / ".runtime"
NUMBA_CACHE = Path(tempfile.gettempdir()) / "vla_sim_numba"
NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.update(
    {
        "HF_HOME": str(RUNTIME / "hf"),
        "HF_DATASETS_CACHE": str(RUNTIME / "hf_datasets"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "USE_TF": "0",
        "NUMBA_CACHE_DIR": str(NUMBA_CACHE),
        "NUMBA_DISABLE_JIT": "1" if os.name == "nt" else "0",
    }
)

import torch  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from vla_sim.envs import UR5eLiftConfig, make_ur5e_lift  # noqa: E402
from vla_sim.lerobot_compat import install_fast_parquet_loader  # noqa: E402
from vla_sim.scenes import load_manifest, load_manifest_metadata  # noqa: E402
from vla_sim.temporal import TemporalEnsemble  # noqa: E402

install_fast_parquet_loader()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    """Fingerprint a file or directory by its relative names and contents."""
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(_sha256(child).encode("ascii"))
    return digest.hexdigest()


def _runtime_metadata() -> dict[str, Any]:
    packages = ("torch", "lerobot", "peft", "mujoco", "robosuite", "numpy")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "packages": versions,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.{os.getpid()}.partial.json")
    serialized = json.dumps(value, indent=2)
    try:
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(path)
    except PermissionError:
        # Some managed Windows sandboxes allow overwriting an approved output
        # file but deny creation of a sibling temporary file.
        path.write_text(serialized, encoding="utf-8")


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "branch": run("branch", "--show-current"),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "diff_sha256": hashlib.sha256(run("diff", "--binary").encode()).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"branch": None, "commit": None, "dirty": None, "diff_sha256": None}


def _validate_resume(
    metadata_path: Path, expected: Mapping[str, Any], scenes: list[Any], results: list[dict[str, Any]]
) -> None:
    """Reject resumes whose immutable benchmark inputs do not exactly match."""
    if not metadata_path.exists():
        raise ValueError("--resume requires the matching rollout metadata file")
    previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    immutable = ("checkpoint", "checkpoint_sha256", "dataset_sha256", "manifest_sha256")
    mismatches = [key for key in immutable if previous.get(key) != expected.get(key)]
    if previous.get("experiment_config", {}).get("sha256") != expected["experiment_config"]["sha256"]:
        mismatches.append("experiment_config.sha256")
    expected_ids = {scene.scene_id for scene in scenes}
    result_ids = [str(result.get("scene_id")) for result in results]
    if len(result_ids) != len(set(result_ids)):
        mismatches.append("duplicate completed scene IDs")
    if not set(result_ids).issubset(expected_ids):
        mismatches.append("completed scenes outside manifest")
    if mismatches:
        raise ValueError("--resume provenance mismatch: " + ", ".join(mismatches))


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _failure_stage(
    *, success: bool, ever_grasped: bool, max_lift_m: float, success_hold_steps: int
) -> str:
    if success:
        return "success"
    if not ever_grasped:
        return "no_grasp"
    if max_lift_m < 0.10:
        return "grasp_no_lift"
    if success_hold_steps < 10:
        return "lift_no_hold"
    return "incomplete"


def _argument_was_supplied(flag: str) -> bool:
    return any(argument == flag or argument.startswith(f"{flag}=") for argument in sys.argv[1:])


def _apply_experiment_config(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    """Apply inference defaults from a versioned JSON preset.

    Explicit command-line values always win; the resolved source is persisted
    in rollout metadata so a preset cannot silently alter a benchmark.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ValueError("--experiment-config must be a schema_version=2 JSON object")
    inference = value.get("inference", {})
    if not isinstance(inference, dict):
        raise ValueError("Experiment config inference section must be an object")
    fields = {
        "temporal_ensemble": "--temporal-ensemble",
        "temporal_ensemble_decay": "--temporal-ensemble-decay",
        "replan_steps": "--replan-steps",
        "samples_per_plan": "--samples-per-plan",
        "gripper_mode": "--gripper-mode",
        "gripper_close_threshold": "--gripper-close-threshold",
        "gripper_confirm_steps": "--gripper-confirm-steps",
        "gripper_hold_steps": "--gripper-hold-steps",
    }
    for name, flag in fields.items():
        if name in inference and not _argument_was_supplied(flag):
            setattr(args, name, inference[name])
    environment = value.get("environment", {})
    if isinstance(environment, dict) and "horizon" in environment and not _argument_was_supplied("--horizon"):
        args.horizon = environment["horizon"]
    return value


def _install_peft_compatibility() -> None:
    if not hasattr(SmolVLAConfig, "get"):
        SmolVLAConfig.get = lambda self, key, default=None: getattr(self, key, default)  # type: ignore[attr-defined]
    if not hasattr(SmolVLAConfig, "__contains__"):
        SmolVLAConfig.__contains__ = lambda self, key: hasattr(self, key)  # type: ignore[attr-defined]


def load_policy(
    checkpoint: Path, dataset: LeRobotDataset, action_steps: int | None = None
):
    _install_peft_compatibility()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = "cuda"
    config.use_amp = True
    if action_steps is not None:
        config.n_action_steps = action_steps
    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset.meta.stats,
    )
    policy.eval()
    return config, policy, preprocessor, postprocessor


def predict_ensemble_chunk(
    observation,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    use_amp: bool,
    samples: int,
) -> np.ndarray:
    """Average independent flow samples before executing an action chunk."""

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and use_amp
        else nullcontext(),
    ):
        batch = prepare_observation_for_inference(
            observation,
            device,
            "Grasp the red object and lift it at least ten centimeters",
            "UR5e",
        )
        batch = preprocessor(batch)
        chunks = [policy.predict_action_chunk(batch) for _ in range(samples)]
        chunk = postprocessor(torch.stack(chunks).mean(dim=0))
    return chunk[0].detach().float().cpu().numpy()


def main() -> int:
    if os.environ.get("VLA_DEBUG_HANG") == "1":
        faulthandler.dump_traceback_later(60, repeat=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        help="Schema-v2 JSON preset whose inference defaults are applied before validation.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/ur5e_custom_lift")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--render", action="store_true", help="show the MuJoCo viewer")
    parser.add_argument(
        "--action-steps",
        type=int,
        help="Execute this many actions from each predicted chunk before replanning.",
    )
    parser.add_argument(
        "--samples-per-plan",
        type=int,
        default=1,
        help="Average this many independently sampled action chunks.",
    )
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="Enable temporal ensembling of action chunks over time.",
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.5,
        help="Exponential decay weight for temporal ensembling.",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=4,
        help="Replan frequency (in steps) when temporal ensembling is enabled.",
    )
    parser.add_argument(
        "--gripper-mode",
        choices=sorted(TemporalEnsemble.MODES),
        default="latest",
        help="How to aggregate the discrete gripper command in temporal mode.",
    )
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=0.5,
        help="Latch the gripper closed above this command value.",
    )
    parser.add_argument(
        "--gripper-confirm-steps",
        type=int,
        default=2,
        help="Consecutive close commands required before debounce mode latches.",
    )
    parser.add_argument(
        "--gripper-hold-steps",
        type=int,
        default=4,
        help="Minimum closed duration after a close command in hold mode.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "rollouts.json")
    parser.add_argument(
        "--policy-seed",
        type=int,
        help="Independent policy RNG seed; defaults to the scene seed plus one million.",
    )
    parser.add_argument(
        "--benchmark-role",
        choices=("legacy", "development", "blind", "diagnostic"),
        help="Expected role of the manifest; mismatches fail fast.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted evaluation without rerunning completed scene IDs.",
    )
    args = parser.parse_args()

    experiment_config: dict[str, Any] | None = None
    if args.experiment_config:
        try:
            experiment_config = _apply_experiment_config(args, args.experiment_config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))

    if args.samples_per_plan < 1:
        parser.error("--samples-per-plan must be at least 1")
    if args.replan_steps < 1:
        parser.error("--replan-steps must be at least 1")
    if not 0 < args.temporal_ensemble_decay <= 1:
        parser.error("--temporal-ensemble-decay must be in (0, 1]")
    if not -1 <= args.gripper_close_threshold <= 1:
        parser.error("--gripper-close-threshold must be in [-1, 1]")
    if args.gripper_confirm_steps < 1:
        parser.error("--gripper-confirm-steps must be at least 1")
    if args.gripper_hold_steps < 1:
        parser.error("--gripper-hold-steps must be at least 1")

    manifest_metadata = load_manifest_metadata(args.manifest)
    if args.benchmark_role and manifest_metadata["role"] not in {
        args.benchmark_role,
        "legacy" if args.benchmark_role == "legacy" else None,
    }:
        parser.error(
            f"Manifest role {manifest_metadata['role']!r} does not match "
            f"--benchmark-role={args.benchmark_role!r}"
        )
    scenes = load_manifest(args.manifest)[: args.episodes]
    metadata_path = Path(f"{args.output}.meta.json")
    results: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        loaded_results = json.loads(args.output.read_text(encoding="utf-8"))
        if not isinstance(loaded_results, list):
            parser.error("--resume output must contain a JSON result list")
        results = loaded_results
    completed_scene_ids = {str(result.get("scene_id")) for result in results}
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    metadata = {
        "schema_version": 2,
        "status": "running",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _tree_sha256(checkpoint),
        "dataset": str(dataset_root),
        "dataset_sha256": _tree_sha256(dataset_root),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "manifest_metadata": manifest_metadata,
        "experiment_config": {
            "path": str(args.experiment_config.resolve()) if args.experiment_config else None,
            "schema_version": experiment_config.get("schema_version") if experiment_config else None,
            "experiment_id": experiment_config.get("experiment_id") if experiment_config else None,
            "sha256": _sha256(args.experiment_config) if args.experiment_config else None,
        },
        "completed_scene_ids": sorted(completed_scene_ids),
        "pre_postprocessor": {
            "factory": "lerobot.policies.factory.make_pre_post_processors",
            "checkpoint_sha256": _tree_sha256(checkpoint),
        },
        "environment": _runtime_metadata(),
        "git": _git_metadata(),
    }
    if args.resume:
        try:
            _validate_resume(metadata_path, metadata, scenes, results)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    _write_json_atomic(metadata_path, metadata)

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    config, policy, preprocessor, postprocessor = load_policy(
        args.checkpoint, dataset, args.action_steps
    )
    env = make_ur5e_lift(
        UR5eLiftConfig(horizon=args.horizon, has_renderer=args.render)
    )
    evaluation_error: BaseException | None = None
    try:
        for scene in scenes:
            if scene.scene_id in completed_scene_ids:
                continue
            episode_started = time.perf_counter()
            policy_seed = args.policy_seed if args.policy_seed is not None else scene.seed + 1_000_000
            np.random.seed(policy_seed)
            torch.manual_seed(policy_seed)
            torch.cuda.manual_seed_all(policy_seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            latencies: list[float] = []
            inference_latencies: list[float] = []
            actions: list[np.ndarray] = []
            initial_cube_pos = np.asarray(env.raw_observation["cube_pos"], dtype=float)
            max_cube_z = float(initial_cube_pos[2])
            max_hold_count = 0
            ensemble_actions: deque[np.ndarray] = deque()
            temporal_ensemble = (
                TemporalEnsemble(
                    chunk_size=config.n_action_steps,
                    action_dim=7,
                    decay=args.temporal_ensemble_decay,
                    gripper_mode=args.gripper_mode,
                    gripper_close_threshold=args.gripper_close_threshold,
                    gripper_confirm_steps=args.gripper_confirm_steps,
                    gripper_hold_steps=args.gripper_hold_steps,
                )
                if args.temporal_ensemble
                else None
            )

            # Diagnostics initialization
            min_dist = float("inf")
            approach_success = False
            grasp_attempted = False
            time_to_approach = -1
            first_gripper_close_step = -1
            gripper_transition_count = 0
            raw_gripper_transition_count = 0
            eef_cube_dist_at_first_close_m: float | None = None
            max_lift_after_first_close_m: float | None = None
            previous_gripper_closed = False
            previous_raw_gripper_closed = False
            ever_grasped = False
            first_grasp_step = -1
            grasp_streak = 0
            max_consecutive_grasp_steps = 0

            success = False
            for step in range(1, args.horizon + 1):
                started = time.perf_counter()
                if args.temporal_ensemble:
                    if (step - 1) % args.replan_steps == 0:
                        inference_started = time.perf_counter()
                        chunk = predict_ensemble_chunk(
                            observation,
                            policy,
                            preprocessor,
                            postprocessor,
                            torch.device("cuda"),
                            config.use_amp,
                            samples=args.samples_per_plan,
                        )
                        inference_latencies.append(time.perf_counter() - inference_started)
                        if len(chunk) < args.replan_steps:
                            raise ValueError(
                                f"replan_steps={args.replan_steps} exceeds predicted chunk "
                                f"length={len(chunk)}"
                            )
                        temporal_ensemble.add_chunk(step, chunk)
                    action_array = temporal_ensemble.get_action(step)
                elif args.samples_per_plan > 1:
                    if not ensemble_actions:
                        inference_started = time.perf_counter()
                        chunk = predict_ensemble_chunk(
                            observation,
                            policy,
                            preprocessor,
                            postprocessor,
                            torch.device("cuda"),
                            config.use_amp,
                            args.samples_per_plan,
                        )
                        inference_latencies.append(time.perf_counter() - inference_started)
                        ensemble_actions.extend(
                            chunk[: config.n_action_steps]
                        )
                    action_array = ensemble_actions.popleft()
                else:
                    inference_started = time.perf_counter()
                    action_tensor = predict_action(
                        observation=observation,
                        policy=policy,
                        device=torch.device("cuda"),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=config.use_amp,
                        task="Grasp the red object and lift it at least ten centimeters",
                        robot_type="UR5e",
                    )
                    action_array = action_tensor.detach().float().cpu().numpy().reshape(-1)
                    inference_latencies.append(time.perf_counter() - inference_started)
                latencies.append(time.perf_counter() - started)
                action = np.clip(
                    action_array[:7],
                    -1.0,
                    1.0,
                ).astype(np.float32)
                actions.append(action.copy())
                raw_gripper = (
                    temporal_ensemble.last_raw_gripper
                    if temporal_ensemble is not None and temporal_ensemble.last_raw_gripper is not None
                    else float(action_array[6])
                )
                pre_cube_pos = np.asarray(env.raw_observation["cube_pos"], dtype=float)
                pre_eef_pos = np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float)
                gripper_closed = bool(action[6] > args.gripper_close_threshold)
                if gripper_closed != previous_gripper_closed:
                    gripper_transition_count += 1
                raw_gripper_closed = bool(raw_gripper > args.gripper_close_threshold)
                if raw_gripper_closed != previous_raw_gripper_closed:
                    raw_gripper_transition_count += 1
                if gripper_closed and first_gripper_close_step == -1:
                    first_gripper_close_step = step
                    eef_cube_dist_at_first_close_m = float(
                        np.linalg.norm(pre_cube_pos - pre_eef_pos)
                    )
                    max_lift_after_first_close_m = 0.0
                previous_gripper_closed = gripper_closed
                previous_raw_gripper_closed = raw_gripper_closed
                observation, _, terminated, truncated, info = env.step(action)
                if args.render:
                    env.render()
                max_cube_z = max(
                    max_cube_z,
                    float(np.asarray(env.raw_observation["cube_pos"])[2]),
                )
                if first_gripper_close_step != -1:
                    max_lift_after_first_close_m = max(
                        float(max_lift_after_first_close_m or 0.0),
                        max_cube_z - float(initial_cube_pos[2]),
                    )
                max_hold_count = max(max_hold_count, int(info["success_hold_count"]))
                success = bool(info["success"])
                grasped = bool(info.get("grasped", False))
                ever_grasped = ever_grasped or grasped
                if grasped:
                    grasp_streak += 1
                    if first_grasp_step == -1:
                        first_grasp_step = step
                else:
                    grasp_streak = 0
                max_consecutive_grasp_steps = max(max_consecutive_grasp_steps, grasp_streak)

                # Update diagnostics
                cube_pos = np.asarray(env.raw_observation["cube_pos"], dtype=float)
                eef_pos = np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float)
                dist = float(np.linalg.norm(cube_pos - eef_pos))
                min_dist = min(min_dist, dist)
                if dist < 0.03:
                    approach_success = True
                    if time_to_approach == -1:
                        time_to_approach = step
                if action[6] > 0.5:
                    grasp_attempted = True

                if terminated or truncated:
                    break
            action_array = np.asarray(actions)
            final_cube_pos = np.asarray(env.raw_observation["cube_pos"], dtype=float)
            final_eef_pos = np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float)
            max_lift_m = max_cube_z - float(initial_cube_pos[2])
            result = {
                "scene_id": scene.scene_id,
                "env_seed": scene.effective_env_seed,
                "policy_seed": policy_seed,
                "success": success,
                "steps": step,
                "episode_wall_time_s": time.perf_counter() - episode_started,
                "mean_policy_latency_s": float(np.mean(latencies)),
                "policy_inference_p50_s": _percentile(inference_latencies, 50),
                "policy_inference_p95_s": _percentile(inference_latencies, 95),
                "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "initial_cube_pos": initial_cube_pos.tolist(),
                "final_cube_pos": final_cube_pos.tolist(),
                "final_eef_pos": final_eef_pos.tolist(),
                "max_lift_m": max_lift_m,
                "max_success_hold_steps": max_hold_count,
                "mean_action": action_array.mean(axis=0).tolist(),
                "std_action": action_array.std(axis=0).tolist(),
                # Diagnostics
                "min_eef_cube_dist_m": min_dist,
                "approach_success": approach_success,
                "grasp_attempted": grasp_attempted,
                "time_to_approach_steps": time_to_approach,
                "first_gripper_close_step": first_gripper_close_step,
                "gripper_transition_count": gripper_transition_count,
                "raw_gripper_transition_count": raw_gripper_transition_count,
                "eef_cube_dist_at_first_close_m": eef_cube_dist_at_first_close_m,
                "max_lift_after_first_close_m": max_lift_after_first_close_m,
                "ever_grasped": ever_grasped,
                "first_grasp_step": first_grasp_step,
                "max_consecutive_grasp_steps": max_consecutive_grasp_steps,
                "failure_stage": _failure_stage(
                    success=success,
                    ever_grasped=ever_grasped,
                    max_lift_m=max_lift_m,
                    success_hold_steps=max_hold_count,
                ),
            }
            results.append(result)
            _write_json_atomic(args.output, results)
            metadata["completed_scene_ids"] = sorted(
                {str(item["scene_id"]) for item in results}
            )
            _write_json_atomic(metadata_path, metadata)
            print(json.dumps(result), flush=True)
    except BaseException as exc:
        evaluation_error = exc
        metadata["status"] = "interrupted"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.close()
        _write_json_atomic(args.output, results)
        if evaluation_error is None:
            metadata["status"] = "completed"
        metadata["completed_scene_ids"] = sorted(
            {str(item["scene_id"]) for item in results}
        )
        _write_json_atomic(metadata_path, metadata)
    successes = sum(int(result["success"]) for result in results)
    print(f"rollout_summary successes={successes}/{len(results)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
