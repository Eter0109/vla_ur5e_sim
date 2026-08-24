"""Evaluate static-prompt PickPlace with no object-pose or phase-supervisor control."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from ctypes import wintypes
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
from vla_sim.artifact_identity import sha256_directory, sha256_file  # noqa: E402
from vla_sim.pick_place_contract import (  # noqa: E402
    PICK_PLACE_GLOBAL_PROMPT,
    validate_pick_place_contract,
)
from vla_sim.pick_place_control import (  # noqa: E402
    VLA_ONLY_CONTROL_MODES,
    VLAOnlyActionCalibration,
    VLAOnlyActionCalibrator,
    filter_vla_only_action,
    scene_policy_seed,
    uses_vla_only_action_calibration,
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
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--rgb-window", action="store_true")
    parser.add_argument("--rgb-delay-ms", type=int, default=1)
    parser.add_argument("--rgb-initial-hold-ms", type=int, default=1)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.5)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--samples-per-plan", type=int, default=1)
    parser.add_argument(
        "--control-mode", choices=VLA_ONLY_CONTROL_MODES, default="vla_raw_safety"
    )
    parser.add_argument("--closed-negative-y-gain", type=float, default=1.0)
    parser.add_argument("--transport-positive-x-gain", type=float, default=1.0)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=1000,
        help="Base seed combined with each environment seed for reproducible flow sampling.",
    )
    parser.add_argument(
        "--overwrite-development",
        action="store_true",
        help="Allow replacing an existing development-only result file.",
    )
    parser.add_argument(
        "--development-trace",
        action="store_true",
        help="Include per-step metrics-only traces; rejected for test and blind manifests.",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Record CPU RSS and CUDA allocator peaks at replanning boundaries; development only.",
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
    if metadata.get("environment_preset") not in {"pick_place_v1", "pick_place_robust_v1"}:
        parser.error("VLA-only evaluation requires a compatible PickPlace manifest")
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.samples_per_plan < 1:
        parser.error("samples-per-plan must be positive")
    scenes = load_manifest(args.manifest)
    role = str(metadata.get("role", ""))
    if role in {"test", "blind"} and args.episodes != len(scenes):
        parser.error(f"PickPlace {role} evaluation must run all {len(scenes)} scenes")
    if role in {"test", "blind"} and args.output.exists():
        parser.error(f"{role.title()} output already exists and cannot be overwritten")
    if args.development_trace and role in {"test", "blind"}:
        parser.error("Per-step traces are limited to development manifests")
    if args.memory_profile and role != "development":
        parser.error("Memory profiling is limited to development manifests")
    if args.output.exists() and not args.overwrite_development:
        parser.error("Output already exists; use --overwrite-development only for development")
    if args.control_mode == "vla_raw_safety" and (
        args.closed_negative_y_gain != 1.0 or args.transport_positive_x_gain != 1.0
    ):
        parser.error("vla_raw_safety requires both action gains to be exactly 1.0")
    if args.scene_index is not None:
        if role != "development" or not 0 <= args.scene_index < len(scenes):
            parser.error("scene-index is limited to a valid development scene")
        scenes = [scenes[args.scene_index]]
    if args.episodes > len(scenes):
        parser.error(f"manifest contains only {len(scenes)} scenes")
    scenes = scenes[: args.episodes]
    checkpoint_sha256 = sha256_directory(args.checkpoint)

    env_config = UR5ePickPlaceConfig(horizon=args.horizon, has_renderer=args.render)
    validate_pick_place_contract(args.dataset_root, env_config)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    config, policy, preprocessor, postprocessor = load_policy(
        args.checkpoint,
        dataset,
        None,
    )
    if not 1 <= args.replan_steps <= config.n_action_steps:
        parser.error(
            f"replan-steps must be in [1, {config.n_action_steps}] for this checkpoint"
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
            calibration_enabled = uses_vla_only_action_calibration(args.control_mode)
            action_calibrator = (
                VLAOnlyActionCalibrator(action_calibration)
                if calibration_enabled
                else None
            )
            actions: list[np.ndarray] = []
            plan_latencies_ms: list[float] = []
            trace: list[dict] = []
            memory_samples: list[dict[str, int | str]] = []
            if args.memory_profile:
                torch.cuda.reset_peak_memory_stats(device)
                memory_samples.append({"phase": "scene_start", **_memory_snapshot(device)})
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
                    if args.memory_profile:
                        memory_samples.append(
                            {"step": step, "phase": "before_plan", **_memory_snapshot(device)}
                        )
                    # ``predict_action_chunk`` populates SmolVLA's internal
                    # observation queues.  This runner executes full chunks
                    # directly, so retaining prior replans only accumulates
                    # large image tensors and is not temporal ensembling.
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
                        task_prompt=PICK_PLACE_GLOBAL_PROMPT,
                    )
                    plan_latencies_ms.append((time.perf_counter() - plan_started) * 1000.0)
                    temporal.add_chunk(step, chunk)
                    del chunk
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if args.memory_profile:
                        memory_samples.append(
                            {"step": step, "phase": "after_plan", **_memory_snapshot(device)}
                        )
                raw_action = temporal.get_action(step)
                model_action = np.asarray(raw_action[:7], dtype=np.float32)
                calibrated_action = (
                    action_calibrator.calibrate(model_action)
                    if action_calibrator is not None
                    else model_action
                )
                action = filter_vla_only_action(
                    calibrated_action,
                    eef_xyz=np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float),
                )
                actions.append(action)
                observation, _, terminated, truncated, info = env.step(action)
                if args.development_trace:
                    trace.append(
                        {
                            "step": step,
                            "model_action": model_action.tolist(),
                            "executed_action": action.tolist(),
                            "eef_xyz_m": np.asarray(
                                env.raw_observation["robot0_eef_pos"], dtype=float
                            ).tolist(),
                            "cube_xyz_m": np.asarray(
                                env.raw_observation["cube_pos"], dtype=float
                            ).tolist(),
                            "target_xyz_m": np.asarray(
                                env.raw_observation["target_zone_pos"], dtype=float
                            ).tolist(),
                        }
                    )
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
                "failure_stage": _failure_stage(success, ever_grasped, info),
                "place_conditions": info.get("place_conditions", {}),
                "mean_action": action_array.mean(axis=0).tolist(),
                "gripper_close_fraction": float((action_array[:, 6] > 0.0).mean()),
                "control_mode": args.control_mode,
                "static_prompt": PICK_PLACE_GLOBAL_PROMPT,
                "oracle_control": False,
                "fixed_rotation_safety": True,
                "workspace_clamp": True,
                "closed_negative_y_gain": args.closed_negative_y_gain,
                "transport_positive_x_gain": args.transport_positive_x_gain,
                "transport_direction_lock": calibration_enabled,
                "samples_per_plan": args.samples_per_plan,
                "policy_seed": policy_seed,
                "plan_latency_ms": {
                    "mean": float(np.mean(plan_latencies_ms)),
                    "p95": float(np.percentile(plan_latencies_ms, 95)),
                    "count": len(plan_latencies_ms),
                },
                "metrics_only": {
                    "final_cube_xyz_m": final_cube.tolist(),
                    "target_xyz_m": target.tolist(),
                    "xy_error_vector_m": (final_cube[:2] - target[:2]).tolist(),
                },
            }
            if args.development_trace:
                result["development_trace"] = trace
            if args.memory_profile:
                memory_samples.append({"phase": "scene_end", **_memory_snapshot(device)})
                result["memory_profile"] = memory_samples
            results.append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _partial_path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
            # SmolVLA's flow sampler and robosuite retain temporary tensors across
            # long scene suites on 6 GB GPUs unless reclaimed explicitly.
            del action_array, actions, plan_latencies_ms, temporal
            del observation, result
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
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
    _partial_path(args.output).replace(args.output)
    _metadata_path(args.output).write_text(
        json.dumps(
            {
                "schema_version": 3,
                "evaluation_fingerprint": _evaluation_fingerprint(
                    args, metadata, checkpoint_sha256
                ),
                "manifest_role": role,
                "manifest_sha256": sha256_file(args.manifest),
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_config_sha256": sha256_file(args.checkpoint / "config.json"),
                "dataset_root": str(args.dataset_root.resolve()),
                "control_mode": args.control_mode,
                "static_prompt": PICK_PLACE_GLOBAL_PROMPT,
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
    return 0


def _failure_stage(success: bool, ever_grasped: bool, info: dict) -> str:
    if success:
        return "success"
    if not ever_grasped:
        return "no_grasp"
    if not bool(info.get("ever_lifted", False)):
        return "no_lift"
    conditions = info.get("place_conditions", {})
    if not bool(conditions.get("gripper_released", False)):
        return "not_released"
    if not bool(conditions.get("in_target_zone", False)):
        return "xy_miss"
    if not bool(conditions.get("on_table", False)):
        return "not_on_table"
    if not bool(conditions.get("objects_stable", False)):
        return "unstable"
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
        "checkpoint_config_sha256": sha256_file(args.checkpoint / "config.json"),
        "dataset": str(args.dataset_root.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_role": metadata.get("role"),
        "control_mode": args.control_mode,
        "prompt": PICK_PLACE_GLOBAL_PROMPT,
        "samples": args.samples_per_plan,
        "replan": args.replan_steps,
        "decay": args.temporal_ensemble_decay,
        "policy_seed": args.policy_seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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
        native_viewer.cam.type = 2
        native_viewer.cam.fixedcamid = env.backend.sim.model.camera_name2id(
            env.config.camera.third_person.name
        )


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    """Return process RSS and CUDA allocator values without retaining tensors."""

    counters = _process_memory_counters()
    snapshot = {"rss_bytes": counters["WorkingSetSize"], "private_bytes": counters["PrivateUsage"]}
    if device.type == "cuda":
        snapshot.update(
            {
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
    return snapshot


def _process_memory_counters() -> dict[str, int]:
    """Read Windows process memory counters without adding a psutil dependency."""

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return {"WorkingSetSize": int(counters.WorkingSetSize), "PrivateUsage": int(counters.PrivateUsage)}


if __name__ == "__main__":
    raise SystemExit(main())
