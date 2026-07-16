"""Run a saved SmolVLA/LoRA checkpoint in the UR5e simulation loop."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import tempfile
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

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
from vla_sim.scenes import load_manifest  # noqa: E402

install_fast_parquet_loader()


class TemporalEnsemble:
    """Temporal ensemble that aggregates action predictions over time using exponential decay weights."""

    def __init__(self, chunk_size: int, action_dim: int, decay: float = 0.5):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.decay = decay
        self._buffer: dict[int, list[tuple[np.ndarray, float]]] = {}

    def add_chunk(self, start_step: int, chunk: np.ndarray):
        weights = self.decay ** np.arange(len(chunk))
        for i, (action, w) in enumerate(zip(chunk, weights)):
            t = start_step + i
            self._buffer.setdefault(t, []).append((action, w))

    def get_action(self, step: int) -> np.ndarray:
        entries = self._buffer.pop(step, None)
        if entries is None:
            raise ValueError(f"No prediction for step {step}")
        actions, weights = zip(*entries)
        weights = np.array(weights)
        return np.average(actions, axis=0, weights=weights)

    def has_action(self, step: int) -> bool:
        return step in self._buffer


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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "rollouts.json")
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    config, policy, preprocessor, postprocessor = load_policy(
        args.checkpoint, dataset, args.action_steps
    )
    env = make_ur5e_lift(
        UR5eLiftConfig(horizon=args.horizon, has_renderer=args.render)
    )
    scenes = load_manifest(args.manifest)[: args.episodes]
    results: list[dict[str, Any]] = []
    try:
        for scene in scenes:
            np.random.seed(scene.seed)
            torch.manual_seed(scene.seed)
            torch.cuda.manual_seed_all(scene.seed)
            observation, _ = env.reset(seed=scene.seed, scene=scene)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            latencies: list[float] = []
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
                )
                if args.temporal_ensemble
                else None
            )

            # Diagnostics initialization
            min_dist = float("inf")
            approach_success = False
            grasp_attempted = False
            time_to_approach = -1

            success = False
            for step in range(1, args.horizon + 1):
                started = time.perf_counter()
                if args.temporal_ensemble:
                    if (step - 1) % args.replan_steps == 0:
                        chunk = predict_ensemble_chunk(
                            observation,
                            policy,
                            preprocessor,
                            postprocessor,
                            torch.device("cuda"),
                            config.use_amp,
                            samples=1,
                        )
                        temporal_ensemble.add_chunk(step, chunk)
                    action_array = temporal_ensemble.get_action(step)
                elif args.samples_per_plan > 1:
                    if not ensemble_actions:
                        chunk = predict_ensemble_chunk(
                            observation,
                            policy,
                            preprocessor,
                            postprocessor,
                            torch.device("cuda"),
                            config.use_amp,
                            args.samples_per_plan,
                        )
                        ensemble_actions.extend(
                            chunk[: config.n_action_steps]
                        )
                    action_array = ensemble_actions.popleft()
                else:
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
                latencies.append(time.perf_counter() - started)
                action = np.clip(
                    action_array[:7],
                    -1.0,
                    1.0,
                ).astype(np.float32)
                actions.append(action.copy())
                observation, _, terminated, truncated, info = env.step(action)
                if args.render:
                    env.render()
                max_cube_z = max(
                    max_cube_z,
                    float(np.asarray(env.raw_observation["cube_pos"])[2]),
                )
                max_hold_count = max(max_hold_count, int(info["success_hold_count"]))
                success = bool(info["success"])

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
            result = {
                "scene_id": scene.scene_id,
                "success": success,
                "steps": step,
                "mean_policy_latency_s": float(np.mean(latencies)),
                "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "initial_cube_pos": initial_cube_pos.tolist(),
                "final_cube_pos": final_cube_pos.tolist(),
                "final_eef_pos": final_eef_pos.tolist(),
                "max_lift_m": max_cube_z - float(initial_cube_pos[2]),
                "max_success_hold_steps": max_hold_count,
                "mean_action": action_array.mean(axis=0).tolist(),
                "std_action": action_array.std(axis=0).tolist(),
                # Diagnostics
                "min_eef_cube_dist_m": min_dist,
                "approach_success": approach_success,
                "grasp_attempted": grasp_attempted,
                "time_to_approach_steps": time_to_approach,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    successes = sum(int(result["success"]) for result in results)
    print(f"rollout_summary successes={successes}/{len(results)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
