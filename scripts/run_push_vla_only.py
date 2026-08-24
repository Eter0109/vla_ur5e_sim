"""Evaluate static-prompt Push with VLA only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.envs import UR5ePushConfig, make_ur5e_push  # noqa: E402
from vla_sim.pick_place_control import filter_vla_only_action  # noqa: E402
from vla_sim.policy_runtime import load_policy, predict_ensemble_chunk  # noqa: E402
from vla_sim.temporal import TemporalEnsemble  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open the robosuite viewer and render each rollout in real time.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "push_vla_only_rollouts.json")
    args = parser.parse_args()

    # Move policy to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset("local/ur5e_push_1000", root="data/lerobot/ur5e_push_1000")
    config, policy, preprocessor, postprocessor = load_policy(args.checkpoint, dataset, None)
    
    env_config = UR5ePushConfig(horizon=250, has_renderer=args.render)
    env = make_ur5e_push(env_config)
    
    results = []
    successes = 0
    
    try:
        for seed in range(args.episodes):
            observation, _ = env.reset(seed=seed)
            ensemble = TemporalEnsemble(16, 7)
            success = False
            
            for step in range(env_config.horizon):
                if step % args.replan_steps == 0:
                    prompt = "push the block into the red target circle"
                    policy.reset()
                    action_chunk = predict_ensemble_chunk(
                        observation,
                        policy,
                        preprocessor,
                        postprocessor,
                        device,
                        config.use_amp,
                        1,
                        task_prompt=prompt,
                    )
                    ensemble.add_chunk(step, action_chunk)
                
                raw_action = ensemble.get_action(step)
                action = filter_vla_only_action(
                    raw_action,
                    eef_xyz=np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float),
                )
                
                observation, _, terminated, truncated, info = env.step(action)
                success = bool(info["success"])
                if terminated or truncated:
                    break
                    
            print(f"seed={seed} success={success}", flush=True)
            if success:
                successes += 1
            results.append({"seed": seed, "success": success})
            
    finally:
        env.close()
        
    print(f"Total successes: {successes}/{args.episodes}")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
