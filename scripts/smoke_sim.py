"""Smoke-test the UR5e Lift simulator with a heuristic or random policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vla_sim.envs import (  # noqa: E402
    CameraConfig,
    PrimitiveObjectConfig,
    UR5eLiftConfig,
    make_ur5e_lift,
)
from vla_sim.sim import (  # noqa: E402
    ContractError,
    HeuristicLiftExpert,
    RobosuiteUnavailableError,
)


def _dimensions(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dimensions must be comma-separated metres") from exc
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("dimensions must contain three positive values")
    return parts  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--shape", choices=("box", "cylinder", "sphere"), default="box")
    parser.add_argument(
        "--dimensions",
        type=_dimensions,
        default=(0.05, 0.05, 0.05),
        help="full X,Y,Z dimensions in metres, e.g. 0.05,0.05,0.05",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--random", action="store_true", help="use random actions instead of the heuristic")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.episodes <= 0 or args.steps <= 0:
        raise ValueError("episodes and steps must be positive")

    config = UR5eLiftConfig(
        camera=CameraConfig(name=args.camera, width=args.width, height=args.height),
        object=PrimitiveObjectConfig(shape=args.shape, dimensions_m=args.dimensions),
        horizon=args.steps,
        seed=args.seed,
        has_renderer=args.render,
    )

    try:
        env = make_ur5e_lift(config)
    except RobosuiteUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    successes = 0
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            expert = HeuristicLiftExpert()
            total_reward = 0.0
            success = False

            for step in range(args.steps):
                action = (
                    rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
                    if args.random
                    else expert.act(env.raw_observation)
                )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                success = bool(info["success"])
                if args.render:
                    env.render()
                if terminated or truncated:
                    break

            successes += int(success)
            image_shape = observation["observation.images.front"].shape
            print(
                f"episode={episode} steps={step + 1} reward={total_reward:.3f} "
                f"success={success} image_shape={image_shape}"
            )
    except ContractError as exc:
        print(f"Simulation contract error: {exc}", file=sys.stderr)
        return 3
    finally:
        env.close()

    print(f"successes={successes}/{args.episodes}")
    return 0 if successes or args.random else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
