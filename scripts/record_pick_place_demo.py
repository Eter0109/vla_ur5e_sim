"""Record one successful dual-RGB PickPlace expert demonstration as MP4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.contracts import IMAGE_KEY, WRIST_IMAGE_KEY  # noqa: E402
from vla_sim.envs import UR5ePickPlaceConfig, make_ur5e_pick_place  # noqa: E402
from vla_sim.scenes import load_manifest  # noqa: E402
from vla_sim.sim import HeuristicPickPlaceExpert  # noqa: E402


def _frame(observation: dict[str, np.ndarray], step: int) -> np.ndarray:
    third_person = observation[IMAGE_KEY].copy()
    wrist = observation[WRIST_IMAGE_KEY].copy()
    for image, label in ((third_person, "third-person (front 45 deg)"), (wrist, "wrist RGB")):
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    cv2.putText(third_person, f"step {step}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1)
    return np.concatenate((third_person, wrist), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "pick_place_v2_native_bin" / "pick_place_demo.mp4",
    )
    args = parser.parse_args()
    scene = load_manifest(args.manifest)[args.scene]
    env = make_ur5e_pick_place(UR5ePickPlaceConfig())
    writer: cv2.VideoWriter | None = None
    success = False
    try:
        observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
        frame = _frame(observation, 0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), env.config.control_frequency_hz,
            (frame.shape[1], frame.shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open MP4 writer for {args.output}")
        expert = HeuristicPickPlaceExpert()
        for step in range(1, env.config.horizon + 1):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            observation, _, terminated, truncated, info = env.step(expert.act(env.raw_observation))
            frame = _frame(observation, step)
            success = bool(info["success"])
            if terminated or truncated:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                break
    finally:
        if writer is not None:
            writer.release()
        env.close()
    print(f"video_ok scene={scene.scene_id} success={success} output={args.output}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
