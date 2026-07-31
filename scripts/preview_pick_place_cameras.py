"""Render a dual-camera preview from a fresh PickPlace expert rollout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.contracts import IMAGE_KEY, WRIST_IMAGE_KEY  # noqa: E402
from vla_sim.envs import UR5ePickPlaceConfig, make_ur5e_pick_place  # noqa: E402
from vla_sim.envs.ur5e_lift import CameraConfig  # noqa: E402
from vla_sim.envs.ur5e_pick_place import PickPlaceCameraConfig  # noqa: E402
from vla_sim.scenes import load_manifest  # noqa: E402
from vla_sim.sim import HeuristicPickPlaceExpert  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--third-person-camera", default="birdview")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "pick_place_v2_native_bin" / "camera_preview.png",
    )
    args = parser.parse_args()
    scene = load_manifest(args.manifest)[args.scene]
    camera = PickPlaceCameraConfig(third_person=CameraConfig(name=args.third_person_camera))
    env = make_ur5e_pick_place(UR5ePickPlaceConfig(camera=camera, horizon=250))
    observations: list[dict] = []
    try:
        observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
        third_camera_id = env.backend.sim.model.camera_name2id(args.third_person_camera)
        print(
            "third_camera="
            + np.array2string(env.backend.sim.data.cam_xpos[third_camera_id], precision=3)
            + " quat="
            + np.array2string(env.backend.sim.data.cam_xmat[third_camera_id].reshape(3, 3), precision=3),
            flush=True,
        )
        expert = HeuristicPickPlaceExpert()
        for _ in range(env.config.horizon):
            observations.append(observation)
            observation, _, terminated, truncated, _ = env.step(expert.act(env.raw_observation))
            if terminated or truncated:
                observations.append(observation)
                break
    finally:
        env.close()
    indices = np.linspace(0, len(observations) - 1, args.frames, dtype=int)
    height, width = observations[0][IMAGE_KEY].shape[:2]
    header = 24
    canvas = Image.new("RGB", (args.frames * width, 2 * height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for column, index in enumerate(indices):
        sample = observations[int(index)]
        canvas.paste(Image.fromarray(sample[IMAGE_KEY]), (column * width, header))
        canvas.paste(Image.fromarray(sample[WRIST_IMAGE_KEY]), (column * width, header + height))
        draw.text((column * width + 4, 4), f"t={int(index):02d}", fill="black")
    draw.text((4, header + 4), "third-person", fill="white")
    draw.text((4, header + height + 4), "wrist", fill="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"camera_preview_ok frames={len(observations)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
