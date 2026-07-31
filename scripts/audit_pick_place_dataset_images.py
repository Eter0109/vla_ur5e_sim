"""Render representative dual-camera frames directly from a LeRobot dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from vla_sim.contracts import IMAGE_KEY, WRIST_IMAGE_KEY  # noqa: E402


def _image(value) -> Image.Image:
    array = value.detach().cpu().permute(1, 2, 0).numpy()
    if array.max() <= 1.0:
        array = array * 255.0
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 499, 999])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    width = height = 256
    header = 28
    canvas = Image.new("RGB", (3 * width, len(args.episodes) * (2 * height + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, episode_index in enumerate(args.episodes):
        episode = dataset.meta.episodes[int(episode_index)]
        start = int(episode["dataset_from_index"])
        length = int(episode["length"])
        indices = (start, start + length // 2, start + length - 1)
        y = row * (2 * height + header)
        for column, index in enumerate(indices):
            sample = dataset[index]
            x = column * width
            canvas.paste(_image(sample[IMAGE_KEY]), (x, y + header))
            canvas.paste(_image(sample[WRIST_IMAGE_KEY]), (x, y + header + height))
            draw.text((x + 5, y + 5), f"ep={episode_index} frame={int(sample['frame_index'])}", fill="black")
            draw.text((x + 5, y + header + 5), "front", fill="white")
            draw.text((x + 5, y + header + height + 5), "wrist", fill="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"dataset_image_audit_ok output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
