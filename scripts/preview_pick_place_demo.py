"""Render a contact sheet for one dual-camera PickPlace demonstration."""

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


def to_image(value: object) -> Image.Image:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating) and array.max(initial=0.0) <= 1.0:
        array = array * 255.0
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/ur5e_pick_place_v2_native_bin")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "pick_place_v2_native_bin" / "demo_preview.png",
    )
    args = parser.parse_args()
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    if not 0 <= args.episode < dataset.num_episodes:
        parser.error(f"episode must be in [0, {dataset.num_episodes - 1}]")
    episode = dataset.meta.episodes[args.episode]
    start = int(episode["dataset_from_index"])
    stop = int(episode["dataset_to_index"])
    indices = np.linspace(start, stop - 1, args.frames, dtype=int)
    first = dataset[int(indices[0])]
    tile = to_image(first[IMAGE_KEY])
    width, height = tile.size
    header = 24
    canvas = Image.new("RGB", (args.frames * width, 2 * height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for column, index in enumerate(indices):
        sample = dataset[int(index)]
        canvas.paste(to_image(sample[IMAGE_KEY]), (column * width, header))
        canvas.paste(to_image(sample[WRIST_IMAGE_KEY]), (column * width, header + height))
        draw.text((column * width + 4, 4), f"t={int(index - start):02d}", fill="black")
    draw.text((4, header + 4), "third-person", fill="white")
    draw.text((4, header + height + 4), "wrist", fill="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"preview_ok episode={args.episode} frames={args.frames} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
