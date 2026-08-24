"""Fail fast unless Push and PickPlace datasets can share one SmolVLA policy."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.contracts import ACTION_KEY, IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY  # noqa: E402

EXPECTED_PROMPTS = {
    "push": "push the block into the red target circle",
    "pick_place": "place the red cube in the blue storage bin",
}
REQUIRED_FEATURES = {IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY, ACTION_KEY}


@dataclass(frozen=True)
class DatasetMetadata:
    features: dict[str, object]
    fps: int
    episodes: int
    frames: int
    prompts: set[str]


def _load(root: Path, repo_id: str) -> DatasetMetadata:
    del repo_id  # Repo IDs are training-time identifiers; local metadata is authoritative here.
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    if not info_path.exists() or not tasks_path.exists():
        raise FileNotFoundError(f"Dataset metadata is incomplete under {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    missing = REQUIRED_FEATURES - set(features)
    if missing:
        raise ValueError(f"{root} is missing required features: {sorted(missing)}")
    if tuple(features[STATE_KEY]["shape"]) != (10,):
        raise ValueError(f"{root} has an incompatible state shape")
    if tuple(features[ACTION_KEY]["shape"]) != (7,):
        raise ValueError(f"{root} has an incompatible action shape")
    import pyarrow.parquet as pq

    prompts = {str(prompt) for prompt in pq.read_table(tasks_path).column("__index_level_0__").to_pylist()}
    return DatasetMetadata(
        features=features,
        fps=int(info["fps"]),
        episodes=int(info["total_episodes"]),
        frames=int(info["total_frames"]),
        prompts=prompts,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push-root", type=Path, required=True)
    parser.add_argument("--push-repo-id", required=True)
    parser.add_argument("--pick-place-root", type=Path, required=True)
    parser.add_argument("--pick-place-repo-id", required=True)
    args = parser.parse_args()
    push = _load(args.push_root, args.push_repo_id)
    pick_place = _load(args.pick_place_root, args.pick_place_repo_id)
    if push.features != pick_place.features:
        raise ValueError("Push and PickPlace features differ; refusing mixed training")
    if push.fps != pick_place.fps:
        raise ValueError("Push and PickPlace FPS differ; refusing mixed training")
    actual = {"push": push.prompts, "pick_place": pick_place.prompts}
    for task, expected in EXPECTED_PROMPTS.items():
        if actual[task] != {expected}:
            raise ValueError(f"{task} prompts must be exactly {{{expected!r}}}, got {sorted(actual[task])}")
    report = {
        "status": "ok",
        "fps": push.fps,
        "features": sorted(push.features),
        "push": {"episodes": push.episodes, "frames": push.frames},
        "pick_place": {"episodes": pick_place.episodes, "frames": pick_place.frames},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
