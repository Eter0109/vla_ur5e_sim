"""Archived cleaner for low-velocity redundant Stack approach frames."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(tempfile.gettempdir()) / "vla_sim_hf_datasets"))
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def _phase_group(prompt: str) -> str:
    prompt = prompt.lower()
    if "above the grasp" in prompt:
        return "approach"
    if "grasp object" in prompt or "down and grasp" in prompt:
        return "grasp"
    if "lift the grasped" in prompt or prompt.startswith("lift "):
        return "lift"
    if "above target" in prompt:
        return "transport"
    if "onto target" in prompt or "for release" in prompt:
        return "place_release"
    raise ValueError(f"Unknown phase prompt: {prompt!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True, help="Path to input LeRobot dataset")
    parser.add_argument("--src-repo-id", default="local/ur5e_stack_v1")
    parser.add_argument("--min-speed-m", type=float, default=0.001, help="Min EEF movement (meters) per frame in approach phase")
    args = parser.parse_args()

    dataset = LeRobotDataset(args.src_repo_id, root=args.src_root)
    raw = dataset.hf_dataset.with_format(None)
    states = np.asarray(raw["observation.state"], dtype=np.float32)
    episodes = np.asarray(raw["episode_index"], dtype=np.int64)
    task_indices = np.asarray(raw["task_index"], dtype=np.int64)

    tasks = dataset.meta.tasks
    prompts = {int(row["task_index"]): str(prompt) for prompt, row in tasks.iterrows()}

    keep_indices = []
    trimmed_count = 0
    total_frames = len(states)

    num_episodes = dataset.num_episodes
    for ep in range(num_episodes):
        ep_mask = episodes == ep
        ep_indices = np.where(ep_mask)[0]
        if len(ep_indices) == 0:
            continue

        ep_states = states[ep_indices]
        ep_tasks = task_indices[ep_indices]

        # EEF positions (indices 6, 7, 8 in 10-D state)
        eef_xyz = ep_states[:, 6:9]
        deltas = np.linalg.norm(np.diff(eef_xyz, axis=0, prepend=eef_xyz[:1]), axis=1)

        for i, idx in enumerate(ep_indices):
            phase = _phase_group(prompts[int(ep_tasks[i])])
            if phase == "approach" and deltas[i] < args.min_speed_m:
                trimmed_count += 1
            else:
                keep_indices.append(idx)

    print(
        f"Trajectory Cleaning Audit:\n"
        f"  Total Frames: {total_frames}\n"
        f"  Trimmed Idle Approach Frames (< {args.min_speed_m * 1000:.1f} mm/step): {trimmed_count} ({trimmed_count / total_frames:.1%})\n"
        f"  Retained Frames: {len(keep_indices)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
