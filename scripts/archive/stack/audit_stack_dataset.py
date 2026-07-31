"""Archived audit for the canonical 3,000-episode Stack v1 dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".runtime" / "hf_datasets"))
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vla_sim.sampling import PHASE_GROUPS, phase_groups_from_indices  # noqa: E402
from vla_sim.scenes import load_manifest, oriented_rectangles_overlap  # noqa: E402


def phase_group(prompt: str) -> str:
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


def load_tokenizer(path: Path):
    """Load a tokenizer from either a Transformers directory or SmolVLA checkpoint."""

    config_path = path / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("type") == "smolvla" and config.get("vlm_model_name"):
            return AutoTokenizer.from_pretrained(
                config["vlm_model_name"], local_files_only=True
            )
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/ur5e_stack_v1")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--phase-conditioned",
        action="store_true",
        help="Require five task-phase prompts per episode instead of one task prompt.",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    if dataset.num_episodes != args.episodes:
        raise ValueError(f"Expected {args.episodes} episodes, found {dataset.num_episodes}")
    raw = dataset.hf_dataset.with_format(None)
    states = np.asarray(raw["observation.state"], dtype=np.float32)
    actions = np.asarray(raw["action"], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 10:
        raise ValueError(f"Expected 10-D Stack state, found {states.shape}")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError("Dataset contains NaN or infinity")

    tasks = dataset.meta.tasks
    prompts = {int(row["task_index"]): str(prompt) for prompt, row in tasks.iterrows()}
    tokenizer = load_tokenizer(args.tokenizer)
    for prompt in prompts.values():
        if len(tokenizer(prompt, add_special_tokens=True)["input_ids"]) > 16:
            raise ValueError(f"Prompt exceeds tokenizer_max_length=16: {prompt!r}")
    phases_by_episode: dict[int, set[str]] = defaultdict(set)
    tasks_by_episode: dict[int, set[int]] = defaultdict(set)
    if "phase_index" in raw.column_names:
        frame_phases = phase_groups_from_indices([int(value) for value in raw["phase_index"]])
    else:
        frame_phases = [
            phase_group(prompts[int(task_index)]) for task_index in raw["task_index"]
        ]
    for episode, task_index, group in zip(
        raw["episode_index"],
        raw["task_index"],
        frame_phases,
        strict=True,
    ):
        episode = int(episode)
        phases_by_episode[episode].add(group)
        tasks_by_episode[episode].add(int(task_index))
    if args.phase_conditioned:
        invalid_tasks = [
            episode
            for episode, values in tasks_by_episode.items()
            if values not in ({0, 1, 2, 3, 4}, {5, 6, 7, 8, 9})
        ]
        if invalid_tasks:
            raise ValueError(f"Invalid task-phase prompts in episodes: {invalid_tasks[:10]}")
    else:
        changing_tasks = [
            episode for episode, values in tasks_by_episode.items() if len(values) != 1
        ]
        if changing_tasks:
            raise ValueError(f"Task instruction changes within episodes: {changing_tasks[:10]}")
    incomplete = [
        episode for episode, phases in phases_by_episode.items() if phases != set(PHASE_GROUPS)
    ]
    if incomplete:
        raise ValueError(f"Episodes missing phase groups: {incomplete[:10]}")

    records = json.loads((args.root / "collection_scenes.json").read_text(encoding="utf-8"))
    accepted = [record for record in records if record["success"]]
    cells = Counter((record["task"], int(record["distance_bin"])) for record in accepted)
    if len(accepted) != args.episodes or set(cells.values()) != {args.episodes // 6}:
        raise ValueError(f"Task/distance cells are not exactly balanced: {dict(cells)}")
    scenes = {scene.scene_id: scene for scene in load_manifest(args.manifest)}
    for record in accepted:
        scene = scenes[record["scene_id"]]
        if oriented_rectangles_overlap(
            (scene.x_m, scene.y_m),
            scene.yaw_rad,
            (0.05, 0.05),
            (float(scene.overrides["cubeB_x_m"]), float(scene.overrides["cubeB_y_m"])),
            float(scene.overrides["cubeB_yaw_rad"]),
            (0.05, 0.05),
            clearance_m=0.005,
        ):
            raise ValueError(f"Initial object overlap in {scene.scene_id}")
    print(
        json.dumps(
            {
                "status": "passed",
                "episodes": dataset.num_episodes,
                "frames": len(dataset),
                "state_dim": states.shape[1],
                "cells": {f"{task}/distance_{distance}": count for (task, distance), count in cells.items()},
                "phase_groups": list(PHASE_GROUPS),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
