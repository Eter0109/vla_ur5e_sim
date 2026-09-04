"""Audit the three Sim2Real-v2 sources and optional canonical combined dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vla_sim.paths import load_catalog, project_root, resolve_asset

ROOT = project_root()

from vla_sim.simulation.provenance import scene_seed_overlap

EXPECTED_PROMPTS = {
    "push the block into the red target circle",
    "place the red cube in the blue storage bin",
    "pick up the red cube",
    "pick up the green cube",
    "pick up the blue cube",
}


def _read_data(root: Path) -> pa.Table:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"dataset has no parquet data: {root}")
    columns = ["index", "episode_index", "task_index", "observation.state", "action"]
    return pa.concat_tables([pq.read_table(path, columns=columns) for path in files])


def _manifest_scenes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, list) else payload["scenes"]


def _reference_manifests(task: str) -> list[Path]:
    benchmarks = load_catalog("simulation")["tasks"][task]["benchmarks"]
    return [resolve_asset(path) for path in benchmarks.values()]


def _source_report(task: str, root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    info = json.loads((root / "meta" / "info.json").read_text())
    provenance = json.loads((root / "meta" / "collection_provenance.json").read_text())
    manifest_path = root / "meta" / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    scenes = manifest["scenes"]
    primary_count = int(manifest["primary_candidates"])
    group_fields = tuple(manifest["group_fields"])
    desired = Counter(
        (
            scene["overrides"]["domain_randomization"]["tier"],
            *(str(scene["overrides"][field]) for field in group_fields),
        )
        for scene in scenes[:primary_count]
    )
    accepted = Counter(
        tuple(entry["cell"].split("|"))
        for entry in provenance["results"]
        if entry["success"]
    )
    if int(info["total_episodes"]) != 1_500:
        errors.append(f"{task}: episode count is not 1500")
    if desired != accepted:
        errors.append(f"{task}: accepted quotas do not match the primary distribution")
    if any(not entry["success"] for entry in provenance["results"] if entry["partition"] == "reserve"):
        errors.append(f"{task}: reserve contains a failed terminal attempt")
    if task == "color_pick" and any(
        entry.get("wrong_color_grasp") for entry in provenance["results"]
    ):
        errors.append("color_pick: wrong-color grasp was recorded")
    gate = provenance.get("first_attempt_gate") or {}
    if not gate.get("passed"):
        errors.append(f"{task}: first-attempt gate did not pass")
    table = _read_data(root)
    indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
    episodes = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if not np.array_equal(indices, np.arange(len(indices))):
        errors.append(f"{task}: frame indices are not contiguous")
    if episodes.min(initial=0) != 0 or episodes.max(initial=-1) != 1_499:
        errors.append(f"{task}: episode indices are not 0..1499")
    if states.shape[1:] != (10,) or not np.all(np.isfinite(states)):
        errors.append(f"{task}: invalid state vectors")
    if actions.shape[1:] != (7,) or not np.all(np.isfinite(actions)):
        errors.append(f"{task}: invalid action vectors")
    if actions.size and (float(actions.min()) < -1.0 or float(actions.max()) > 1.0):
        errors.append(f"{task}: action outside [-1, 1]")
    references = [
        scene
        for path in _reference_manifests(task)
        for scene in _manifest_scenes(path)
    ]
    overlap = scene_seed_overlap(scenes, references)
    if any(overlap.values()):
        errors.append(f"{task}: collection/evaluation seeds overlap")
    prompts = set(
        pq.read_table(root / "meta" / "tasks.parquet")["__index_level_0__"].to_pylist()
    )
    return (
        {
            "episodes": int(info["total_episodes"]),
            "frames": int(info["total_frames"]),
            "tier_counts": dict(Counter(cell[0] for cell in accepted.elements())),
            "accepted_cells": {"|".join(key): value for key, value in sorted(accepted.items())},
            "first_attempt_gate": gate,
            "prompts": sorted(prompts),
            "seed_overlap": overlap,
            "action_min": actions.min(axis=0).tolist(),
            "action_max": actions.max(axis=0).tolist(),
        },
        errors,
    )


def _combined_report(root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    info = json.loads((root / "meta" / "info.json").read_text())
    table = _read_data(root)
    prompts = set(
        pq.read_table(root / "meta" / "tasks.parquet")["__index_level_0__"].to_pylist()
    )
    indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
    episodes = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    if int(info["total_episodes"]) != 4_500 or int(info["total_tasks"]) != 5:
        errors.append("combined: expected 4500 episodes and 5 tasks")
    if prompts != EXPECTED_PROMPTS:
        errors.append("combined: prompt set does not match the five-task contract")
    if not np.array_equal(indices, np.arange(len(indices))):
        errors.append("combined: frame indices are not contiguous")
    if episodes.min(initial=0) != 0 or episodes.max(initial=-1) != 4_499:
        errors.append("combined: episode indices are not 0..4499")
    return {
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "tasks": int(info["total_tasks"]),
        "prompts": sorted(prompts),
    }, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push-root", type=Path, required=True)
    parser.add_argument("--pick-place-root", type=Path, required=True)
    parser.add_argument("--color-pick-root", type=Path, required=True)
    parser.add_argument("--combined-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {"schema_version": 2, "sources": {}}
    errors: list[str] = []
    for task, root in (
        ("push", args.push_root),
        ("pick_place", args.pick_place_root),
        ("color_pick", args.color_pick_root),
    ):
        source_report, source_errors = _source_report(task, root)
        report["sources"][task] = source_report
        errors.extend(source_errors)
    if args.combined_root:
        report["combined"], combined_errors = _combined_report(args.combined_root)
        errors.extend(combined_errors)
    report["errors"] = errors
    report["passed"] = not errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "errors": errors}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
