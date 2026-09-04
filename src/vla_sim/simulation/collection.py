"""Collect one resumable, quota-balanced Sim2Real-v2 LeRobot source dataset."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vla_sim.paths import project_root

ROOT = project_root()
RUNTIME = ROOT / ".runtime"
os.environ.update(
    {
        "HF_HOME": str(RUNTIME / "hf"),
        "HF_DATASETS_CACHE": str(RUNTIME / "hf_datasets"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "USE_TF": "0",
        "NUMBA_CACHE_DIR": str(Path(tempfile.gettempdir()) / "vla_sim_numba"),
        "NUMBA_DISABLE_JIT": "1" if os.name == "nt" else "0",
    }
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla_sim.simulation.color_pick_contract import (
    color_pick_prompt,
    write_color_pick_contract,
)
from vla_sim.simulation.contracts import (
    ACTION_KEY,
    IMAGE_KEY,
    STATE_KEY,
    TASK_KEY,
    WRIST_IMAGE_KEY,
)
from vla_sim.simulation.experts import (
    HeuristicColorPickExpert,
    HeuristicPickPlaceExpert,
    HeuristicPushExpert,
)
from vla_sim.simulation.pick_place_contract import (
    PICK_PLACE_FPS,
    PICK_PLACE_GLOBAL_PROMPT,
    write_pick_place_contract,
)
from vla_sim.simulation.pick_place_control import filter_vla_only_action
from vla_sim.simulation.scenes import load_manifest, load_manifest_metadata
from vla_sim.simulation.tasks import (
    UR5eColorPickConfig,
    UR5ePickPlaceConfig,
    UR5ePushConfig,
    make_ur5e_color_pick,
    make_ur5e_pick_place,
    make_ur5e_push,
)

PUSH_PROMPT = "push the block into the red target circle"


def _features() -> dict[str, Any]:
    image = {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        IMAGE_KEY: image,
        WRIST_IMAGE_KEY: image.copy(),
        STATE_KEY: {
            "dtype": "float32",
            "shape": (10,),
            "names": [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "eef_x",
                "eef_y",
                "eef_z",
                "gripper",
            ],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"],
        },
    }


def _cell(scene: Any, group_fields: tuple[str, ...]) -> tuple[str, ...]:
    tier = str(scene.overrides["domain_randomization"]["tier"])
    return (tier, *(str(scene.overrides[field]) for field in group_fields))


def _cell_name(cell: tuple[str, ...]) -> str:
    return "|".join(cell)


def _write_progress(path: Path, progress: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    temporary.replace(path)


def _recover_pending(dataset: Any, progress: dict[str, Any]) -> None:
    pending = progress.get("pending")
    committed = sum(bool(entry["success"]) for entry in progress["results"])
    if pending is None:
        if dataset.num_episodes != committed:
            raise RuntimeError("dataset/progress episode counts disagree")
        return
    if dataset.num_episodes == committed:
        progress["next_source_index"] = int(pending["source_index"])
    elif dataset.num_episodes == committed + 1:
        progress["results"].append(pending)
        progress["next_source_index"] = int(pending["source_index"]) + 1
    else:
        raise RuntimeError("cannot recover interrupted episode commit")
    progress["pending"] = None


def _make_task(task: str, horizon: int):
    if task == "push":
        return make_ur5e_push(UR5ePushConfig(horizon=horizon))
    if task == "pick_place":
        return make_ur5e_pick_place(UR5ePickPlaceConfig(horizon=horizon))
    return make_ur5e_color_pick(UR5eColorPickConfig(horizon=horizon))


def _expert_action(task: str, env: Any, expert: Any) -> np.ndarray:
    action = np.asarray(expert.act(env.raw_observation), dtype=np.float32)
    if task == "push":
        return filter_vla_only_action(
            action,
            eef_xyz=np.asarray(env.raw_observation["robot0_eef_pos"], dtype=float),
        )
    return action


def _expert_and_prompt(task: str, scene: Any) -> tuple[Any, str]:
    if task == "push":
        return HeuristicPushExpert(), PUSH_PROMPT
    if task == "pick_place":
        return HeuristicPickPlaceExpert(), PICK_PLACE_GLOBAL_PROMPT
    target_color = str(scene.overrides["target_color"])
    return HeuristicColorPickExpert(target_color), color_pick_prompt(target_color)


def _first_attempt_gate(
    results: list[dict[str, Any]],
    *,
    primary_count: int,
) -> dict[str, Any]:
    primary = [entry for entry in results if entry["source_index"] < primary_count]
    if len(primary) != primary_count:
        raise RuntimeError("primary first-attempt gate ran before every primary scene")
    groups: dict[str, list[bool]] = defaultdict(list)
    for entry in primary:
        groups[entry["cell"]].append(bool(entry["first_attempt_success"]))
    overall = sum(bool(entry["first_attempt_success"]) for entry in primary) / len(primary)
    rates = {key: sum(values) / len(values) for key, values in sorted(groups.items())}
    passed = overall >= 0.95 and all(rate >= 0.90 for rate in rates.values())
    return {"passed": passed, "overall": overall, "cells": rates}


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("push", "pick_place", "color_pick"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-env-every", type=int, default=20)
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stop cleanly after this many newly accepted episodes, leaving a resumable dataset.",
    )
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")

    metadata = load_manifest_metadata(args.manifest)
    expected_preset = f"{args.task}_sim2real_v2"
    if metadata.get("role") != "collection" or metadata.get("environment_preset") != expected_preset:
        parser.error(f"manifest must be a collection manifest for {expected_preset}")
    primary_count = int(metadata["primary_candidates"])
    target_count = int(metadata["accepted_episode_target"])
    group_fields = tuple(str(value) for value in metadata["group_fields"])
    scenes = load_manifest(args.manifest)
    desired = Counter(_cell(scene, group_fields) for scene in scenes[:primary_count])
    progress_path = args.root / "collection_progress.json"
    if args.root.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite dataset root: {args.root}")
    if args.resume:
        dataset = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        _recover_pending(dataset, progress)
        _write_progress(progress_path, progress)
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=PICK_PLACE_FPS,
            root=args.root,
            robot_type="UR5e",
            features=_features(),
            use_videos=False,
            image_writer_threads=4,
        )
        progress = {
            "schema_version": 2,
            "task": args.task,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "next_source_index": 0,
            "pending": None,
            "gate": None,
            "results": [],
        }
        _write_progress(progress_path, progress)

    accepted = Counter(
        tuple(entry["cell"].split("|"))
        for entry in progress["results"]
        if bool(entry["success"])
    )
    accepted_at_start = sum(accepted.values())
    stopped_early = False
    horizons = {"push": 250, "pick_place": 250, "color_pick": 200}
    env = _make_task(args.task, horizons[args.task])
    try:
        for source_index in range(int(progress["next_source_index"]), len(scenes)):
            scene = scenes[source_index]
            cell = _cell(scene, group_fields)
            if accepted[cell] >= desired[cell]:
                progress["next_source_index"] = source_index + 1
                _write_progress(progress_path, progress)
                continue
            success = False
            wrong_grasp = False
            attempt = 0
            first_attempt_success = False
            frames = 0
            for attempt in (1, 2):
                observation, _ = env.reset(seed=scene.effective_env_seed, scene=scene)
                expert, prompt = _expert_and_prompt(args.task, scene)
                info: dict[str, Any] = {}
                for step in range(env.config.horizon):
                    requested_action = _expert_action(args.task, env, expert)
                    dataset.add_frame(
                        {
                            IMAGE_KEY: observation[IMAGE_KEY],
                            WRIST_IMAGE_KEY: observation[WRIST_IMAGE_KEY],
                            STATE_KEY: observation[STATE_KEY],
                            ACTION_KEY: requested_action,
                            TASK_KEY: prompt,
                        }
                    )
                    observation, _, terminated, truncated, info = env.step(requested_action)
                    if terminated or truncated:
                        break
                frames = step + 1
                wrong_grasp = bool(info.get("ever_wrong_object_grasped", False))
                success = bool(info.get("success", False)) and not wrong_grasp
                if attempt == 1:
                    first_attempt_success = success
                if success:
                    break
                dataset.clear_episode_buffer()
                env.close()
                env = _make_task(args.task, horizons[args.task])
            entry = {
                "scene_id": scene.scene_id,
                "source_index": source_index,
                "partition": scene.overrides["candidate_partition"],
                "cell": _cell_name(cell),
                "tier": cell[0],
                "success": success,
                "first_attempt_success": first_attempt_success,
                "attempts": attempt,
                "frames": frames,
                "wrong_color_grasp": wrong_grasp,
                "randomization": scene.overrides["domain_randomization"],
            }
            if success:
                progress["pending"] = entry
                _write_progress(progress_path, progress)
                dataset.save_episode(parallel_encoding=True)
                progress["results"].append(entry)
                progress["pending"] = None
                accepted[cell] += 1
            else:
                dataset.clear_episode_buffer()
                progress["results"].append(entry)
            progress["next_source_index"] = source_index + 1
            if source_index + 1 == primary_count:
                progress["gate"] = _first_attempt_gate(
                    progress["results"], primary_count=primary_count
                )
            _write_progress(progress_path, progress)
            if source_index + 1 == primary_count and not progress["gate"]["passed"]:
                raise RuntimeError(f"first-attempt collection gate failed: {progress['gate']}")
            if sum(accepted.values()) % args.reset_env_every == 0:
                env.close()
                del env
                gc.collect()
                env = _make_task(args.task, horizons[args.task])
            if sum(accepted.values()) % 100 == 0 and success:
                print(
                    f"accepted={sum(accepted.values())}/{target_count} "
                    f"source={source_index} task={args.task}",
                    flush=True,
                )
            if sum(accepted.values()) == target_count:
                break
            if (
                args.stop_after is not None
                and sum(accepted.values()) - accepted_at_start >= args.stop_after
            ):
                stopped_early = True
                break
    finally:
        env.close()
    if stopped_early:
        print(
            f"collection_paused task={args.task} accepted={sum(accepted.values())} "
            f"resume_with=--resume",
            flush=True,
        )
        return 75
    if accepted != desired:
        missing = {str(key): desired[key] - accepted[key] for key in desired if accepted[key] < desired[key]}
        raise RuntimeError(f"collection exhausted before exact quotas were filled: {missing}")

    dataset.finalize()
    if args.task == "pick_place":
        write_pick_place_contract(args.root, UR5ePickPlaceConfig(horizon=250))
    elif args.task == "color_pick":
        write_color_pick_contract(args.root, UR5eColorPickConfig(horizon=200))
    meta = args.root / "meta"
    shutil.copyfile(args.manifest, meta / "collection_manifest.json")
    provenance = {
        "schema_version": 2,
        "randomization_schema_version": 2,
        "task": args.task,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "git_commit": _git_commit(),
        "collection_command": [sys.executable, *sys.argv],
        "accepted_episodes": target_count,
        "accepted_cells": {_cell_name(key): value for key, value in sorted(accepted.items())},
        "first_attempt_gate": progress["gate"],
        "action_label": "expert_requested_before_execution_gain_or_delay",
        "policy_observation": "sim2real_v2_perturbed",
        "failure_statistics": {
            "failed_candidates": sum(not bool(entry["success"]) for entry in progress["results"]),
            "retried_candidates": sum(int(entry["attempts"]) > 1 for entry in progress["results"]),
            "wrong_color_grasps": sum(
                bool(entry["wrong_color_grasp"]) for entry in progress["results"]
            ),
        },
        "results": progress["results"],
    }
    (meta / "collection_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    (meta / "sim2real_v2_environment.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "fps": PICK_PLACE_FPS,
                "randomization_source": "collection_manifest.json per-scene overrides",
                "policy_observation_keys": [IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY],
                "action_shape": [7],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.root / "collection.complete").write_text(
        f"episodes={target_count} randomization_schema_version=2\n",
        encoding="utf-8",
    )
    print(f"dataset_ok task={args.task} root={args.root} episodes={target_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
