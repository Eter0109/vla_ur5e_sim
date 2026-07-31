"""Archived relabeler for the immutable Stack dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from vla_sim.sampling import PHASE_INDEX_TO_GROUP
from vla_sim.stack_control import task_phase_prompt


TASK_TO_INDEX = {"red_on_blue": 0, "blue_on_red": 1}
TASK_TO_INSTRUCTION = {
    "red_on_blue": "stack the red block on the blue block",
    "blue_on_red": "stack the blue block on the red block",
}
GROUP_TO_INDEX = {
    "approach": 0,
    "grasp": 1,
    "lift": 2,
    "transport": 3,
    "place_release": 4,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": [float(array.min())],
        "max": [float(array.max())],
        "mean": [float(array.mean())],
        "std": [float(array.std())],
        "count": [int(array.size)],
        "q01": [float(np.quantile(array, 0.01))],
        "q10": [float(np.quantile(array, 0.10))],
        "q50": [float(np.quantile(array, 0.50))],
        "q90": [float(np.quantile(array, 0.90))],
        "q99": [float(np.quantile(array, 0.99))],
    }


def _episode_tasks(scene_log: Path, expected_episodes: int) -> np.ndarray:
    records = json.loads(scene_log.read_text(encoding="utf-8"))
    tasks = np.full(expected_episodes, -1, dtype=np.int64)
    for record in records:
        episode = record.get("saved_episode_index")
        if episode is None:
            continue
        task = str(record["task"])
        if task not in TASK_TO_INDEX:
            raise ValueError(f"Unknown Stack task in scene log: {task}")
        episode = int(episode)
        if not 0 <= episode < expected_episodes or tasks[episode] != -1:
            raise ValueError(f"Invalid or duplicate saved_episode_index: {episode}")
        tasks[episode] = TASK_TO_INDEX[task]
    missing = np.flatnonzero(tasks < 0)
    if missing.size:
        raise ValueError(f"Scene log is missing {missing.size} saved episodes")
    return tasks


def _rewrite_parquet(
    source: Path,
    destination: Path,
    episode_tasks: np.ndarray,
    *,
    phase_conditioned: bool = False,
) -> tuple[int, list[int], list[int], dict[str, Any]]:
    table = pq.read_table(source)
    episode_indices = table.column("episode_index").to_numpy()
    if episode_indices.size and int(episode_indices.max()) >= len(episode_tasks):
        raise ValueError(f"Out-of-range episode index in {source}")
    old_task = table.column("task_index").to_numpy().astype(np.int64, copy=False)
    phases = (
        table.column("phase_index").to_numpy().astype(np.int64, copy=False)
        if "phase_index" in table.column_names
        else old_task
    )
    if phase_conditioned:
        group_indices = np.asarray(
            [GROUP_TO_INDEX[PHASE_INDEX_TO_GROUP[int(phase)]] for phase in phases],
            dtype=np.int64,
        )
        new_task = episode_tasks[episode_indices] * len(GROUP_TO_INDEX) + group_indices
    else:
        new_task = episode_tasks[episode_indices]
    task_column = table.schema.get_field_index("task_index")
    rewritten = table.set_column(task_column, "task_index", pa.array(new_task, type=pa.int64()))
    if "phase_index" not in rewritten.column_names:
        rewritten = rewritten.append_column("phase_index", pa.array(phases, type=pa.int64()))

    for name in ("observation.images.front", "observation.state", "action"):
        if not table.column(name).equals(rewritten.column(name)):
            raise RuntimeError(f"Content changed unexpectedly for {name} in {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(rewritten, destination, compression="zstd")
    record = {
        "path": source.name,
        "source_sha256": _sha256(source),
        "derived_sha256": _sha256(destination),
        "rows": len(table),
    }
    return len(table), phases.tolist(), new_task.tolist(), record


def _rewrite_episode_metadata(
    source: Path,
    destination: Path,
    episode_tasks: np.ndarray,
    *,
    phase_conditioned: bool = False,
) -> dict[str, Any]:
    table = pq.read_table(source)
    episodes = table.column("episode_index").to_numpy().astype(np.int64, copy=False)
    lengths = table.column("length").to_numpy().astype(np.int64, copy=False)
    task_indices = episode_tasks[episodes]
    if phase_conditioned:
        task_names = [
            [
                task_phase_prompt(
                    "red_on_blue" if index == 0 else "blue_on_red",
                    group,
                )
                for group in GROUP_TO_INDEX
            ]
            for index in task_indices
        ]
    else:
        task_names = [
            [TASK_TO_INSTRUCTION["red_on_blue" if index == 0 else "blue_on_red"]]
            for index in task_indices
        ]
    task_column = table.schema.get_field_index("tasks")
    rewritten = table.set_column(
        task_column,
        "tasks",
        pa.array(task_names, type=table.schema.field("tasks").type),
    )
    quantiles = ("q01", "q10", "q50", "q90", "q99")
    scalar_values: dict[str, list[list[float | int]]] = {
        "min": [[int(index)] for index in task_indices],
        "max": [[int(index)] for index in task_indices],
        "mean": [[float(index)] for index in task_indices],
        "std": [[0.0] for _ in task_indices],
        "count": [[int(length)] for length in lengths],
        **{
            quantile: [[float(index)] for index in task_indices]
            for quantile in quantiles
        },
    }
    for statistic, values in scalar_values.items():
        name = f"stats/task_index/{statistic}"
        column = rewritten.schema.get_field_index(name)
        if column >= 0:
            rewritten = rewritten.set_column(
                column,
                name,
                pa.array(values, type=rewritten.schema.field(name).type),
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(rewritten, destination, compression="zstd")
    return {
        "path": source.as_posix(),
        "source_sha256": _sha256(source),
        "derived_sha256": _sha256(destination),
        "rows": len(table),
    }


def _write_episode_metadata(
    source: Path,
    destination: Path,
    episode_tasks: np.ndarray,
    *,
    phase_conditioned: bool = False,
) -> list[dict[str, Any]]:
    records = []
    for source_file in sorted((source / "meta" / "episodes").rglob("*.parquet")):
        relative = source_file.relative_to(source)
        records.append(
            _rewrite_episode_metadata(
                source_file,
                destination / relative,
                episode_tasks,
                phase_conditioned=phase_conditioned,
            )
        )
    if not records:
        raise FileNotFoundError(f"No episode metadata found under {source / 'meta' / 'episodes'}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--repair-metadata-only",
        action="store_true",
        help="Add or replace episode metadata in an already transformed destination.",
    )
    parser.add_argument(
        "--phase-conditioned",
        action="store_true",
        help="Use eight task-by-phase-group prompts instead of two episode prompts.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if args.repair_metadata_only:
        if not destination.exists():
            raise FileNotFoundError(f"Destination does not exist: {destination}")
        info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
        episode_tasks = _episode_tasks(
            source / "collection_scenes.json",
            int(info["total_episodes"]),
        )
        records = _write_episode_metadata(
            source,
            destination,
            episode_tasks,
            phase_conditioned=args.phase_conditioned,
        )
        provenance_path = destination / "transformation_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["episode_metadata_files"] = records
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"repaired={destination} episode_metadata_files={len(records)}")
        return 0
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    episode_tasks = _episode_tasks(
        source / "collection_scenes.json",
        int(info["total_episodes"]),
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    file_records: list[dict[str, Any]] = []
    phase_values: list[int] = []
    task_values: list[int] = []
    total_rows = 0
    try:
        for source_file in sorted((source / "data").rglob("*.parquet")):
            relative = source_file.relative_to(source)
            rows, phases, tasks, record = _rewrite_parquet(
                source_file,
                temporary / relative,
                episode_tasks,
                phase_conditioned=args.phase_conditioned,
            )
            record["path"] = relative.as_posix()
            file_records.append(record)
            phase_values.extend(phases)
            task_values.extend(tasks)
            total_rows += rows

        if total_rows != int(info["total_frames"]):
            raise ValueError(f"Expected {info['total_frames']} frames, transformed {total_rows}")

        (temporary / "meta").mkdir(parents=True, exist_ok=True)
        derived_info = dict(info)
        derived_info["total_tasks"] = 10 if args.phase_conditioned else 2
        derived_info["features"] = dict(info["features"])
        derived_info["features"]["phase_index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        }
        (temporary / "meta" / "info.json").write_text(
            json.dumps(derived_info, indent=4),
            encoding="utf-8",
        )

        if args.phase_conditioned:
            instructions = [
                task_phase_prompt(task, group)
                for task in TASK_TO_INDEX
                for group in GROUP_TO_INDEX
            ]
        else:
            instructions = [
                TASK_TO_INSTRUCTION["red_on_blue"],
                TASK_TO_INSTRUCTION["blue_on_red"],
            ]
        task_frame = pd.DataFrame(
            {"task_index": list(range(len(instructions)))},
            index=instructions,
        )
        task_frame.to_parquet(temporary / "meta" / "tasks.parquet")

        stats = json.loads((source / "meta" / "stats.json").read_text(encoding="utf-8"))
        stats["task_index"] = _scalar_stats(np.asarray(task_values))
        stats["phase_index"] = _scalar_stats(np.asarray(phase_values))
        (temporary / "meta" / "stats.json").write_text(
            json.dumps(stats, indent=4),
            encoding="utf-8",
        )
        episode_metadata_records = _write_episode_metadata(
            source,
            temporary,
            episode_tasks,
            phase_conditioned=args.phase_conditioned,
        )

        for name in ("collection_scenes.json", "collection_provenance.json"):
            shutil.copy2(source / name, temporary / name)
        provenance = {
            "schema_version": 1,
            "transform": (
                "stack_task_phase_conditioning_v3"
                if args.phase_conditioned
                else "stack_task_conditioning_v1"
            ),
            "source": str(source),
            "source_collection_provenance_sha256": _sha256(
                source / "collection_provenance.json"
            ),
            "instructions": instructions,
            "episodes": int(info["total_episodes"]),
            "frames": total_rows,
            "task_episode_counts": {
                task: int((episode_tasks == index).sum())
                for task, index in TASK_TO_INDEX.items()
            },
            "phase_frame_counts": {
                str(index): int((np.asarray(phase_values) == index).sum())
                for index in range(6)
            },
            "files": file_records,
            "episode_metadata_files": episode_metadata_records,
        }
        (temporary / "transformation_provenance.json").write_text(
            json.dumps(provenance, indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"created={destination} episodes={len(episode_tasks)} frames={total_rows} "
        f"red_on_blue={(episode_tasks == 0).sum()} blue_on_red={(episode_tasks == 1).sum()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
