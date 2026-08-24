"""Build a single standard LeRobot dataset from the robust Push and PickPlace sets.

The two source datasets store images inside parquet files.  Keeping them as
separate online-replay sources causes Windows to materialize both image tables
at the same time.  This builder retains the Push parquet files as hard links
and rewrites only PickPlace parquet metadata (global frame, episode, and task
indices), producing one ordinary LeRobot v3 dataset for the established
single-dataset training path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PUSH_PROMPT = "push the block into the red target circle"
PICK_PLACE_PROMPT = "place the red cube in the blue storage bin"


def _parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    return files


def _read_all(directory: Path) -> pa.Table:
    return pa.concat_tables([pq.read_table(path) for path in _parquet_files(directory)])


def _replace_numeric_column(table: pa.Table, name: str, offset: int | None = None, value: int | None = None) -> pa.Table:
    if (offset is None) == (value is None):
        raise ValueError("Specify exactly one of offset or value")
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Missing required column {name!r}")
    old = table.column(index)
    if offset is not None:
        replacement = pc.add(old, pa.scalar(offset, old.type))
    else:
        replacement = pa.chunked_array([pa.array([value] * len(table), type=old.type)])
    return table.set_column(index, name, replacement)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _combine_numbers(left: Any, right: Any, left_count: Any, right_count: Any) -> Any:
    if isinstance(left, list):
        if not isinstance(left_count, list):
            left_count = [left_count] * len(left)
        elif len(left_count) == 1 and len(left) != 1:
            left_count = [left_count[0]] * len(left)
        if not isinstance(right_count, list):
            right_count = [right_count] * len(right)
        elif len(right_count) == 1 and len(right) != 1:
            right_count = [right_count[0]] * len(right)
        return [
            _combine_numbers(a, b, c, d)
            for a, b, c, d in zip(left, right, left_count, right_count, strict=True)
        ]
    total = left_count + right_count
    return (left * left_count + right * right_count) / total if total else 0.0


def _combine_std(left: Any, right: Any, left_mean: Any, right_mean: Any, left_count: Any, right_count: Any, mean: Any) -> Any:
    if isinstance(left, list):
        if not isinstance(left_count, list):
            left_count = [left_count] * len(left)
        elif len(left_count) == 1 and len(left) != 1:
            left_count = [left_count[0]] * len(left)
        if not isinstance(right_count, list):
            right_count = [right_count] * len(right)
        elif len(right_count) == 1 and len(right) != 1:
            right_count = [right_count[0]] * len(right)
        return [
            _combine_std(a, b, lm, rm, c, d, m)
            for a, b, lm, rm, c, d, m in zip(
                left, right, left_mean, right_mean, left_count, right_count, mean, strict=True
            )
        ]
    total = left_count + right_count
    if total <= 1:
        return 0.0
    variance = (
        max(left, 0.0) ** 2 * left_count
        + max(right, 0.0) ** 2 * right_count
        + (left_mean - mean) ** 2 * left_count
        + (right_mean - mean) ** 2 * right_count
    ) / total
    return variance**0.5


def _combine_min_max(left: Any, right: Any, function) -> Any:
    if isinstance(left, list):
        return [_combine_min_max(a, b, function) for a, b in zip(left, right, strict=True)]
    return function(left, right)


def _combine_stats(push: dict[str, Any], pick: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, left in push.items():
        right = pick[name]
        count = _combine_min_max(left["count"], right["count"], lambda a, b: a + b)
        mean = _combine_numbers(left["mean"], right["mean"], left["count"], right["count"])
        result = {"count": count, "mean": mean}
        result["std"] = _combine_std(
            left["std"], right["std"], left["mean"], right["mean"], left["count"], right["count"], mean
        )
        result["min"] = _combine_min_max(left["min"], right["min"], min)
        result["max"] = _combine_min_max(left["max"], right["max"], max)
        for quantile in ("q01", "q10", "q50", "q90", "q99"):
            result[quantile] = _combine_numbers(
                left[quantile], right[quantile], left["count"], right["count"]
            )
        output[name] = result
    return output


def build(push_root: Path, pick_root: Path, output_root: Path, *, resume: bool = False) -> None:
    if output_root.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    push_info = json.loads((push_root / "meta" / "info.json").read_text(encoding="utf-8"))
    pick_info = json.loads((pick_root / "meta" / "info.json").read_text(encoding="utf-8"))
    if push_info["features"] != pick_info["features"] or push_info["fps"] != pick_info["fps"]:
        raise ValueError("Push and PickPlace feature contracts must match")
    output_root.mkdir(parents=True, exist_ok=resume)
    push_frames = int(push_info["total_frames"])
    push_episodes = int(push_info["total_episodes"])

    # Hard-link untouched Push data to avoid another 14.7 GB copy.
    for source in _parquet_files(push_root / "data"):
        relative = source.relative_to(push_root / "data")
        destination = output_root / "data" / relative
        if not destination.exists():
            _link_or_copy(source, destination)

    # PickPlace is rewritten into a separate chunk with globally unique ids.
    pick_files = _parquet_files(pick_root / "data")
    for file_index, source in enumerate(pick_files):
        table = pq.read_table(source)
        table = _replace_numeric_column(table, "episode_index", offset=push_episodes)
        table = _replace_numeric_column(table, "index", offset=push_frames)
        table = _replace_numeric_column(table, "task_index", value=1)
        destination = output_root / "data" / "chunk-001" / f"file-{file_index:03d}.parquet"
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")

    push_episodes_table = _read_all(push_root / "meta" / "episodes")
    pick_episodes_table = _read_all(pick_root / "meta" / "episodes")
    pick_episodes_table = _replace_numeric_column(pick_episodes_table, "episode_index", offset=push_episodes)
    pick_episodes_table = _replace_numeric_column(
        pick_episodes_table, "dataset_from_index", offset=push_frames
    )
    pick_episodes_table = _replace_numeric_column(
        pick_episodes_table, "dataset_to_index", offset=push_frames
    )
    pick_episodes_table = _replace_numeric_column(pick_episodes_table, "data/chunk_index", value=1)
    pick_episodes_table = _replace_numeric_column(pick_episodes_table, "data/file_index", value=0)
    # File index maps each newly written file's contiguous episode block.
    offsets = []
    cursor = 0
    for source in pick_files:
        offsets.append(cursor)
        cursor += pq.ParquetFile(source).metadata.num_rows
    starts = pick_episodes_table["dataset_from_index"].to_pylist()
    starts = [start - push_frames for start in starts]
    file_ids = [max(index for index, offset in enumerate(offsets) if offset <= start) for start in starts]
    pick_episodes_table = pick_episodes_table.set_column(
        pick_episodes_table.schema.get_field_index("data/file_index"),
        "data/file_index",
        pa.array(file_ids, type=pa.int64()),
    )
    episode_output = output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables([push_episodes_table, pick_episodes_table]), episode_output)

    # Preserve the source parquet's pandas index metadata.  LeRobot uses that
    # index (rather than the task_index column) as the actual language prompt.
    push_tasks = pq.read_table(push_root / "meta" / "tasks.parquet")
    pick_tasks = pq.read_table(pick_root / "meta" / "tasks.parquet")
    pick_tasks = _replace_numeric_column(pick_tasks, "task_index", value=1)
    tasks = pa.concat_tables([push_tasks, pick_tasks])
    pq.write_table(tasks, output_root / "meta" / "tasks.parquet")
    with (push_root / "meta" / "stats.json").open(encoding="utf-8") as handle:
        push_stats = json.load(handle)
    with (pick_root / "meta" / "stats.json").open(encoding="utf-8") as handle:
        pick_stats = json.load(handle)
    (output_root / "meta" / "stats.json").write_text(
        json.dumps(_combine_stats(push_stats, pick_stats), indent=2), encoding="utf-8"
    )
    info = dict(push_info)
    info["total_episodes"] = push_episodes + int(pick_info["total_episodes"])
    info["total_frames"] = push_frames + int(pick_info["total_frames"])
    info["total_tasks"] = 2
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    (output_root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (output_root / "meta" / "build_provenance.json").write_text(
        json.dumps(
            {
                "push_root": str(push_root),
                "pick_place_root": str(pick_root),
                "push_prompt": PUSH_PROMPT,
                "pick_place_prompt": PICK_PLACE_PROMPT,
                "push_data_hard_linked": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push-root", type=Path, required=True)
    parser.add_argument("--pick-place-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Finish metadata for an interrupted build")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.push_root, args.pick_place_root, args.output_root, resume=args.resume)
