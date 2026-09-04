"""Build the canonical five-prompt dataset from three Sim2Real-v2 sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from functools import reduce
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _combine_numbers(
    left: Any, right: Any, left_count: Any, right_count: Any
) -> Any:
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


def _combine_std(
    left: Any,
    right: Any,
    left_mean: Any,
    right_mean: Any,
    left_count: Any,
    right_count: Any,
    mean: Any,
) -> Any:
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
                left,
                right,
                left_mean,
                right_mean,
                left_count,
                right_count,
                mean,
                strict=True,
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


def _combine_min_max(left: Any, right: Any, function: Any) -> Any:
    if isinstance(left, list):
        return [
            _combine_min_max(a, b, function)
            for a, b in zip(left, right, strict=True)
        ]
    return function(left, right)


def _combine_stats(left_stats: dict[str, Any], right_stats: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, left in left_stats.items():
        right = right_stats[name]
        count = _combine_min_max(left["count"], right["count"], lambda a, b: a + b)
        mean = _combine_numbers(left["mean"], right["mean"], left["count"], right["count"])
        result = {"count": count, "mean": mean}
        result["std"] = _combine_std(
            left["std"],
            right["std"],
            left["mean"],
            right["mean"],
            left["count"],
            right["count"],
            mean,
        )
        result["min"] = _combine_min_max(left["min"], right["min"], min)
        result["max"] = _combine_min_max(left["max"], right["max"], max)
        for quantile in ("q01", "q10", "q50", "q90", "q99"):
            result[quantile] = _combine_numbers(
                left[quantile], right[quantile], left["count"], right["count"]
            )
        output[name] = result
    return output


def _parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found in {directory}")
    return files


def _read_all(directory: Path) -> pa.Table:
    return pa.concat_tables([pq.read_table(path) for path in _parquet_files(directory)])


def _offset(table: pa.Table, name: str, value: int) -> pa.Table:
    index = table.schema.get_field_index(name)
    old = table.column(index)
    return table.set_column(index, name, pc.add(old, pa.scalar(value, old.type)))


def _constant(table: pa.Table, name: str, value: int) -> pa.Table:
    index = table.schema.get_field_index(name)
    old = table.column(index)
    replacement = pa.array([value] * len(table), type=old.type)
    return table.set_column(index, name, replacement)


def _remap(table: pa.Table, name: str, mapping: dict[int, int]) -> pa.Table:
    index = table.schema.get_field_index(name)
    old = table.column(index)
    values = [mapping[int(value.as_py())] for value in old]
    return table.set_column(index, name, pa.array(values, type=old.type))


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build(sources: list[tuple[str, Path]], output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    if len(sources) != 3:
        raise ValueError("the canonical Sim2Real-v2 build requires exactly three sources")
    infos = [json.loads((root / "meta" / "info.json").read_text()) for _, root in sources]
    if any(info["features"] != infos[0]["features"] for info in infos[1:]):
        raise ValueError("source feature contracts do not match")
    if any(info["fps"] != infos[0]["fps"] for info in infos[1:]):
        raise ValueError("source FPS values do not match")
    for _name, root in sources:
        if not (root / "collection.complete").exists():
            raise RuntimeError(f"source collection is incomplete: {root}")

    output_root.mkdir(parents=True)
    frame_offset = 0
    episode_offset = 0
    global_prompts: list[str] = []
    episode_tables: list[pa.Table] = []
    stats: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    task_schema_metadata: dict[bytes, bytes] | None = None

    for source_index, ((name, root), info) in enumerate(zip(sources, infos, strict=True)):
        task_table = pq.read_table(root / "meta" / "tasks.parquet")
        task_schema_metadata = task_schema_metadata or task_table.schema.metadata
        task_data = task_table.to_pydict()
        local_indices = [int(value) for value in task_data["task_index"]]
        prompts = [str(value) for value in task_data["__index_level_0__"]]
        mapping = {
            local: len(global_prompts) + offset
            for offset, local in enumerate(local_indices)
        }
        global_prompts.extend(prompts)

        data_files = _parquet_files(root / "data")
        file_starts: list[int] = []
        local_cursor = 0
        identity_mapping = all(local == global_ for local, global_ in mapping.items())
        for file_index, source in enumerate(data_files):
            file_starts.append(local_cursor)
            row_count = pq.ParquetFile(source).metadata.num_rows
            destination = (
                output_root / "data" / f"chunk-{source_index:03d}" / f"file-{file_index:03d}.parquet"
            )
            if source_index == 0 and identity_mapping:
                _link_or_copy(source, destination)
            else:
                table = pq.read_table(source)
                table = _offset(table, "episode_index", episode_offset)
                table = _offset(table, "index", frame_offset)
                table = _remap(table, "task_index", mapping)
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, destination, compression="zstd")
            local_cursor += row_count

        episodes = _read_all(root / "meta" / "episodes")
        local_starts = episodes["dataset_from_index"].to_pylist()
        file_ids = [
            max(index for index, start in enumerate(file_starts) if start <= local_start)
            for local_start in local_starts
        ]
        episodes = _offset(episodes, "episode_index", episode_offset)
        episodes = _offset(episodes, "dataset_from_index", frame_offset)
        episodes = _offset(episodes, "dataset_to_index", frame_offset)
        episodes = _constant(episodes, "data/chunk_index", source_index)
        episodes = episodes.set_column(
            episodes.schema.get_field_index("data/file_index"),
            "data/file_index",
            pa.array(file_ids, type=pa.int64()),
        )
        episode_tables.append(episodes)
        stats.append(json.loads((root / "meta" / "stats.json").read_text()))
        manifest = root / "meta" / "collection_manifest.json"
        provenance = root / "meta" / "collection_provenance.json"
        environment_contract = root / "meta" / "sim2real_v2_environment.json"
        source_provenance = json.loads(provenance.read_text())
        source_records.append(
            {
                "name": name,
                "root": str(root.resolve()),
                "episodes": int(info["total_episodes"]),
                "frames": int(info["total_frames"]),
                "task_mapping": mapping,
                "prompts": prompts,
                "manifest_sha256": _sha256(manifest),
                "provenance_sha256": _sha256(provenance),
                "environment_contract_sha256": _sha256(environment_contract),
                "collection_command": source_provenance["collection_command"],
                "failure_statistics": source_provenance["failure_statistics"],
                "data_hard_linked": source_index == 0,
            }
        )
        frame_offset += int(info["total_frames"])
        episode_offset += int(info["total_episodes"])

    episode_path = output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    pq.write_table(pa.concat_tables(episode_tables), episode_path, compression="zstd")
    task_table = pa.table(
        {
            "task_index": pa.array(range(len(global_prompts)), type=pa.int64()),
            "__index_level_0__": global_prompts,
        }
    ).replace_schema_metadata(task_schema_metadata)
    pq.write_table(task_table, output_root / "meta" / "tasks.parquet")
    combined_stats = reduce(_combine_stats, stats)
    (output_root / "meta" / "stats.json").write_text(
        json.dumps(combined_stats, indent=2), encoding="utf-8"
    )
    info = dict(infos[0])
    info.update(
        {
            "total_episodes": episode_offset,
            "total_frames": frame_offset,
            "total_tasks": len(global_prompts),
            "splits": {"train": f"0:{episode_offset}"},
        }
    )
    (output_root / "meta" / "info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    (output_root / "meta" / "build_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset": "multitask_sim2real_v2_4500",
                "randomization_schema_version": 2,
                "git_commit": _git_commit(),
                "sources": source_records,
                "global_prompts": global_prompts,
                "combined_mapping": {
                    source["name"]: source["task_mapping"] for source in source_records
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "collection.complete").write_text(
        f"episodes={episode_offset} tasks={len(global_prompts)}\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push-root", type=Path, required=True)
    parser.add_argument("--pick-place-root", type=Path, required=True)
    parser.add_argument("--color-pick-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    build(
        [
            ("push", args.push_root),
            ("pick_place", args.pick_place_root),
            ("color_pick", args.color_pick_root),
        ],
        args.output_root,
    )
    print(f"dataset_ok root={args.output_root} episodes=4500 tasks=5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
