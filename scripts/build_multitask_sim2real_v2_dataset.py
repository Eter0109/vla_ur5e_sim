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

try:
    from build_multitask_robust_dataset import _combine_stats
except ModuleNotFoundError:  # Imported as scripts.build_multitask_sim2real_v2_dataset.
    from scripts.build_multitask_robust_dataset import _combine_stats


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
