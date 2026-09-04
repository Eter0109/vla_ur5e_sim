from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vla_sim.simulation.dataset_build import build


def _stats(value: float, count: int) -> dict:
    return {
        "action": {
            "count": [count],
            "mean": [value],
            "std": [0.0],
            "min": [value],
            "max": [value],
            "q01": [value],
            "q10": [value],
            "q50": [value],
            "q90": [value],
            "q99": [value],
        }
    }


def _source(root: Path, prompts: list[str], value: float) -> None:
    episodes = len(prompts)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    data = pa.table(
        {
            "index": pa.array(range(episodes), type=pa.int64()),
            "episode_index": pa.array(range(episodes), type=pa.int64()),
            "task_index": pa.array(range(episodes), type=pa.int64()),
        }
    )
    pq.write_table(data, root / "data/chunk-000/file-000.parquet")
    episode_table = pa.table(
        {
            "episode_index": pa.array(range(episodes), type=pa.int64()),
            "dataset_from_index": pa.array(range(episodes), type=pa.int64()),
            "dataset_to_index": pa.array(range(1, episodes + 1), type=pa.int64()),
            "data/chunk_index": pa.array([0] * episodes, type=pa.int64()),
            "data/file_index": pa.array([0] * episodes, type=pa.int64()),
        }
    )
    pq.write_table(episode_table, root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array(range(episodes), type=pa.int64()),
                "__index_level_0__": prompts,
            }
        ),
        root / "meta/tasks.parquet",
    )
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "features": {"dummy": {"dtype": "float32", "shape": [1]}},
                "total_episodes": episodes,
                "total_frames": episodes,
                "total_tasks": episodes,
            }
        ),
        encoding="utf-8",
    )
    (root / "meta/stats.json").write_text(
        json.dumps(_stats(value, episodes)), encoding="utf-8"
    )
    (root / "meta/collection_manifest.json").write_text("{}", encoding="utf-8")
    (root / "meta/sim2real_v2_environment.json").write_text("{}", encoding="utf-8")
    (root / "meta/collection_provenance.json").write_text(
        json.dumps(
            {
                "collection_command": ["python", "collect.py"],
                "failure_statistics": {"failed_candidates": 0},
            }
        ),
        encoding="utf-8",
    )
    (root / "collection.complete").write_text("ok\n", encoding="utf-8")


def test_three_source_build_remaps_indices_prompts_and_combines_stats(tmp_path: Path) -> None:
    push = tmp_path / "push"
    pick = tmp_path / "pick"
    color = tmp_path / "color"
    output = tmp_path / "combined"
    _source(push, ["push the block into the red target circle"], 1.0)
    _source(pick, ["place the red cube in the blue storage bin"], 2.0)
    _source(
        color,
        ["pick up the red cube", "pick up the green cube", "pick up the blue cube"],
        3.0,
    )
    build([("push", push), ("pick_place", pick), ("color_pick", color)], output)
    files = sorted((output / "data").rglob("*.parquet"))
    tables = [pq.read_table(path) for path in files]
    assert pa.concat_tables(tables)["index"].to_pylist() == list(range(5))
    assert pa.concat_tables(tables)["episode_index"].to_pylist() == list(range(5))
    assert pa.concat_tables(tables)["task_index"].to_pylist() == list(range(5))
    tasks = pq.read_table(output / "meta/tasks.parquet")["__index_level_0__"].to_pylist()
    assert tasks == [
        "push the block into the red target circle",
        "place the red cube in the blue storage bin",
        "pick up the red cube",
        "pick up the green cube",
        "pick up the blue cube",
    ]
    assert os.stat(files[0]).st_ino == os.stat(push / "data/chunk-000/file-000.parquet").st_ino
    assert json.loads((output / "meta/info.json").read_text())["total_episodes"] == 5
    assert json.loads((output / "meta/stats.json").read_text())["action"]["count"] == [5]
