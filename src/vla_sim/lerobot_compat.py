"""Small runtime compatibility helpers for the pinned LeRobot stack."""

from __future__ import annotations

import os
from pathlib import Path


def install_fast_parquet_loader() -> None:
    """Avoid a datasets.from_parquet cache hang observed on Windows."""

    if os.name != "nt":
        return

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from datasets import Dataset
    from lerobot.datasets import lerobot_dataset, utils

    if getattr(utils.load_nested_dataset, "_vla_sim_fast_loader", False):
        return

    def load_nested_dataset(
        pq_dir: Path, features=None, episodes: list[int] | None = None
    ) -> Dataset:
        paths = sorted(Path(pq_dir).glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {pq_dir}")
        table = pa.concat_tables([pq.read_table(path) for path in paths])
        if episodes is not None:
            mask = pc.is_in(table["episode_index"], value_set=pa.array(episodes))
            table = table.filter(mask)
        return Dataset(table)

    load_nested_dataset._vla_sim_fast_loader = True  # type: ignore[attr-defined]
    utils.load_nested_dataset = load_nested_dataset
    lerobot_dataset.load_nested_dataset = load_nested_dataset
