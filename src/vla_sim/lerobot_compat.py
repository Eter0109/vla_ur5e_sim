"""Small runtime compatibility helpers for the pinned LeRobot stack."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any


class _LazyParquetColumn:
    def __init__(self, dataset: "_LazyParquetDataset", name: str) -> None:
        self.dataset = dataset
        self.name = name

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            indices = range(*indices.indices(len(self.dataset)))
        if not isinstance(indices, (list, tuple, range)):
            return self.dataset._column_value(self.name, int(indices))
        return [self.dataset._column_value(self.name, int(index)) for index in indices]


class _LazyParquetDataset:
    """Minimal HF-Dataset-compatible, bounded-memory parquet reader."""

    def __init__(self, paths: list[Path], features: Any) -> None:
        import pyarrow.parquet as pq

        self.features = features
        self.column_names = list(features)
        self._paths = paths
        self._files = [pq.ParquetFile(path) for path in paths]
        self._starts: list[int] = []
        total = 0
        for parquet_file in self._files:
            self._starts.append(total)
            total += parquet_file.metadata.num_rows
        self._length = total
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._cache_size = max(1, int(os.environ.get("VLA_LAZY_PARQUET_CACHE_FILES", "2")))
        self._transform = None

    def __len__(self) -> int:
        return self._length

    def set_transform(self, transform) -> None:
        self._transform = transform

    def with_format(self, _format=None):
        return self

    def unique(self, name: str) -> list[int]:
        values: set[int] = set()
        for index, parquet_file in enumerate(self._files):
            table = parquet_file.read(columns=[name])
            values.update(int(value) for value in table[name].to_pylist())
            del table
        return sorted(values)

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError("Parquet dataset index is outside the dataset")
        file_index = 0
        while file_index + 1 < len(self._starts) and self._starts[file_index + 1] <= index:
            file_index += 1
        return file_index, index - self._starts[file_index]

    def _table(self, file_index: int):
        if file_index in self._cache:
            self._cache.move_to_end(file_index)
            return self._cache[file_index]
        table = self._files[file_index].read()
        self._cache[file_index] = table
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return table

    def _raw_value(self, name: str, index: int):
        file_index, local_index = self._locate(index)
        return self._table(file_index)[name][local_index].as_py()

    def _column_value(self, name: str, index: int):
        import torch

        value = self._raw_value(name, index)
        if name.startswith("observation.images."):
            return self.features[name].decode_example(value)
        return torch.tensor(value)

    def _item(self, index: int) -> dict[str, Any]:
        file_index, local_index = self._locate(index)
        row = self._table(file_index).slice(local_index, 1).to_pylist()[0]
        decoded = self.features.decode_example(row)
        if self._transform is None:
            return decoded
        transformed = self._transform({key: [value] for key, value in decoded.items()})
        return {key: values[0] for key, values in transformed.items()}

    def __getitem__(self, index):
        if isinstance(index, str):
            return _LazyParquetColumn(self, index)
        if isinstance(index, slice):
            index = range(*index.indices(self._length))
        if isinstance(index, (list, tuple, range)):
            rows = [self._item(int(value)) for value in index]
            return {key: [row[key] for row in rows] for key in self.column_names}
        return self._item(int(index))


def install_fast_parquet_loader() -> None:
    """Avoid a datasets.from_parquet cache hang observed on Windows."""

    # The eager compatibility loader is useful for small legacy datasets, but
    # it materializes every embedded image byte in RAM.  Large multi-task image
    # datasets must use LeRobot's normal memory-mapped parquet path instead.
    if os.name != "nt":
        return

    from lerobot.datasets import lerobot_dataset, utils

    if os.environ.get("VLA_LAZY_PARQUET_LOADER", "0") == "1":
        if getattr(utils.load_nested_dataset, "_vla_sim_lazy_loader", False):
            return

        def load_nested_dataset(
            pq_dir: Path, features=None, episodes: list[int] | None = None
        ):
            paths = sorted(Path(pq_dir).glob("*/*.parquet"))
            if not paths:
                raise FileNotFoundError(f"No parquet files found in {pq_dir}")
            # LeRobot metadata (episodes/tasks) is small and intentionally has
            # no feature schema; keep its native Dataset behavior.
            if features is None:
                from datasets import Dataset

                return Dataset.from_parquet([str(path) for path in paths])
            dataset = _LazyParquetDataset(paths, features)
            if episodes is not None:
                raise NotImplementedError("Lazy parquet loader does not support episode subsets")
            return dataset

        load_nested_dataset._vla_sim_lazy_loader = True  # type: ignore[attr-defined]
        utils.load_nested_dataset = load_nested_dataset
        lerobot_dataset.load_nested_dataset = load_nested_dataset
        return

    if os.environ.get("VLA_FAST_PARQUET_LOADER", "1") != "1":
        return

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from datasets import Dataset

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
