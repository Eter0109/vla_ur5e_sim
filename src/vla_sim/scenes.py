"""Deterministic scene manifests for data collection and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    seed: int
    x_m: float
    y_m: float
    yaw_rad: float

    @classmethod
    def from_dict(cls, value: dict) -> "SceneSpec":
        return cls(**value)


def generate_scenes(split: str, count: int, seed: int) -> list[SceneSpec]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    return [
        SceneSpec(
            scene_id=f"{split}-{index:04d}",
            seed=seed + index,
            x_m=float(rng.uniform(-0.06, 0.06)),
            y_m=float(rng.uniform(-0.06, 0.06)),
            yaw_rad=float(rng.uniform(-np.pi, np.pi)),
        )
        for index in range(count)
    ]


def save_manifest(path: str | Path, scenes: list[SceneSpec]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(scene) for scene in scenes], indent=2), encoding="utf-8"
    )
    return destination


def load_manifest(path: str | Path) -> list[SceneSpec]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SceneSpec.from_dict(value) for value in values]
