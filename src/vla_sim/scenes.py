"""Deterministic scene manifests for data collection and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    seed: int
    x_m: float
    y_m: float
    yaw_rad: float
    env_seed: int | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict) -> "SceneSpec":
        normalized = dict(value)
        # Version-1 manifests used one seed for the environment and policy.
        # Keep that field so existing datasets and reports remain readable.
        normalized.setdefault("env_seed", normalized.get("seed"))
        normalized.setdefault("overrides", {})
        return cls(**normalized)

    @property
    def effective_env_seed(self) -> int:
        return self.seed if self.env_seed is None else self.env_seed


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


def save_manifest(
    path: str | Path,
    scenes: list[SceneSpec],
    *,
    benchmark_id: str | None = None,
    role: str | None = None,
    generator_seed: int | None = None,
    environment_preset: str | None = None,
) -> Path:
    """Write a legacy list manifest or a schema-v2 benchmark manifest.

    Passing no metadata preserves the version-1 list format. New benchmark
    manifests must supply ``benchmark_id`` and ``role`` so that their intended
    selection policy is explicit in the file itself.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if benchmark_id is None and role is None:
        payload: object = [asdict(scene) for scene in scenes]
    else:
        if not benchmark_id or not role:
            raise ValueError("benchmark_id and role must be provided together")
        payload = {
            "schema_version": 2,
            "benchmark_id": benchmark_id,
            "role": role,
            "generator_seed": generator_seed,
            "environment_preset": environment_preset,
            "scenes": [asdict(scene) for scene in scenes],
        }
    destination.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return destination


def load_manifest(path: str | Path) -> list[SceneSpec]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    scenes = values if isinstance(values, list) else values.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("Manifest must be a scene list or a schema-v2 object with scenes")
    return [SceneSpec.from_dict(value) for value in scenes]


def load_manifest_metadata(path: str | Path) -> dict[str, Any]:
    """Return a normalized metadata record without changing legacy manifests."""
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(values, list):
        return {"schema_version": 1, "benchmark_id": None, "role": "legacy"}
    if not isinstance(values, dict):
        raise ValueError("Manifest root must be a JSON list or object")
    return {
        "schema_version": values.get("schema_version"),
        "benchmark_id": values.get("benchmark_id"),
        "role": values.get("role"),
        "generator_seed": values.get("generator_seed"),
        "environment_preset": values.get("environment_preset"),
    }
