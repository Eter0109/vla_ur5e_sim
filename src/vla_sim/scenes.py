"""Deterministic scene manifests for data collection and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


STACK_TASKS = ("red_on_blue", "blue_on_red")
STACK_DISTANCE_BINS_M = ((0.075, 0.095), (0.095, 0.115), (0.115, 0.140))
STACK_WORKSPACE_LIMIT_M = 0.052
STACK_OBJECT_XY_M = (0.05, 0.05)
STACK_GRIPPER_CLEARANCE_M = 0.030
PICK_PLACE_TASK = "red_to_storage_bin"
PICK_PLACE_DISTANCE_BINS_M = ((0.180, 0.215), (0.215, 0.250))
PICK_PLACE_SOURCE_X_M = (-0.060, -0.035)
PICK_PLACE_SOURCE_Y_M = (-0.040, 0.040)
PICK_PLACE_TARGET_X_M = (0.145, 0.175)
PICK_PLACE_TARGET_Y_M = (-0.160, 0.160)
PICK_PLACE_MIN_LATERAL_SEPARATION_M = 0.100


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


def _rectangle_axes(yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
    cosine = float(np.cos(yaw_rad))
    sine = float(np.sin(yaw_rad))
    return (
        np.asarray([cosine, sine], dtype=np.float64),
        np.asarray([-sine, cosine], dtype=np.float64),
    )


def oriented_rectangles_overlap(
    center_a: tuple[float, float],
    yaw_a: float,
    size_a: tuple[float, float],
    center_b: tuple[float, float],
    yaw_b: float,
    size_b: tuple[float, float],
    *,
    clearance_m: float = 0.0,
) -> bool:
    """Return whether two oriented rectangles overlap within a clearance margin."""

    if clearance_m < 0:
        raise ValueError("clearance_m must be non-negative")
    axes_a = _rectangle_axes(yaw_a)
    axes_b = _rectangle_axes(yaw_b)
    half_a = np.asarray(size_a, dtype=np.float64) / 2.0 + clearance_m / 2.0
    half_b = np.asarray(size_b, dtype=np.float64) / 2.0 + clearance_m / 2.0
    delta = np.asarray(center_b, dtype=np.float64) - np.asarray(center_a, dtype=np.float64)
    for axis in (*axes_a, *axes_b):
        distance = abs(float(np.dot(delta, axis)))
        radius_a = sum(half_a[i] * abs(float(np.dot(axes_a[i], axis))) for i in range(2))
        radius_b = sum(half_b[i] * abs(float(np.dot(axes_b[i], axis))) for i in range(2))
        if distance >= radius_a + radius_b:
            return False
    return True


def _balanced_stack_assignments(count: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    if count % len(STACK_TASKS):
        raise ValueError("Stack scene count must be even for 50/50 task balance")
    assignments: list[tuple[str, int]] = []
    per_task = count // len(STACK_TASKS)
    for task in STACK_TASKS:
        assignments.extend((task, index % 3) for index in range(per_task))
    rng.shuffle(assignments)
    return assignments


def generate_stack_scenes(split: str, count: int, seed: int) -> list[SceneSpec]:
    """Generate balanced, non-overlapping Stack scenes in three distance strata."""

    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    scenes: list[SceneSpec] = []
    if split == "stack_collect_v1" and count == 3200:
        cells = [(task, bin_index) for task in STACK_TASKS for bin_index in range(3)]
        first = [cell for cell in cells for _ in range(400)]
        supplement = [cell for cell in cells for _ in range(100)]
        reserve = _balanced_stack_assignments(200, rng)
        rng.shuffle(first)
        rng.shuffle(supplement)
        assignments = first + supplement + reserve
    else:
        assignments = _balanced_stack_assignments(count, rng)
    for index, (task, bin_index) in enumerate(assignments):
        distance_min, distance_max = STACK_DISTANCE_BINS_M[bin_index]
        for _attempt in range(10_000):
            a_x = float(rng.uniform(-STACK_WORKSPACE_LIMIT_M, STACK_WORKSPACE_LIMIT_M))
            a_y = float(rng.uniform(-STACK_WORKSPACE_LIMIT_M, STACK_WORKSPACE_LIMIT_M))
            direction = float(rng.uniform(-np.pi, np.pi))
            distance = float(rng.uniform(distance_min, distance_max))
            # The fixed-orientation Robotiq fingers close along the world Y
            # axis. At very short distances a target cube on that axis blocks
            # the fingers, so retain 75--87 mm cases only along the clear X
            # approach axis. This is a physical feasibility constraint, not a
            # policy shortcut; the expert and policy still receive no poses.
            if distance < 0.087 and abs(float(np.cos(direction))) < 0.75:
                continue
            b_x = a_x + distance * float(np.cos(direction))
            b_y = a_y + distance * float(np.sin(direction))
            if max(abs(b_x), abs(b_y)) > STACK_WORKSPACE_LIMIT_M:
                continue
            # Stack v1 deliberately fixes yaw because the deployed action
            # contract fixes rotation and trains only XYZ. A future oriented
            # object benchmark must introduce rotation actions and labels.
            yaw_a = 0.0
            yaw_b = 0.0
            if oriented_rectangles_overlap(
                (a_x, a_y),
                yaw_a,
                STACK_OBJECT_XY_M,
                (b_x, b_y),
                yaw_b,
                STACK_OBJECT_XY_M,
                clearance_m=STACK_GRIPPER_CLEARANCE_M,
            ):
                continue
            scenes.append(
                SceneSpec(
                    scene_id=f"{split}-{index:04d}",
                    seed=seed + index,
                    x_m=a_x,
                    y_m=a_y,
                    yaw_rad=yaw_a,
                    overrides={
                        "task": task,
                        "cubeB_x_m": b_x,
                        "cubeB_y_m": b_y,
                        "cubeB_yaw_rad": yaw_b,
                        "distance_bin": bin_index,
                    },
                )
            )
            break
        else:
            raise RuntimeError(f"Could not sample valid Stack scene {index}")
    return scenes


def generate_pick_place_scenes(
    split: str,
    count: int,
    seed: int,
    *,
    target_y_bounds_m: tuple[float, float] = PICK_PLACE_TARGET_Y_M,
) -> list[SceneSpec]:
    """Generate red-cube-to-blue-storage-bin scenes balanced by transport distance."""

    if count < 1 or count % len(PICK_PLACE_DISTANCE_BINS_M):
        raise ValueError("PickPlace scene count must be positive and divisible by two")
    target_y_min, target_y_max = target_y_bounds_m
    if not (
        PICK_PLACE_TARGET_Y_M[0] <= target_y_min < target_y_max <= PICK_PLACE_TARGET_Y_M[1]
    ):
        raise ValueError("target_y_bounds_m must be ordered and inside the PickPlace workspace")
    rng = np.random.default_rng(seed)
    bins = [index % len(PICK_PLACE_DISTANCE_BINS_M) for index in range(count)]
    rng.shuffle(bins)
    scenes: list[SceneSpec] = []
    for index, bin_index in enumerate(bins):
        lower, upper = PICK_PLACE_DISTANCE_BINS_M[bin_index]
        for _attempt in range(10_000):
            source_x = float(rng.uniform(*PICK_PLACE_SOURCE_X_M))
            source_y = float(rng.uniform(*PICK_PLACE_SOURCE_Y_M))
            target_x = float(rng.uniform(*PICK_PLACE_TARGET_X_M))
            target_y = float(rng.uniform(target_y_min, target_y_max))
            if abs(target_y - source_y) < PICK_PLACE_MIN_LATERAL_SEPARATION_M:
                continue
            distance = float(np.hypot(target_x - source_x, target_y - source_y))
            if lower <= distance <= upper:
                scenes.append(
                    SceneSpec(
                        scene_id=f"{split}-{index:04d}",
                        seed=seed + index,
                        x_m=source_x,
                        y_m=source_y,
                        yaw_rad=0.0,
                        overrides={
                            "task": PICK_PLACE_TASK,
                            "target_x_m": target_x,
                            "target_y_m": target_y,
                            "distance_bin": bin_index,
                        },
                    )
                )
                break
        else:
            raise RuntimeError(f"Could not sample valid PickPlace scene {index}")
    return scenes


def generate_scenes(split: str, count: int, seed: int) -> list[SceneSpec]:
    """Backward-compatible alias for the current Stack scene generator."""

    return generate_stack_scenes(split, count, seed)


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
