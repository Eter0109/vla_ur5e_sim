"""Deterministic scene manifests for data collection and evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vla_sim.domain_randomization import (
    sample_domain_randomization,
    sample_sim2real_v2,
)

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
PUSH_ANGLE_BINS_RAD = tuple(
    (float(np.deg2rad(lower)), float(np.deg2rad(upper)))
    for lower, upper in ((-15.0, -9.0), (-9.0, -3.0), (-3.0, 3.0), (3.0, 9.0), (9.0, 15.0))
)
PUSH_DISTANCE_BINS_M = ((0.100, 0.125), (0.125, 0.150))
PUSH_OBJECT_XY_M = ((-0.055, 0.025), (-0.055, 0.055))
PUSH_TABLE_LIMIT_M = 0.250
COLOR_PICK_TASK = "pick_requested_color"
COLOR_PICK_COLORS = ("red", "green", "blue")
COLOR_PICK_WORKSPACE_X_M = (-0.080, 0.080)
COLOR_PICK_WORKSPACE_Y_M = (-0.110, 0.110)
COLOR_PICK_OBJECT_XY_M = (0.050, 0.050)
COLOR_PICK_GRIPPER_CLEARANCE_M = 0.035


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
    def from_dict(cls, value: dict) -> SceneSpec:
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
    source_y_bounds_m: tuple[float, float] = PICK_PLACE_SOURCE_Y_M,
    distance_bins: tuple[int, ...] = tuple(range(len(PICK_PLACE_DISTANCE_BINS_M))),
) -> list[SceneSpec]:
    """Generate red-cube-to-blue-storage-bin scenes balanced by transport distance."""

    if count < 1 or not distance_bins or count % len(distance_bins):
        raise ValueError("count must be positive and divisible by selected distance bins")
    if len(set(distance_bins)) != len(distance_bins) or any(
        index < 0 or index >= len(PICK_PLACE_DISTANCE_BINS_M) for index in distance_bins
    ):
        raise ValueError("distance_bins must be distinct valid PickPlace distance-bin indices")
    target_y_min, target_y_max = target_y_bounds_m
    if not (
        PICK_PLACE_TARGET_Y_M[0] <= target_y_min < target_y_max <= PICK_PLACE_TARGET_Y_M[1]
    ):
        raise ValueError("target_y_bounds_m must be ordered and inside the PickPlace workspace")
    source_y_min, source_y_max = source_y_bounds_m
    if not (
        PICK_PLACE_SOURCE_Y_M[0] <= source_y_min < source_y_max <= PICK_PLACE_SOURCE_Y_M[1]
    ):
        raise ValueError("source_y_bounds_m must be ordered and inside the PickPlace workspace")
    rng = np.random.default_rng(seed)
    bins = [distance_bins[index % len(distance_bins)] for index in range(count)]
    rng.shuffle(bins)
    scenes: list[SceneSpec] = []
    for index, bin_index in enumerate(bins):
        lower, upper = PICK_PLACE_DISTANCE_BINS_M[bin_index]
        for _attempt in range(10_000):
            source_x = float(rng.uniform(*PICK_PLACE_SOURCE_X_M))
            source_y = float(rng.uniform(source_y_min, source_y_max))
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


def generate_color_pick_scenes(split: str, count: int, seed: int) -> list[SceneSpec]:
    """Generate balanced three-cube scenes for language-conditioned color selection.

    Red, green, and blue cubes are placed independently in a fixed workspace while
    preserving enough clearance for the Robotiq fingers. The requested target color
    is exactly balanced and is carried only in the scene task metadata / language prompt.
    """

    if count < len(COLOR_PICK_COLORS) or count % len(COLOR_PICK_COLORS):
        raise ValueError("count must be positive and divisible by three target colors")
    rng = np.random.default_rng(seed)
    targets = [
        color
        for color in COLOR_PICK_COLORS
        for _ in range(count // len(COLOR_PICK_COLORS))
    ]
    rng.shuffle(targets)
    scenes: list[SceneSpec] = []
    for index, target_color in enumerate(targets):
        for _attempt in range(10_000):
            positions = {
                color: (
                    float(rng.uniform(*COLOR_PICK_WORKSPACE_X_M)),
                    float(rng.uniform(*COLOR_PICK_WORKSPACE_Y_M)),
                )
                for color in COLOR_PICK_COLORS
            }
            pairs = (("red", "green"), ("red", "blue"), ("green", "blue"))
            if any(
                oriented_rectangles_overlap(
                    positions[left],
                    0.0,
                    COLOR_PICK_OBJECT_XY_M,
                    positions[right],
                    0.0,
                    COLOR_PICK_OBJECT_XY_M,
                    clearance_m=COLOR_PICK_GRIPPER_CLEARANCE_M,
                )
                for left, right in pairs
            ):
                continue
            red_x, red_y = positions["red"]
            green_x, green_y = positions["green"]
            blue_x, blue_y = positions["blue"]
            scenes.append(
                SceneSpec(
                    scene_id=f"{split}-{index:04d}",
                    seed=seed + index,
                    env_seed=seed + index,
                    x_m=red_x,
                    y_m=red_y,
                    yaw_rad=0.0,
                    overrides={
                        "task": COLOR_PICK_TASK,
                        "target_color": target_color,
                        "green_x_m": green_x,
                        "green_y_m": green_y,
                        "green_yaw_rad": 0.0,
                        "blue_x_m": blue_x,
                        "blue_y_m": blue_y,
                        "blue_yaw_rad": 0.0,
                    },
                )
            )
            break
        else:
            raise RuntimeError(f"Could not sample valid ColorPick scene {index}")
    return scenes


def generate_push_scenes(split: str, count: int, seed: int) -> list[SceneSpec]:
    """Generate deterministic forward-push scenes balanced by angle and distance.

    The target, object geometry, and object appearance are carried in the scene
    overrides so development and held-out manifests do not depend on the
    simulator's implicit random state.
    """

    strata = len(PUSH_ANGLE_BINS_RAD) * len(PUSH_DISTANCE_BINS_M)
    if count < 1 or count % strata:
        raise ValueError(f"count must be positive and divisible by {strata} for Push balancing")
    rng = np.random.default_rng(seed)
    assignments = [(angle_bin, distance_bin) for angle_bin in range(len(PUSH_ANGLE_BINS_RAD)) for distance_bin in range(len(PUSH_DISTANCE_BINS_M))]
    assignments *= count // strata
    rng.shuffle(assignments)
    scenes: list[SceneSpec] = []
    for index, (angle_bin, distance_bin) in enumerate(assignments):
        angle = float(rng.uniform(*PUSH_ANGLE_BINS_RAD[angle_bin]))
        distance = float(rng.uniform(*PUSH_DISTANCE_BINS_M[distance_bin]))
        for _attempt in range(1_000):
            x_m = float(rng.uniform(*PUSH_OBJECT_XY_M[0]))
            y_m = float(rng.uniform(*PUSH_OBJECT_XY_M[1]))
            target_x_m = x_m + float(np.cos(angle) * distance)
            target_y_m = y_m + float(np.sin(angle) * distance)
            if abs(target_x_m) > PUSH_TABLE_LIMIT_M or abs(target_y_m) > PUSH_TABLE_LIMIT_M:
                continue
            dimensions = tuple(float(value) for value in rng.uniform(0.030, 0.060, size=3))
            rgba = tuple(float(value) for value in rng.uniform(0.10, 0.90, size=3)) + (1.0,)
            scenes.append(
                SceneSpec(
                    scene_id=f"{split}-{index:04d}",
                    seed=seed + index,
                    env_seed=seed + index,
                    x_m=x_m,
                    y_m=y_m,
                    yaw_rad=0.0,
                    overrides={
                        "task": "push_block_to_red_circle",
                        "target_x_m": target_x_m,
                        "target_y_m": target_y_m,
                        "target_angle_rad": angle,
                        "target_distance_m": distance,
                        "angle_bin": angle_bin,
                        "distance_bin": distance_bin,
                        "object_shape": "box",
                        "object_dimensions_m": dimensions,
                        "object_rgba": rgba,
                    },
                )
            )
            break
        else:
            raise RuntimeError(f"Could not sample valid Push scene {index}")
    return scenes


def attach_domain_randomization(
    scenes: list[SceneSpec], *, tier_counts: dict[str, int], seed: int
) -> list[SceneSpec]:
    """Attach deterministic render/contact samples while preserving geometry.

    ``tier_counts`` must account for every input scene exactly. The assignment
    is shuffled once with ``seed`` so tiers cannot correlate with geometry
    strata or source ordering.
    """

    if sum(tier_counts.values()) != len(scenes):
        raise ValueError("tier_counts must sum to the number of scenes")
    if any(count < 0 for count in tier_counts.values()):
        raise ValueError("tier counts must be non-negative")
    assignments = [tier for tier, count in tier_counts.items() for _ in range(count)]
    rng = np.random.default_rng(seed)
    rng.shuffle(assignments)
    result: list[SceneSpec] = []
    for index, (scene, tier) in enumerate(zip(scenes, assignments, strict=True)):
        overrides = dict(scene.overrides)
        overrides["domain_randomization"] = sample_domain_randomization(tier, seed + index).as_overrides()
        result.append(
            SceneSpec(
                scene_id=scene.scene_id,
                seed=scene.seed,
                env_seed=scene.env_seed,
                x_m=scene.x_m,
                y_m=scene.y_m,
                yaw_rad=scene.yaw_rad,
                overrides=overrides,
            )
        )
    return result


def attach_sim2real_v2_randomization(
    scenes: list[SceneSpec],
    *,
    group_fields: tuple[str, ...],
    seed: int,
    color_sensitive: bool = False,
) -> list[SceneSpec]:
    """Attach an exact 20/50/30 tier mix independently within every group."""

    if not scenes or not group_fields:
        raise ValueError("scenes and group_fields must be non-empty")
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, scene in enumerate(scenes):
        try:
            key = tuple(scene.overrides[field] for field in group_fields)
        except KeyError as error:
            raise ValueError(f"scene is missing grouping field {error.args[0]!r}") from error
        groups.setdefault(key, []).append(index)
    assignments: list[str | None] = [None] * len(scenes)
    rng = np.random.default_rng(seed)
    for indices in groups.values():
        if len(indices) % 10:
            raise ValueError("each sim2real-v2 group size must be divisible by 10")
        unit = len(indices) // 10
        tiers = ["nominal"] * (2 * unit) + ["light"] * (5 * unit) + ["medium"] * (3 * unit)
        rng.shuffle(tiers)
        for index, tier in zip(indices, tiers, strict=True):
            assignments[index] = tier
    randomized: list[SceneSpec] = []
    for index, (scene, tier) in enumerate(zip(scenes, assignments, strict=True)):
        if tier is None:
            raise AssertionError("sim2real-v2 tier assignment is incomplete")
        overrides = dict(scene.overrides)
        overrides["domain_randomization"] = sample_sim2real_v2(
            tier,
            seed + index,
            color_sensitive=color_sensitive,
        ).as_overrides()
        randomized.append(
            SceneSpec(
                scene_id=scene.scene_id,
                seed=scene.seed,
                env_seed=scene.env_seed,
                x_m=scene.x_m,
                y_m=scene.y_m,
                yaw_rad=scene.yaw_rad,
                overrides=overrides,
            )
        )
    return randomized


def select_targeted_push_recovery_scenes(
    scenes: list[SceneSpec], *, count: int
) -> list[SceneSpec]:
    """Select an interleaved hard Push subset without consulting evaluation outcomes.

    The current development funnel shows that the deployment distribution is
    weakest for the far-distance stratum under medium domain randomization,
    especially in angle bins 1 and 4.  Collection uses fresh source seeds and
    balances those two cells so replay corrects both sides of the workspace
    instead of memorizing one direction.
    """

    if count < 2 or count % 2:
        raise ValueError("count must be a positive even number")
    groups: dict[int, list[SceneSpec]] = {1: [], 4: []}
    for scene in scenes:
        randomization = scene.overrides.get("domain_randomization", {})
        angle_bin = int(scene.overrides.get("angle_bin", -1))
        if (
            randomization.get("tier") == "medium"
            and int(scene.overrides.get("distance_bin", -1)) == 1
            and angle_bin in groups
        ):
            groups[angle_bin].append(scene)
    per_group = count // 2
    if any(len(group) < per_group for group in groups.values()):
        available = ", ".join(f"angle{key}={len(value)}" for key, value in groups.items())
        raise ValueError(f"insufficient targeted Push scenes: {available}")
    selected: list[SceneSpec] = []
    for left, right in zip(groups[1][:per_group], groups[4][:per_group], strict=True):
        selected.extend((left, right))
    return selected


def select_stratified_scenes(
    scenes: list[SceneSpec],
    *,
    count: int,
    stratum: Callable[[SceneSpec], Hashable],
) -> list[SceneSpec]:
    """Select a deterministic round-robin subset spanning every available stratum."""

    if not 1 <= count <= len(scenes):
        raise ValueError("count must be within the scene collection")
    groups: dict[Hashable, list[SceneSpec]] = {}
    for scene in scenes:
        groups.setdefault(stratum(scene), []).append(scene)
    ordered_keys = sorted(groups, key=repr)
    selected: list[SceneSpec] = []
    offset = 0
    while len(selected) < count:
        added = False
        for key in ordered_keys:
            group = groups[key]
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError("stratified selection exhausted scenes before reaching count")
        offset += 1
    return selected


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
    extra_metadata: dict[str, Any] | None = None,
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
        if extra_metadata:
            reserved = set(payload) & set(extra_metadata)
            if reserved:
                raise ValueError(f"extra metadata overrides reserved fields: {sorted(reserved)}")
            payload.update(extra_metadata)
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
    return {key: value for key, value in values.items() if key != "scenes"}
