"""Deterministic visual and contact domain randomization for UR5e tasks.

The randomization sample is stored in every scene manifest.  This keeps data
collection, development evaluation, and blind evaluation reproducible and
prevents renderer state from leaking between episodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray


PUSH_FRONT_CAMERA_POSITION_M = (0.50, 0.0, 1.35)
PUSH_FRONT_CAMERA_LOOK_AT_M = (0.0, 0.0, 0.80)
PUSH_FRONT_CAMERA_FOVY_DEG = 45.0
PICK_FRONT_CAMERA_POSITION_M = (0.70, 0.0, 1.46)
PICK_FRONT_CAMERA_LOOK_AT_M = (0.05, 0.0, 0.81)
PICK_FRONT_CAMERA_FOVY_DEG = 55.0

TRAIN_TABLE_COLORS = (
    (0.34, 0.34, 0.34),
    (0.72, 0.70, 0.64),
    (0.50, 0.36, 0.22),
    (0.18, 0.21, 0.24),
    (0.30, 0.42, 0.40),
    (0.56, 0.48, 0.34),
)
BLIND_TABLE_COLORS = (
    (0.21, 0.31, 0.25),
    (0.60, 0.62, 0.64),
    (0.25, 0.29, 0.38),
)


@dataclass(frozen=True)
class DomainRandomizationSample:
    """One immutable rendering/contact sample attached to a scene."""

    tier: str
    seed: int
    front_position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    front_look_at_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    front_fovy_offset_deg: float = 0.0
    wrist_position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wrist_rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wrist_fovy_offset_deg: float = 0.0
    light_position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    light_diffuse_scale: float = 1.0
    light_ambient: float = 0.0
    light_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    table_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
    table_friction_scale: float = 1.0
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    hue_shift: float = 0.0
    gaussian_noise_std: float = 0.0
    blur_probability: float = 0.0

    def as_overrides(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DomainRandomizationSample":
        normalized = dict(value)
        for name in (
            "front_position_offset_m",
            "front_look_at_offset_m",
            "wrist_position_offset_m",
            "wrist_rotation_deg",
            "light_position_offset_m",
            "light_tint",
            "table_rgb",
        ):
            if name in normalized:
                normalized[name] = tuple(float(item) for item in normalized[name])
        return cls(**normalized)


@dataclass(frozen=True)
class RenderBaseline:
    """Mutable MuJoCo render fields restored before each episode sample."""

    camera_positions: NDArray[np.float64]
    camera_quaternions: NDArray[np.float64]
    camera_fovy: NDArray[np.float64]
    light_positions: NDArray[np.float64]
    light_diffuse: NDArray[np.float64]
    light_ambient: NDArray[np.float64]
    light_specular: NDArray[np.float64]
    table_rgba: dict[int, NDArray[np.float64]]
    table_friction: dict[int, NDArray[np.float64]]


def sample_domain_randomization(tier: str, seed: int) -> DomainRandomizationSample:
    """Sample a deterministic nominal, training, blind, or stress profile."""

    rng = np.random.default_rng(seed)
    if tier == "nominal":
        return DomainRandomizationSample(tier=tier, seed=seed)
    if tier == "light":
        return _sample(
            tier,
            seed,
            rng,
            front_xy=0.03,
            front_z=0.02,
            look_xy=0.02,
            look_z=0.02,
            front_fovy=3.0,
            wrist_position=0.002,
            wrist_rotation=1.0,
            wrist_fovy=1.0,
            light_position=0.20,
            diffuse=(0.8, 1.2),
            ambient=(0.0, 0.10),
            tint=0.05,
            friction=(0.8, 1.2),
            colors=TRAIN_TABLE_COLORS[:4],
            brightness=(0.90, 1.10),
            contrast=(0.90, 1.10),
            saturation=(0.92, 1.08),
            hue=0.01,
            noise=0.01,
            blur_probability=0.0,
        )
    if tier == "medium":
        return _sample(
            tier,
            seed,
            rng,
            front_xy=0.08,
            front_z=0.05,
            look_xy=0.04,
            look_z=0.02,
            front_fovy=7.0,
            wrist_position=0.005,
            wrist_rotation=3.0,
            wrist_fovy=3.0,
            light_position=0.50,
            diffuse=(0.5, 1.5),
            ambient=(0.0, 0.25),
            tint=0.12,
            friction=(0.6, 1.4),
            colors=TRAIN_TABLE_COLORS,
            brightness=(0.75, 1.25),
            contrast=(0.75, 1.25),
            saturation=(0.80, 1.20),
            hue=0.03,
            noise=0.02,
            blur_probability=0.20,
        )
    if tier == "blind":
        return _sample_blind(tier, seed, rng)
    if tier == "stress":
        return _sample(
            tier,
            seed,
            rng,
            front_xy=0.15,
            front_z=0.10,
            look_xy=0.07,
            look_z=0.04,
            front_fovy=11.0,
            wrist_position=0.010,
            wrist_rotation=6.0,
            wrist_fovy=6.0,
            light_position=0.70,
            diffuse=(0.35, 1.70),
            ambient=(0.0, 0.35),
            tint=0.18,
            friction=(0.45, 1.55),
            colors=BLIND_TABLE_COLORS,
            brightness=(0.60, 1.35),
            contrast=(0.60, 1.35),
            saturation=(0.70, 1.30),
            hue=0.05,
            noise=0.03,
            blur_probability=0.35,
        )
    raise ValueError(f"Unknown domain-randomization tier: {tier}")


def _sample(
    tier: str,
    seed: int,
    rng: np.random.Generator,
    *,
    front_xy: float,
    front_z: float,
    look_xy: float,
    look_z: float,
    front_fovy: float,
    wrist_position: float,
    wrist_rotation: float,
    wrist_fovy: float,
    light_position: float,
    diffuse: tuple[float, float],
    ambient: tuple[float, float],
    tint: float,
    friction: tuple[float, float],
    colors: tuple[tuple[float, float, float], ...],
    brightness: tuple[float, float],
    contrast: tuple[float, float],
    saturation: tuple[float, float],
    hue: float,
    noise: float,
    blur_probability: float,
) -> DomainRandomizationSample:
    def uniform(low: float, high: float) -> float:
        return float(rng.uniform(low, high))

    return DomainRandomizationSample(
        tier=tier,
        seed=seed,
        front_position_offset_m=(uniform(-front_xy, front_xy), uniform(-front_xy, front_xy), uniform(-front_z, front_z)),
        front_look_at_offset_m=(uniform(-look_xy, look_xy), uniform(-look_xy, look_xy), uniform(-look_z, look_z)),
        front_fovy_offset_deg=uniform(-front_fovy, front_fovy),
        wrist_position_offset_m=(uniform(-wrist_position, wrist_position), uniform(-wrist_position, wrist_position), uniform(-wrist_position, wrist_position)),
        wrist_rotation_deg=(uniform(-wrist_rotation, wrist_rotation), uniform(-wrist_rotation, wrist_rotation), uniform(-wrist_rotation, wrist_rotation)),
        wrist_fovy_offset_deg=uniform(-wrist_fovy, wrist_fovy),
        light_position_offset_m=(uniform(-light_position, light_position), uniform(-light_position, light_position), uniform(-light_position, light_position)),
        light_diffuse_scale=uniform(*diffuse),
        light_ambient=uniform(*ambient),
        light_tint=(uniform(1.0 - tint, 1.0 + tint), uniform(1.0 - tint, 1.0 + tint), uniform(1.0 - tint, 1.0 + tint)),
        table_rgb=colors[int(rng.integers(len(colors)))],
        table_friction_scale=uniform(*friction),
        brightness=uniform(*brightness),
        contrast=uniform(*contrast),
        saturation=uniform(*saturation),
        hue_shift=uniform(-hue, hue),
        gaussian_noise_std=uniform(0.0, noise),
        blur_probability=blur_probability,
    )


def _sample_blind(tier: str, seed: int, rng: np.random.Generator) -> DomainRandomizationSample:
    """Use held-out materials and edge combinations near/just beyond training."""

    def sign() -> float:
        return -1.0 if int(rng.integers(2)) else 1.0

    diffuse = (float(rng.uniform(0.40, 0.60)) if int(rng.integers(2)) else float(rng.uniform(1.40, 1.60)))
    friction = (float(rng.uniform(0.50, 0.70)) if int(rng.integers(2)) else float(rng.uniform(1.30, 1.50)))
    return DomainRandomizationSample(
        tier=tier,
        seed=seed,
        front_position_offset_m=(sign() * float(rng.uniform(0.06, 0.10)), sign() * float(rng.uniform(0.06, 0.10)), sign() * float(rng.uniform(0.04, 0.07))),
        front_look_at_offset_m=(sign() * float(rng.uniform(0.03, 0.05)), sign() * float(rng.uniform(0.03, 0.05)), sign() * float(rng.uniform(0.01, 0.03))),
        front_fovy_offset_deg=sign() * float(rng.uniform(6.0, 9.0)),
        wrist_position_offset_m=tuple(sign() * float(rng.uniform(0.004, 0.007)) for _ in range(3)),
        wrist_rotation_deg=tuple(sign() * float(rng.uniform(2.0, 4.0)) for _ in range(3)),
        wrist_fovy_offset_deg=sign() * float(rng.uniform(2.0, 4.0)),
        light_position_offset_m=tuple(sign() * float(rng.uniform(0.35, 0.55)) for _ in range(3)),
        light_diffuse_scale=diffuse,
        light_ambient=float(rng.uniform(0.20, 0.30)),
        light_tint=tuple(float(rng.uniform(0.85, 1.15)) for _ in range(3)),
        table_rgb=BLIND_TABLE_COLORS[int(rng.integers(len(BLIND_TABLE_COLORS)))],
        table_friction_scale=friction,
        brightness=float(rng.uniform(0.70, 1.30)),
        contrast=float(rng.uniform(0.70, 1.30)),
        saturation=float(rng.uniform(0.75, 1.25)),
        hue_shift=float(rng.uniform(-0.04, 0.04)),
        gaussian_noise_std=float(rng.uniform(0.01, 0.025)),
        blur_probability=0.25,
    )


def capture_render_baseline(backend: Any) -> RenderBaseline | None:
    model = getattr(getattr(backend, "sim", None), "model", None)
    required_attributes = (
        "cam_pos",
        "cam_quat",
        "cam_fovy",
        "light_pos",
        "light_diffuse",
        "light_ambient",
        "light_specular",
    )
    if model is None or any(not hasattr(model, attribute) for attribute in required_attributes):
        # Unit-test and non-MuJoCo backends can implement task state without a
        # mutable render model. They should retain task behavior while simply
        # opting out of visual randomization.
        return None
    table_rgba: dict[int, NDArray[np.float64]] = {}
    table_friction: dict[int, NDArray[np.float64]] = {}
    for index in range(int(getattr(model, "ngeom", 0))):
        name = model.geom_id2name(index) or ""
        if "table" in name.lower():
            table_rgba[index] = np.asarray(model.geom_rgba[index], dtype=np.float64).copy()
            table_friction[index] = np.asarray(model.geom_friction[index], dtype=np.float64).copy()
    return RenderBaseline(
        camera_positions=np.asarray(model.cam_pos, dtype=np.float64).copy(),
        camera_quaternions=np.asarray(model.cam_quat, dtype=np.float64).copy(),
        camera_fovy=np.asarray(model.cam_fovy, dtype=np.float64).copy(),
        light_positions=np.asarray(model.light_pos, dtype=np.float64).copy(),
        light_diffuse=np.asarray(model.light_diffuse, dtype=np.float64).copy(),
        light_ambient=np.asarray(model.light_ambient, dtype=np.float64).copy(),
        light_specular=np.asarray(model.light_specular, dtype=np.float64).copy(),
        table_rgba=table_rgba,
        table_friction=table_friction,
    )


def apply_domain_randomization(
    backend: Any,
    baseline: RenderBaseline | None,
    sample: DomainRandomizationSample | None,
    *,
    front_camera_name: str,
    front_look_at_m: Iterable[float],
    wrist_camera_name: str = "robot0_eye_in_hand",
) -> None:
    """Restore baseline state, then apply a sample to mutable MuJoCo fields."""

    if baseline is None:
        return
    model = getattr(getattr(backend, "sim", None), "model", None)
    if model is None:
        return
    model.cam_pos[:] = baseline.camera_positions
    model.cam_quat[:] = baseline.camera_quaternions
    model.cam_fovy[:] = baseline.camera_fovy
    model.light_pos[:] = baseline.light_positions
    model.light_diffuse[:] = baseline.light_diffuse
    model.light_ambient[:] = baseline.light_ambient
    model.light_specular[:] = baseline.light_specular
    for index, value in baseline.table_rgba.items():
        model.geom_rgba[index] = value
    for index, value in baseline.table_friction.items():
        model.geom_friction[index] = value
    if sample is None or sample.tier == "nominal":
        backend.sim.forward()
        return

    front_id = model.camera_name2id(front_camera_name)
    nominal_position = baseline.camera_positions[front_id]
    position = nominal_position + np.asarray(sample.front_position_offset_m, dtype=np.float64)
    look_at = np.asarray(front_look_at_m, dtype=np.float64) + np.asarray(sample.front_look_at_offset_m, dtype=np.float64)
    model.cam_pos[front_id] = position
    model.cam_quat[front_id] = look_at_quaternion(position, look_at)
    model.cam_fovy[front_id] = np.clip(baseline.camera_fovy[front_id] + sample.front_fovy_offset_deg, 20.0, 120.0)

    try:
        wrist_id = model.camera_name2id(wrist_camera_name)
    except (KeyError, ValueError):
        wrist_id = -1
    if wrist_id >= 0:
        model.cam_pos[wrist_id] = baseline.camera_positions[wrist_id] + np.asarray(sample.wrist_position_offset_m, dtype=np.float64)
        model.cam_quat[wrist_id] = quaternion_multiply(
            baseline.camera_quaternions[wrist_id], euler_xyz_quaternion(sample.wrist_rotation_deg)
        )
        model.cam_fovy[wrist_id] = np.clip(baseline.camera_fovy[wrist_id] + sample.wrist_fovy_offset_deg, 30.0, 140.0)

    for light_id in range(int(getattr(model, "nlight", 0))):
        model.light_pos[light_id] = baseline.light_positions[light_id] + np.asarray(sample.light_position_offset_m, dtype=np.float64)
        model.light_diffuse[light_id] = np.clip(baseline.light_diffuse[light_id] * sample.light_diffuse_scale * np.asarray(sample.light_tint), 0.0, 1.0)
        model.light_ambient[light_id] = np.clip(sample.light_ambient * np.asarray(sample.light_tint), 0.0, 1.0)
    for index, rgba in baseline.table_rgba.items():
        model.geom_rgba[index, :3] = np.asarray(sample.table_rgb, dtype=np.float64)
        model.geom_rgba[index, 3] = rgba[3]
        model.geom_friction[index] = baseline.table_friction[index] * sample.table_friction_scale
    backend.sim.forward()


def policy_observation_randomized(
    observation: Mapping[str, NDArray[Any]], sample: DomainRandomizationSample | None, step: int
) -> dict[str, NDArray[Any]]:
    """Apply deterministic photometric-only perturbations to policy images."""

    if sample is None or sample.tier == "nominal":
        return dict(observation)
    transformed: dict[str, NDArray[Any]] = dict(observation)
    for index, (key, value) in enumerate(observation.items()):
        image = np.asarray(value)
        if image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1] == 3:
            transformed[key] = photometric_randomize(image, sample, stream=index + step * 17)
    return transformed


def photometric_randomize(
    image: NDArray[np.uint8], sample: DomainRandomizationSample, *, stream: int) -> NDArray[np.uint8]:
    pixels = image.astype(np.float32) / 255.0
    gray = pixels.mean(axis=-1, keepdims=True)
    pixels = (pixels - gray) * sample.saturation + gray
    pixels = (pixels - 0.5) * sample.contrast + 0.5
    pixels *= sample.brightness
    if sample.hue_shift:
        pixels = np.roll(pixels, 1 if sample.hue_shift > 0 else -1, axis=-1) * abs(sample.hue_shift) + pixels * (1.0 - abs(sample.hue_shift))
    rng = np.random.default_rng(sample.seed + stream)
    if sample.gaussian_noise_std:
        pixels += rng.normal(0.0, sample.gaussian_noise_std, size=pixels.shape)
    if sample.blur_probability and float(rng.random()) < sample.blur_probability:
        pixels = _box_blur_3x3(pixels)
    return np.clip(pixels * 255.0, 0.0, 255.0).astype(np.uint8)


def points_visible(
    points_m: Iterable[Iterable[float]],
    position_m: Iterable[float],
    look_at_m: Iterable[float],
    fovy_deg: float,
    *,
    aspect_ratio: float = 1.0,
    margin_fraction: float = 10.0 / 256.0,
) -> bool:
    """Analytic visibility check for task landmarks before expensive rollout."""

    rotation = quaternion_to_matrix(look_at_quaternion(np.asarray(position_m), np.asarray(look_at_m)))
    tangent_y = float(np.tan(np.deg2rad(fovy_deg) / 2.0))
    tangent_x = tangent_y * aspect_ratio
    position = np.asarray(position_m, dtype=np.float64)
    for point in points_m:
        local = rotation.T @ (np.asarray(point, dtype=np.float64) - position)
        depth = -float(local[2])
        if depth <= 1e-6:
            return False
        x = float(local[0]) / depth / tangent_x
        y = float(local[1]) / depth / tangent_y
        if abs(x) > 1.0 - 2.0 * margin_fraction or abs(y) > 1.0 - 2.0 * margin_fraction:
            return False
    return True


def look_at_quaternion(position_m: NDArray[np.float64], target_m: NDArray[np.float64]) -> NDArray[np.float64]:
    forward = np.asarray(target_m, dtype=np.float64) - np.asarray(position_m, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    x_axis = np.cross(np.asarray([0.0, 0.0, 1.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return matrix_to_quaternion(np.column_stack((x_axis, y_axis, z_axis)))


def matrix_to_quaternion(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.asarray([0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale])
    else:
        index = int(np.argmax(np.diag(matrix)))
        next_index, last_index = (index + 1) % 3, (index + 2) % 3
        scale = 2.0 * np.sqrt(1.0 + matrix[index, index] - matrix[next_index, next_index] - matrix[last_index, last_index])
        quat = np.zeros(4, dtype=np.float64)
        quat[index + 1] = 0.25 * scale
        quat[0] = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
        quat[next_index + 1] = (matrix[next_index, index] + matrix[index, next_index]) / scale
        quat[last_index + 1] = (matrix[last_index, index] + matrix[index, last_index]) / scale
    return quat / np.linalg.norm(quat)


def quaternion_to_matrix(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_multiply(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = np.asarray([lw * rw - lx * rx - ly * ry - lz * rz, lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw])
    return result / np.linalg.norm(result)


def euler_xyz_quaternion(degrees: Iterable[float]) -> NDArray[np.float64]:
    half = np.deg2rad(np.asarray(tuple(degrees), dtype=np.float64)) / 2.0
    cx, cy, cz = np.cos(half)
    sx, sy, sz = np.sin(half)
    return np.asarray([cx * cy * cz + sx * sy * sz, sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz], dtype=np.float64)


def _box_blur_3x3(pixels: NDArray[np.float32]) -> NDArray[np.float32]:
    padded = np.pad(pixels, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return sum(padded[row : row + pixels.shape[0], column : column + pixels.shape[1]] for row in range(3) for column in range(3)) / 9.0
