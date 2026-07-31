"""PickPlace-specific names over the shared visual phase supervisor."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .stack_control import StackSupervisor, StackSupervisorConfig


@dataclass(frozen=True)
class PickPlaceSupervisorConfig(StackSupervisorConfig):
    target_xy_m: float = 0.030
    place_reacquire_xy_m: float = 0.040
    pick_visual_servo_gain: float = 1.0
    pick_visual_servo_max_action: float = 0.8
    transport_visual_servo_gain: float = 1.0
    transport_visual_servo_max_action: float = 0.8
    place_height_offset_m: float = 0.027
    release_height_error_m: float = 0.015


class PickPlaceSupervisor(StackSupervisor):
    def __init__(self, config: PickPlaceSupervisorConfig | None = None) -> None:
        super().__init__(config or PickPlaceSupervisorConfig(), task="red_to_storage_bin")


@dataclass(frozen=True)
class VLAOnlySafetyConfig:
    """Non-semantic safety limits that do not use task-object poses."""

    workspace_low: tuple[float, float, float] = (-0.25, -0.25, 0.76)
    workspace_high: tuple[float, float, float] = (0.25, 0.25, 1.20)
    position_action_scale_m: float = 0.05


@dataclass(frozen=True)
class VLAOnlyActionCalibration:
    """Fixed model-output calibration that consumes no object or target pose."""

    closed_negative_y_gain: float = 1.0
    transport_positive_x_gain: float = 1.0
    close_command_threshold: float = 0.0
    transport_xy_action_threshold: float = 0.30
    transport_abs_z_action_max: float = 0.25
    transport_direction_y_threshold: float = 0.15
    transport_direction_lock_steps: int = 3

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.closed_negative_y_gain)
            or self.closed_negative_y_gain < 1.0
        ):
            raise ValueError("closed_negative_y_gain must be finite and at least 1")
        if (
            not math.isfinite(self.transport_positive_x_gain)
            or not 0.0 < self.transport_positive_x_gain <= 1.0
        ):
            raise ValueError("transport_positive_x_gain must be finite and in (0, 1]")
        if not math.isfinite(self.close_command_threshold):
            raise ValueError("close_command_threshold must be finite")
        if (
            not math.isfinite(self.transport_xy_action_threshold)
            or self.transport_xy_action_threshold < 0.0
        ):
            raise ValueError("transport_xy_action_threshold must be finite and non-negative")
        if (
            not math.isfinite(self.transport_abs_z_action_max)
            or self.transport_abs_z_action_max < 0.0
        ):
            raise ValueError("transport_abs_z_action_max must be finite and non-negative")
        if (
            not math.isfinite(self.transport_direction_y_threshold)
            or self.transport_direction_y_threshold < 0.0
        ):
            raise ValueError("transport_direction_y_threshold must be finite and non-negative")
        if self.transport_direction_lock_steps < 1:
            raise ValueError("transport_direction_lock_steps must be positive")


class VLAOnlyActionCalibrator:
    """Stateful phase-aware calibration using only the model's own action history."""

    def __init__(self, config: VLAOnlyActionCalibration | None = None) -> None:
        self.config = config or VLAOnlyActionCalibration()
        self.reset()

    def reset(self) -> None:
        self._transport_y_direction = 0
        self._transport_y_sum = 0.0
        self._transport_direction_samples = 0

    @property
    def transport_y_direction(self) -> int:
        return self._transport_y_direction

    def calibrate(self, action: NDArray[np.floating]) -> NDArray[np.float32]:
        calibrated = _validated_vla_only_action(action)
        config = self.config
        closed = calibrated[6] > config.close_command_threshold
        if not closed:
            self.reset()
            return calibrated

        horizontal_norm = float(np.linalg.norm(calibrated[:2]))
        horizontal_transport = (
            horizontal_norm >= config.transport_xy_action_threshold
            and abs(float(calibrated[2])) <= config.transport_abs_z_action_max
        )
        if not horizontal_transport:
            return calibrated

        if self._transport_y_direction == 0:
            self._transport_y_sum += float(calibrated[1])
            self._transport_direction_samples += 1
            if (
                self._transport_direction_samples >= config.transport_direction_lock_steps
                and abs(self._transport_y_sum) >= config.transport_direction_y_threshold
            ):
                self._transport_y_direction = 1 if self._transport_y_sum > 0.0 else -1

        if calibrated[0] > 0.0:
            calibrated[0] *= config.transport_positive_x_gain
        if self._transport_y_direction < 0 and calibrated[1] < 0.0:
            calibrated[1] *= config.closed_negative_y_gain
        return calibrated


def scene_policy_seed(base_seed: int, environment_seed: int) -> int:
    """Derive a stable positive PyTorch seed for one evaluation scene."""

    return int((base_seed + environment_seed) % (2**31 - 1))


def calibrate_vla_only_action(
    action: NDArray[np.floating],
    config: VLAOnlyActionCalibration | None = None,
) -> NDArray[np.float32]:
    """Calibrate one action; callers needing direction lock should reuse a calibrator."""

    return VLAOnlyActionCalibrator(config).calibrate(action)


def _validated_vla_only_action(
    action: NDArray[np.floating],
) -> NDArray[np.float32]:
    calibrated = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if calibrated.size < 7 or not np.all(np.isfinite(calibrated[:7])):
        raise ValueError("VLA-only calibration requires a finite 7-D action")
    return calibrated[:7]


def filter_vla_only_action(
    action: NDArray[np.floating],
    *,
    eef_xyz: NDArray[np.floating],
    config: VLAOnlySafetyConfig | None = None,
) -> NDArray[np.float32]:
    """Apply fixed-orientation and workspace safety without target corrections."""

    effective = config or VLAOnlySafetyConfig()
    filtered = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    eef = np.asarray(eef_xyz, dtype=np.float64).reshape(-1)
    if filtered.size < 7 or eef.size != 3:
        raise ValueError("VLA-only safety requires a 7-D action and 3-D EEF position")
    if not np.all(np.isfinite(filtered[:7])) or not np.all(np.isfinite(eef)):
        raise ValueError("VLA-only action and EEF position must be finite")
    filtered[3:6] = 0.0
    low = np.asarray(effective.workspace_low, dtype=np.float64)
    high = np.asarray(effective.workspace_high, dtype=np.float64)
    requested = eef + filtered[:3] * effective.position_action_scale_m
    bounded = np.clip(requested, low, high)
    filtered[:3] = (bounded - eef) / effective.position_action_scale_m
    return np.clip(filtered[:7], -1.0, 1.0).astype(np.float32)
