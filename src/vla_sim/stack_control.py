"""Deployable perception and safety supervision for the Stack task."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray


class StackPhase(str, Enum):
    APPROACH = "approach"
    GRASP = "grasp"
    GRIPPER_CLOSE = "gripper_close"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    RELEASE = "release"
    VERIFY = "verify"
    DONE = "done"
    FAILED = "failed"


PHASE_PROMPTS = {
    StackPhase.APPROACH: "move above the grasp object",
    StackPhase.GRASP: "move down to grasp object",
    StackPhase.GRIPPER_CLOSE: "hold gripper closed",
    StackPhase.LIFT: "lift the grasped object",
    StackPhase.TRANSPORT: "move object above target",
    StackPhase.PLACE: "lower object onto target",
    StackPhase.RELEASE: "hold position for release",
    StackPhase.VERIFY: "move upward after release",
    StackPhase.DONE: "hold after successful stack",
    StackPhase.FAILED: "hold safely",
}

TASK_INSTRUCTIONS = {
    "red_on_blue": "stack the red block on the blue block",
    "blue_on_red": "stack the blue block on the red block",
    "red_to_storage_bin": "place the red cube in the blue storage bin",
}

TASK_OBJECTS = {
    "red_on_blue": ("red", "blue"),
    "blue_on_red": ("blue", "red"),
    "red_to_storage_bin": ("red", "blue storage bin"),
}

PHASE_GROUP_BY_PHASE = {
    StackPhase.APPROACH: "approach",
    StackPhase.GRASP: "grasp",
    StackPhase.GRIPPER_CLOSE: "grasp",
    StackPhase.LIFT: "lift",
    StackPhase.TRANSPORT: "transport",
    StackPhase.PLACE: "place_release",
    StackPhase.RELEASE: "place_release",
    StackPhase.VERIFY: "place_release",
    StackPhase.DONE: "place_release",
    StackPhase.FAILED: "failed",
}


def task_phase_prompt(task: str, group: str) -> str:
    """Return the task-and-phase prompt shared by training and rollout."""

    if task not in TASK_OBJECTS:
        raise ValueError(f"task must be one of {sorted(TASK_OBJECTS)}")
    pick, target = TASK_OBJECTS[task]
    prompts = {
        "approach": f"move above {pick} block",
        "grasp": f"move down and grasp {pick} block",
        "lift": f"lift {pick} block",
        "transport": f"move {pick} block above {target} block",
        "place_release": f"place {pick} block on {target} and release",
        "failed": "hold safely",
    }
    if task == "red_to_storage_bin":
        prompts["approach"] = "move above the red cube"
        prompts["grasp"] = "move down and grasp the red cube"
        prompts["lift"] = "lift the red cube"
        prompts["transport"] = "move the red cube above the blue storage bin"
        prompts["place_release"] = "place the red cube in the blue storage bin and release"
    if group not in prompts:
        raise ValueError(f"Unknown Stack phase group: {group}")
    return prompts[group]


@dataclass(frozen=True)
class ObjectPoseEstimate:
    pick_xyz: NDArray[np.float64] | None
    target_xyz: NDArray[np.float64] | None
    confidence: float
    target_confidence: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        target_confidence = (
            self.confidence if self.target_confidence is None else self.target_confidence
        )
        if not 0.0 <= target_confidence <= 1.0:
            raise ValueError("target_confidence must be in [0, 1]")
        object.__setattr__(self, "target_confidence", target_confidence)
        for field_name in ("pick_xyz", "target_xyz"):
            value = getattr(self, field_name)
            if value is None:
                continue
            position = np.asarray(value, dtype=np.float64).reshape(-1)
            if position.size != 3 or not np.all(np.isfinite(position)):
                raise ValueError("Object positions must each contain three finite values")
            object.__setattr__(self, field_name, position)
        if self.pick_xyz is None and self.target_xyz is None:
            raise ValueError("At least one object position must be available")


class ObjectPoseProvider(Protocol):
    """Estimate task object poses from deployable sensors, never simulator state."""

    def estimate(
        self, observation: Mapping[str, Any], *, task: str, simulator: Any
    ) -> ObjectPoseEstimate | None: ...


@dataclass(frozen=True)
class ColorDepthPoseConfig:
    camera_name: str = "agentview"
    flip_vertical: bool = True
    minimum_pixels: int = 20
    dominance_ratio: float = 1.25
    minimum_channel: int = 70
    world_offset_m: tuple[float, float, float] = (-0.011225, 0.000990, -0.013923)


class ColorDepthObjectPoseProvider:
    """Locate red/blue objects with RGB masks and calibrated depth unprojection."""

    def __init__(self, config: ColorDepthPoseConfig | None = None) -> None:
        self.config = config or ColorDepthPoseConfig()

    def estimate(
        self, observation: Mapping[str, Any], *, task: str, simulator: Any
    ) -> ObjectPoseEstimate | None:
        image_key = f"{self.config.camera_name}_image"
        depth_key = f"{self.config.camera_name}_depth"
        if image_key not in observation or depth_key not in observation:
            return None
        image = np.asarray(observation[image_key])
        depth = np.asarray(observation[depth_key], dtype=np.float64).squeeze()
        if image.ndim != 3 or image.shape[-1] != 3 or depth.shape != image.shape[:2]:
            return None
        if self.config.flip_vertical:
            image = np.flipud(image)
            depth = np.flipud(depth)

        red = self._mask(image, 0)
        blue = self._mask(image, 2)
        green = self._mask(image, 1)
        red_xyz, red_confidence = self._unproject(red, depth, simulator)
        blue_xyz, blue_confidence = self._unproject(blue, depth, simulator)
        green_xyz, green_confidence = self._unproject(green, depth, simulator)
        if red_xyz is None and blue_xyz is None and green_xyz is None:
            return None
        if task == "red_to_storage_bin":
            target_value = observation.get("target_zone_pos")
            target = None if target_value is None else np.asarray(target_value, dtype=np.float64)[:3]
            pick, target = red_xyz, target
            pick_confidence, target_confidence = red_confidence, 1.0 if target is not None else 0.0
        elif task == "blue_on_red":
            pick, target = blue_xyz, red_xyz
            pick_confidence, target_confidence = blue_confidence, red_confidence
        else:
            pick, target = red_xyz, blue_xyz
            pick_confidence, target_confidence = red_confidence, blue_confidence
        return ObjectPoseEstimate(pick, target, pick_confidence, target_confidence)

    def _mask(self, image: NDArray[Any], channel: int) -> NDArray[np.bool_]:
        pixels = image.astype(np.float64)
        selected = pixels[..., channel]
        others = np.delete(pixels, channel, axis=-1)
        return (selected >= self.config.minimum_channel) & (
            selected[..., None] >= self.config.dominance_ratio * others
        ).all(axis=-1)

    def _unproject(
        self, mask: NDArray[np.bool_], depth: NDArray[np.float64], simulator: Any
    ) -> tuple[NDArray[np.float64] | None, float]:
        rows, columns = np.nonzero(mask)
        if len(rows) < self.config.minimum_pixels:
            return None, 0.0
        from robosuite.utils.camera_utils import (  # imported only with simulator runtime
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
            get_real_depth_map,
        )

        metric_depth = np.asarray(get_real_depth_map(simulator, depth), dtype=np.float64)
        values = metric_depth[rows, columns]
        valid = np.isfinite(values) & (values > 0)
        if int(valid.sum()) < self.config.minimum_pixels:
            return None, 0.0
        rows = rows[valid]
        columns = columns[valid]
        z = float(np.median(values[valid]))
        row = float(np.median(rows))
        column = float(np.median(columns))
        height, width = depth.shape
        intrinsic = get_camera_intrinsic_matrix(
            simulator, self.config.camera_name, camera_height=height, camera_width=width
        )
        extrinsic = get_camera_extrinsic_matrix(simulator, self.config.camera_name)
        camera_point = np.asarray(
            [
                (column - intrinsic[0, 2]) * z / intrinsic[0, 0],
                (row - intrinsic[1, 2]) * z / intrinsic[1, 1],
                z,
                1.0,
            ]
        )
        world = extrinsic @ camera_point
        confidence = min(1.0, float(valid.sum()) / (4 * self.config.minimum_pixels))
        calibrated = world[:3] + np.asarray(self.config.world_offset_m, dtype=np.float64)
        return calibrated.astype(np.float64), confidence


@dataclass(frozen=True)
class StackSupervisorConfig:
    confidence_threshold: float = 0.8
    confidence_frames: int = 2
    # RGB-D calibration removes the former 11-14 mm systematic pose offset.
    # A 30 mm gate now covers the observed reachable 18-27 mm approach band
    # without reproducing the old uncalibrated 35-47 mm premature closes.
    grasp_distance_m: float = 0.030
    approach_xy_m: float = 0.040
    lift_height_m: float = 0.05
    target_xy_m: float = 0.025
    place_reacquire_xy_m: float = 0.035
    target_pose_ema_alpha: float = 0.2
    pick_visual_servo_gain: float = 0.0
    pick_visual_servo_max_action: float = 0.8
    transport_visual_servo_gain: float = 0.0
    transport_visual_servo_max_action: float = 0.8
    release_height_error_m: float = 0.018
    place_height_offset_m: float = 0.040
    max_retries: int = 2
    unreliable_retry_steps: int = 10
    open_command: float = -1.0
    close_command: float = 1.0
    workspace_low: tuple[float, float, float] = (-0.25, -0.25, 0.76)
    workspace_high: tuple[float, float, float] = (0.25, 0.25, 1.20)
    position_action_scale_m: float = 0.05
    gripper_dwell_steps: int = 5
    grasp_confirmation_threshold: float = 0.30
    attachment_confirmation_lift_m: float = 0.012
    # The phase-conditioned lift policy needs roughly 40-90 control steps to
    # produce a visually reliable 12 mm rise. Keep the physical rise check,
    # but do not retry a valid grasp before that motion can develop.
    attachment_confirmation_steps: int = 35
    release_opening_threshold: float = 0.25
    grasp_retry_max: int = 2
    phase_timeout_steps: int = 180
    phase_timeouts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.gripper_dwell_steps < 0:
            raise ValueError("gripper_dwell_steps must be non-negative")
        if self.place_reacquire_xy_m <= self.target_xy_m:
            raise ValueError("place_reacquire_xy_m must exceed target_xy_m")
        if not 0.0 < self.target_pose_ema_alpha <= 1.0:
            raise ValueError("target_pose_ema_alpha must be in (0, 1]")
        if self.pick_visual_servo_gain < 0.0:
            raise ValueError("pick_visual_servo_gain must be non-negative")
        if not 0.0 < self.pick_visual_servo_max_action <= 1.0:
            raise ValueError("pick_visual_servo_max_action must be in (0, 1]")
        if self.transport_visual_servo_gain < 0.0:
            raise ValueError("transport_visual_servo_gain must be non-negative")
        if not 0.0 < self.transport_visual_servo_max_action <= 1.0:
            raise ValueError("transport_visual_servo_max_action must be in (0, 1]")
        if not 0.0 < self.grasp_confirmation_threshold <= 1.0:
            raise ValueError("grasp_confirmation_threshold must be in (0, 1]")
        if self.attachment_confirmation_lift_m <= 0:
            raise ValueError("attachment_confirmation_lift_m must be positive")
        if self.attachment_confirmation_steps < 1:
            raise ValueError("attachment_confirmation_steps must be at least 1")
        if not 0.0 <= self.release_opening_threshold < 1.0:
            raise ValueError("release_opening_threshold must be in [0, 1)")
        if self.grasp_retry_max < 0:
            raise ValueError("grasp_retry_max must be non-negative")
        if self.phase_timeout_steps < 1:
            raise ValueError("phase_timeout_steps must be at least 1")
        if self.phase_timeouts is not None:
            for phase_name, timeout in self.phase_timeouts.items():
                if timeout < 1:
                    raise ValueError(
                        f"phase_timeouts[{phase_name!r}] must be at least 1"
                    )
        if self.place_height_offset_m <= 0:
            raise ValueError("place_height_offset_m must be positive")


class StackSupervisor:
    """Gate VLA actions using sensor-derived state and robot feedback."""

    def __init__(
        self,
        config: StackSupervisorConfig | None = None,
        *,
        task: str = "red_on_blue",
    ) -> None:
        if task not in TASK_INSTRUCTIONS:
            raise ValueError(f"task must be one of {sorted(TASK_INSTRUCTIONS)}")
        self.config = config or StackSupervisorConfig()
        self.task = task
        self.reset()

    def reset(self) -> None:
        self.phase = StackPhase.APPROACH
        self.retries = 0
        self._confidence_streak = 0
        self._unreliable_steps = 0
        self._initial_pick_z: float | None = None
        self._grasp_retries: int = 0
        self._phase_step: int = 0
        self._eef_to_pick_offset: NDArray[np.float64] | None = None
        self._prelift_pick_z: float | None = None
        self._attachment_confirmed = False
        self._last_pick_xyz: NDArray[np.float64] | None = None
        self._last_target_xyz: NDArray[np.float64] | None = None
        self.last_perception_reliable = False
        self.retry_epoch = 0

    @property
    def prompt(self) -> str:
        return task_phase_prompt(self.task, PHASE_GROUP_BY_PHASE[self.phase])

    def _start_grasp_retry(self) -> None:
        if self._grasp_retries >= self.config.grasp_retry_max:
            self.phase = StackPhase.FAILED
            return
        self._grasp_retries += 1
        self.retry_epoch += 1
        self._initial_pick_z = None
        self._prelift_pick_z = None
        self._eef_to_pick_offset = None
        self._attachment_confirmed = False
        self.phase = StackPhase.APPROACH
        self._phase_step = 0

    def filter_action(
        self,
        action: NDArray[Any],
        *,
        eef_xyz: NDArray[Any],
        estimate: ObjectPoseEstimate | None,
        gripper_opening: float,
    ) -> NDArray[np.float32]:
        filtered = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        eef = np.asarray(eef_xyz, dtype=np.float64).reshape(-1)
        if filtered.size < 7 or eef.size != 3:
            raise ValueError("Stack supervisor requires a 7-D action and 3-D EEF position")
        filtered[3:6] = 0.0

        pick_visible = (
            estimate is not None
            and estimate.pick_xyz is not None
            and estimate.confidence >= self.config.confidence_threshold
        )
        target_visible = (
            estimate is not None
            and estimate.target_xyz is not None
            and estimate.target_confidence is not None
            and estimate.target_confidence >= self.config.confidence_threshold
        )
        if pick_visible:
            assert estimate is not None and estimate.pick_xyz is not None
            self._last_pick_xyz = estimate.pick_xyz.copy()
        if target_visible:
            assert estimate is not None and estimate.target_xyz is not None
            if self._last_target_xyz is None:
                self._last_target_xyz = estimate.target_xyz.copy()
            else:
                alpha = self.config.target_pose_ema_alpha
                self._last_target_xyz = (
                    (1.0 - alpha) * self._last_target_xyz
                    + alpha * estimate.target_xyz
                )

        visual_pick = self._last_pick_xyz if pick_visible else None
        tracked_pick = (
            eef + self._eef_to_pick_offset
            if self._attachment_confirmed and self._eef_to_pick_offset is not None
            else visual_pick
        )
        target = self._last_target_xyz if target_visible else None
        needs_pick = self.phase in {StackPhase.APPROACH, StackPhase.GRASP} or (
            self.phase is StackPhase.LIFT and not self._attachment_confirmed
        )
        needs_target = self.phase in {StackPhase.TRANSPORT, StackPhase.PLACE}
        reliable = (not needs_pick or visual_pick is not None) and (
            not needs_target or target is not None
        )
        self.last_perception_reliable = reliable
        self._confidence_streak = self._confidence_streak + 1 if pick_visible else 0
        if not reliable:
            self._unreliable_steps += 1
            if self._unreliable_steps >= self.config.unreliable_retry_steps:
                if self.retries < self.config.max_retries:
                    self.retries += 1
                    self.retry_epoch += 1
                    self.phase = StackPhase.APPROACH
                    self._unreliable_steps = 0
                    self._initial_pick_z = None
                    self._prelift_pick_z = None
                    self._eef_to_pick_offset = None
                    self._attachment_confirmed = False
                    self._phase_step = 0
                else:
                    self.phase = StackPhase.FAILED
            filtered[:3] = 0.0
            filtered[6] = self.config.open_command
            self._phase_step += 1
            return filtered[:7]

        self._unreliable_steps = 0
        pick = tracked_pick
        if pick is None:
            pick = self._last_pick_xyz
        if self._initial_pick_z is None:
            if pick is not None:
                self._initial_pick_z = float(pick[2])
        pick_distance = float(np.linalg.norm(eef - pick)) if pick is not None else float("inf")
        pick_xy = (
            float(np.linalg.norm(eef[:2] - pick[:2])) if pick is not None else float("inf")
        )
        target_xy = (
            float(np.linalg.norm(pick[:2] - target[:2]))
            if pick is not None and target is not None
            else float("inf")
        )
        object_height = (
            max(0.001, float(pick[2] - self._initial_pick_z))
            if pick is not None and self._initial_pick_z is not None
            else 0.0
        )

        if self.phase is StackPhase.APPROACH and pick_xy <= self.config.approach_xy_m:
            self.phase = StackPhase.GRASP
            self._phase_step = 0
        if self.phase is StackPhase.GRASP and (
            pick_distance <= self.config.grasp_distance_m
            and self._confidence_streak >= self.config.confidence_frames
        ):
            self.phase = StackPhase.GRIPPER_CLOSE
            self._phase_step = 0
        if self.phase is StackPhase.GRIPPER_CLOSE and self._phase_step >= self.config.gripper_dwell_steps:
            if gripper_opening >= self.config.grasp_confirmation_threshold:
                if self._last_pick_xyz is not None:
                    self._prelift_pick_z = float(self._last_pick_xyz[2])
                self.phase = StackPhase.LIFT
                self._phase_step = 0
            else:
                self._start_grasp_retry()
        if (
            self.phase is StackPhase.LIFT
            and not self._attachment_confirmed
            and pick_visible
            and pick is not None
            and self._prelift_pick_z is not None
            and float(pick[2] - self._prelift_pick_z)
            >= self.config.attachment_confirmation_lift_m
        ):
            self._attachment_confirmed = True
            self._eef_to_pick_offset = pick - eef
        if (
            self.phase is StackPhase.LIFT
            and not self._attachment_confirmed
            and self._phase_step >= self.config.attachment_confirmation_steps
        ):
            self._start_grasp_retry()
        if (
            self.phase is StackPhase.LIFT
            and self._attachment_confirmed
            and object_height >= self.config.lift_height_m
        ):
            self.phase = StackPhase.TRANSPORT
            self._phase_step = 0
        if self.phase is StackPhase.TRANSPORT and target_xy <= self.config.target_xy_m:
            self.phase = StackPhase.PLACE
            self._phase_step = 0
        if (
            self.phase is StackPhase.PLACE
            and target_xy > self.config.place_reacquire_xy_m
        ):
            self.phase = StackPhase.TRANSPORT
            self._phase_step = 0
        expected_z = float(target[2] + self.config.place_height_offset_m) if target is not None else float("inf")
        if self.phase is StackPhase.PLACE and (
            target_xy <= self.config.target_xy_m
            and pick is not None
            and abs(float(pick[2]) - expected_z) <= self.config.release_height_error_m
        ):
            self.phase = StackPhase.RELEASE
            self._phase_step = 0
        if (
            self.phase is StackPhase.RELEASE
            and gripper_opening <= self.config.release_opening_threshold
        ):
            self.phase = StackPhase.VERIFY
            self._phase_step = 0

        # Early manipulation timeouts trigger a safe retry; later failures stop.
        if self.phase not in {StackPhase.DONE, StackPhase.FAILED}:
            timeout = (self.config.phase_timeouts or {}).get(
                self.phase.value, self.config.phase_timeout_steps
            )
            if self._phase_step >= timeout:
                if self.phase in {StackPhase.APPROACH, StackPhase.GRASP, StackPhase.LIFT}:
                    self._start_grasp_retry()
                else:
                    self.phase = StackPhase.FAILED

        self._phase_step += 1

        closed = self.phase in {
            StackPhase.GRIPPER_CLOSE,
            StackPhase.LIFT,
            StackPhase.TRANSPORT,
            StackPhase.PLACE,
        }
        filtered[6] = self.config.close_command if closed else self.config.open_command
        if self.phase in {StackPhase.GRIPPER_CLOSE, StackPhase.RELEASE, StackPhase.FAILED}:
            filtered[:3] = 0.0
        elif self.phase is StackPhase.VERIFY:
            filtered[0:2] = 0.0
            filtered[2] = 0.02  # Ultra-gentle upward retract after fully opening
        else:
            if (
                self.phase in {StackPhase.APPROACH, StackPhase.GRASP}
                and visual_pick is not None
                and self.config.pick_visual_servo_gain > 0.0
            ):
                correction = (
                    self.config.pick_visual_servo_gain
                    * (visual_pick[:2] - eef[:2])
                    / self.config.position_action_scale_m
                )
                filtered[:2] = np.clip(
                    correction,
                    -self.config.pick_visual_servo_max_action,
                    self.config.pick_visual_servo_max_action,
                )
            if (
                self.phase is StackPhase.TRANSPORT
                and pick is not None
                and target is not None
                and self.config.transport_visual_servo_gain > 0.0
            ):
                correction = (
                    self.config.transport_visual_servo_gain
                    * (target[:2] - pick[:2])
                    / self.config.position_action_scale_m
                )
                filtered[:2] = np.clip(
                    correction,
                    -self.config.transport_visual_servo_max_action,
                    self.config.transport_visual_servo_max_action,
                )
            low = np.asarray(self.config.workspace_low)
            high = np.asarray(self.config.workspace_high)
            requested = eef + filtered[:3] * self.config.position_action_scale_m
            bounded = np.clip(requested, low, high)
            filtered[:3] = (bounded - eef) / self.config.position_action_scale_m
        return np.clip(filtered[:7], -1.0, 1.0).astype(np.float32)
