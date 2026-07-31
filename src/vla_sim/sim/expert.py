"""A small privileged-state heuristic for smoke-testing the lift task."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from vla_sim.stack_control import task_phase_prompt

from .contracts import DEFAULT_ACTION_SPEC, ContractError


class LiftPhase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    DONE = auto()


@dataclass(frozen=True)
class HeuristicExpertConfig:
    """Tunable values for the OSC-position lift heuristic."""

    approach_height_m: float = 0.10
    grasp_height_m: float = 0.015
    lift_distance_m: float = 0.14
    success_delta_m: float = 0.11
    position_action_scale_m: float = 0.05
    position_tolerance_m: float = 0.012
    place_release_tolerance_m: float = 0.020
    place_release_xy_tolerance_m: float | None = None
    close_steps: int = 12
    open_command: float = -1.0
    close_command: float = 1.0
    max_position_action: float = 0.8
    object_height_m: float = 0.05
    release_clearance_m: float = 0.001

    def __post_init__(self) -> None:
        if self.position_action_scale_m <= 0 or self.place_release_tolerance_m <= 0:
            raise ValueError("position_action_scale_m must be positive.")
        if (
            self.place_release_xy_tolerance_m is not None
            and self.place_release_xy_tolerance_m <= 0
        ):
            raise ValueError("place_release_xy_tolerance_m must be positive when provided.")
        if self.close_steps < 1:
            raise ValueError("close_steps must be at least one.")
        if self.object_height_m <= 0 or self.release_clearance_m < 0:
            raise ValueError("Object height must be positive and release clearance non-negative.")


class HeuristicLiftExpert:
    """State machine that approaches, closes the gripper, and lifts.

    It intentionally consumes robosuite's privileged ``cube_pos`` and
    ``robot0_eef_pos`` observations. It is a simulator smoke-test expert, not a
    policy input or a source of ground-truth data at deployment time.
    """

    def __init__(self, config: HeuristicExpertConfig | None = None) -> None:
        self.config = config or HeuristicExpertConfig()
        self.reset()

    def reset(self) -> None:
        self.phase = LiftPhase.APPROACH
        self._close_counter = 0
        self._initial_object_z: float | None = None
        self._lift_target: NDArray[np.float64] | None = None

    @property
    def done(self) -> bool:
        return self.phase is LiftPhase.DONE

    def act(self, raw_observation: Mapping[str, Any]) -> NDArray[np.float32]:
        cube = self._vector(raw_observation, "cube_pos", 3)
        eef = self._vector(raw_observation, "robot0_eef_pos", 3)
        if self._initial_object_z is None:
            self._initial_object_z = float(cube[2])

        if float(cube[2]) >= self._initial_object_z + self.config.success_delta_m:
            self.phase = LiftPhase.DONE

        action = np.zeros(7, dtype=np.float32)
        action[-1] = self.config.open_command

        if self.phase is LiftPhase.APPROACH:
            target = cube.copy()
            target[2] += self.config.approach_height_m
            action[:3] = self._position_delta(target, eef)
            if np.linalg.norm(target - eef) <= self.config.position_tolerance_m:
                self.phase = LiftPhase.DESCEND

        elif self.phase is LiftPhase.DESCEND:
            target = cube.copy()
            target[2] += self.config.grasp_height_m
            action[:3] = self._position_delta(target, eef)
            if np.linalg.norm(target - eef) <= self.config.position_tolerance_m:
                self.phase = LiftPhase.CLOSE

        elif self.phase is LiftPhase.CLOSE:
            action[-1] = self.config.close_command
            self._close_counter += 1
            if self._close_counter >= self.config.close_steps:
                self.phase = LiftPhase.LIFT
                self._lift_target = eef.copy()
                self._lift_target[2] += self.config.lift_distance_m

        elif self.phase is LiftPhase.LIFT:
            action[-1] = self.config.close_command
            if self._lift_target is None:
                self._lift_target = eef.copy()
                self._lift_target[2] += self.config.lift_distance_m
            action[:3] = self._position_delta(self._lift_target, eef)

        elif self.phase is LiftPhase.DONE:
            # Hold the grasp without issuing further Cartesian motion.
            action[-1] = self.config.close_command

        return DEFAULT_ACTION_SPEC.validate(action, clip=True)

    def _position_delta(
        self, target: NDArray[np.float64], current: NDArray[np.float64]
    ) -> NDArray[np.float32]:
        normalized = (target - current) / self.config.position_action_scale_m
        normalized = np.clip(
            normalized,
            -self.config.max_position_action,
            self.config.max_position_action,
        )
        return normalized.astype(np.float32)

    @staticmethod
    def _vector(
        observation: Mapping[str, Any], key: str, size: int
    ) -> NDArray[np.float64]:
        if key not in observation:
            available = ", ".join(sorted(observation.keys()))
            raise ContractError(
                f"Heuristic expert requires privileged key '{key}'. "
                f"Available keys: {available or '<none>'}."
            )
        value = np.asarray(observation[key], dtype=np.float64).reshape(-1)
        if value.size < size or not np.all(np.isfinite(value[:size])):
            raise ContractError(f"'{key}' must contain {size} finite values.")
        return value[:size].copy()


class PickPlacePhase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    TRANSPORT = auto()
    PLACE = auto()
    OPEN = auto()
    DONE = auto()


class HeuristicPickPlaceExpert:
    """Privileged expert for red cube to blue storage-bin demonstrations."""

    def __init__(self, config: HeuristicExpertConfig | None = None) -> None:
        self.config = config or HeuristicExpertConfig()
        self.reset()

    def reset(self) -> None:
        self.phase = PickPlacePhase.APPROACH
        self._counter = 0
        self._lift_target: NDArray[np.float64] | None = None

    @property
    def done(self) -> bool:
        return self.phase is PickPlacePhase.DONE

    @property
    def prompt(self) -> str:
        groups = {
            PickPlacePhase.APPROACH: "approach",
            PickPlacePhase.DESCEND: "grasp",
            PickPlacePhase.CLOSE: "grasp",
            PickPlacePhase.LIFT: "lift",
            PickPlacePhase.TRANSPORT: "transport",
            PickPlacePhase.PLACE: "place_release",
            PickPlacePhase.OPEN: "place_release",
            PickPlacePhase.DONE: "failed",
        }
        return task_phase_prompt("red_to_storage_bin", groups[self.phase])

    def act(self, observation: Mapping[str, Any]) -> NDArray[np.float32]:
        cube = HeuristicStackExpert._vector(observation, "cube_pos", 3)
        target = HeuristicStackExpert._vector(observation, "target_zone_pos", 3)
        eef = HeuristicStackExpert._vector(observation, "robot0_eef_pos", 3)
        action = np.zeros(7, dtype=np.float32)
        action[-1] = self.config.open_command
        if self.phase is PickPlacePhase.APPROACH:
            goal = cube.copy()
            goal[2] += self.config.approach_height_m
            action[:3] = self._delta(goal, eef)
            if np.linalg.norm(goal - eef) <= self.config.position_tolerance_m:
                self.phase = PickPlacePhase.DESCEND
        elif self.phase is PickPlacePhase.DESCEND:
            goal = cube.copy()
            goal[2] += self.config.grasp_height_m
            action[:3] = self._delta(goal, eef)
            if np.linalg.norm(goal - eef) <= self.config.position_tolerance_m:
                self.phase = PickPlacePhase.CLOSE
                self._counter = 0
        elif self.phase is PickPlacePhase.CLOSE:
            action[-1] = self.config.close_command
            self._counter += 1
            if self._counter >= self.config.close_steps:
                self._lift_target = eef.copy()
                self._lift_target[2] += self.config.lift_distance_m
                self.phase = PickPlacePhase.LIFT
        elif self.phase is PickPlacePhase.LIFT:
            action[-1] = self.config.close_command
            assert self._lift_target is not None
            action[:3] = self._delta(self._lift_target, eef)
            if np.linalg.norm(self._lift_target - eef) <= self.config.position_tolerance_m:
                self.phase = PickPlacePhase.TRANSPORT
        elif self.phase is PickPlacePhase.TRANSPORT:
            action[-1] = self.config.close_command
            goal = np.array([target[0], target[1], eef[2]])
            action[:3] = self._delta(goal, eef)
            if np.linalg.norm(goal[:2] - eef[:2]) <= self.config.position_tolerance_m:
                self.phase = PickPlacePhase.PLACE
        elif self.phase is PickPlacePhase.PLACE:
            action[-1] = self.config.close_command
            goal = target.copy()
            goal[2] += self.config.object_height_m / 2 + self.config.release_clearance_m
            action[:3] = 0.45 * self._delta(goal, eef)
            xy_tolerance = self.config.place_release_xy_tolerance_m
            if xy_tolerance is None:
                release_aligned = (
                    np.linalg.norm(goal - eef) <= self.config.place_release_tolerance_m
                )
            else:
                release_aligned = (
                    np.linalg.norm(goal[:2] - eef[:2]) <= xy_tolerance
                    and abs(float(goal[2] - eef[2]))
                    <= self.config.place_release_tolerance_m
                )
            if release_aligned:
                self.phase = PickPlacePhase.OPEN
                self._counter = 0
        elif self.phase is PickPlacePhase.OPEN:
            self._counter += 1
            if self._counter >= self.config.close_steps:
                self.phase = PickPlacePhase.DONE
        return DEFAULT_ACTION_SPEC.validate(action, clip=True)

    def _delta(self, target: NDArray[np.float64], current: NDArray[np.float64]) -> NDArray[np.float32]:
        return HeuristicStackExpert._position_delta(target, current, self.config)


class StackPhase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    MOVE_TO_TARGET = auto()
    DESCEND_TO_TARGET = auto()
    OPEN = auto()
    RETRY = auto()
    DONE = auto()


class HeuristicStackExpert:
    """State machine that picks cubeA and places it on cubeB."""

    def __init__(self, config: HeuristicExpertConfig | None = None) -> None:
        if config is None:
            self.config = HeuristicExpertConfig(
                position_tolerance_m=0.003,
                grasp_height_m=0.010,
            )
        else:
            self.config = config
        self.reset()

    def reset(self) -> None:
        self.phase = StackPhase.APPROACH
        self._close_counter = 0
        self._open_counter = 0
        self._initial_object_z: float | None = None
        self._lift_target: NDArray[np.float64] | None = None
        self._phase_steps = 0
        self._retries = 0
        self._last_cubeA: NDArray[np.float64] | None = None

    @property
    def prompt(self) -> str:
        prompts = {
            StackPhase.APPROACH: "move above the grasp object",
            StackPhase.DESCEND: "move down to grasp object",
            StackPhase.CLOSE: "move down to grasp object",
            StackPhase.LIFT: "lift the grasped object",
            StackPhase.MOVE_TO_TARGET: "move object above target",
            StackPhase.DESCEND_TO_TARGET: "lower object onto target",
            StackPhase.OPEN: "hold position for release",
            StackPhase.RETRY: "move above the grasp object",
            StackPhase.DONE: "hold position for release",
        }
        return prompts[self.phase]

    @property
    def retries(self) -> int:
        return self._retries

    def _set_phase(self, phase: StackPhase) -> None:
        if phase is not self.phase:
            self.phase = phase
            self._phase_steps = 0

    @property
    def done(self) -> bool:
        return self.phase is StackPhase.DONE

    def act(self, raw_observation: Mapping[str, Any]) -> NDArray[np.float32]:
        cubeA = self._vector(raw_observation, "cubeA_pos", 3)
        cubeB = self._vector(raw_observation, "cubeB_pos", 3)
        eef = self._vector(raw_observation, "robot0_eef_pos", 3)

        if self._initial_object_z is None:
            self._initial_object_z = float(cubeA[2])
        self._phase_steps += 1

        action = np.zeros(7, dtype=np.float32)
        # Default gripper action
        if self.phase in (StackPhase.APPROACH, StackPhase.DESCEND, StackPhase.OPEN, StackPhase.DONE):
            action[-1] = self.config.open_command
        else:
            action[-1] = self.config.close_command

        if self.phase is StackPhase.APPROACH:
            target = cubeA.copy()
            target[2] += self.config.approach_height_m
            action[:3] = self._position_delta(target, eef, self.config)
            if np.linalg.norm(target - eef) <= self.config.position_tolerance_m:
                self._set_phase(StackPhase.DESCEND)

        elif self.phase is StackPhase.DESCEND:
            target = cubeA.copy()
            target[2] += self.config.grasp_height_m
            action[:3] = self._position_delta(target, eef, self.config)
            if np.linalg.norm(target - eef) <= self.config.position_tolerance_m:
                self._set_phase(StackPhase.CLOSE)

        elif self.phase is StackPhase.CLOSE:
            action[-1] = self.config.close_command
            self._close_counter += 1
            if self._close_counter >= self.config.close_steps:
                grasp_geometry_ok = np.linalg.norm(cubeA - eef) <= 0.020
                if grasp_geometry_ok:
                    self._set_phase(StackPhase.LIFT)
                    self._lift_target = eef.copy()
                    self._lift_target[2] += self.config.lift_distance_m
                else:
                    self._set_phase(StackPhase.RETRY)

        elif self.phase is StackPhase.LIFT:
            action[-1] = self.config.close_command
            if self._lift_target is None:
                self._lift_target = eef.copy()
                self._lift_target[2] += self.config.lift_distance_m
            action[:3] = self._position_delta(self._lift_target, eef, self.config)
            lifted = float(cubeA[2]) >= float(self._initial_object_z) + 0.04
            if lifted:
                self._set_phase(StackPhase.MOVE_TO_TARGET)
            elif self._phase_steps >= 35:
                self._set_phase(StackPhase.RETRY)

        elif self.phase is StackPhase.MOVE_TO_TARGET:
            # Move cubeA to a position high above cubeB
            cubeA_target = cubeB.copy()
            cubeA_target[2] += 0.15
            delta = cubeA_target - cubeA
            target_eef = eef + delta
            action[:3] = self._position_delta(target_eef, eef, self.config)
            if np.linalg.norm(target_eef - eef) <= self.config.position_tolerance_m:
                self._set_phase(StackPhase.DESCEND_TO_TARGET)

        elif self.phase is StackPhase.DESCEND_TO_TARGET:
            # Move cubeA to place it on top of cubeB (1mm clearance)
            cubeA_target = cubeB.copy()
            cubeA_target[2] += self.config.object_height_m + self.config.release_clearance_m
            delta = cubeA_target - cubeA
            target_eef = eef + delta
            action[:3] = 0.45 * self._position_delta(target_eef, eef, self.config)
            if np.linalg.norm(target_eef - eef) <= self.config.position_tolerance_m:
                self._set_phase(StackPhase.OPEN)

        elif self.phase is StackPhase.OPEN:
            action[-1] = self.config.open_command
            # Hold still while the fingers clear the cube; retreating during
            # release can drag an otherwise aligned placement sideways.
            action[:3] = 0.0
            self._open_counter += 1
            # Wait a few steps for the cube to drop and settle
            if self._open_counter >= self.config.close_steps:
                self._set_phase(StackPhase.DONE)

        elif self.phase is StackPhase.RETRY:
            action[-1] = self.config.open_command
            target = cubeA.copy()
            target[2] += self.config.approach_height_m
            action[:3] = self._position_delta(target, eef, self.config)
            if np.linalg.norm(target - eef) <= self.config.position_tolerance_m:
                if self._retries >= 2:
                    self._set_phase(StackPhase.DONE)
                else:
                    self._retries += 1
                    self._close_counter = 0
                    self._lift_target = None
                    self._set_phase(StackPhase.DESCEND)

        elif self.phase is StackPhase.DONE:
            action[-1] = self.config.open_command

        self._last_cubeA = cubeA
        return DEFAULT_ACTION_SPEC.validate(action, clip=True)

    @staticmethod
    def _position_delta(
        target: NDArray[np.float64], current: NDArray[np.float64], config: HeuristicExpertConfig
    ) -> NDArray[np.float32]:
        normalized = (target - current) / config.position_action_scale_m
        normalized = np.clip(
            normalized,
            -config.max_position_action,
            config.max_position_action,
        )
        return normalized.astype(np.float32)

    @staticmethod
    def _vector(
        observation: Mapping[str, Any], key: str, size: int
    ) -> NDArray[np.float64]:
        if key not in observation:
            available = ", ".join(sorted(observation.keys()))
            raise ContractError(
                f"Heuristic expert requires privileged key '{key}'. "
                f"Available keys: {available or '<none>'}."
            )
        value = np.asarray(observation[key], dtype=np.float64).reshape(-1)
        if value.size < size or not np.all(np.isfinite(value[:size])):
            raise ContractError(f"'{key}' must contain {size} finite values.")
        return value[:size].copy()
