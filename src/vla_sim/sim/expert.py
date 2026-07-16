"""A small privileged-state heuristic for smoke-testing the lift task."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

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
    close_steps: int = 12
    open_command: float = -1.0
    close_command: float = 1.0
    max_position_action: float = 0.8

    def __post_init__(self) -> None:
        if self.position_action_scale_m <= 0:
            raise ValueError("position_action_scale_m must be positive.")
        if self.close_steps < 1:
            raise ValueError("close_steps must be at least one.")


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
