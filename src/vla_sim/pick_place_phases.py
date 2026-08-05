"""Privileged phase labels for pick-place training-data collection only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .stack_control import task_phase_prompt


@dataclass(frozen=True)
class PickPlacePhaseLabelConfig:
    approach_xy_m: float = 0.035
    approach_height_m: float = 0.055
    transport_xy_m: float = 0.040
    close_command_threshold: float = 0.0


def pick_place_phase_group(
    raw_observation: Mapping[str, Any],
    action: np.ndarray,
    info: Mapping[str, Any],
    config: PickPlacePhaseLabelConfig | None = None,
) -> str:
    """Label one recorded frame from simulator state, never policy inputs.

    ``info`` describes the state before the recorded action. The label is used
    only to route replay samples; rollout inference continues to receive the
    static global task prompt.
    """

    effective = config or PickPlacePhaseLabelConfig()
    cube = _xyz(raw_observation, "cube_pos")
    target = _xyz(raw_observation, "target_zone_pos")
    eef = _xyz(raw_observation, "robot0_eef_pos")
    command = np.asarray(action, dtype=np.float32).reshape(-1)
    if command.size < 7 or not np.all(np.isfinite(command[:7])):
        raise ValueError("Pick-place phase labeling requires a finite 7-D action")

    ever_lifted = bool(info.get("ever_lifted", False))
    grasped = bool(info.get("grasped", False))
    ever_grasped = bool(info.get("ever_grasped", False))
    closed = float(command[6]) > effective.close_command_threshold

    if ever_lifted:
        return (
            "transport"
            if float(np.linalg.norm(cube[:2] - target[:2])) > effective.transport_xy_m
            else "place_release"
        )
    if grasped or ever_grasped:
        return "lift"
    if closed:
        return "grasp"
    cube_xy_error = float(np.linalg.norm(eef[:2] - cube[:2]))
    eef_height = float(eef[2] - cube[2])
    if cube_xy_error > effective.approach_xy_m or eef_height > effective.approach_height_m:
        return "approach"
    return "grasp"


def pick_place_phase_prompt(
    raw_observation: Mapping[str, Any],
    action: np.ndarray,
    info: Mapping[str, Any],
) -> str:
    return task_phase_prompt(
        "red_to_storage_bin",
        pick_place_phase_group(raw_observation, action, info),
    )


def _xyz(observation: Mapping[str, Any], key: str) -> np.ndarray:
    value = np.asarray(observation.get(key), dtype=np.float64).reshape(-1)
    if value.size < 3 or not np.all(np.isfinite(value[:3])):
        raise ValueError(f"Pick-place phase labeling requires finite {key}")
    return value[:3]
