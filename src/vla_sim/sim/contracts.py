"""Backward-compatible imports for the canonical :mod:`vla_sim.contracts` module."""

from vla_sim.contracts import (
    ACTION_DIM,
    ACTION_KEY,
    DEFAULT_ACTION_SPEC,
    IMAGE_KEY,
    STATE_DIM,
    STATE_KEY,
    TASK_KEY,
    ActionSpec,
    ContractError,
    ObservationAdapter,
    validate_observation,
)

__all__ = [
    "ACTION_DIM",
    "ACTION_KEY",
    "DEFAULT_ACTION_SPEC",
    "IMAGE_KEY",
    "STATE_DIM",
    "STATE_KEY",
    "TASK_KEY",
    "ActionSpec",
    "ContractError",
    "ObservationAdapter",
    "validate_observation",
]
