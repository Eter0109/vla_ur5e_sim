"""Three-task UR5e VLA simulation, policy, and evaluation package."""

from .simulation.contracts import (
    ACTION_DIM,
    ACTION_KEY,
    IMAGE_KEY,
    STATE_DIM,
    STATE_KEY,
    TASK_KEY,
    WRIST_IMAGE_KEY,
)

__all__ = [
    "ACTION_DIM",
    "ACTION_KEY",
    "IMAGE_KEY",
    "STATE_DIM",
    "STATE_KEY",
    "TASK_KEY",
    "WRIST_IMAGE_KEY",
]
