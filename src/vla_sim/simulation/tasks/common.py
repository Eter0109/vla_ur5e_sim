"""Shared robosuite configuration for the three supported tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraConfig:
    """Single policy camera configuration."""

    name: str = "agentview"
    width: int = 256
    height: int = 256
    flip_vertical: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Camera name cannot be empty.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera dimensions must be positive.")


def controller_config(suite: Any) -> Any:
    """Load OSC pose control across supported robosuite releases."""

    try:
        from robosuite.controllers import load_composite_controller_config

        return load_composite_controller_config(controller="BASIC")
    except ImportError:
        return suite.load_controller_config(default_controller="OSC_POSE")
