"""The single policy-facing contract used by simulation, datasets and rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

ACTION_DIM = 7
STATE_DIM = 7
IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_KEY = "task"


class ContractError(ValueError):
    """Raised when data crosses a simulator/policy boundary in an invalid form."""


@dataclass(frozen=True)
class ActionSpec:
    """Normalized OSC pose action.

    Components are ``dx, dy, dz, dRx, dRy, dRz, gripper``.  The first six are
    normalized deltas in ``[-1, 1]``.  The final component follows robosuite's
    Robotiq convention: ``-1`` opens and ``+1`` closes the gripper.
    """

    minimum: float = -1.0
    maximum: float = 1.0

    @property
    def shape(self) -> tuple[int]:
        return (ACTION_DIM,)

    def validate(self, action: Any, *, clip: bool = False) -> NDArray[np.float32]:
        array = np.asarray(action, dtype=np.float32)
        if array.shape != self.shape:
            raise ContractError(
                f"Expected action shape {self.shape}; got {array.shape}. "
                "Action order is [dx, dy, dz, dRx, dRy, dRz, gripper]."
            )
        if not np.all(np.isfinite(array)):
            raise ContractError("Action contains NaN or infinity.")
        if clip:
            array = np.clip(array, self.minimum, self.maximum)
        elif np.any(array < self.minimum) or np.any(array > self.maximum):
            raise ContractError("Normalized action values must stay in [-1, 1].")
        return np.ascontiguousarray(array, dtype=np.float32)


DEFAULT_ACTION_SPEC = ActionSpec()


@dataclass(frozen=True)
class ObservationAdapter:
    """Translate raw robosuite observations to the policy-visible fields."""

    camera_name: str = "agentview"
    flip_vertical: bool = False
    gripper_min: float = -0.04
    gripper_max: float = 0.04

    @property
    def source_image_key(self) -> str:
        return f"{self.camera_name}_image"

    def convert(self, raw: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
        return {IMAGE_KEY: self._extract_image(raw), STATE_KEY: self._extract_state(raw)}

    def _extract_image(self, raw: Mapping[str, Any]) -> NDArray[np.uint8]:
        if self.source_image_key not in raw:
            raise ContractError(f"Missing camera observation '{self.source_image_key}'.")
        image = np.asarray(raw[self.source_image_key])
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ContractError(f"Camera image must be HxWxC; got {image.shape}.")
        image = image[..., :3]
        if np.issubdtype(image.dtype, np.floating):
            scale = 255.0 if image.size and float(np.nanmax(image)) <= 1.0 else 1.0
            image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if self.flip_vertical:
            image = np.flip(image, axis=0)
        return np.ascontiguousarray(image)

    def _extract_state(self, raw: Mapping[str, Any]) -> NDArray[np.float32]:
        joints = np.asarray(raw.get("robot0_joint_pos"), dtype=np.float32).reshape(-1)
        if joints.size < 6 or not np.all(np.isfinite(joints[:6])):
            raise ContractError("UR5e observation requires six finite joint positions.")
        aperture = None
        for key in ("robot0_gripper_qpos", "robot0_gripper_pos"):
            if key in raw:
                values = np.asarray(raw[key], dtype=np.float32).reshape(-1)
                if values.size:
                    aperture = float(values[0])
                    break
        if aperture is None:
            raise ContractError("Missing Robotiq gripper position observation.")
        denominator = self.gripper_max - self.gripper_min
        opening = float(np.clip((aperture - self.gripper_min) / denominator, 0.0, 1.0))
        return np.ascontiguousarray(np.r_[joints[:6], np.float32(opening)], dtype=np.float32)


def validate_observation(observation: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
    """Validate the two policy-visible observation fields at process boundaries."""

    if set(observation) != {IMAGE_KEY, STATE_KEY}:
        raise ContractError(f"Observation keys must be exactly '{IMAGE_KEY}' and '{STATE_KEY}'.")
    image = np.asarray(observation[IMAGE_KEY])
    state = np.asarray(observation[STATE_KEY], dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ContractError("Policy image must be uint8 HxWx3 RGB.")
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise ContractError("Policy state must be a finite float32 7-vector.")
    return {IMAGE_KEY: np.ascontiguousarray(image), STATE_KEY: np.ascontiguousarray(state)}
