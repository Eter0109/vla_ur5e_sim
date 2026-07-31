"""Robosuite Stack environment with a stable VLA-facing API."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from vla_sim.sim.contracts import (
    DEFAULT_ACTION_SPEC,
    ActionSpec,
    ContractError,
    StackObservationAdapter,
)
from vla_sim.sim.dependencies import require_robosuite
from vla_sim.scenes import SceneSpec

from .objects import PrimitiveObjectConfig
from .ur5e_lift import CameraConfig, _controller_config


@dataclass(frozen=True)
class UR5eStackConfig:
    """Configuration for the Stack environment."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    objectA: PrimitiveObjectConfig = field(default_factory=lambda: PrimitiveObjectConfig(rgba=(0.85, 0.12, 0.08, 1.0))) # Red default
    objectB: PrimitiveObjectConfig = field(default_factory=lambda: PrimitiveObjectConfig(rgba=(0.12, 0.2, 0.85, 1.0)))  # Blue default
    control_frequency_hz: int = 10
    horizon: int = 200
    seed: int | None = 0
    reward_shaping: bool = True
    terminate_on_success: bool = True
    has_renderer: bool = False
    success_hold_steps: int = 10
    stack_xy_tolerance_m: float = 0.015
    stack_height_tolerance_m: float = 0.012
    base_height_tolerance_m: float = 0.012
    max_linear_speed_m_s: float = 0.02
    max_angular_speed_rad_s: float = 0.25

    def __post_init__(self) -> None:
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.success_hold_steps < 1:
            raise ValueError("success_hold_steps must be at least one.")
        thresholds = (
            self.stack_xy_tolerance_m,
            self.stack_height_tolerance_m,
            self.base_height_tolerance_m,
            self.max_linear_speed_m_s,
            self.max_angular_speed_rad_s,
        )
        if any(value <= 0 for value in thresholds):
            raise ValueError("Stack geometry and velocity thresholds must be positive.")


@lru_cache(maxsize=1)
def _configurable_stack_class() -> type[Any]:
    """Create the robosuite subclass lazily so module import stays optional."""

    from robosuite.environments.manipulation.stack import Stack

    class ConfigurablePrimitiveStack(Stack):
        def __init__(
            self,
            *args: Any,
            primitive_object_config_A: PrimitiveObjectConfig,
            primitive_object_config_B: PrimitiveObjectConfig,
            **kwargs: Any,
        ) -> None:
            self._primitive_object_config_A = primitive_object_config_A
            self._primitive_object_config_B = primitive_object_config_B
            super().__init__(*args, **kwargs)

        def _load_model(self) -> None:
            super()._load_model()
            self._primitive_object_config_A.apply_to_robosuite_object(self.cubeA)
            self._primitive_object_config_B.apply_to_robosuite_object(self.cubeB)

            # ManipulationTask merges a copy of the object MJCF. Update that
            # merged body as well so the compiled model always sees the change.
            for cube, config in [(self.cubeA, self._primitive_object_config_A), (self.cubeB, self._primitive_object_config_B)]:
                root_body_name = cube.root_body
                merged_body = self.model.worldbody.find(
                    f".//body[@name='{root_body_name}']"
                )
                if merged_body is None:
                    raise RuntimeError(
                        f"Unable to locate merged Stack object body '{root_body_name}'."
                    )
                config.apply_to_xml(merged_body)

    return ConfigurablePrimitiveStack


def create_robosuite_backend(config: UR5eStackConfig | None = None) -> Any:
    """Create the raw robosuite backend, importing the dependency on demand."""

    config = config or UR5eStackConfig()
    suite = require_robosuite()
    environment_class = _configurable_stack_class()
    return environment_class(
        primitive_object_config_A=config.objectA,
        primitive_object_config_B=config.objectB,
        robots="UR5e",
        gripper_types="Robotiq85Gripper",
        controller_configs=_controller_config(suite),
        control_freq=config.control_frequency_hz,
        horizon=config.horizon,
        reward_shaping=config.reward_shaping,
        has_renderer=config.has_renderer,
        render_camera=config.camera.name,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=config.camera.name,
        camera_widths=config.camera.width,
        camera_heights=config.camera.height,
        camera_depths=True,
        camera_segmentations=None,
        hard_reset=False,
        ignore_done=False,
        seed=config.seed,
    )


class UR5eStackEnv:
    """Gymnasium-style adapter around robosuite's four-return environment API."""

    action_spec: ActionSpec = DEFAULT_ACTION_SPEC

    def __init__(self, backend: Any, config: UR5eStackConfig | None = None) -> None:
        self.config = config or UR5eStackConfig()
        self.backend = backend
        self._observation_adapter = StackObservationAdapter(
            camera_name=self.config.camera.name,
            flip_vertical=self.config.camera.flip_vertical,
        )
        self._raw_observation: Mapping[str, Any] | None = None
        self._success_hold_count = 0
        self._initial_cubeA_z: float | None = None
        self._initial_cubeB_z: float | None = None
        self._validate_backend_action_space()

    @classmethod
    def create(cls, config: UR5eStackConfig | None = None) -> "UR5eStackEnv":
        effective_config = config or UR5eStackConfig()
        return cls(create_robosuite_backend(effective_config), effective_config)

    @property
    def raw_observation(self) -> Mapping[str, Any]:
        """Latest privileged backend observation, intended for smoke experts."""

        if self._raw_observation is None:
            raise RuntimeError("reset() must be called before reading observations.")
        return self._raw_observation

    def reset(
        self, *, seed: int | None = None, scene: SceneSpec | None = None
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        result = self.backend.reset()
        raw = result[0] if isinstance(result, tuple) else result
        if scene is not None:
            self._apply_scene(scene, raw)
            raw = self.backend._get_observations(force_update=True)
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite reset() did not return an observation mapping.")
        self._raw_observation = raw
        self._success_hold_count = 0
        cube_a = self._position("cubeA_pos")
        cube_b = self._position("cubeB_pos")
        self._initial_cubeA_z = float(cube_a[2])
        self._initial_cubeB_z = float(cube_b[2])
        return self._observation_adapter.convert(raw), {"success": False}

    def _apply_scene(self, scene: SceneSpec, raw: Mapping[str, Any]) -> None:
        """Place the free-joint object at the exact manifest pose."""

        cubeA_pos = np.asarray(raw.get("cubeA_pos"), dtype=np.float64).reshape(-1)
        if cubeA_pos.size < 3:
            raise ContractError("Cannot apply scene without cubeA_pos.")

        # Cube A placement
        half_yaw = scene.yaw_rad / 2.0
        qpos_A = np.asarray(
            [
                scene.x_m,
                scene.y_m,
                float(cubeA_pos[2]),
                np.cos(half_yaw),
                0.0,
                0.0,
                np.sin(half_yaw),
            ],
            dtype=np.float64,
        )
        self.backend.sim.data.set_joint_qpos(self.backend.cubeA.joints[0], qpos_A)

        # Cube B placement if provided in overrides, else leave as randomly initialized
        if "cubeB_x_m" in scene.overrides and "cubeB_y_m" in scene.overrides:
            cubeB_pos = np.asarray(raw.get("cubeB_pos"), dtype=np.float64).reshape(-1)
            b_yaw = scene.overrides.get("cubeB_yaw_rad", 0.0)
            b_half_yaw = b_yaw / 2.0
            qpos_B = np.asarray(
                [
                    scene.overrides["cubeB_x_m"],
                    scene.overrides["cubeB_y_m"],
                    float(cubeB_pos[2]),
                    np.cos(b_half_yaw),
                    0.0,
                    0.0,
                    np.sin(b_half_yaw),
                ],
                dtype=np.float64,
            )
            self.backend.sim.data.set_joint_qpos(self.backend.cubeB.joints[0], qpos_B)

        self.backend.sim.forward()

    def step(
        self, action: Any
    ) -> tuple[
        dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]
    ]:
        normalized_action = self.action_spec.validate(action)
        raw, reward, done, backend_info = self.backend.step(normalized_action)
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite step() did not return an observation mapping.")
        self._raw_observation = raw

        info = dict(backend_info or {})
        lifted, grasped, lift_m = self._success_state()
        success, stack_conditions = self._check_success()
        info["success"] = success
        info["success_hold_count"] = self._success_hold_count
        info["grasped"] = grasped
        info["lifted"] = lifted
        info["object_lift_m"] = lift_m
        info["stack_conditions"] = stack_conditions
        info["target_xy_error_m"] = stack_conditions["xy_error_m"]
        info["target_height_error_m"] = stack_conditions["height_error_m"]

        terminated = bool(success and self.config.terminate_on_success)
        truncated = bool(done and not terminated)
        return (
            self._observation_adapter.convert(raw),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self) -> Any:
        return self.backend.render()

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    def _check_success(self) -> tuple[bool, dict[str, Any]]:
        conditions = self._strict_stack_conditions()
        is_success = all(
            bool(conditions[name])
            for name in (
                "objects_touching",
                "above_target",
                "xy_aligned",
                "height_aligned",
                "gripper_released",
                "base_on_table",
                "objects_stable",
            )
        )
        self._success_hold_count = self._success_hold_count + 1 if is_success else 0
        return self._success_hold_count >= self.config.success_hold_steps, conditions

    def _strict_stack_conditions(self) -> dict[str, Any]:
        cube_a = self._position("cubeA_pos")
        cube_b = self._position("cubeB_pos")
        xy_error = float(np.linalg.norm(cube_a[:2] - cube_b[:2]))
        expected_height = float(self.config.objectB.dimensions_m[2])
        height_delta = float(cube_a[2] - cube_b[2])
        height_error = abs(height_delta - expected_height)
        linear_a, angular_a = self._body_speeds(getattr(self.backend, "cubeA", None))
        linear_b, angular_b = self._body_speeds(getattr(self.backend, "cubeB", None))
        initial_b_z = self._initial_cubeB_z if self._initial_cubeB_z is not None else cube_b[2]
        return {
            "objects_touching": self._objects_touching(),
            "above_target": height_delta > 0,
            "xy_aligned": xy_error <= self.config.stack_xy_tolerance_m,
            "height_aligned": height_error <= self.config.stack_height_tolerance_m,
            "gripper_released": not self._is_grasped(),
            "base_on_table": abs(float(cube_b[2]) - float(initial_b_z))
            <= self.config.base_height_tolerance_m,
            "objects_stable": max(linear_a, linear_b) <= self.config.max_linear_speed_m_s
            and max(angular_a, angular_b) <= self.config.max_angular_speed_rad_s,
            "xy_error_m": xy_error,
            "height_error_m": height_error,
            "linear_speed_m_s": max(linear_a, linear_b),
            "angular_speed_rad_s": max(angular_a, angular_b),
        }

    def _success_state(self) -> tuple[bool, bool, float]:
        """Return current lift and grasp state without mutating hold counters."""
        if self._raw_observation is None:
            return False, False, 0.0
        cubeA_pos = self._position("cubeA_pos")
        initial_z = self._initial_cubeA_z if self._initial_cubeA_z is not None else cubeA_pos[2]
        lift_m = float(cubeA_pos[2]) - float(initial_z)
        lifted = lift_m >= 0.04
        grasped = self._is_grasped()
        return lifted, grasped, lift_m

    def _position(self, key: str) -> NDArray[np.float64]:
        if self._raw_observation is None:
            raise ContractError("reset() must be called before reading object positions.")
        value = np.asarray(self._raw_observation.get(key), dtype=np.float64).reshape(-1)
        if value.size < 3 or not np.all(np.isfinite(value[:3])):
            raise ContractError(f"Stack observation requires three finite values for {key!r}.")
        return value[:3]

    def _is_grasped(self) -> bool:
        grasp_checker = getattr(self.backend, "_check_grasp", None)
        robots = getattr(self.backend, "robots", None)
        cubeA = getattr(self.backend, "cubeA", None)
        if callable(grasp_checker) and robots and cubeA is not None:
            return bool(grasp_checker(robots[0].gripper, cubeA))
        return False

    def _objects_touching(self) -> bool:
        checker = getattr(self.backend, "check_contact", None)
        cube_a = getattr(self.backend, "cubeA", None)
        cube_b = getattr(self.backend, "cubeB", None)
        return bool(callable(checker) and cube_a is not None and cube_b is not None and checker(cube_a, cube_b))

    def _body_speeds(self, body: Any) -> tuple[float, float]:
        root_body = getattr(body, "root_body", None)
        data = getattr(getattr(self.backend, "sim", None), "data", None)
        if root_body is None or data is None:
            return float("inf"), float("inf")
        try:
            linear = np.asarray(data.get_body_xvelp(root_body), dtype=np.float64)
            angular = np.asarray(data.get_body_xvelr(root_body), dtype=np.float64)
        except (AttributeError, KeyError, ValueError):
            return float("inf"), float("inf")
        return float(np.linalg.norm(linear)), float(np.linalg.norm(angular))

    def _validate_backend_action_space(self) -> None:
        raw_spec = getattr(self.backend, "action_spec", None)
        if raw_spec is None:
            return
        try:
            low, high = raw_spec
            low_array = np.asarray(low).reshape(-1)
            high_array = np.asarray(high).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ContractError("Backend action_spec is not a (low, high) pair.") from exc
        if low_array.size != 7 or high_array.size != 7:
            raise ContractError(
                "UR5e + Robotiq85 with OSC_POSE must expose seven actions; "
                f"backend reported {low_array.size}."
            )


def make_ur5e_stack(config: UR5eStackConfig | None = None) -> UR5eStackEnv:
    """Public environment factory."""

    return UR5eStackEnv.create(config)
