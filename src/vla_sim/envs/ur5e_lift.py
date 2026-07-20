"""Robosuite Lift environment with a stable VLA-facing API."""

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
    ObservationAdapter,
)
from vla_sim.sim.dependencies import require_robosuite
from vla_sim.scenes import SceneSpec

from .objects import PrimitiveObjectConfig


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


@dataclass(frozen=True)
class UR5eLiftConfig:
    """Configuration for the Windows-first robosuite environment."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    object: PrimitiveObjectConfig = field(default_factory=PrimitiveObjectConfig)
    control_frequency_hz: int = 10
    horizon: int = 200
    seed: int | None = 0
    reward_shaping: bool = True
    terminate_on_success: bool = True
    has_renderer: bool = False
    success_lift_height_m: float = 0.10
    success_hold_steps: int = 10

    def __post_init__(self) -> None:
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.success_lift_height_m <= 0:
            raise ValueError("success_lift_height_m must be positive.")
        if self.success_hold_steps < 1:
            raise ValueError("success_hold_steps must be at least one.")


def _controller_config(suite: Any) -> Any:
    """Load OSC pose control across robosuite 1.4 and 1.5 APIs."""

    try:
        from robosuite.controllers import load_composite_controller_config

        # BASIC uses OSC pose control for fixed-base arms in robosuite 1.5.
        return load_composite_controller_config(controller="BASIC")
    except ImportError:
        return suite.load_controller_config(default_controller="OSC_POSE")


@lru_cache(maxsize=1)
def _configurable_lift_class() -> type[Any]:
    """Create the robosuite subclass lazily so module import stays optional."""

    from robosuite.environments.manipulation.lift import Lift

    class ConfigurablePrimitiveLift(Lift):
        def __init__(
            self,
            *args: Any,
            primitive_object_config: PrimitiveObjectConfig,
            **kwargs: Any,
        ) -> None:
            self._primitive_object_config = primitive_object_config
            super().__init__(*args, **kwargs)

        def _load_model(self) -> None:
            super()._load_model()
            self._primitive_object_config.apply_to_robosuite_object(self.cube)

            # ManipulationTask merges a copy of the object MJCF. Update that
            # merged body as well so the compiled model always sees the change.
            root_body_name = self.cube.root_body
            merged_body = self.model.worldbody.find(
                f".//body[@name='{root_body_name}']"
            )
            if merged_body is None:
                raise RuntimeError(
                    f"Unable to locate merged Lift object body '{root_body_name}'."
                )
            self._primitive_object_config.apply_to_xml(merged_body)

    return ConfigurablePrimitiveLift


def create_robosuite_backend(config: UR5eLiftConfig | None = None) -> Any:
    """Create the raw robosuite backend, importing the dependency on demand."""

    config = config or UR5eLiftConfig()
    suite = require_robosuite()
    environment_class = _configurable_lift_class()
    return environment_class(
        primitive_object_config=config.object,
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
        camera_depths=False,
        camera_segmentations=None,
        # Recompiling the MuJoCo model every episode leaks address space on
        # Windows during long evaluations. Scene placement is applied after
        # reset, so a soft reset is both sufficient and reproducible.
        hard_reset=False,
        ignore_done=False,
        seed=config.seed,
    )


class UR5eLiftEnv:
    """Gymnasium-style adapter around robosuite's four-return environment API."""

    action_spec: ActionSpec = DEFAULT_ACTION_SPEC

    def __init__(self, backend: Any, config: UR5eLiftConfig | None = None) -> None:
        self.config = config or UR5eLiftConfig()
        self.backend = backend
        self._observation_adapter = ObservationAdapter(
            camera_name=self.config.camera.name,
            flip_vertical=self.config.camera.flip_vertical,
        )
        self._raw_observation: Mapping[str, Any] | None = None
        self._initial_object_z: float | None = None
        self._success_hold_count = 0
        self._validate_backend_action_space()

    @classmethod
    def create(cls, config: UR5eLiftConfig | None = None) -> "UR5eLiftEnv":
        effective_config = config or UR5eLiftConfig()
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
            # robosuite 1.5 seeds at construction. This still gives deterministic
            # numpy-driven placement for lightweight fake/test backends.
            np.random.seed(seed)
        result = self.backend.reset()
        raw = result[0] if isinstance(result, tuple) else result
        if scene is not None:
            self._apply_scene(scene, raw)
            raw = self.backend._get_observations(force_update=True)
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite reset() did not return an observation mapping.")
        self._raw_observation = raw
        cube_pos = np.asarray(raw.get("cube_pos"), dtype=np.float64).reshape(-1)
        if cube_pos.size < 3:
            raise ContractError("robosuite reset observation is missing cube_pos.")
        self._initial_object_z = float(cube_pos[2])
        self._success_hold_count = 0
        return self._observation_adapter.convert(raw), {"success": False}

    def _apply_scene(self, scene: SceneSpec, raw: Mapping[str, Any]) -> None:
        """Place the free-joint object at the exact manifest pose."""

        cube_pos = np.asarray(raw.get("cube_pos"), dtype=np.float64).reshape(-1)
        if cube_pos.size < 3:
            raise ContractError("Cannot apply scene without cube_pos.")
        half_yaw = scene.yaw_rad / 2.0
        qpos = np.asarray(
            [
                scene.x_m,
                scene.y_m,
                float(cube_pos[2]),
                np.cos(half_yaw),
                0.0,
                0.0,
                np.sin(half_yaw),
            ],
            dtype=np.float64,
        )
        self.backend.sim.data.set_joint_qpos(self.backend.cube.joints[0], qpos)
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
        self._success_hold_count = self._success_hold_count + 1 if lifted and grasped else 0
        success = self._success_hold_count >= self.config.success_hold_steps
        info["success"] = success
        info["success_hold_count"] = self._success_hold_count
        # These privileged fields are for simulator diagnostics only. They make
        # rollout failure categories reflect the environment's real grasp check
        # instead of an arbitrary end-effector distance threshold.
        info["grasped"] = grasped
        info["lifted"] = lifted
        info["object_lift_m"] = lift_m
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

    def _check_success(self) -> bool:
        lifted, grasped, _ = self._success_state()
        self._success_hold_count = self._success_hold_count + 1 if lifted and grasped else 0
        return self._success_hold_count >= self.config.success_hold_steps

    def _success_state(self) -> tuple[bool, bool, float]:
        """Return current lift and grasp state without mutating hold counters."""
        if self._raw_observation is None or self._initial_object_z is None:
            return False, False, 0.0
        cube_pos = np.asarray(self._raw_observation.get("cube_pos"), dtype=np.float64).reshape(-1)
        lift_m = 0.0
        if cube_pos.size >= 3:
            lift_m = float(cube_pos[2]) - self._initial_object_z
        lifted = bool(
            cube_pos.size >= 3
            and lift_m >= self.config.success_lift_height_m
        )
        grasped = False
        grasp_checker = getattr(self.backend, "_check_grasp", None)
        robots = getattr(self.backend, "robots", None)
        cube = getattr(self.backend, "cube", None)
        if callable(grasp_checker) and robots and cube is not None:
            grasped = bool(grasp_checker(robots[0].gripper, cube))
        elif not callable(grasp_checker):
            # Lightweight fake backends used by contract tests have no contacts.
            grasped = bool(getattr(self.backend, "_check_success", lambda: False)())
        return lifted, grasped, lift_m

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


def make_ur5e_lift(config: UR5eLiftConfig | None = None) -> UR5eLiftEnv:
    """Public environment factory."""

    return UR5eLiftEnv.create(config)
