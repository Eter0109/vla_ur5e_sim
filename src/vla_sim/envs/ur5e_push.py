"""Forward-cone robosuite Push environment with a stable VLA-facing API."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from xml.etree.ElementTree import SubElement

import numpy as np
from numpy.typing import NDArray

from vla_sim.domain_randomization import (
    DomainRandomizationSample,
    Sim2RealEpisodeRuntime,
    apply_domain_randomization,
    capture_render_baseline,
)
from vla_sim.scenes import SceneSpec
from vla_sim.sim.contracts import (
    DEFAULT_ACTION_SPEC,
    ActionSpec,
    ContractError,
)
from vla_sim.sim.dependencies import require_robosuite

from .objects import PrimitiveObjectConfig
from .ur5e_lift import CameraConfig, _controller_config


@dataclass(frozen=True)
class UR5ePushConfig:
    """Configuration for the Windows-first robosuite Push environment."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    object: PrimitiveObjectConfig | None = None  # None means random unless a scene freezes it.
    control_frequency_hz: int = 10
    horizon: int = 200
    seed: int | None = 0
    reward_shaping: bool = True
    terminate_on_success: bool = True
    has_renderer: bool = False
    
    target_radius_m: float = 0.05
    target_angle_range_rad: tuple[float, float] = (-math.pi / 12, math.pi / 12)
    target_distance_range_m: tuple[float, float] = (0.10, 0.15)
    table_xy_limit_m: float = 0.25
    success_hold_steps: int = 5

    def __post_init__(self) -> None:
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.success_hold_steps < 1:
            raise ValueError("success_hold_steps must be at least one.")
        if self.target_radius_m <= 0 or self.table_xy_limit_m <= 0:
            raise ValueError("target_radius_m and table_xy_limit_m must be positive.")
        low_angle, high_angle = self.target_angle_range_rad
        if not -math.pi <= low_angle < high_angle <= math.pi:
            raise ValueError("target_angle_range_rad must be ordered within [-pi, pi].")
        low_distance, high_distance = self.target_distance_range_m
        if not 0 < low_distance < high_distance:
            raise ValueError("target_distance_range_m must be positive and ordered.")


def generate_random_object_config() -> PrimitiveObjectConfig:
    """Generate a fully random PrimitiveObjectConfig."""
    shape = "box"
    
    # Random size between 3cm and 6cm
    size = float(np.random.uniform(0.03, 0.06))
    dimensions_m = (
        size,
        float(np.random.uniform(0.03, 0.06)),
        float(np.random.uniform(0.03, 0.06))
    )
        
    rgba = tuple(float(c) for c in np.random.uniform(0.1, 0.9, 4))
    # Alpha must be 1.0 to avoid transparency artifacts
    rgba = (*rgba[:3], 1.0)
    
    return PrimitiveObjectConfig(
        shape=shape,  # type: ignore
        dimensions_m=dimensions_m,
        rgba=rgba,
        friction=(1.0, 0.5, 0.1),
    )


@lru_cache(maxsize=1)
def _configurable_push_class() -> type[Any]:
    """Create the robosuite subclass lazily so module import stays optional."""

    from robosuite.environments.manipulation.lift import Lift

    class ConfigurablePrimitivePush(Lift):
        def __init__(
            self,
            *args: Any,
            push_config: UR5ePushConfig,
            **kwargs: Any,
        ) -> None:
            self._push_config = push_config
            self._current_object_config = push_config.object or generate_random_object_config()
            self.target_pos = np.array([0.0, 0.0, 0.8]) # Will be updated in reset
            super().__init__(*args, **kwargs)

        def _load_model(self) -> None:
            super()._load_model()
            
            # Apply object configuration
            self._current_object_config.apply_to_robosuite_object(self.cube)

            # Update the merged body in worldbody
            root_body_name = self.cube.root_body
            merged_body = self.model.worldbody.find(
                f".//body[@name='{root_body_name}']"
            )
            if merged_body is None:
                raise RuntimeError(
                    f"Unable to locate merged object body '{root_body_name}'."
                )
            self._current_object_config.apply_to_xml(merged_body)
            
            # Add target visual marker to worldbody (table height is approx 0.8 in robosuite)
            target_site = SubElement(self.model.worldbody, "site")
            target_site.set("name", "push_target_zone")
            target_site.set("type", "cylinder")
            target_site.set("size", f"{self._push_config.target_radius_m} 0.001")
            target_site.set("pos", "0 0 0.801") # Slightly above table
            target_site.set("rgba", "1 0 0 0.5") # Semi-transparent red
            
    return ConfigurablePrimitivePush


def create_robosuite_backend(config: UR5ePushConfig | None = None) -> Any:
    """Create the raw robosuite backend, importing the dependency on demand."""

    config = config or UR5ePushConfig()
    suite = require_robosuite()
    environment_class = _configurable_push_class()
    
    return environment_class(
        push_config=config,
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
        camera_names=[config.camera.name, "robot0_eye_in_hand"],
        camera_widths=[config.camera.width, config.camera.width],
        camera_heights=[config.camera.height, config.camera.height],
        camera_depths=False,
        camera_segmentations=None,
        # Recompiling MuJoCo once per episode leaks native allocations on the
        # Windows build used for collection. Episode geometry is updated in
        # place by UR5ePushEnv instead, so one compiled simulation is reused.
        hard_reset=False,
        ignore_done=False,
        seed=config.seed,
    )


class UR5ePushEnv:
    """Gymnasium-style adapter around robosuite's four-return environment API."""

    action_spec: ActionSpec = DEFAULT_ACTION_SPEC

    def __init__(self, backend: Any, config: UR5ePushConfig | None = None) -> None:
        self.config = config or UR5ePushConfig()
        self.backend = backend
        
        from vla_sim.sim.contracts import PickPlaceObservationAdapter
        self._observation_adapter = PickPlaceObservationAdapter(
            camera_name=self.config.camera.name,
            flip_vertical=self.config.camera.flip_vertical,
        )
        self._raw_observation: Mapping[str, Any] | None = None
        self._success_hold_count = 0
        self._target_pos: NDArray[np.float64] | None = None
        self._render_baseline = capture_render_baseline(backend)
        self._randomization_sample: DomainRandomizationSample | None = None
        self._sim2real_runtime = Sim2RealEpisodeRuntime()
        self._visual_step = 0
        self._validate_backend_action_space()

    @classmethod
    def create(cls, config: UR5ePushConfig | None = None) -> UR5ePushEnv:
        effective_config = config or UR5ePushConfig()
        return cls(create_robosuite_backend(effective_config), effective_config)

    @property
    def raw_observation(self) -> Mapping[str, Any]:
        """Latest privileged backend observation, intended for smoke experts."""
        if self._raw_observation is None:
            raise RuntimeError("reset() must be called before reading observations.")
        return self._raw_observation

    @property
    def target_pos(self) -> NDArray[np.float64]:
        if self._target_pos is None:
            raise RuntimeError("reset() must be called before reading target_pos.")
        return self._target_pos.copy()

    def _set_episode_object(self, scene: SceneSpec | None) -> None:
        if not hasattr(self.backend, "_current_object_config"):
            return
        if scene is not None and "object_dimensions_m" in scene.overrides:
            dimensions = tuple(float(value) for value in scene.overrides["object_dimensions_m"])
            rgba = tuple(float(value) for value in scene.overrides["object_rgba"])
            self.backend._current_object_config = PrimitiveObjectConfig(
                shape=str(scene.overrides.get("object_shape", "box")),  # type: ignore[arg-type]
                dimensions_m=dimensions,  # type: ignore[arg-type]
                rgba=rgba,  # type: ignore[arg-type]
            )
        elif self.config.object is None:
            self.backend._current_object_config = generate_random_object_config()
        self._apply_runtime_object_config()

    def _apply_runtime_object_config(self) -> None:
        """Update box geometry without recompiling the MuJoCo model."""
        sim_model = getattr(getattr(self.backend, "sim", None), "model", None)
        cube = getattr(self.backend, "cube", None)
        contact_geoms = getattr(cube, "contact_geoms", ())
        geom_name2id = getattr(sim_model, "geom_name2id", None)
        if sim_model is None or not callable(geom_name2id):
            return
        config = self._object_config()
        if config.shape != "box":
            raise ContractError("Push runtime geometry updates currently support only box objects.")
        for geom_name in contact_geoms:
            geom_id = geom_name2id(geom_name)
            sim_model.geom_size[geom_id, :3] = np.asarray(config.mujoco_size, dtype=np.float64)
            sim_model.geom_rgba[geom_id] = np.asarray(config.rgba, dtype=np.float32)

    def _object_config(self) -> PrimitiveObjectConfig:
        current = getattr(self.backend, "_current_object_config", None)
        if isinstance(current, PrimitiveObjectConfig):
            return current
        return self.config.object or PrimitiveObjectConfig()

    def _table_z(self) -> float:
        model = getattr(self.backend, "model", None)
        offset = getattr(getattr(model, "mujoco_arena", None), "table_offset", None)
        if offset is None:
            offset = getattr(self.backend, "table_offset", (0.0, 0.0, 0.8))
        values = np.asarray(offset, dtype=np.float64).reshape(-1)
        if values.size < 3:
            raise ContractError("Push backend table offset must contain XYZ coordinates.")
        return float(values[2])

    def _sample_target_xy(self, cube_pos: NDArray[np.float64]) -> tuple[float, float]:
        for _attempt in range(1_000):
            angle = float(np.random.uniform(*self.config.target_angle_range_rad))
            distance = float(np.random.uniform(*self.config.target_distance_range_m))
            target_x = float(cube_pos[0] + math.cos(angle) * distance)
            target_y = float(cube_pos[1] + math.sin(angle) * distance)
            limit = self.config.table_xy_limit_m
            if -limit <= target_x <= limit and -limit <= target_y <= limit:
                return target_x, target_y
        raise RuntimeError("Could not sample a Push target inside the table bounds.")

    def _scene_target(
        self, scene: SceneSpec | None, cube_pos: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if scene is not None and {"target_x_m", "target_y_m"} <= set(scene.overrides):
            target_x = float(scene.overrides["target_x_m"])
            target_y = float(scene.overrides["target_y_m"])
        else:
            target_x, target_y = self._sample_target_xy(cube_pos)
        limit = self.config.table_xy_limit_m
        if not (-limit <= target_x <= limit and -limit <= target_y <= limit):
            raise ContractError("Push target falls outside configured table bounds.")
        return np.asarray([target_x, target_y, self._table_z()], dtype=np.float64)

    def _set_target_site(self) -> None:
        model = getattr(getattr(self.backend, "sim", None), "model", None)
        site_name2id = getattr(model, "site_name2id", None)
        if not callable(site_name2id):
            return
        model.site_pos[site_name2id("push_target_zone")] = self.target_pos
        self.backend.sim.forward()

    def _augment_raw(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        augmented = dict(raw)
        augmented["target_pos"] = self.target_pos
        augmented["object_half_extents_m"] = np.asarray(self._object_config().half_extents_m)
        return augmented

    def reset(
        self, *, seed: int | None = None, scene: SceneSpec | None = None
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self._set_episode_object(scene)
        result = self.backend.reset()
        raw = result[0] if isinstance(result, tuple) else result
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite reset did not return a mapping.")
        if scene is not None:
            self._apply_scene(scene, raw)
        raw = self.backend._get_observations(force_update=True)
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite reset did not return a mapping after scene placement.")
        cube_pos = np.asarray(raw.get("cube_pos"), dtype=np.float64).reshape(-1)
        if cube_pos.size < 3:
            raise ContractError("Push reset observation is missing cube_pos.")
        self._target_pos = self._scene_target(scene, cube_pos[:3])
        self._set_target_site()
        randomization = scene.overrides.get("domain_randomization") if scene else None
        self._randomization_sample = (
            DomainRandomizationSample.from_mapping(randomization)
            if isinstance(randomization, Mapping)
            else None
        )
        self._sim2real_runtime.reset(self._randomization_sample)
        cube = getattr(self.backend, "cube", None)
        apply_domain_randomization(
            self.backend,
            self._render_baseline,
            self._randomization_sample,
            front_camera_name=self.config.camera.name,
            front_look_at_m=(0.0, 0.0, self._table_z()),
            object_body_names=(getattr(cube, "root_body", ""),),
            object_geom_names=tuple(getattr(cube, "contact_geoms", ())),
        )
        raw = self.backend._get_observations(force_update=True)
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite reset did not return a mapping after target placement.")
        self._raw_observation = self._augment_raw(raw)
        self._success_hold_count = 0
        self._visual_step = 0
        return self._policy_observation(self._raw_observation), {"success": False}
            
        if self.config.object is None:
            # Randomize object for next reload
            self.backend._current_object_config = generate_random_object_config()
            
        result = self.backend.reset()
        raw = result[0] if isinstance(result, tuple) else result
        
        # Set target position at least 15cm away from the object
        table_offset = self.backend.model.mujoco_arena.table_offset
        cube_pos = np.asarray(raw.get("cube_pos"), dtype=np.float64).reshape(-1)
        
        while True:
            # Sample random angle in forward cone [-36°, +36°]
            angle = np.random.uniform(-math.pi / 5, math.pi / 5)
            distance = np.random.uniform(0.10, 0.15)
            
            target_x = cube_pos[0] + math.cos(angle) * distance
            target_y = cube_pos[1] + math.sin(angle) * distance
            
            # Ensure target is well within the table boundaries to prevent falling off
            if -0.25 <= target_x <= 0.25 and -0.25 <= target_y <= 0.25:
                self._target_pos = np.array([target_x, target_y, table_offset[2]])
                break
                
        
        # Update site position in MuJoCo
        site_id = self.backend.sim.model.site_name2id("push_target_zone")
        self.backend.sim.model.site_pos[site_id] = self._target_pos
        self.backend.sim.forward()
        
        if scene is not None:
            self._apply_scene(scene, raw)
            
        raw = self.backend._get_observations(force_update=True)
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite reset did not return a mapping.")
            
        self._raw_observation = self._augment_raw(raw)
        
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
                self._table_z() + self._object_config().half_extents_m[2],
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
        execution_action = self._sim2real_runtime.execution_action(normalized_action)
        raw, _, done, backend_info = self.backend.step(execution_action)
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite step() did not return an observation mapping.")
            
        self._raw_observation = self._augment_raw(raw)

        info = dict(backend_info or {})
        
        success = self._check_success()
        
        # Custom reward for pushing
        cube_pos = np.asarray(raw.get("cube_pos"), dtype=np.float64).reshape(-1)
        dist_to_target = float(np.linalg.norm(cube_pos[:2] - self.target_pos[:2]))
        reward = 1.0 if success else -dist_to_target
        
        info["success"] = success
        info["success_hold_count"] = self._success_hold_count
        info["dist_to_target"] = dist_to_target
        info["target_pos"] = self.target_pos
        info["in_target_zone"] = dist_to_target <= self.config.target_radius_m
        
        terminated = bool(success and self.config.terminate_on_success)
        truncated = bool(done and not terminated)
        self._visual_step += 1
        return (
            self._policy_observation(self._raw_observation),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def _policy_observation(self, raw: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
        observation = self._observation_adapter.convert(raw)
        return self._sim2real_runtime.policy_observation(observation, self._visual_step)

    def render(self) -> Any:
        return self.backend.render()

    def close(self) -> None:
        sim = getattr(self.backend, "sim", None)
        free = getattr(sim, "free", None)
        if callable(free):
            try:
                free()
            except AttributeError:
                # robosuite may not create an offscreen render context, while
                # its MjSim.free() still unconditionally deletes the field.
                pass
        close = getattr(self.backend, "close", None)
        if callable(close):
            try:
                close()
            except AttributeError:
                pass

    def _check_success(self) -> bool:
        if self._raw_observation is None or self._target_pos is None:
            return False
        cube_pos = np.asarray(self._raw_observation.get("cube_pos"), dtype=np.float64).reshape(-1)
        if cube_pos.size < 3:
            return False
            
        dist_to_target = float(np.linalg.norm(cube_pos[:2] - self._target_pos[:2]))
        is_in_target = dist_to_target <= self.config.target_radius_m
        
        self._success_hold_count = self._success_hold_count + 1 if is_in_target else 0
        return self._success_hold_count >= self.config.success_hold_steps

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

def make_ur5e_push(config: UR5ePushConfig | None = None) -> UR5ePushEnv:
    """Public environment factory."""
    return UR5ePushEnv.create(config)
