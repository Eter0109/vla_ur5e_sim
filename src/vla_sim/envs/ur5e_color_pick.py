"""Language-conditioned UR5e task for selecting one of three colored cubes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vla_sim.contracts import (
    DEFAULT_ACTION_SPEC,
    ActionSpec,
    ContractError,
    PickPlaceObservationAdapter,
)
from vla_sim.domain_randomization import (
    DomainRandomizationSample,
    Sim2RealEpisodeRuntime,
    apply_domain_randomization,
    capture_render_baseline,
)
from vla_sim.scenes import COLOR_PICK_COLORS, COLOR_PICK_TASK, SceneSpec
from vla_sim.sim.dependencies import require_robosuite

from .objects import PrimitiveObjectConfig
from .ur5e_lift import _controller_config
from .ur5e_pick_place import PickPlaceCameraConfig, _look_at_quaternion


def _colored_cube(rgba: tuple[float, float, float, float]) -> PrimitiveObjectConfig:
    return PrimitiveObjectConfig(dimensions_m=(0.05, 0.05, 0.05), rgba=rgba)


@dataclass(frozen=True)
class UR5eColorPickConfig:
    """Configuration and success contract for three-color target selection."""

    camera: PickPlaceCameraConfig = field(default_factory=PickPlaceCameraConfig)
    red_object: PrimitiveObjectConfig = field(
        default_factory=lambda: _colored_cube((0.85, 0.08, 0.05, 1.0))
    )
    green_object: PrimitiveObjectConfig = field(
        default_factory=lambda: _colored_cube((0.08, 0.70, 0.18, 1.0))
    )
    blue_object: PrimitiveObjectConfig = field(
        default_factory=lambda: _colored_cube((0.08, 0.22, 0.85, 1.0))
    )
    control_frequency_hz: int = 10
    horizon: int = 200
    seed: int | None = 0
    reward_shaping: bool = True
    terminate_on_success: bool = True
    terminate_on_wrong_grasp: bool = True
    has_renderer: bool = False
    use_camera_depths: bool = True
    required_lift_m: float = 0.080
    success_hold_steps: int = 5

    def __post_init__(self) -> None:
        if self.control_frequency_hz <= 0 or self.horizon <= 0:
            raise ValueError("Control frequency and horizon must be positive.")
        if self.required_lift_m <= 0 or self.success_hold_steps < 1:
            raise ValueError("Lift threshold and success hold steps must be positive.")
        objects = tuple(self.object_configs.values())
        if any(obj.shape != "box" for obj in objects):
            raise ValueError("ColorPick v1 requires three box objects.")
        if len({obj.dimensions_m for obj in objects}) != 1:
            raise ValueError("ColorPick cubes must share one geometry.")

    @property
    def object_configs(self) -> dict[str, PrimitiveObjectConfig]:
        return {
            "red": self.red_object,
            "green": self.green_object,
            "blue": self.blue_object,
        }


@lru_cache(maxsize=1)
def _configurable_color_pick_class() -> type[Any]:
    """Create a Lift derivative with two additional free-joint cubes."""

    from robosuite.environments.manipulation.lift import Lift

    class ConfigurableColorPickLift(Lift):
        def __init__(
            self,
            *args: Any,
            color_object_configs: Mapping[str, PrimitiveObjectConfig],
            **kwargs: Any,
        ) -> None:
            self._color_object_configs = dict(color_object_configs)
            super().__init__(*args, **kwargs)

        def _load_model(self) -> None:
            super()._load_model()
            from robosuite.models.objects import BoxObject

            red_config = self._color_object_configs["red"]
            red_config.apply_to_robosuite_object(self.cube)
            merged_red = self.model.worldbody.find(f".//body[@name='{self.cube.root_body}']")
            if merged_red is None:
                raise RuntimeError("Unable to locate the merged red cube body.")
            red_config.apply_to_xml(merged_red)

            extra_cubes: dict[str, Any] = {}
            for color in ("green", "blue"):
                config = self._color_object_configs[color]
                extra_cubes[color] = BoxObject(
                    name=f"{color}_cube",
                    size=config.half_extents_m,
                    density=config.density_kg_m3,
                    friction=config.friction,
                    rgba=config.rgba,
                    joints="default",
                )
            self.green_cube = extra_cubes["green"]
            self.blue_cube = extra_cubes["blue"]
            self.model.merge_objects([self.green_cube, self.blue_cube])
            self.color_cubes = {
                "red": self.cube,
                "green": self.green_cube,
                "blue": self.blue_cube,
            }

    return ConfigurableColorPickLift


def create_color_pick_backend(config: UR5eColorPickConfig | None = None) -> Any:
    config = config or UR5eColorPickConfig()
    suite = require_robosuite()
    camera = config.camera
    backend = _configurable_color_pick_class()(
        color_object_configs=config.object_configs,
        robots="UR5e",
        gripper_types="Robotiq85Gripper",
        controller_configs=_controller_config(suite),
        control_freq=config.control_frequency_hz,
        horizon=config.horizon,
        reward_shaping=config.reward_shaping,
        has_renderer=config.has_renderer,
        render_camera=camera.third_person.name,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[camera.third_person.name, "all-eye_in_hand"],
        camera_widths=[camera.third_person.width, camera.wrist_width],
        camera_heights=[camera.third_person.height, camera.wrist_height],
        camera_depths=[config.use_camera_depths, config.use_camera_depths],
        camera_segmentations=[None, None],
        hard_reset=False,
        ignore_done=False,
        seed=config.seed,
    )
    third_camera_id = backend.sim.model.camera_name2id(camera.third_person.name)
    third_position = np.asarray(camera.third_person_position_m, dtype=np.float64)
    third_target = np.asarray(camera.third_person_look_at_m, dtype=np.float64)
    backend.sim.model.cam_pos[third_camera_id] = third_position
    backend.sim.model.cam_quat[third_camera_id] = _look_at_quaternion(
        third_position, third_target
    )
    backend.sim.model.cam_fovy[third_camera_id] = camera.third_person_fovy_deg
    wrist_camera_id = backend.sim.model.camera_name2id("robot0_eye_in_hand")
    backend.sim.model.cam_pos[wrist_camera_id] = np.asarray(
        [camera.wrist_forward_offset_m, 0.0, 0.0], dtype=np.float64
    )
    backend.sim.model.cam_fovy[wrist_camera_id] = camera.wrist_fovy_deg
    return backend


class UR5eColorPickEnv:
    """Gymnasium-style wrapper that never exposes target poses to the policy."""

    action_spec: ActionSpec = DEFAULT_ACTION_SPEC

    def __init__(self, backend: Any, config: UR5eColorPickConfig | None = None) -> None:
        self.config = config or UR5eColorPickConfig()
        self.backend = backend
        camera = self.config.camera
        self._observation_adapter = PickPlaceObservationAdapter(
            camera_name=camera.third_person.name,
            flip_vertical=camera.third_person.flip_vertical,
            wrist_camera_name=camera.wrist_name,
            wrist_flip_vertical=camera.wrist_flip_vertical,
        )
        self._raw_observation: Mapping[str, Any] | None = None
        self._target_color = "red"
        self._initial_z: dict[str, float] = {}
        self._success_hold_count = 0
        self._ever_target_grasped = False
        self._ever_wrong_object_grasped = False
        self._render_baseline = capture_render_baseline(backend)
        self._randomization_sample: DomainRandomizationSample | None = None
        self._sim2real_runtime = Sim2RealEpisodeRuntime()
        self._visual_step = 0
        self._validate_backend_action_space()

    @classmethod
    def create(cls, config: UR5eColorPickConfig | None = None) -> UR5eColorPickEnv:
        effective = config or UR5eColorPickConfig()
        return cls(create_color_pick_backend(effective), effective)

    @property
    def raw_observation(self) -> Mapping[str, Any]:
        if self._raw_observation is None:
            raise RuntimeError("reset() must be called before reading observations.")
        return self._raw_observation

    @property
    def target_color(self) -> str:
        return self._target_color

    def reset(
        self, *, seed: int | None = None, scene: SceneSpec | None = None
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        result = self.backend.reset()
        self._reset_robot_pose()
        raw = self._backend_observations(result)
        effective_scene = scene or self._default_scene(seed)
        self._validate_scene(effective_scene)
        self._target_color = str(effective_scene.overrides["target_color"])
        self._apply_scene(effective_scene)
        randomization = effective_scene.overrides.get("domain_randomization")
        self._randomization_sample = (
            DomainRandomizationSample.from_mapping(randomization)
            if isinstance(randomization, Mapping)
            else None
        )
        self._sim2real_runtime.reset(self._randomization_sample)
        cubes = self._color_cubes()
        apply_domain_randomization(
            self.backend,
            self._render_baseline,
            self._randomization_sample,
            front_camera_name=self.config.camera.third_person.name,
            front_look_at_m=self.config.camera.third_person_look_at_m,
            wrist_camera_name=self.config.camera.wrist_name,
            object_body_names=tuple(cube.root_body for cube in cubes.values()),
            object_geom_names=tuple(
                geom_name
                for cube in cubes.values()
                for geom_name in getattr(cube, "contact_geoms", ())
            ),
        )
        raw = self._backend_observations(raw)
        self._raw_observation = self._augment_raw(raw)
        self._initial_z = {
            color: float(self._position(color)[2]) for color in COLOR_PICK_COLORS
        }
        self._success_hold_count = 0
        self._ever_target_grasped = False
        self._ever_wrong_object_grasped = False
        self._visual_step = 0
        return self._policy_observation(self._raw_observation), {
            "success": False,
            "target_color": self._target_color,
        }

    def step(
        self, action: Any
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        requested_action = self.action_spec.validate(action)
        execution_action = self._sim2real_runtime.execution_action(requested_action)
        raw, _backend_reward, done, backend_info = self.backend.step(execution_action)
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite step() did not return an observation mapping.")
        self._raw_observation = self._augment_raw(raw)
        grasped = {color: self._is_grasped(color) for color in COLOR_PICK_COLORS}
        target_grasped = grasped[self._target_color]
        wrong_colors = [
            color for color in COLOR_PICK_COLORS if color != self._target_color and grasped[color]
        ]
        self._ever_target_grasped = self._ever_target_grasped or target_grasped
        self._ever_wrong_object_grasped = (
            self._ever_wrong_object_grasped or bool(wrong_colors)
        )
        target_lift_m = float(
            self._position(self._target_color)[2] - self._initial_z[self._target_color]
        )
        valid = (
            target_grasped
            and target_lift_m >= self.config.required_lift_m
            and not self._ever_wrong_object_grasped
        )
        self._success_hold_count = self._success_hold_count + 1 if valid else 0
        success = self._success_hold_count >= self.config.success_hold_steps
        wrong_grasp_failure = bool(
            self.config.terminate_on_wrong_grasp and self._ever_wrong_object_grasped
        )
        terminated = bool(
            (success and self.config.terminate_on_success) or wrong_grasp_failure
        )
        info = dict(backend_info or {})
        info.update(
            {
                "success": success,
                "target_color": self._target_color,
                "target_grasped": target_grasped,
                "ever_target_grasped": self._ever_target_grasped,
                "wrong_object_grasped": bool(wrong_colors),
                "wrong_colors_grasped": wrong_colors,
                "ever_wrong_object_grasped": self._ever_wrong_object_grasped,
                "target_lift_m": target_lift_m,
                "target_lifted": target_lift_m >= self.config.required_lift_m,
                "success_hold_count": self._success_hold_count,
            }
        )
        reward = self._reward(success, target_grasped, target_lift_m, wrong_grasp_failure)
        self._visual_step += 1
        return (
            self._policy_observation(self._raw_observation),
            reward,
            terminated,
            bool(done and not terminated),
            info,
        )

    def _policy_observation(
        self, raw: Mapping[str, Any]
    ) -> dict[str, NDArray[Any]]:
        observation = self._observation_adapter.convert(raw)
        return self._sim2real_runtime.policy_observation(observation, self._visual_step)

    def _reward(
        self, success: bool, target_grasped: bool, lift_m: float, wrong_grasp: bool
    ) -> float:
        if wrong_grasp:
            return -1.0
        if success:
            return 1.0
        if not self.config.reward_shaping:
            return 0.0
        progress = float(np.clip(lift_m / self.config.required_lift_m, 0.0, 1.0))
        return 0.25 * float(target_grasped) + 0.5 * progress

    def _default_scene(self, seed: int | None) -> SceneSpec:
        effective_seed = self.config.seed if seed is None else seed
        return SceneSpec(
            scene_id="color_pick_default",
            seed=0 if effective_seed is None else int(effective_seed),
            x_m=-0.065,
            y_m=-0.075,
            yaw_rad=0.0,
            overrides={
                "task": COLOR_PICK_TASK,
                "target_color": "red",
                "green_x_m": 0.065,
                "green_y_m": -0.075,
                "blue_x_m": 0.0,
                "blue_y_m": 0.080,
            },
        )

    def _validate_scene(self, scene: SceneSpec) -> None:
        if scene.overrides.get("task") != COLOR_PICK_TASK:
            raise ContractError(f"ColorPick scene task must be {COLOR_PICK_TASK!r}.")
        if scene.overrides.get("target_color") not in COLOR_PICK_COLORS:
            raise ContractError("ColorPick target_color must be red, green, or blue.")
        required = ("green_x_m", "green_y_m", "blue_x_m", "blue_y_m")
        if any(key not in scene.overrides for key in required):
            raise ContractError("ColorPick scene is missing a distractor cube position.")

    def _apply_scene(self, scene: SceneSpec) -> None:
        table_z = float(np.asarray(self.backend.table_offset, dtype=np.float64)[2])
        positions = {
            "red": (scene.x_m, scene.y_m, scene.yaw_rad),
            "green": (
                float(scene.overrides["green_x_m"]),
                float(scene.overrides["green_y_m"]),
                float(scene.overrides.get("green_yaw_rad", 0.0)),
            ),
            "blue": (
                float(scene.overrides["blue_x_m"]),
                float(scene.overrides["blue_y_m"]),
                float(scene.overrides.get("blue_yaw_rad", 0.0)),
            ),
        }
        cubes = self._color_cubes()
        for color, (x_m, y_m, yaw_rad) in positions.items():
            half_yaw = yaw_rad / 2.0
            object_z = table_z + self.config.object_configs[color].half_extents_m[2]
            qpos = np.asarray(
                [x_m, y_m, object_z, np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
                dtype=np.float64,
            )
            self.backend.sim.data.set_joint_qpos(cubes[color].joints[0], qpos)
            try:
                self.backend.sim.data.set_joint_qvel(
                    cubes[color].joints[0], np.zeros(6, dtype=np.float64)
                )
            except (AttributeError, KeyError, ValueError):
                pass
        self.backend.sim.forward()

    def _augment_raw(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        augmented = dict(raw)
        for color in COLOR_PICK_COLORS:
            augmented[f"{color}_cube_pos"] = self._sim_position(color, raw)
        augmented["target_color"] = self._target_color
        augmented["target_cube_pos"] = augmented[
            f"{self._target_color}_cube_pos"
        ].copy()
        return augmented

    def _position(self, color: str) -> NDArray[np.float64]:
        if self._raw_observation is None:
            raise ContractError("reset() must be called before reading cube positions.")
        value = np.asarray(
            self._raw_observation.get(f"{color}_cube_pos"), dtype=np.float64
        ).reshape(-1)
        if value.size < 3 or not np.all(np.isfinite(value[:3])):
            raise ContractError(f"ColorPick observation requires {color}_cube_pos.")
        return value[:3].copy()

    def _sim_position(
        self, color: str, raw: Mapping[str, Any]
    ) -> NDArray[np.float64]:
        key = f"{color}_cube_pos"
        if key in raw:
            value = np.asarray(raw[key], dtype=np.float64).reshape(-1)
            if value.size >= 3 and np.all(np.isfinite(value[:3])):
                return value[:3].copy()
        if color == "red" and "cube_pos" in raw:
            value = np.asarray(raw["cube_pos"], dtype=np.float64).reshape(-1)
            if value.size >= 3 and np.all(np.isfinite(value[:3])):
                return value[:3].copy()
        cube = self._color_cubes()[color]
        data = self.backend.sim.data
        try:
            return np.asarray(data.get_body_xpos(cube.root_body), dtype=np.float64).copy()
        except (AttributeError, KeyError, ValueError):
            model = self.backend.sim.model
            body_id = model.body_name2id(cube.root_body)
            return np.asarray(data.body_xpos[body_id], dtype=np.float64).copy()

    def _color_cubes(self) -> dict[str, Any]:
        cubes = getattr(self.backend, "color_cubes", None)
        if isinstance(cubes, Mapping) and set(cubes) == set(COLOR_PICK_COLORS):
            return dict(cubes)
        red = getattr(self.backend, "cube", None)
        green = getattr(self.backend, "green_cube", None)
        blue = getattr(self.backend, "blue_cube", None)
        if any(value is None for value in (red, green, blue)):
            raise ContractError("ColorPick backend must expose red, green, and blue cubes.")
        return {"red": red, "green": green, "blue": blue}

    def _is_grasped(self, color: str) -> bool:
        checker = getattr(self.backend, "_check_grasp", None)
        robots = getattr(self.backend, "robots", None)
        cube = self._color_cubes()[color]
        return bool(callable(checker) and robots and checker(robots[0].gripper, cube))

    def _reset_robot_pose(self) -> None:
        robots = getattr(self.backend, "robots", ())
        if not robots:
            raise ContractError("ColorPick backend must expose one robot for reset.")
        reset = getattr(robots[0], "reset", None)
        if callable(reset):
            reset(deterministic=True)
            self.backend.sim.forward()

    def _backend_observations(self, fallback: Any) -> Mapping[str, Any]:
        getter = getattr(self.backend, "_get_observations", None)
        raw = getter(force_update=True) if callable(getter) else fallback
        if isinstance(raw, tuple):
            raw = raw[0]
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite reset() did not return an observation mapping.")
        return raw

    def _validate_backend_action_space(self) -> None:
        spec = getattr(self.backend, "action_spec", None)
        if spec is None:
            return
        low, high = spec
        if np.asarray(low).size != 7 or np.asarray(high).size != 7:
            raise ContractError("UR5e + Robotiq85 must expose seven actions.")

    def render(self) -> Any:
        return self.backend.render()

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


def make_ur5e_color_pick(config: UR5eColorPickConfig | None = None) -> UR5eColorPickEnv:
    return UR5eColorPickEnv.create(config)
