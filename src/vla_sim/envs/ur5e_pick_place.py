"""Dual-camera UR5e red-cube-to-storage-bin PickPlace environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from vla_sim.contracts import (
    ActionSpec,
    ContractError,
    DEFAULT_ACTION_SPEC,
    PickPlaceObservationAdapter,
)
from vla_sim.domain_randomization import (
    DomainRandomizationSample,
    apply_domain_randomization,
    capture_render_baseline,
    policy_observation_randomized,
)
from vla_sim.scenes import SceneSpec
from vla_sim.sim.dependencies import require_robosuite

from .objects import PrimitiveObjectConfig
from .ur5e_lift import CameraConfig, _controller_config


@dataclass(frozen=True)
class PickPlaceCameraConfig:
    """Camera contract shared by simulation, data collection and rollout."""

    third_person: CameraConfig = field(default_factory=CameraConfig)
    # Front RGB camera: 45-degree downward view of the workspace center.
    third_person_position_m: tuple[float, float, float] = (0.70, 0.0, 1.46)
    third_person_look_at_m: tuple[float, float, float] = (0.05, 0.0, 0.81)
    third_person_fovy_deg: float = 55.0
    wrist_name: str = "robot0_eye_in_hand"
    wrist_width: int = 256
    wrist_height: int = 256
    wrist_fovy_deg: float = 100.0
    wrist_forward_offset_m: float = 0.10
    wrist_flip_vertical: bool = False

    def __post_init__(self) -> None:
        if (
            not self.wrist_name
            or self.wrist_width <= 0
            or self.wrist_height <= 0
            or not 1.0 < self.wrist_fovy_deg < 179.0
            or self.wrist_forward_offset_m <= 0.0
            or not 1.0 < self.third_person_fovy_deg < 179.0
        ):
            raise ValueError("Wrist camera name and dimensions must be positive.")


@dataclass(frozen=True)
class UR5ePickPlaceConfig:
    camera: PickPlaceCameraConfig = field(default_factory=PickPlaceCameraConfig)
    object: PrimitiveObjectConfig = field(default_factory=PrimitiveObjectConfig)
    target_size_m: float = 0.12
    target_marker_height_m: float = 0.002
    control_frequency_hz: int = 10
    horizon: int = 250
    seed: int | None = 0
    reward_shaping: bool = True
    terminate_on_success: bool = True
    has_renderer: bool = False
    use_camera_depths: bool = True
    success_hold_steps: int = 10
    placement_tolerance_m: float = 0.030
    table_height_tolerance_m: float = 0.012
    required_lift_m: float = 0.040
    max_linear_speed_m_s: float = 0.020
    max_angular_speed_rad_s: float = 0.25

    def __post_init__(self) -> None:
        if self.control_frequency_hz <= 0 or self.horizon <= 0 or self.success_hold_steps < 1:
            raise ValueError("Control frequency, horizon and hold steps must be positive.")
        values = (
            self.target_size_m,
            self.target_marker_height_m,
            self.placement_tolerance_m,
            self.table_height_tolerance_m,
            self.required_lift_m,
            self.max_linear_speed_m_s,
            self.max_angular_speed_rad_s,
        )
        if any(value <= 0 for value in values):
            raise ValueError("PickPlace geometry and threshold values must be positive.")
        if self.placement_tolerance_m > (self.target_size_m - self.object.dimensions_m[0]) / 2:
            raise ValueError("placement_tolerance_m must keep the cube fully inside the target zone.")


@lru_cache(maxsize=1)
def _configurable_pick_place_class() -> type[Any]:
    from robosuite.environments.manipulation.lift import Lift

    class ConfigurablePickPlaceLift(Lift):
        def __init__(
            self,
            *args: Any,
            primitive_object_config: PrimitiveObjectConfig,
            target_size_m: float,
            target_marker_height_m: float,
            **kwargs: Any,
        ) -> None:
            self._primitive_object_config = primitive_object_config
            self._target_size_m = target_size_m
            self._target_marker_height_m = target_marker_height_m
            super().__init__(*args, **kwargs)

        def _load_model(self) -> None:
            super()._load_model()
            from robosuite.models.objects.composite import Bin

            self._primitive_object_config.apply_to_robosuite_object(self.cube)
            merged = self.model.worldbody.find(f".//body[@name='{self.cube.root_body}']")
            if merged is None:
                raise RuntimeError("Unable to locate merged PickPlace cube body.")
            self._primitive_object_config.apply_to_xml(merged)
            self.target_bin = Bin(
                name="target_storage_bin",
                bin_size=(self._target_size_m, self._target_size_m, 0.040),
                wall_thickness=0.005,
                transparent_walls=False,
                friction=(1.0, 0.005, 0.0001),
                use_texture=False,
                rgba=(0.12, 0.30, 0.75, 1.0),
            )
            self.model.merge_objects([self.target_bin])

    return ConfigurablePickPlaceLift


def create_pick_place_backend(config: UR5ePickPlaceConfig | None = None) -> Any:
    config = config or UR5ePickPlaceConfig()
    suite = require_robosuite()
    camera = config.camera
    backend = _configurable_pick_place_class()(
        primitive_object_config=config.object,
        target_size_m=config.target_size_m,
        target_marker_height_m=config.target_marker_height_m,
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
        # Keep the training-time RGB-D render configuration by default. Even the
        # RGB policy input is sensitive to the renderer's observation lifecycle.
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
    backend.sim.model.cam_quat[third_camera_id] = _look_at_quaternion(third_position, third_target)
    backend.sim.model.cam_fovy[third_camera_id] = camera.third_person_fovy_deg
    wrist_camera_id = backend.sim.model.camera_name2id("robot0_eye_in_hand")
    backend.sim.model.cam_pos[wrist_camera_id] = np.asarray(
        [camera.wrist_forward_offset_m, 0.0, 0.0], dtype=np.float64
    )
    backend.sim.model.cam_fovy[wrist_camera_id] = camera.wrist_fovy_deg
    # Bin is merged as a free-joint composite object.
    # Compensate gravity to keep the storage bin fixed on the table instead of
    # letting it fall through the table after reset.
    bin_body_id = backend.sim.model.body_name2id(backend.target_bin.root_body)
    backend.sim.model.body_gravcomp[bin_body_id] = 1.0
    return backend


def _look_at_quaternion(position: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a MuJoCo wxyz camera quaternion looking from ``position`` at ``target``."""

    forward = target - position
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        next_index, last_index = (index + 1) % 3, (index + 2) % 3
        scale = 2.0 * np.sqrt(1.0 + rotation[index, index] - rotation[next_index, next_index] - rotation[last_index, last_index])
        quaternion = np.zeros(4, dtype=np.float64)
        quaternion[index + 1] = 0.25 * scale
        quaternion[0] = (rotation[last_index, next_index] - rotation[next_index, last_index]) / scale
        quaternion[next_index + 1] = (rotation[next_index, index] + rotation[index, next_index]) / scale
        quaternion[last_index + 1] = (rotation[last_index, index] + rotation[index, last_index]) / scale
    return quaternion / np.linalg.norm(quaternion)


class UR5ePickPlaceEnv:
    action_spec: ActionSpec = DEFAULT_ACTION_SPEC

    def __init__(self, backend: Any, config: UR5ePickPlaceConfig | None = None) -> None:
        self.config = config or UR5ePickPlaceConfig()
        self.backend = backend
        camera = self.config.camera
        self._observation_adapter = PickPlaceObservationAdapter(
            camera_name=camera.third_person.name,
            flip_vertical=camera.third_person.flip_vertical,
            wrist_camera_name=camera.wrist_name,
            wrist_flip_vertical=camera.wrist_flip_vertical,
        )
        self._raw_observation: Mapping[str, Any] | None = None
        self._initial_cube_z: float | None = None
        self._target_xyz = np.zeros(3, dtype=np.float64)
        self._target_bin_qpos = np.zeros(7, dtype=np.float64)
        self._success_hold_count = 0
        self._ever_grasped = False
        self._ever_lifted = False
        self._render_baseline = capture_render_baseline(backend)
        self._randomization_sample: DomainRandomizationSample | None = None
        self._visual_step = 0
        self._validate_backend_action_space()

    @classmethod
    def create(cls, config: UR5ePickPlaceConfig | None = None) -> "UR5ePickPlaceEnv":
        effective = config or UR5ePickPlaceConfig()
        return cls(create_pick_place_backend(effective), effective)

    @property
    def raw_observation(self) -> Mapping[str, Any]:
        if self._raw_observation is None:
            raise RuntimeError("reset() must be called before reading observations.")
        return self._raw_observation

    @property
    def target_xyz(self) -> NDArray[np.float64]:
        return self._target_xyz.copy()

    def reset(self, *, seed: int | None = None, scene: SceneSpec | None = None):
        if seed is not None:
            np.random.seed(seed)
        self.backend.reset()
        self._reset_robot_pose()
        raw = self.backend._get_observations(force_update=True)
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite reset() did not return an observation mapping.")
        target_x = float(scene.overrides.get("target_x_m", 0.060)) if scene else 0.060
        target_y = float(scene.overrides.get("target_y_m", 0.0)) if scene else 0.0
        if scene is not None:
            self._apply_cube_scene(scene, raw)
        table_z = float(np.asarray(self.backend.table_offset, dtype=np.float64)[2])
        self._set_target(target_x, target_y, table_z)
        randomization = scene.overrides.get("domain_randomization") if scene else None
        self._randomization_sample = (
            DomainRandomizationSample.from_mapping(randomization)
            if isinstance(randomization, Mapping)
            else None
        )
        apply_domain_randomization(
            self.backend,
            self._render_baseline,
            self._randomization_sample,
            front_camera_name=self.config.camera.third_person.name,
            front_look_at_m=self.config.camera.third_person_look_at_m,
            wrist_camera_name=self.config.camera.wrist_name,
        )
        refreshed = self.backend._get_observations(force_update=True)
        self._raw_observation = self._with_target(refreshed)
        cube = self._cube_position()
        self._initial_cube_z = float(cube[2])
        self._success_hold_count = 0
        self._ever_grasped = False
        self._ever_lifted = False
        self._visual_step = 0
        return self._policy_observation(self._raw_observation), {"success": False}

    def _reset_robot_pose(self) -> None:
        """Restore robot joints and controller goals between independent episodes."""
        robots = getattr(self.backend, "robots", ())
        if not robots:
            raise ContractError("PickPlace backend must expose one robot for reset.")
        robots[0].reset(deterministic=True)
        self.backend.sim.forward()

    def _apply_cube_scene(self, scene: SceneSpec, raw: Mapping[str, Any]) -> None:
        cube = np.asarray(raw.get("cube_pos"), dtype=np.float64).reshape(-1)
        if cube.size < 3:
            raise ContractError("Cannot apply PickPlace scene without cube_pos.")
        half_yaw = scene.yaw_rad / 2.0
        qpos = np.asarray([scene.x_m, scene.y_m, cube[2], np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
        self.backend.sim.data.set_joint_qpos(self.backend.cube.joints[0], qpos)
        self.backend.sim.forward()

    def _set_target(self, x: float, y: float, table_z: float) -> None:
        self._target_xyz = np.asarray([x, y, table_z + self.config.target_marker_height_m], dtype=np.float64)
        self._target_bin_qpos = np.asarray([x, y, table_z + 0.020, 1.0, 0.0, 0.0, 0.0])
        self._lock_target_bin()
        self.backend.sim.forward()

    def _lock_target_bin(self) -> bool:
        """Keep the render-only free-joint Bin at its configured table pose."""
        target_bin = getattr(self.backend, "target_bin", None)
        if target_bin is None:
            return False
        bin_joint = target_bin.joints[0]
        self.backend.sim.data.set_joint_qpos(bin_joint, self._target_bin_qpos)
        self.backend.sim.data.set_joint_qvel(bin_joint, np.zeros(6, dtype=np.float64))
        return True

    def _with_target(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        augmented = dict(raw)
        augmented["target_zone_pos"] = self._target_xyz.copy()
        return augmented

    def step(self, action: Any):
        raw, reward, done, backend_info = self.backend.step(self.action_spec.validate(action))
        if self._lock_target_bin():
            self.backend.sim.forward()
            raw = self.backend._get_observations(force_update=True)
        if not isinstance(raw, Mapping):
            raise ContractError("robosuite step() did not return an observation mapping.")
        self._raw_observation = self._with_target(raw)
        cube = self._cube_position()
        grasped = self._is_grasped()
        lift_m = float(cube[2] - (self._initial_cube_z if self._initial_cube_z is not None else cube[2]))
        self._ever_grasped = self._ever_grasped or grasped
        self._ever_lifted = self._ever_lifted or lift_m >= self.config.required_lift_m
        success, conditions = self._check_success()
        info = dict(backend_info or {})
        info.update({
            "success": success,
            "success_hold_count": self._success_hold_count,
            "grasped": grasped,
            "lifted": lift_m >= self.config.required_lift_m,
            "object_lift_m": lift_m,
            "ever_grasped": self._ever_grasped,
            "ever_lifted": self._ever_lifted,
            "place_conditions": conditions,
            "target_xy_error_m": conditions["xy_error_m"],
        })
        terminated = bool(success and self.config.terminate_on_success)
        self._visual_step += 1
        return self._policy_observation(self._raw_observation), float(reward), terminated, bool(done and not terminated), info

    def _policy_observation(self, raw: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
        observation = self._observation_adapter.convert(raw)
        return policy_observation_randomized(observation, self._randomization_sample, self._visual_step)

    def _check_success(self) -> tuple[bool, dict[str, Any]]:
        cube = self._cube_position()
        delta = cube - self._target_xyz
        linear, angular = self._body_speeds(getattr(self.backend, "cube", None))
        table_z = self._target_xyz[2] - self.config.target_marker_height_m
        conditions = {
            "ever_grasped": self._ever_grasped,
            "ever_lifted": self._ever_lifted,
            "in_target_zone": abs(float(delta[0])) <= self.config.placement_tolerance_m and abs(float(delta[1])) <= self.config.placement_tolerance_m,
            "on_table": abs(float(cube[2] - (table_z + self.config.object.half_extents_m[2]))) <= self.config.table_height_tolerance_m,
            "gripper_released": not self._is_grasped(),
            "objects_stable": linear <= self.config.max_linear_speed_m_s and angular <= self.config.max_angular_speed_rad_s,
            "xy_error_m": float(np.linalg.norm(delta[:2])),
            "linear_speed_m_s": linear,
            "angular_speed_rad_s": angular,
        }
        valid = all(bool(conditions[name]) for name in ("ever_grasped", "ever_lifted", "in_target_zone", "on_table", "gripper_released", "objects_stable"))
        self._success_hold_count = self._success_hold_count + 1 if valid else 0
        return self._success_hold_count >= self.config.success_hold_steps, conditions

    def _cube_position(self) -> NDArray[np.float64]:
        if self._raw_observation is None:
            raise ContractError("reset() must be called before reading cube position.")
        cube = np.asarray(self._raw_observation.get("cube_pos"), dtype=np.float64).reshape(-1)
        if cube.size < 3 or not np.all(np.isfinite(cube[:3])):
            raise ContractError("PickPlace observation requires cube_pos.")
        return cube[:3]

    def _is_grasped(self) -> bool:
        checker = getattr(self.backend, "_check_grasp", None)
        robots = getattr(self.backend, "robots", None)
        cube = getattr(self.backend, "cube", None)
        return bool(callable(checker) and robots and cube is not None and checker(robots[0].gripper, cube))

    def _body_speeds(self, body: Any) -> tuple[float, float]:
        root = getattr(body, "root_body", None)
        data = getattr(getattr(self.backend, "sim", None), "data", None)
        if root is None or data is None:
            return float("inf"), float("inf")
        try:
            linear = np.asarray(data.get_body_xvelp(root), dtype=np.float64)
            angular = np.asarray(data.get_body_xvelr(root), dtype=np.float64)
        except (AttributeError, KeyError, ValueError):
            return float("inf"), float("inf")
        return float(np.linalg.norm(linear)), float(np.linalg.norm(angular))

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


def make_ur5e_pick_place(config: UR5ePickPlaceConfig | None = None) -> UR5ePickPlaceEnv:
    return UR5ePickPlaceEnv.create(config)
