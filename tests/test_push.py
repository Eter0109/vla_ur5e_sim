from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vla_sim.envs.ur5e_push import UR5ePushConfig, UR5ePushEnv
from vla_sim.scenes import PUSH_ANGLE_BINS_RAD, PUSH_DISTANCE_BINS_M, generate_push_scenes
from vla_sim.sim.expert import HeuristicPushExpert, PushPhase


ROOT = Path(__file__).resolve().parents[1]


def _raw() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(6, dtype=np.float32),
        "robot0_eef_pos": np.asarray([-0.15, 0.0, 0.84], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.0, 0.0], dtype=np.float32),
        "cube_pos": np.asarray([-0.03, 0.0, 0.825], dtype=np.float32),
    }


class _Data:
    def __init__(self, raw: dict[str, np.ndarray]) -> None:
        self.raw = raw

    def set_joint_qpos(self, _joint: str, qpos: np.ndarray) -> None:
        self.raw["cube_pos"] = np.asarray(qpos[:3], dtype=np.float32)


class _Backend:
    action_spec = (-np.ones(7), np.ones(7))

    def __init__(self) -> None:
        self.raw = _raw()
        self.data = _Data(self.raw)
        self.sim = SimpleNamespace(data=self.data, forward=lambda: None, model=SimpleNamespace())
        self.model = SimpleNamespace(mujoco_arena=SimpleNamespace(table_offset=np.asarray([0.0, 0.0, 0.8])))
        self.cube = SimpleNamespace(joints=["cube_joint"])

    def reset(self) -> dict[str, np.ndarray]:
        return self.raw

    def _get_observations(self, *, force_update: bool) -> dict[str, np.ndarray]:
        assert force_update
        return self.raw

    def step(self, _action: np.ndarray):
        return self.raw, 0.0, False, {}


def test_push_scene_generation_is_deterministic_and_balanced() -> None:
    scenes = generate_push_scenes("push", 20, 52000)
    assert scenes == generate_push_scenes("push", 20, 52000)
    assert {scene.overrides["angle_bin"] for scene in scenes} == set(range(len(PUSH_ANGLE_BINS_RAD)))
    assert {scene.overrides["distance_bin"] for scene in scenes} == set(range(len(PUSH_DISTANCE_BINS_M)))
    for scene in scenes:
        target = scene.overrides
        distance = np.hypot(float(target["target_x_m"]) - scene.x_m, float(target["target_y_m"]) - scene.y_m)
        lower, upper = PUSH_DISTANCE_BINS_M[int(target["distance_bin"])]
        assert lower <= distance <= upper


def test_push_reset_places_scene_before_target_and_requires_stable_hold() -> None:
    scene = generate_push_scenes("push", 20, 52000)[0]
    backend = _Backend()
    env = UR5ePushEnv(backend, UR5ePushConfig(success_hold_steps=2))
    env.reset(scene=scene)
    np.testing.assert_allclose(backend.raw["cube_pos"][:2], [scene.x_m, scene.y_m])
    np.testing.assert_allclose(env.target_pos[:2], [scene.overrides["target_x_m"], scene.overrides["target_y_m"]])
    assert "object_half_extents_m" in env.raw_observation
    backend.raw["cube_pos"][:2] = env.target_pos[:2]
    _, _, terminated, _, info = env.step(np.zeros(7, dtype=np.float32))
    assert not terminated and not info["success"]
    _, _, terminated, _, info = env.step(np.zeros(7, dtype=np.float32))
    assert terminated and info["success"]


def test_push_expert_uses_fixed_rotation_and_closed_loop_goal() -> None:
    expert = HeuristicPushExpert()
    observation = {
        "cube_pos": np.asarray([0.0, 0.0, 0.825]),
        "target_pos": np.asarray([0.12, 0.0, 0.8]),
        "object_half_extents_m": np.asarray([0.025, 0.025, 0.025]),
        "robot0_eef_pos": np.asarray([-0.080, 0.0, 0.84]),
    }
    expert.act(observation)
    assert expert.phase is PushPhase.PUSH
    action = expert.act(observation)
    assert action[0] > 0 and action[6] == 1.0
    np.testing.assert_allclose(action[3:6], 0.0)
    observation["cube_pos"] = np.asarray([0.105, 0.0, 0.825])
    expert.act(observation)
    assert expert.phase is PushPhase.HOLD


def test_push_operational_scripts_configure_offline_cache_before_lerobot_import() -> None:
    for name in ("run_push_vla_only_benchmark.py", "collect_push_demos_v2.py"):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        environment_setup = source.index('"HF_HUB_OFFLINE": "1"')
        lerobot_import = source.index(
            "from lerobot.datasets.lerobot_dataset import LeRobotDataset"
        )

        assert environment_setup < lerobot_import
        assert '"TRANSFORMERS_OFFLINE": "1"' in source
        assert '"HF_HOME": str(RUNTIME / "hf")' in source


def test_push_benchmark_records_full_checkpoint_and_sampling_identity() -> None:
    source = (ROOT / "scripts" / "run_push_vla_only_benchmark.py").read_text(
        encoding="utf-8"
    )

    assert '"checkpoint_sha256": sha256_directory(args.checkpoint)' in source
    assert '"samples_per_plan": args.samples_per_plan' in source
