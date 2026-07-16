from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree.ElementTree import Element, SubElement

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vla_sim.envs import PrimitiveObjectConfig, UR5eLiftConfig, UR5eLiftEnv  # noqa: E402
from vla_sim.sim import (  # noqa: E402
    ActionSpec,
    ContractError,
    HeuristicLiftExpert,
    ObservationAdapter,
    RobosuiteUnavailableError,
)
from vla_sim.sim.dependencies import require_robosuite  # noqa: E402


def _raw_observation() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.full((12, 16, 3), 0.5, dtype=np.float32),
        "robot0_joint_pos": np.arange(6, dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([-0.024, -0.024], dtype=np.float32),
        "robot0_eef_pos": np.asarray([0.0, 0.0, 0.9], dtype=np.float32),
        "cube_pos": np.asarray([0.1, 0.1, 0.8], dtype=np.float32),
    }


class FakeBackend:
    action_spec = (-np.ones(7, dtype=np.float32), np.ones(7, dtype=np.float32))

    def __init__(self) -> None:
        self.raw = _raw_observation()
        self.closed = False

    def reset(self) -> dict[str, np.ndarray]:
        return self.raw

    def step(self, action: np.ndarray):
        assert action.shape == (7,)
        return self.raw, 0.25, False, {}

    def _check_success(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


def test_action_contract_rejects_bad_shape_and_can_clip() -> None:
    spec = ActionSpec()
    with pytest.raises(ContractError, match="shape"):
        spec.validate(np.zeros(6))
    with pytest.raises(ContractError, match="NaN"):
        spec.validate([0, 0, 0, 0, 0, np.nan, 0])
    clipped = spec.validate([2, -2, 0, 0, 0, 0, 0], clip=True)
    np.testing.assert_array_equal(clipped[:2], np.asarray([1, -1], dtype=np.float32))


def test_observation_adapter_produces_one_rgb_camera_and_seven_states() -> None:
    observation = ObservationAdapter().convert(_raw_observation())
    assert set(observation) == {"observation.images.front", "observation.state"}
    assert observation["observation.images.front"].shape == (12, 16, 3)
    assert observation["observation.images.front"].dtype == np.uint8
    assert observation["observation.state"].shape == (7,)
    assert observation["observation.state"].dtype == np.float32
    assert observation["observation.state"][-1] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("shape", "dimensions", "expected_size"),
    [
        ("box", (0.04, 0.06, 0.08), (0.02, 0.03, 0.04)),
        ("cylinder", (0.04, 0.04, 0.08), (0.02, 0.04)),
        ("sphere", (0.04, 0.04, 0.04), (0.02,)),
    ],
)
def test_primitive_object_config_mutates_mjcf(
    shape: str,
    dimensions: tuple[float, float, float],
    expected_size: tuple[float, ...],
) -> None:
    root = Element("worldbody")
    body = SubElement(root, "body", name="cube_main")
    collision = SubElement(body, "geom", name="cube_g0", contype="1", conaffinity="1")
    visual = SubElement(body, "geom", name="cube_g0_vis", contype="0", conaffinity="0", material="red")
    config = PrimitiveObjectConfig(shape=shape, dimensions_m=dimensions)  # type: ignore[arg-type]
    fake_object = SimpleNamespace(worldbody=root, root_body="cube_main", size=[1.0] * 3)

    assert config.apply_to_xml(body) == 2
    config.apply_to_robosuite_object(fake_object)
    assert collision.get("type") == shape
    assert tuple(float(value) for value in collision.get("size", "").split()) == expected_size
    assert collision.get("density") == "400"
    assert "material" not in visual.attrib
    np.testing.assert_allclose(fake_object.size, np.asarray(dimensions) / 2.0)


def test_fake_backend_exercises_full_env_contract_without_robosuite() -> None:
    backend = FakeBackend()
    env = UR5eLiftEnv(backend, UR5eLiftConfig())
    observation, info = env.reset(seed=7)
    assert observation["observation.state"].shape == (7,)
    assert info == {"success": False}

    _, reward, terminated, truncated, step_info = env.step(np.zeros(7))
    assert reward == pytest.approx(0.25)
    assert not terminated
    assert not truncated
    assert step_info == {"success": False, "success_hold_count": 0}
    env.close()
    assert backend.closed


def test_heuristic_expert_emits_executable_normalized_action() -> None:
    expert = HeuristicLiftExpert()
    action = expert.act(_raw_observation())
    assert action.shape == (7,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
    assert action[-1] == -1.0


def test_missing_robosuite_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = importlib.import_module

    def fail_robosuite(name: str, package: str | None = None):
        if name == "robosuite":
            raise ModuleNotFoundError("robosuite")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fail_robosuite)
    with pytest.raises(RobosuiteUnavailableError, match=r"pip install -e"):
        require_robosuite()
