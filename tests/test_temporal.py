from __future__ import annotations

import numpy as np
import pytest

from vla_sim.temporal import TemporalEnsemble


def _chunk(first_gripper: float, second_gripper: float) -> np.ndarray:
    values = np.zeros((2, 7), dtype=np.float32)
    values[:, 0] = [1.0, 3.0]
    values[:, 6] = [first_gripper, second_gripper]
    return values


def test_continuous_actions_are_decay_weighted_and_latest_gripper_wins() -> None:
    ensemble = TemporalEnsemble(2, 7, decay=1.0, gripper_mode="latest")
    ensemble.add_chunk(1, _chunk(-1.0, -1.0))
    ensemble.add_chunk(2, _chunk(1.0, 1.0))
    action = ensemble.get_action(2)
    assert action[0] == pytest.approx(2.0)
    assert action[6] == pytest.approx(1.0)


def test_latch_never_reopens_after_close_threshold() -> None:
    ensemble = TemporalEnsemble(
        2,
        7,
        gripper_mode="latch",
        gripper_close_threshold=0.5,
    )
    ensemble.add_chunk(1, _chunk(0.75, -1.0))
    assert ensemble.get_action(1)[6] == pytest.approx(1.0)
    assert ensemble.get_action(2)[6] == pytest.approx(1.0)


def test_debounce_requires_consecutive_close_commands_then_latches() -> None:
    ensemble = TemporalEnsemble(
        2,
        7,
        gripper_mode="debounce",
        gripper_close_threshold=0.5,
        gripper_confirm_steps=2,
    )
    ensemble.add_chunk(1, _chunk(0.75, -1.0))
    assert ensemble.get_action(1)[6] == pytest.approx(-1.0)
    assert ensemble.get_action(2)[6] == pytest.approx(-1.0)
    ensemble.add_chunk(3, _chunk(0.75, 0.75))
    assert ensemble.get_action(3)[6] == pytest.approx(-1.0)
    assert ensemble.get_action(4)[6] == pytest.approx(1.0)


def test_confirm_requires_consecutive_close_commands_and_can_reopen() -> None:
    ensemble = TemporalEnsemble(
        1,
        7,
        gripper_mode="confirm",
        gripper_confirm_steps=2,
    )
    for step, gripper in enumerate((0.8, 0.9, -0.2, 0.8), start=1):
        action = np.zeros((1, 7), dtype=np.float32)
        action[0, 6] = gripper
        ensemble.add_chunk(step, action)
        expected = 1.0 if step == 2 else -1.0
        assert ensemble.get_action(step)[6] == pytest.approx(expected)


def test_invalid_gripper_confirm_steps_are_rejected() -> None:
    with pytest.raises(ValueError, match="gripper_confirm_steps"):
        TemporalEnsemble(2, 7, gripper_confirm_steps=0)


def test_hysteresis_executes_first_close_then_latches_after_confirmation() -> None:
    ensemble = TemporalEnsemble(
        2,
        7,
        gripper_mode="hysteresis",
        gripper_close_threshold=0.5,
        gripper_confirm_steps=2,
    )
    ensemble.add_chunk(1, _chunk(0.75, 0.75))
    assert ensemble.get_action(1)[6] == pytest.approx(0.75)
    assert ensemble.get_action(2)[6] == pytest.approx(1.0)
    ensemble.add_chunk(3, _chunk(-1.0, -1.0))
    assert ensemble.get_action(3)[6] == pytest.approx(1.0)


def test_hold_filters_short_reopen_then_releases() -> None:
    ensemble = TemporalEnsemble(
        4,
        7,
        gripper_mode="hold",
        gripper_close_threshold=0.5,
        gripper_hold_steps=3,
    )
    values = np.zeros((4, 7), dtype=np.float32)
    values[:, 6] = [0.75, -1.0, -1.0, -1.0]
    ensemble.add_chunk(1, values)
    assert ensemble.get_action(1)[6] == pytest.approx(1.0)
    assert ensemble.get_action(2)[6] == pytest.approx(1.0)
    assert ensemble.get_action(3)[6] == pytest.approx(1.0)
    assert ensemble.get_action(4)[6] == pytest.approx(-1.0)


def test_confirm_then_hold_requires_confirmation_then_can_reopen() -> None:
    ensemble = TemporalEnsemble(
        1,
        7,
        gripper_mode="confirm_then_hold",
        gripper_confirm_steps=2,
        gripper_hold_steps=2,
    )
    outputs = []
    for step, gripper in enumerate((0.8, 0.9, -1.0, -1.0), start=1):
        action = np.zeros((1, 7), dtype=np.float32)
        action[0, 6] = gripper
        ensemble.add_chunk(step, action)
        outputs.append(ensemble.get_action(step)[6])
    assert outputs == pytest.approx([-1.0, 1.0, 1.0, -1.0])
    assert ensemble.last_raw_gripper == pytest.approx(-1.0)


@pytest.mark.parametrize("decay", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_decay_is_rejected(decay: float) -> None:
    with pytest.raises(ValueError, match="decay"):
        TemporalEnsemble(2, 7, decay=decay)


def test_missing_or_malformed_predictions_are_rejected() -> None:
    ensemble = TemporalEnsemble(2, 7)
    with pytest.raises(ValueError, match="Expected"):
        ensemble.add_chunk(1, np.zeros((2, 6)))
    with pytest.raises(ValueError, match="No prediction"):
        ensemble.get_action(1)
