from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vla_sim.policy.losses import action_dimension_weights, weighted_action_loss


def test_standard_weights_match_plain_mean() -> None:
    losses = torch.arange(1, 15, dtype=torch.float32).reshape(1, 2, 7)
    weights = action_dimension_weights(
        7,
        rotation_weight=1.0,
        gripper_weight=1.0,
        device=losses.device,
        dtype=losses.dtype,
    )
    assert weighted_action_loss(losses, None, weights) == pytest.approx(losses.mean().item())


def test_rotation_mask_removes_rotation_gradients_and_gripper_can_be_weighted() -> None:
    losses = torch.ones((1, 1, 7), dtype=torch.float32, requires_grad=True)
    weights = action_dimension_weights(
        7,
        rotation_weight=0.0,
        gripper_weight=2.0,
        device=losses.device,
        dtype=losses.dtype,
    )
    result = weighted_action_loss(losses, None, weights)
    result.backward()
    assert result.item() == pytest.approx(1.0)
    assert torch.count_nonzero(losses.grad[0, 0, 3:6]) == 0
    assert losses.grad[0, 0, 6] == pytest.approx(0.4)


def test_padding_is_excluded_from_weighted_denominator() -> None:
    losses = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])
    padding = torch.tensor([[False, True]])
    weights = torch.tensor([1.0, 1.0])
    assert weighted_action_loss(losses, padding, weights) == pytest.approx(2.0)
    per_sample = weighted_action_loss(losses, padding, weights, reduction="none")
    torch.testing.assert_close(per_sample, torch.tensor([2.0]))


@pytest.mark.parametrize("rotation,gripper", [(-1.0, 1.0), (1.0, float("nan"))])
def test_invalid_dimension_weights_are_rejected(rotation: float, gripper: float) -> None:
    with pytest.raises(ValueError, match="weights"):
        action_dimension_weights(
            7,
            rotation_weight=rotation,
            gripper_weight=gripper,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
