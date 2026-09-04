from __future__ import annotations

import pytest
import torch

from vla_sim.policy.runtime import (
    aggregate_action_chunks,
    postprocess_and_aggregate_action_chunks,
)


def test_aggregate_action_chunks_averages_xyz_and_votes_gripper() -> None:
    chunks = torch.tensor(
        [
            [[[-0.4, 0.2, 0.1, 0.0, 0.0, 0.0, -0.8]]],
            [[[-0.2, 0.4, 0.3, 0.0, 0.0, 0.0, 0.9]]],
        ]
    )
    result = aggregate_action_chunks(chunks, action_dim=7)
    assert torch.allclose(result[..., :3], torch.tensor([[[-0.3, 0.3, 0.2]]]))
    assert result[..., 6].item() == pytest.approx(0.9)


def test_aggregate_action_chunks_uses_majority_gripper_vote() -> None:
    chunks = torch.tensor(
        [
            [[[-0.4, 0.2, 0.1, 0.0, 0.0, 0.0, -0.8]]],
            [[[-0.2, 0.4, 0.3, 0.0, 0.0, 0.0, 0.9]]],
            [[[-0.1, 0.5, 0.4, 0.0, 0.0, 0.0, -0.7]]],
        ]
    )
    assert aggregate_action_chunks(chunks, action_dim=7)[..., 6].item() == -1.0


def test_gripper_vote_uses_postprocessed_action_domain() -> None:
    chunks = torch.zeros((3, 1, 1, 7))
    chunks[:, 0, 0, 6] = torch.tensor([-0.4, -0.1, -0.05])

    def shifted_postprocessor(sample: torch.Tensor) -> torch.Tensor:
        result = sample.clone()
        result[..., 6] += 0.224
        return result

    result = postprocess_and_aggregate_action_chunks(
        chunks, shifted_postprocessor, action_dim=7
    )
    assert result[..., 6].item() == 1.0
