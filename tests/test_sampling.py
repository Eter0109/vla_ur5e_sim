from __future__ import annotations

import pytest
import torch

from vla_sim.sampling import transition_sampling_weights


def test_transition_window_is_upweighted_without_crossing_episode_boundary() -> None:
    actions = torch.zeros(8, 7)
    actions[:, 6] = torch.tensor([-1, -1, 1, 1, 1, 1, -1, -1])
    episodes = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    weights = transition_sampling_weights(actions, episodes, factor=4, window=1)
    assert weights.tolist() == [1, 4, 4, 4, 1, 4, 4, 4]


@pytest.mark.parametrize("factor", [0.0, 0.5, float("nan")])
def test_invalid_transition_factor_is_rejected(factor: float) -> None:
    with pytest.raises(ValueError, match="factor"):
        transition_sampling_weights(torch.zeros(2, 7), torch.zeros(2), factor=factor)
