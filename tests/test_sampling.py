from __future__ import annotations

import pytest
import torch

from vla_sim.policy.sampling import task_sampling_weights


def test_task_sampling_weights_match_requested_probability_mass() -> None:
    labels = ["push"] * 6 + ["pick"] * 3 + ["blue", "green", "red"]
    targets = {
        "push": 1 / 3,
        "pick": 1 / 3,
        "blue": 1 / 9,
        "green": 1 / 9,
        "red": 1 / 9,
    }
    weights = task_sampling_weights(labels, targets)
    normalized = weights / weights.sum()
    for label, target in targets.items():
        mask = torch.tensor([value == label for value in labels])
        assert float(normalized[mask].sum()) == pytest.approx(target)


def test_task_sampling_weights_reject_contract_drift() -> None:
    with pytest.raises(ValueError, match="contract mismatch"):
        task_sampling_weights(["push", "pick"], {"push": 1.0})
    with pytest.raises(ValueError, match="sum to 1"):
        task_sampling_weights(["push", "pick"], {"push": 0.4, "pick": 0.4})
