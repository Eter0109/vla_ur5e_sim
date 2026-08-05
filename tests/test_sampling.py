from __future__ import annotations

import pytest
import torch
from datasets import Dataset

from vla_sim.sampling import (
    GlobalTaskPromptDataset,
    PhaseActionMaskedDataset,
    ReplayMixDataset,
    ReplayMultiMixDataset,
    phase_action_pad_mask,
    phase_chunk_safe_mask,
    phase_groups_from_indices,
    phase_sampling_weights,
    transition_sampling_weights,
)


def test_replay_mix_dataset_routes_items_and_records_sampling_multipliers() -> None:
    class FakeDataset:
        features = {"value": {"dtype": "int64"}}
        fps = 10

        def __init__(self, values: list[int]) -> None:
            self.values = values
            self.hf_dataset = Dataset.from_dict({"value": values})
            self.meta = object()

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int):
            return {"value": self.values[index]}

    mixed = ReplayMixDataset(FakeDataset([1, 2, 3]), FakeDataset([10, 11]), 1.25)
    assert len(mixed) == 5
    assert [mixed[index]["value"] for index in range(len(mixed))] == [1, 2, 3, 10, 11]
    assert mixed.hf_dataset["value"] == [1, 2, 3, 10, 11]
    assert mixed.sampling_multipliers.tolist() == [1.0, 1.0, 1.0, 1.25, 1.25]
    assert mixed._vla_replay_mixed is True
    assert GlobalTaskPromptDataset(mixed, "task")._vla_replay_mixed is True


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan")])
def test_replay_mix_dataset_rejects_invalid_auxiliary_weight(weight: float) -> None:
    class FakeDataset:
        features = {}
        fps = 10
        hf_dataset = Dataset.from_dict({"value": []})
        meta = object()

        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="auxiliary_sample_weight"):
        ReplayMixDataset(FakeDataset(), FakeDataset(), weight)


def test_replay_multi_mix_dataset_routes_each_source_with_independent_weights() -> None:
    class FakeDataset:
        features = {"value": {"dtype": "int64"}}
        fps = 10

        def __init__(self, values: list[int]) -> None:
            self.values = values
            self.hf_dataset = Dataset.from_dict({"value": values})
            self.meta = object()

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int):
            return {"value": self.values[index]}

    mixed = ReplayMultiMixDataset(
        FakeDataset([1, 2]), [FakeDataset([10]), FakeDataset([20, 21])], [0.2, 0.3]
    )
    assert mixed.base_length == 2
    assert [mixed[index]["value"] for index in range(len(mixed))] == [1, 2, 10, 20, 21]
    assert mixed.sampling_multipliers.tolist() == [1.0, 1.0, 0.2, 0.3, 0.3]


def test_global_task_prompt_dataset_relabels_text_without_changing_actions() -> None:
    action = torch.tensor([0.1, 0.2])
    dataset = [{"task": "phase prompt", "action": action}]
    wrapped = GlobalTaskPromptDataset(dataset, "complete the whole task")
    assert wrapped[0]["task"] == "complete the whole task"
    assert wrapped[0]["action"] is action


def test_global_task_prompt_dataset_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        GlobalTaskPromptDataset([], " ")


def test_phase_action_pad_mask_keeps_short_phase_start_and_masks_remainder() -> None:
    groups = ["approach", "approach", "lift", "lift"]
    episodes = [0, 0, 0, 0]
    assert phase_action_pad_mask(groups, episodes, 0, 4).tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert phase_action_pad_mask(groups, episodes, 1, 4).tolist() == [
        False,
        True,
        True,
        True,
    ]


def test_phase_action_masked_dataset_combines_phase_and_episode_padding() -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.values = [
                {"value": index, "action_is_pad": torch.tensor([False, False, False])}
                for index in range(3)
            ]

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int):
            return self.values[index]

    dataset = PhaseActionMaskedDataset(
        FakeDataset(),
        ["approach", "approach", "lift"],
        [0, 0, 0],
        3,
    )
    assert dataset[0]["action_is_pad"].tolist() == [False, False, True]
    assert dataset[1]["action_is_pad"].tolist() == [False, True, True]
    assert dataset[2]["action_is_pad"].tolist() == [False, True, True]


def test_phase_chunk_safe_mask_rejects_exact_phase_and_episode_crossings() -> None:
    groups = [1, 1, 1, 2, 2, 2]
    episodes = [0, 0, 0, 0, 0, 1]
    assert phase_chunk_safe_mask(groups, episodes, 2).tolist() == [
        True,
        True,
        False,
        True,
        False,
        False,
    ]


def test_phase_groups_from_indices_maps_stack_phases() -> None:
    groups = phase_groups_from_indices([0, 1, 2, 3, 4, 5])
    assert groups == [
        "grasp",
        "approach",
        "lift",
        "transport",
        "place_release",
        "place_release",
    ]


def test_phase_groups_from_indices_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="Unknown Stack phase indices"):
        phase_groups_from_indices([6])


def test_phase_sampling_balances_unequal_groups():
    groups = (
        ["approach"] * 4
        + ["grasp"] * 2
        + ["lift"] * 2
        + ["transport"]
        + ["place_release"]
    )
    weights = phase_sampling_weights(groups)
    totals = {
        group: float(weights[[index for index, value in enumerate(groups) if value == group]].sum())
        for group in set(groups)
    }
    assert totals == {
        "approach": 1.0,
        "grasp": 1.0,
        "lift": 1.0,
        "transport": 1.0,
        "place_release": 1.0,
    }


def test_phase_sampling_requires_every_group():
    try:
        phase_sampling_weights(["approach", "grasp", "lift", "transport"])
    except ValueError as error:
        assert "place_release" in str(error)
    else:
        raise AssertionError("Expected missing phase group to fail")


def test_phase_sampling_preserves_targets_after_source_phase_filtering() -> None:
    groups = (
        ["approach"] * 3
        + ["grasp"] * 3
        + ["lift"] * 3
        + ["transport"] * 4
        + ["place_release"] * 3
    )
    # The final transport entries represent teacher data. Other teacher phase
    # entries were filtered to zero before phase normalization.
    multipliers = torch.tensor(
        [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 2, 2, 1, 1, 0],
        dtype=torch.double,
    )
    targets = {
        "approach": 0.18,
        "grasp": 0.20,
        "lift": 0.20,
        "transport": 0.27,
        "place_release": 0.15,
    }
    weights = phase_sampling_weights(
        groups,
        target_proportions=targets,
        sampling_multipliers=multipliers,
    )
    normalized = weights / weights.sum()
    totals = {
        group: float(normalized[
            torch.as_tensor([value == group for value in groups], dtype=torch.bool)
        ].sum())
        for group in targets
    }
    assert totals == pytest.approx(targets)
    assert torch.all(weights[multipliers == 0] == 0)


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
