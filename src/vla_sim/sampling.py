"""Sampling utilities for sparse robot action transitions."""

from __future__ import annotations

import math

import torch
from datasets import concatenate_datasets


PHASE_GROUPS = ("approach", "grasp", "lift", "transport", "place_release")
PHASE_INDEX_TO_GROUP = {
    0: "grasp",
    1: "approach",
    2: "lift",
    3: "transport",
    4: "place_release",
    5: "place_release",
}


class ReplayMixDataset(torch.utils.data.Dataset):
    """Concatenate base and auxiliary data while retaining source sampling weights."""

    def __init__(self, base, auxiliary, auxiliary_sample_weight: float) -> None:
        if not math.isfinite(auxiliary_sample_weight) or auxiliary_sample_weight <= 0:
            raise ValueError("auxiliary_sample_weight must be finite and positive")
        if getattr(base, "features", None) != getattr(auxiliary, "features", None):
            raise ValueError("Replay datasets must have identical features")
        if getattr(base, "fps", None) != getattr(auxiliary, "fps", None):
            raise ValueError("Replay datasets must have identical FPS")
        self.base = base
        self.auxiliary = auxiliary
        self.auxiliary_sample_weight = float(auxiliary_sample_weight)
        self._vla_replay_mixed = True
        self.base_length = len(base)
        self.hf_dataset = concatenate_datasets([base.hf_dataset, auxiliary.hf_dataset])
        self.sampling_multipliers = torch.cat(
            (
                torch.ones(self.base_length, dtype=torch.double),
                torch.full(
                    (len(auxiliary),),
                    self.auxiliary_sample_weight,
                    dtype=torch.double,
                ),
            )
        )
        self.meta = base.meta

    def __len__(self) -> int:
        return self.base_length + len(self.auxiliary)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError("ReplayMixDataset index is outside the dataset")
        if index < self.base_length:
            return self.base[index]
        return self.auxiliary[index - self.base_length]

    def __getattr__(self, name: str):
        return getattr(self.base, name)


class ReplayMultiMixDataset(torch.utils.data.Dataset):
    """Replay a base dataset with independently weighted correction datasets."""

    def __init__(self, base, auxiliaries: list, auxiliary_sample_weights: list[float]) -> None:
        if not auxiliaries or len(auxiliaries) != len(auxiliary_sample_weights):
            raise ValueError("auxiliaries and auxiliary_sample_weights must be non-empty and aligned")
        if any(
            not math.isfinite(weight) or weight <= 0 for weight in auxiliary_sample_weights
        ):
            raise ValueError("auxiliary_sample_weights must be finite and positive")
        for auxiliary in auxiliaries:
            if getattr(base, "features", None) != getattr(auxiliary, "features", None):
                raise ValueError("Replay datasets must have identical features")
            if getattr(base, "fps", None) != getattr(auxiliary, "fps", None):
                raise ValueError("Replay datasets must have identical FPS")
        self.base = base
        self.auxiliaries = tuple(auxiliaries)
        self.auxiliary_sample_weights = tuple(float(weight) for weight in auxiliary_sample_weights)
        self.sources = (base, *self.auxiliaries)
        self._vla_replay_mixed = True
        self.source_lengths = tuple(len(source) for source in self.sources)
        self.base_length = self.source_lengths[0]
        self.source_ends = tuple(torch.as_tensor(self.source_lengths).cumsum(0).tolist())
        self.hf_dataset = concatenate_datasets([source.hf_dataset for source in self.sources])
        self.sampling_multipliers = torch.cat(
            (
                torch.ones(self.source_lengths[0], dtype=torch.double),
                *(
                    torch.full((length,), weight, dtype=torch.double)
                    for length, weight in zip(
                        self.source_lengths[1:], self.auxiliary_sample_weights, strict=True
                    )
                ),
            )
        )
        self.meta = base.meta

    def __len__(self) -> int:
        return sum(self.source_lengths)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError("Replay dataset index is outside the dataset")
        start = 0
        for source, end in zip(self.sources, self.source_ends, strict=True):
            if index < end:
                return source[index - start]
            start = end
        raise AssertionError("Replay source boundaries do not cover the dataset")

    def __getattr__(self, name: str):
        return getattr(self.base, name)


class GlobalTaskPromptDataset(torch.utils.data.Dataset):
    """Expose one task-level prompt while preserving the original action sequence."""

    def __init__(self, dataset, prompt: str, task_key: str = "task") -> None:
        if not prompt.strip():
            raise ValueError("Global task prompt must not be empty")
        self.dataset = dataset
        self.prompt = prompt
        self.task_key = task_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return {**self.dataset[index], self.task_key: self.prompt}

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)


class PhaseActionMaskedDataset(torch.utils.data.Dataset):
    """Mask action targets after the current prompt phase ends."""

    def __init__(
        self,
        dataset,
        phase_groups: list[str],
        episode_indices: list[int],
        chunk_size: int,
    ) -> None:
        if len(dataset) != len(phase_groups) or len(dataset) != len(episode_indices):
            raise ValueError("Dataset, phase groups, and episode indices must have equal length")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.dataset = dataset
        self.phase_groups = phase_groups
        self.episode_indices = episode_indices
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        phase_pad = phase_action_pad_mask(
            self.phase_groups,
            self.episode_indices,
            index,
            self.chunk_size,
        )
        existing = item.get("action_is_pad")
        if existing is not None:
            existing = torch.as_tensor(existing, dtype=torch.bool)
            if existing.shape != phase_pad.shape:
                raise ValueError(
                    "Existing action_is_pad shape does not match configured action chunk: "
                    f"{tuple(existing.shape)} != {tuple(phase_pad.shape)}"
                )
            phase_pad |= existing
        return {**item, "action_is_pad": phase_pad}

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)


def phase_action_pad_mask(
    phase_groups: list[str],
    episode_indices: list[int],
    start_index: int,
    chunk_size: int,
) -> torch.Tensor:
    """Pad action targets once a chunk leaves its starting prompt or episode."""

    if len(phase_groups) != len(episode_indices):
        raise ValueError("phase_groups and episode_indices must have the same length")
    if not 0 <= start_index < len(phase_groups):
        raise IndexError("start_index is outside the dataset")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    start_group = phase_groups[start_index]
    start_episode = episode_indices[start_index]
    mask = torch.ones(chunk_size, dtype=torch.bool)
    for offset in range(chunk_size):
        index = start_index + offset
        if (
            index >= len(phase_groups)
            or phase_groups[index] != start_group
            or episode_indices[index] != start_episode
        ):
            break
        mask[offset] = False
    return mask


def phase_groups_from_indices(phase_indices: list[int]) -> list[str]:
    """Map immutable Stack phase labels to the four sampling groups."""

    unknown = sorted(set(phase_indices) - set(PHASE_INDEX_TO_GROUP))
    if unknown:
        raise ValueError(f"Unknown Stack phase indices: {unknown}")
    return [PHASE_INDEX_TO_GROUP[index] for index in phase_indices]


def phase_chunk_safe_mask(
    phase_labels: list[int] | list[str],
    episode_indices: list[int],
    chunk_size: int,
) -> torch.Tensor:
    """Mark starts whose full action chunk stays in one episode and exact phase."""

    if len(phase_labels) != len(episode_indices):
        raise ValueError("phase_labels and episode_indices must have the same length")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not phase_labels:
        return torch.zeros(0, dtype=torch.bool)
    safe = torch.ones(len(phase_labels), dtype=torch.bool)
    label_codes = {label: index for index, label in enumerate(dict.fromkeys(phase_labels))}
    group_codes = torch.as_tensor(
        [label_codes[label] for label in phase_labels],
        dtype=torch.int16,
    )
    episodes = torch.as_tensor(episode_indices, dtype=torch.int64)
    for offset in range(1, chunk_size):
        if offset >= len(safe):
            safe[:] = False
            break
        safe[:-offset] &= (group_codes[:-offset] == group_codes[offset:]) & (
            episodes[:-offset] == episodes[offset:]
        )
        safe[-offset:] = False
    return safe


def phase_sampling_weights(
    phase_groups: list[str],
    target_proportions: dict[str, float] | None = None,
    sampling_multipliers: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return weights with exact phase mass after source weighting or filtering."""

    if not phase_groups:
        raise ValueError("phase_groups must not be empty")
    unknown = sorted(set(phase_groups) - set(PHASE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown phase groups: {', '.join(unknown)}")
    if sampling_multipliers is None:
        multipliers = torch.ones(len(phase_groups), dtype=torch.double)
    else:
        multipliers = torch.as_tensor(sampling_multipliers, dtype=torch.double).clone()
        if multipliers.shape != (len(phase_groups),):
            raise ValueError("sampling_multipliers must have one value per phase group")
        if not torch.isfinite(multipliers).all() or (multipliers < 0).any():
            raise ValueError("sampling_multipliers must be finite and non-negative")
    group_mass = {
        name: float(multipliers[
            torch.as_tensor([group == name for group in phase_groups], dtype=torch.bool)
        ].sum())
        for name in PHASE_GROUPS
    }
    missing = [name for name, mass in group_mass.items() if mass <= 0]
    if missing:
        raise ValueError(f"Dataset is missing required phase groups: {', '.join(missing)}")
    if target_proportions is None:
        return torch.as_tensor(
            [multipliers[index] / group_mass[name] for index, name in enumerate(phase_groups)],
            dtype=torch.double,
        )
    scale = float(len(PHASE_GROUPS))
    return torch.as_tensor(
        [
            multipliers[index] * target_proportions[name] * scale / group_mass[name]
            for index, name in enumerate(phase_groups)
        ],
        dtype=torch.double,
    )


def transition_sampling_weights(
    actions: torch.Tensor,
    episode_indices: torch.Tensor,
    *,
    factor: float = 1.0,
    window: int = 0,
    gripper_index: int = 6,
) -> torch.Tensor:
    """Upweight frames near within-episode gripper state transitions."""
    actions = torch.as_tensor(actions)
    episode_indices = torch.as_tensor(episode_indices)
    if actions.ndim != 2 or not 0 <= gripper_index < actions.shape[1]:
        raise ValueError("actions must be [frames, dims] and contain the gripper index")
    if episode_indices.ndim != 1 or len(episode_indices) != len(actions):
        raise ValueError("episode_indices must have one entry per action frame")
    if not math.isfinite(factor) or factor < 1:
        raise ValueError("factor must be finite and at least 1")
    if window < 0:
        raise ValueError("window must be non-negative")

    weights = torch.ones(len(actions), dtype=torch.double)
    if len(actions) < 2 or factor == 1:
        return weights

    same_episode = episode_indices[1:] == episode_indices[:-1]
    changed = actions[1:, gripper_index] != actions[:-1, gripper_index]
    centers = torch.nonzero(same_episode & changed, as_tuple=False).flatten() + 1
    selected = torch.zeros(len(actions), dtype=torch.bool)
    for offset in range(-window, window + 1):
        indices = centers + offset
        valid = (indices >= 0) & (indices < len(actions))
        indices = indices[valid]
        same = episode_indices[indices] == episode_indices[centers[valid]]
        selected[indices[same]] = True
    weights[selected] = factor
    return weights
