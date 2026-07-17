"""Sampling utilities for sparse robot action transitions."""

from __future__ import annotations

import math

import torch


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
