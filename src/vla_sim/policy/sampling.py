"""Sampling for the frozen five-prompt, three-task training contract."""

from __future__ import annotations

import math

import torch


def task_sampling_weights(
    task_labels: list[str],
    target_proportions: dict[str, float],
) -> torch.Tensor:
    """Return per-frame weights whose normalized mass matches each task target."""

    if not task_labels:
        raise ValueError("task_labels must not be empty")
    observed = set(task_labels)
    configured = set(target_proportions)
    if observed != configured:
        missing = sorted(observed - configured)
        extra = sorted(configured - observed)
        raise ValueError(
            f"Task sampling contract mismatch: missing={missing}, extra={extra}"
        )
    if any(not math.isfinite(value) or value <= 0 for value in target_proportions.values()):
        raise ValueError("Task sampling proportions must be finite and positive")
    if not math.isclose(sum(target_proportions.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Task sampling proportions must sum to 1")

    counts = {label: task_labels.count(label) for label in observed}
    return torch.tensor(
        [target_proportions[label] / counts[label] for label in task_labels],
        dtype=torch.double,
    )
