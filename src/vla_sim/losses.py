"""Loss helpers for the local SmolVLA compatibility entrypoint."""

from __future__ import annotations

import torch


def action_dimension_weights(
    action_dim: int,
    *,
    rotation_weight: float,
    gripper_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build ``[translation, rotation, gripper]`` weights for a 7-D action."""

    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    values = torch.ones(action_dim, device=device, dtype=dtype)
    if action_dim >= 7:
        values[3:6] = rotation_weight
        values[6] = gripper_weight
    if not torch.isfinite(values).all() or torch.any(values < 0) or values.sum() <= 0:
        raise ValueError("Action loss weights must be finite, non-negative, and not all zero")
    return values


def weighted_action_loss(
    losses: torch.Tensor,
    action_is_pad: torch.Tensor | None,
    weights: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Reduce per-step/per-dimension losses with padding and dimension weights."""

    if losses.ndim != 3:
        raise ValueError(f"Expected [batch, time, action] losses; got {tuple(losses.shape)}")
    if weights.ndim != 1 or weights.shape[0] != losses.shape[-1]:
        raise ValueError("weights must be a vector matching the action dimension")
    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    weights = weights.to(device=losses.device, dtype=losses.dtype)
    weight_sum = weights.sum()
    if not torch.isfinite(weights).all() or torch.any(weights < 0) or weight_sum <= 0:
        raise ValueError("Action loss weights must be finite, non-negative, and not all zero")

    weighted = losses * weights.view(1, 1, -1)
    if action_is_pad is None:
        per_sample = weighted.sum(dim=(1, 2)) / (losses.shape[1] * weight_sum)
    else:
        if action_is_pad.shape != losses.shape[:2]:
            raise ValueError("action_is_pad must match the batch and time dimensions")
        valid = (~action_is_pad).to(losses.dtype).unsqueeze(-1)
        weighted = weighted * valid
        denominator = ((~action_is_pad).sum(dim=1).to(losses.dtype) * weight_sum).clamp_min(1)
        per_sample = weighted.sum(dim=(1, 2)) / denominator

    return per_sample if reduction == "none" else per_sample.mean()
