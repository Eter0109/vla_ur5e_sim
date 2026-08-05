"""Shared SmolVLA checkpoint loading and action-chunk inference."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.utils import prepare_observation_for_inference

from .lerobot_compat import install_fast_parquet_loader

install_fast_parquet_loader()


def aggregate_action_chunks(chunks: torch.Tensor, *, action_dim: int) -> torch.Tensor:
    """Aggregate postprocessed flow samples while preserving the gripper decision.

    The caller must first convert every sample back to the deployment action
    domain. Continuous Cartesian dimensions use their arithmetic mean. Gripper
    samples vote by sign; an even-sample tie falls back to the sample with the
    largest magnitude, preserving the most confident discrete command.
    """

    if chunks.ndim != 4 or chunks.shape[0] < 1:
        raise ValueError("Expected samples shaped [sample, batch, time, action].")
    if not 1 <= action_dim <= chunks.shape[-1]:
        raise ValueError("action_dim must be within the predicted action width.")

    aggregated = chunks.mean(dim=0)
    if action_dim < 7:
        return aggregated

    grippers = chunks[..., 6]
    votes = torch.sign(grippers).sum(dim=0)
    largest_magnitude = grippers.abs().argmax(dim=0, keepdim=True)
    tie_breaker = grippers.gather(0, largest_magnitude).squeeze(0)
    aggregated[..., 6] = torch.where(votes == 0, tie_breaker, torch.sign(votes))
    return aggregated


def postprocess_and_aggregate_action_chunks(
    chunks: torch.Tensor, postprocessor: Any, *, action_dim: int
) -> torch.Tensor:
    """Postprocess each flow sample before aggregating discrete action semantics."""

    postprocessed = torch.stack([postprocessor(sample) for sample in chunks])
    return aggregate_action_chunks(postprocessed, action_dim=action_dim)


def _install_peft_compatibility() -> None:
    """Backfill mapping methods expected by older PEFT versions."""

    if not hasattr(SmolVLAConfig, "get"):
        SmolVLAConfig.get = lambda self, key, default=None: getattr(  # type: ignore[attr-defined]
            self, key, default
        )
    if not hasattr(SmolVLAConfig, "__contains__"):
        SmolVLAConfig.__contains__ = lambda self, key: hasattr(  # type: ignore[attr-defined]
            self, key
        )


def load_policy(
    checkpoint: Path,
    dataset: LeRobotDataset,
    action_steps: int | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Load a local SmolVLA checkpoint and its preprocessing pipeline."""

    _install_peft_compatibility()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = "cuda"
    config.use_amp = True
    if action_steps is not None:
        config.n_action_steps = action_steps
    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset.meta.stats,
    )
    policy.eval()
    return config, policy, preprocessor, postprocessor


def predict_ensemble_chunk(
    observation: dict[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    device: torch.device,
    use_amp: bool,
    samples: int,
    task_prompt: str,
) -> np.ndarray:
    """Average independent flow samples before executing an action chunk."""

    if samples < 1:
        raise ValueError("samples must be positive")
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and use_amp
        else nullcontext(),
    ):
        batch = prepare_observation_for_inference(
            observation,
            device,
            task_prompt,
            "UR5e",
        )
        batch = preprocessor(batch)
        chunks = torch.stack([policy.predict_action_chunk(batch) for _ in range(samples)])
        chunk = postprocess_and_aggregate_action_chunks(
            chunks,
            postprocessor,
            action_dim=int(configured_action_dim(policy)),
        )
    action_steps = int(policy.config.n_action_steps)
    return chunk[0, :action_steps].detach().float().cpu().numpy()


def configured_action_dim(policy: Any) -> int:
    """Return the task action width without exposing SmolVLA internals to callers."""

    return int(policy.config.action_feature.shape[0])
