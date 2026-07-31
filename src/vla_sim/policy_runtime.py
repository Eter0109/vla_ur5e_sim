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
        chunks = [policy.predict_action_chunk(batch) for _ in range(samples)]
        chunk = postprocessor(torch.stack(chunks).mean(dim=0))
    return chunk[0].detach().float().cpu().numpy()
