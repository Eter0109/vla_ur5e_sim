"""Shared SmolVLA checkpoint loading and action-chunk inference."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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

from vla_sim.paths import load_catalog, project_root, resolve_asset

from .lerobot_compat import install_fast_parquet_loader

install_fast_parquet_loader()


def _ensure_runtime_policy_config(checkpoint: Path) -> None:
    """Recover LeRobot's policy config from a PEFT training checkpoint."""

    config_path = checkpoint / "config.json"
    if config_path.is_file():
        return
    train_config_path = checkpoint / "train_config.json"
    if not train_config_path.is_file():
        raise FileNotFoundError(f"Checkpoint has neither config.json nor train_config.json: {checkpoint}")
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    policy_config = train_config.get("policy")
    if not isinstance(policy_config, dict) or policy_config.get("type") != "smolvla":
        raise ValueError("Checkpoint train_config.json has no SmolVLA policy configuration")
    policy_config = dict(policy_config)
    policy_config["pretrained_path"] = str(checkpoint)
    config_path.write_text(json.dumps(policy_config, indent=2) + "\n", encoding="utf-8")


def resolve_checkpoint(checkpoint: Path) -> Path:
    """Materialize a portable PEFT checkpoint with an absolute local base path.

    PEFT interprets ``base_model_name_or_path`` relative to the process working
    directory. Durable assets keep repository-relative paths; this hard-linked
    runtime view makes loading deterministic without copying model weights.
    """

    checkpoint = checkpoint.expanduser().resolve()
    adapter_config = checkpoint / "adapter_config.json"
    if not adapter_config.is_file():
        return checkpoint
    configured = json.loads(adapter_config.read_text(encoding="utf-8"))
    policy_catalog = load_catalog("policy")
    base = resolve_asset(policy_catalog["base"])
    if not base.is_dir():
        raise FileNotFoundError(f"Base policy asset is missing: {base}")
    adapter_model = checkpoint / "adapter_model.safetensors"
    identity = hashlib.sha256(
        f"{checkpoint}:{adapter_model.stat().st_size}:{adapter_model.stat().st_mtime_ns}:{base}".encode()
    ).hexdigest()[:16]
    target = project_root() / ".runtime" / "resolved_checkpoints" / identity
    if target.is_dir():
        _ensure_runtime_policy_config(target)
        return target
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint, temporary, copy_function=os.link)
    configured["base_model_name_or_path"] = str(base)
    (temporary / "adapter_config.json").write_text(
        json.dumps(configured, indent=2) + "\n", encoding="utf-8"
    )
    _ensure_runtime_policy_config(temporary)
    try:
        temporary.rename(target)
    except FileExistsError:
        shutil.rmtree(temporary)
    return target


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
    device: str | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Load a local SmolVLA checkpoint and its preprocessing pipeline."""

    _install_peft_compatibility()
    checkpoint = resolve_checkpoint(checkpoint)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config.use_amp = config.device == "cuda"
    if action_steps is not None:
        config.n_action_steps = action_steps
    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": config.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
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
