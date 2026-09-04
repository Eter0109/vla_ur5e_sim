"""LeRobot training entrypoint for the current three-task SmolVLA policy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from vla_sim.policy.lerobot_compat import install_fast_parquet_loader
from vla_sim.policy.losses import action_dimension_weights, weighted_action_loss
from vla_sim.policy.sampling import task_sampling_weights

install_fast_parquet_loader()

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts import lerobot_train
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

if not hasattr(SmolVLAConfig, "get"):

    def _config_get(self: SmolVLAConfig, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    SmolVLAConfig.get = _config_get  # type: ignore[attr-defined]

if not hasattr(SmolVLAConfig, "__contains__"):

    def _config_contains(self: SmolVLAConfig, key: str) -> bool:
        return hasattr(self, key)

    SmolVLAConfig.__contains__ = _config_contains  # type: ignore[attr-defined]


def _install_smolvla_loss_fix() -> None:
    """Train only seven real action dimensions and ignore padded timesteps."""

    if getattr(SmolVLAPolicy, "_vla_sim_loss_fix", False):
        return

    def forward(
        self: SmolVLAPolicy,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, image_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        losses = self.model.forward(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            self.prepare_action(batch),
            noise,
            time,
        )
        action_is_pad = batch.get("action_is_pad")
        action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :action_dim]
        xyz_weight = float(os.environ.get("VLA_XYZ_LOSS_WEIGHT", "1.0"))
        rotation_weight = float(os.environ.get("VLA_ROTATION_LOSS_WEIGHT", "1.0"))
        gripper_weight = float(os.environ.get("VLA_GRIPPER_LOSS_WEIGHT", "1.0"))
        weights = action_dimension_weights(
            action_dim,
            xyz_weight=xyz_weight,
            rotation_weight=rotation_weight,
            gripper_weight=gripper_weight,
            device=losses.device,
            dtype=losses.dtype,
        )
        loss = weighted_action_loss(losses, action_is_pad, weights, reduction=reduction)
        metrics = {
            "loss": loss.detach().mean().item(),
            "losses_after_forward": losses.detach().mean().item(),
            "xyz_loss_weight": xyz_weight,
            "rotation_loss_weight": rotation_weight,
            "gripper_loss_weight": gripper_weight,
        }
        for index, name in enumerate(("dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper")):
            metrics[f"loss_{name}"] = losses[:, :, index].detach().mean().item()
        if action_is_pad is not None:
            valid = (~action_is_pad).unsqueeze(-1)
            metrics["losses_after_in_ep_bound"] = (losses * valid).detach().mean().item()
        return loss, metrics

    SmolVLAPolicy.forward = forward  # type: ignore[method-assign]
    SmolVLAPolicy._vla_sim_loss_fix = True


def _install_task_sampler() -> None:
    raw_contract = os.environ.get("VLA_TASK_SAMPLING_PROPORTIONS", "").strip()
    if not raw_contract or raw_contract == "{}":
        return
    targets = {str(key): float(value) for key, value in json.loads(raw_contract).items()}
    seed = int(os.environ.get("VLA_SAMPLING_SEED", "1000"))
    dataloader_class = torch.utils.data.DataLoader
    original_init = dataloader_class.__init__

    def data_loader_init(self, dataset, *args, **kwargs):
        if kwargs.get("batch_sampler") is None and hasattr(dataset, "hf_dataset"):
            raw = dataset.hf_dataset.with_format(None)
            prompts = {
                int(row["task_index"]): str(prompt)
                for prompt, row in dataset.meta.tasks.iterrows()
            }
            labels = [prompts[int(index)] for index in raw["task_index"]]
            weights = task_sampling_weights(labels, targets)
            generator = torch.Generator().manual_seed(seed)
            kwargs["sampler"] = torch.utils.data.WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
            kwargs["shuffle"] = False
            print(
                "task_sampler "
                + " ".join(
                    f"{label!r}={targets[label]:.6f}({labels.count(label)} frames)"
                    for label in targets
                ),
                flush=True,
            )
        original_init(self, dataset, *args, **kwargs)

    dataloader_class.__init__ = data_loader_init  # type: ignore[method-assign]


_install_smolvla_loss_fix()
_install_task_sampler()

_continue_adapter = os.environ.get("VLA_CONTINUE_PEFT_ADAPTER", "0") == "1"
if _continue_adapter:
    _original_make_policy = lerobot_train.make_policy

    def _make_trainable_adapter(*args, **kwargs):
        from peft import PeftModel

        policy = _original_make_policy(*args, **kwargs)
        if not isinstance(policy, PeftModel):
            raise TypeError("Expected a PEFT checkpoint")
        if list(policy.peft_config) != ["default"]:
            raise RuntimeError("Expected exactly one default PEFT adapter")
        config = policy.peft_config["default"]
        policy.set_adapter("default")
        config.inference_mode = False
        trainable = [name for name, value in policy.named_parameters() if value.requires_grad]
        if not trainable or any("lora_" not in name for name in trainable):
            raise RuntimeError("Adapter continuation must train LoRA parameters only")
        print(f"continue_peft_adapter rank={config.r} trainable_tensors={len(trainable)}")
        return policy

    lerobot_train.make_policy = _make_trainable_adapter

_resume_checkpoint = os.environ.get("VLA_RESUME_CHECKPOINT")
if _resume_checkpoint:
    _original_validate = TrainPipelineConfig.validate

    def _validate_resume(self: TrainPipelineConfig) -> None:
        self.checkpoint_path = Path(_resume_checkpoint)
        _original_validate(self)

    TrainPipelineConfig.validate = _validate_resume  # type: ignore[method-assign]

if os.name == "nt":
    lerobot_train.update_last_checkpoint = lambda _checkpoint_dir: None

_original_optimizer = lerobot_train.make_optimizer_and_scheduler


def _optimizer_with_resume(cfg, policy):
    optimizer, scheduler = _original_optimizer(cfg, policy)
    if _resume_checkpoint and scheduler is not None:
        step_path = Path(_resume_checkpoint) / "training_state" / "training_step.json"
        start_step = int(json.loads(step_path.read_text(encoding="utf-8"))["step"])
        for _ in range(start_step):
            scheduler.step()
        print(f"resume_scheduler_step={start_step}", flush=True)
    return optimizer, scheduler


lerobot_train.make_optimizer_and_scheduler = _optimizer_with_resume
main = lerobot_train.main


if __name__ == "__main__":
    main()
