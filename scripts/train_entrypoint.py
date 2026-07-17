"""LeRobot training entrypoint with compatibility fixes for SmolVLA 0.4.4."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.lerobot_compat import install_fast_parquet_loader  # noqa: E402
from vla_sim.losses import action_dimension_weights, weighted_action_loss  # noqa: E402
from vla_sim.sampling import transition_sampling_weights  # noqa: E402

install_fast_parquet_loader()

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
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
    """Backport the action-loss fixes from current LeRobot to version 0.4.4.

    SmolVLA 0.4.4 looks up ``actions_id_pad`` instead of the dataset's
    ``action_is_pad`` and averages loss across all 32 padded action channels.
    This project uses 7-D actions, so both behaviours disproportionately train
    the model to predict padded zeros and repeated terminal actions.
    """

    if getattr(SmolVLAPolicy, "_vla_sim_loss_fix", False):
        return

    def forward(
        self: SmolVLAPolicy,
        batch: dict[str, torch.Tensor],
        noise=None,
        time=None,
        reduction="mean",
    ):
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)

        # Do not train on model-internal action padding. The action head stays
        # 32-D for checkpoint compatibility, while this task has 7 real dims.
        action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :action_dim]
        rotation_weight = float(os.environ.get("VLA_ROTATION_LOSS_WEIGHT", "1.0"))
        gripper_weight = float(os.environ.get("VLA_GRIPPER_LOSS_WEIGHT", "1.0"))
        dim_weights = action_dimension_weights(
            action_dim,
            rotation_weight=rotation_weight,
            gripper_weight=gripper_weight,
            device=losses.device,
            dtype=losses.dtype,
        )
        metrics = {
            "losses_after_forward": losses.detach().mean().item(),
            "rotation_loss_weight": rotation_weight,
            "gripper_loss_weight": gripper_weight,
        }
        names = ("dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper")
        for index, name in enumerate(names[:action_dim]):
            metrics[f"loss_{name}"] = losses[:, :, index].detach().mean().item()

        loss = weighted_action_loss(
            losses,
            action_is_pad,
            dim_weights,
            reduction=reduction,
        )

        if action_is_pad is not None:
            valid = (~action_is_pad).unsqueeze(-1)
            metrics["losses_after_in_ep_bound"] = (losses * valid).detach().mean().item()

        metrics["loss"] = loss.detach().mean().item()
        return loss, metrics

    SmolVLAPolicy.forward = forward  # type: ignore[method-assign]
    SmolVLAPolicy._vla_sim_loss_fix = True


_install_smolvla_loss_fix()


def _install_transition_sampler() -> None:
    factor = float(os.environ.get("VLA_TRANSITION_OVERSAMPLE_FACTOR", "1.0"))
    window = int(os.environ.get("VLA_TRANSITION_OVERSAMPLE_WINDOW", "0"))
    seed = int(os.environ.get("VLA_SAMPLING_SEED", "1000"))
    if factor == 1:
        return

    dataloader_class = torch.utils.data.DataLoader
    original_init = dataloader_class.__init__

    def data_loader_init(self, dataset, *args, **kwargs):
        if (
            kwargs.get("batch_sampler") is None
            and hasattr(dataset, "hf_dataset")
            and "action" in dataset.hf_dataset.column_names
        ):
            unformatted = dataset.hf_dataset.with_format(None)
            actions = torch.as_tensor(unformatted["action"])
            episode_indices = torch.as_tensor(unformatted["episode_index"])
            weights = transition_sampling_weights(
                actions,
                episode_indices,
                factor=factor,
                window=window,
            )
            generator = torch.Generator().manual_seed(seed)
            kwargs["sampler"] = torch.utils.data.WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
            kwargs["shuffle"] = False
            print(
                "transition_sampler "
                f"factor={factor:g} window={window} "
                f"weighted_frames={int((weights > 1).sum())}/{len(weights)}",
                flush=True,
            )
        original_init(self, dataset, *args, **kwargs)

    dataloader_class.__init__ = data_loader_init  # type: ignore[method-assign]


_install_transition_sampler()

from lerobot.scripts import lerobot_train  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402

_resume_checkpoint = os.environ.get("VLA_RESUME_CHECKPOINT")
if _resume_checkpoint:
    _original_validate = TrainPipelineConfig.validate

    def _validate_with_resume_checkpoint(self: TrainPipelineConfig) -> None:
        self.checkpoint_path = Path(_resume_checkpoint)
        _original_validate(self)

    TrainPipelineConfig.validate = _validate_with_resume_checkpoint  # type: ignore[method-assign]

if os.name == "nt":
    # LeRobot's ``checkpoints/last`` is only a convenience symlink. Creating it
    # requires Windows Developer Mode or administrator privileges; numbered
    # checkpoints are already complete and are used directly by this project.
    lerobot_train.update_last_checkpoint = lambda _checkpoint_dir: None

main = lerobot_train.main


if __name__ == "__main__":
    main()
