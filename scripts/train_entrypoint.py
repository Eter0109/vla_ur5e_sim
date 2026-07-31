"""LeRobot training entrypoint with compatibility fixes for SmolVLA 0.4.4."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.lerobot_compat import install_fast_parquet_loader  # noqa: E402
from vla_sim.losses import action_dimension_weights, weighted_action_loss  # noqa: E402
from vla_sim.sampling import (  # noqa: E402
    GlobalTaskPromptDataset,
    PhaseActionMaskedDataset,
    ReplayMixDataset,
    phase_groups_from_indices,
    phase_sampling_weights,
    transition_sampling_weights,
)

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
        xyz_weight = float(os.environ.get("VLA_XYZ_LOSS_WEIGHT", "1.0"))
        dim_weights = action_dimension_weights(
            action_dim,
            rotation_weight=rotation_weight,
            gripper_weight=gripper_weight,
            xyz_weight=xyz_weight,
            device=losses.device,
            dtype=losses.dtype,
        )
        metrics = {
            "losses_after_forward": losses.detach().mean().item(),
            "xyz_loss_weight": xyz_weight,
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


def _install_global_task_prompt() -> None:
    prompt = os.environ.get("VLA_GLOBAL_TASK_PROMPT", "").strip()
    if not prompt:
        return

    dataloader_class = torch.utils.data.DataLoader
    original_init = dataloader_class.__init__

    def data_loader_init(self, dataset, *args, **kwargs):
        if hasattr(dataset, "hf_dataset"):
            dataset = GlobalTaskPromptDataset(dataset, prompt)
            print(f"global_task_prompt prompt={prompt!r}", flush=True)
        original_init(self, dataset, *args, **kwargs)

    dataloader_class.__init__ = data_loader_init  # type: ignore[method-assign]


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


def _phase_group(prompt: str) -> str:
    normalized = prompt.lower()
    if "above the grasp" in normalized or normalized.startswith("move above"):
        return "approach"
    if "grasp object" in normalized or "down and grasp" in normalized:
        return "grasp"
    if "lift the grasped" in normalized or normalized.startswith("lift "):
        return "lift"
    if "above target" in normalized or "above the blue storage bin" in normalized:
        return "transport"
    if (
        "onto target" in normalized
        or "for release" in normalized
        or "in the blue storage bin" in normalized
        or "hold safely" in normalized
    ):
        return "place_release"
    raise ValueError(f"Unrecognized task phase prompt: {prompt!r}")


def _phase_groups_and_episodes(dataset) -> tuple[list[str], list[int]]:
    groups: list[str] = []
    episode_indices: list[int] = []
    episode_offset = 0
    sources = (
        (dataset.base, dataset.auxiliary)
        if isinstance(dataset, ReplayMixDataset)
        else (dataset,)
    )
    for source in sources:
        raw = source.hf_dataset.with_format(None)
        if "phase_index" in raw.column_names:
            phase_indices = [int(value) for value in raw["phase_index"]]
            source_groups = phase_groups_from_indices(phase_indices)
        else:
            task_indices = raw["task_index"]
            tasks = source.meta.tasks
            prompts_by_index = {
                int(row["task_index"]): str(prompt) for prompt, row in tasks.iterrows()
            }
            source_groups = [_phase_group(prompts_by_index[int(index)]) for index in task_indices]
        source_episodes = [int(value) + episode_offset for value in raw["episode_index"]]
        groups.extend(source_groups)
        episode_indices.extend(source_episodes)
        episode_offset = max(source_episodes, default=episode_offset - 1) + 1
    return groups, episode_indices


def _install_phase_sampler() -> None:
    if os.environ.get("VLA_PHASE_BALANCED", "0") != "1":
        return
    seed = int(os.environ.get("VLA_SAMPLING_SEED", "1000"))
    chunk_size = int(os.environ.get("VLA_PHASE_CHUNK_SIZE", "1"))
    target_proportions = {
        "approach": float(os.environ.get("VLA_APPROACH_WEIGHT", "0.25")),
        "grasp": float(os.environ.get("VLA_GRASP_WEIGHT", "0.20")),
        "lift": float(os.environ.get("VLA_LIFT_WEIGHT", "0.25")),
        "transport": float(os.environ.get("VLA_TRANSPORT_WEIGHT", "0.20")),
        "place_release": float(os.environ.get("VLA_PLACE_RELEASE_WEIGHT", "0.10")),
    }
    if any(not 0.0 < weight < 1.0 for weight in target_proportions.values()):
        raise ValueError("VLA phase weights must each be in (0, 1)")
    if abs(sum(target_proportions.values()) - 1.0) > 1e-9:
        raise ValueError("VLA phase weights must sum to 1")
    dataloader_class = torch.utils.data.DataLoader
    original_init = dataloader_class.__init__

    def data_loader_init(self, dataset, *args, **kwargs):
        if kwargs.get("batch_sampler") is None and hasattr(dataset, "hf_dataset"):
            groups, episode_indices = _phase_groups_and_episodes(dataset)
            weights = phase_sampling_weights(groups, target_proportions=target_proportions)
            sampling_multipliers = getattr(dataset, "sampling_multipliers", None)
            if sampling_multipliers is not None:
                sampling_multipliers = torch.as_tensor(
                    sampling_multipliers,
                    dtype=weights.dtype,
                )
                if sampling_multipliers.shape != weights.shape:
                    raise ValueError("Replay sampling multipliers must match dataset length")
                weights *= sampling_multipliers
            global_prompt = bool(os.environ.get("VLA_GLOBAL_TASK_PROMPT", "").strip())
            if not global_prompt:
                dataset = PhaseActionMaskedDataset(
                    dataset,
                    groups,
                    episode_indices,
                    chunk_size,
                )
            generator = torch.Generator().manual_seed(seed)
            kwargs["sampler"] = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True, generator=generator
            )
            kwargs["shuffle"] = False
            print(
                "phase_sampler "
                + "/".join(
                    f"{group}({target_proportions[group]:.0%})"
                    for group in ("approach", "grasp", "lift", "transport", "place_release")
                )
                + f" prompt_aligned_mask={str(not global_prompt).lower()} "
                f"frames={len(weights)} chunk={chunk_size}",
                flush=True,
            )
        original_init(self, dataset, *args, **kwargs)

    dataloader_class.__init__ = data_loader_init  # type: ignore[method-assign]


def _install_auxiliary_replay_dataset() -> None:
    auxiliary_root = os.environ.get("VLA_AUXILIARY_DATASET", "").strip()
    if not auxiliary_root:
        return
    if os.environ.get("VLA_PHASE_BALANCED", "0") != "1":
        raise ValueError("Auxiliary replay requires phase-balanced sampling")
    auxiliary_repo_id = os.environ.get("VLA_AUXILIARY_REPO_ID", "").strip()
    if not auxiliary_repo_id:
        raise ValueError("VLA_AUXILIARY_REPO_ID is required for auxiliary replay")
    auxiliary_sample_weight = float(os.environ.get("VLA_AUXILIARY_SAMPLE_WEIGHT", "1.0"))
    if not math.isfinite(auxiliary_sample_weight) or auxiliary_sample_weight <= 0:
        raise ValueError("VLA_AUXILIARY_SAMPLE_WEIGHT must be finite and positive")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataloader_class = torch.utils.data.DataLoader
    original_init = dataloader_class.__init__

    def data_loader_init(self, dataset, *args, **kwargs):
        if hasattr(dataset, "hf_dataset") and not getattr(
            dataset,
            "_vla_replay_mixed",
            False,
        ):
            auxiliary = LeRobotDataset(
                auxiliary_repo_id,
                root=Path(auxiliary_root),
                delta_timestamps=dataset.delta_timestamps,
                image_transforms=dataset.image_transforms,
                video_backend=dataset.video_backend,
                tolerance_s=dataset.tolerance_s,
            )
            dataset = ReplayMixDataset(dataset, auxiliary, auxiliary_sample_weight)
            effective_auxiliary_fraction = float(
                dataset.sampling_multipliers[dataset.base_length :].sum()
                / dataset.sampling_multipliers.sum()
            )
            print(
                "auxiliary_replay "
                f"base_frames={dataset.base_length} auxiliary_frames={len(auxiliary)} "
                f"auxiliary_weight={auxiliary_sample_weight:g} "
                f"effective_fraction={effective_auxiliary_fraction:.1%}",
                flush=True,
            )
        original_init(self, dataset, *args, **kwargs)

    dataloader_class.__init__ = data_loader_init  # type: ignore[method-assign]


_install_global_task_prompt()
_install_transition_sampler()
_install_phase_sampler()
_install_auxiliary_replay_dataset()

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

_original_make_optimizer_and_scheduler = lerobot_train.make_optimizer_and_scheduler


def _make_optimizer_and_scheduler_with_resume(cfg, policy):
    optimizer, lr_scheduler = _original_make_optimizer_and_scheduler(cfg, policy)
    if _resume_checkpoint and lr_scheduler is not None:
        try:
            start_step = int(Path(_resume_checkpoint).name)
            print(f"Resume patch: Fast-forwarding scheduler by {start_step} steps...", flush=True)
            for _ in range(start_step):
                lr_scheduler.step()
        except ValueError:
            pass
    return optimizer, lr_scheduler


lerobot_train.make_optimizer_and_scheduler = _make_optimizer_and_scheduler_with_resume

main = lerobot_train.main


if __name__ == "__main__":
    main()
