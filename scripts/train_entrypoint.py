"""LeRobot training entrypoint with compatibility fixes for SmolVLA 0.4.4."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vla_sim.lerobot_compat import install_fast_parquet_loader
from vla_sim.losses import action_dimension_weights, weighted_action_loss
from vla_sim.sampling import (
    GlobalTaskPromptDataset,
    PhaseActionMaskedDataset,
    ReplayMixDataset,
    ReplayMultiMixDataset,
    apply_replay_task_prompts,
    phase_groups_from_indices,
    phase_sampling_weights,
    task_sampling_weights,
    transition_sampling_weights,
)

install_fast_parquet_loader()

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
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
            # The public column accessor materializes a Python list of
            # hundreds of thousands of tiny action arrays.  The on-disk
            # feature is a fixed-size Arrow list, so read its contiguous
            # values buffer directly when available.  This preserves action
            # order exactly while keeping transition-weight construction
            # practical for the replay-mixed dataset.
            arrow_actions = getattr(unformatted, "data", None)
            if arrow_actions is not None:
                action_column = arrow_actions.column("action").combine_chunks()
                action_dim = int(action_column.type.list_size)
                action_values = action_column.values.to_numpy(zero_copy_only=False)
                actions = torch.from_numpy(
                    np.asarray(action_values, dtype=np.float32).reshape(-1, action_dim)
                )
                episode_values = (
                    arrow_actions.column("episode_index")
                    .combine_chunks()
                    .to_numpy(zero_copy_only=False)
                )
                episode_indices = torch.from_numpy(np.asarray(episode_values))
            else:
                actions = torch.from_numpy(np.asarray(unformatted["action"], dtype=np.float32))
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


def _install_task_sampler() -> None:
    raw_contract = os.environ.get("VLA_TASK_SAMPLING_PROPORTIONS", "").strip()
    if not raw_contract:
        return
    target_proportions = {
        str(prompt): float(proportion)
        for prompt, proportion in json.loads(raw_contract).items()
    }
    seed = int(os.environ.get("VLA_SAMPLING_SEED", "1000"))
    dataloader_class = torch.utils.data.DataLoader
    original_init = dataloader_class.__init__

    def data_loader_init(self, dataset, *args, **kwargs):
        if kwargs.get("batch_sampler") is None and hasattr(dataset, "hf_dataset"):
            raw = dataset.hf_dataset.with_format(None)
            task_indices = [int(value) for value in raw["task_index"]]
            prompts_by_index = {
                int(row["task_index"]): str(prompt)
                for prompt, row in dataset.meta.tasks.iterrows()
            }
            labels = [prompts_by_index[index] for index in task_indices]
            weights = task_sampling_weights(labels, target_proportions)
            generator = torch.Generator().manual_seed(seed)
            kwargs["sampler"] = torch.utils.data.WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
            kwargs["shuffle"] = False
            counts = {label: labels.count(label) for label in target_proportions}
            print(
                "task_sampler "
                + " ".join(
                    f"{label!r}={target_proportions[label]:.6f}({counts[label]} frames)"
                    for label in target_proportions
                ),
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
    sources = dataset.sources if isinstance(dataset, ReplayMultiMixDataset) else (
        (dataset.base, dataset.auxiliary) if isinstance(dataset, ReplayMixDataset) else (dataset,)
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
            sampling_multipliers = getattr(dataset, "sampling_multipliers", None)
            if sampling_multipliers is not None:
                sampling_multipliers = torch.as_tensor(
                    sampling_multipliers,
                    dtype=torch.double,
                ).clone()
                if sampling_multipliers.shape != (len(groups),):
                    raise ValueError("Replay sampling multipliers must match dataset length")
                auxiliary_phase_groups = {
                    value.strip()
                    for value in os.environ.get("VLA_AUXILIARY_PHASE_GROUPS", "").split(",")
                    if value.strip()
                }
                if auxiliary_phase_groups:
                    unknown = auxiliary_phase_groups - set(target_proportions)
                    if unknown:
                        raise ValueError(
                            "Unknown VLA_AUXILIARY_PHASE_GROUPS: " + ", ".join(sorted(unknown))
                        )
                    base_length = int(getattr(dataset, "base_length", len(dataset)))
                    for index in range(base_length, len(groups)):
                        if groups[index] not in auxiliary_phase_groups:
                            sampling_multipliers[index] = 0
                    selected = int((sampling_multipliers[base_length:] > 0).sum())
                    if selected == 0:
                        raise ValueError("Auxiliary phase filter selected no replay frames")
                    print(
                        "auxiliary_phase_filter "
                        f"groups={','.join(sorted(auxiliary_phase_groups))} "
                        f"selected_frames={selected}/{len(groups) - base_length}",
                        flush=True,
                    )
            weights = phase_sampling_weights(
                groups,
                target_proportions=target_proportions,
                sampling_multipliers=sampling_multipliers,
            )
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
    auxiliary_roots = [
        value.strip() for value in os.environ.get("VLA_AUXILIARY_DATASET", "").split(";") if value.strip()
    ]
    if not auxiliary_roots:
        return
    auxiliary_repo_ids = [
        value.strip() for value in os.environ.get("VLA_AUXILIARY_REPO_ID", "").split(";") if value.strip()
    ]
    if len(auxiliary_repo_ids) != len(auxiliary_roots):
        raise ValueError("VLA_AUXILIARY_REPO_ID is required for auxiliary replay")
    auxiliary_sample_weights = [
        float(value) for value in os.environ.get("VLA_AUXILIARY_SAMPLE_WEIGHT", "1.0").split(";")
    ]
    if len(auxiliary_sample_weights) != len(auxiliary_roots) or any(
        not math.isfinite(weight) or weight <= 0 for weight in auxiliary_sample_weights
    ):
        raise ValueError("VLA_AUXILIARY_SAMPLE_WEIGHT must be finite and positive")
    base_task_prompt = os.environ.get("VLA_BASE_TASK_PROMPT", "").strip()
    auxiliary_task_prompts = [
        value.strip() for value in os.environ.get("VLA_AUXILIARY_TASK_PROMPTS", "").split(";")
    ]
    if auxiliary_task_prompts and len(auxiliary_task_prompts) != len(auxiliary_roots):
        raise ValueError("one VLA_AUXILIARY_TASK_PROMPTS value per auxiliary is required")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataloader_class = torch.utils.data.DataLoader
    original_init = dataloader_class.__init__

    def data_loader_init(self, dataset, *args, **kwargs):
        if hasattr(dataset, "hf_dataset") and not getattr(
            dataset,
            "_vla_replay_mixed",
            False,
        ):
            auxiliaries = [
                LeRobotDataset(
                    repo_id,
                    root=Path(root),
                    delta_timestamps=dataset.delta_timestamps,
                    image_transforms=dataset.image_transforms,
                    video_backend=dataset.video_backend,
                    tolerance_s=dataset.tolerance_s,
                )
                for root, repo_id in zip(auxiliary_roots, auxiliary_repo_ids, strict=True)
            ]
            base, prompted_auxiliaries = apply_replay_task_prompts(
                dataset,
                auxiliaries,
                base_prompt=base_task_prompt,
                auxiliary_prompts=auxiliary_task_prompts,
            )
            dataset = (
                ReplayMixDataset(base, prompted_auxiliaries[0], auxiliary_sample_weights[0])
                if len(auxiliaries) == 1
                else ReplayMultiMixDataset(base, prompted_auxiliaries, auxiliary_sample_weights)
            )
            effective_auxiliary_fraction = float(
                dataset.sampling_multipliers[dataset.base_length :].sum()
                / dataset.sampling_multipliers.sum()
            )
            print(
                "auxiliary_replay "
                f"base_frames={len(dataset.base)} auxiliary_frames={sum(map(len, auxiliaries))} "
                f"auxiliary_weights={auxiliary_sample_weights} "
                f"effective_fraction={effective_auxiliary_fraction:.1%}",
                flush=True,
            )
        original_init(self, dataset, *args, **kwargs)

    dataloader_class.__init__ = data_loader_init  # type: ignore[method-assign]


_install_global_task_prompt()
_install_transition_sampler()
_install_phase_sampler()
_install_auxiliary_replay_dataset()
_install_task_sampler()

from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import lerobot_train

_continue_peft_adapter = os.environ.get("VLA_CONTINUE_PEFT_ADAPTER", "0") == "1"
if _continue_peft_adapter:
    _original_make_policy = lerobot_train.make_policy

    def _make_trainable_peft_policy(*args, **kwargs):
        policy = _original_make_policy(*args, **kwargs)
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("PEFT is required to continue an existing LoRA adapter") from error
        if not isinstance(policy, PeftModel):
            raise TypeError("Expected a PEFT checkpoint, but the loaded policy is not a PeftModel")
        adapters = list(policy.peft_config)
        if adapters != ["default"]:
            raise RuntimeError(f"Expected exactly one default PEFT adapter, got {adapters}")
        config = policy.peft_config["default"]
        if str(config.peft_type).upper().split(".")[-1] != "LORA":
            raise RuntimeError(f"Expected LoRA adapter, got {config.peft_type}")
        policy.set_adapter("default")
        config.inference_mode = False
        trainable = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
        lora_trainable = sum(
            parameter.numel()
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad and "lora_" in name
        )
        if trainable == 0 or lora_trainable != trainable:
            raise RuntimeError(
                "Existing adapter continuation must train LoRA parameters only: "
                f"trainable={trainable}, lora_trainable={lora_trainable}"
            )
        print(
            f"continue_peft_adapter adapter=default rank={config.r} "
            f"trainable_params={trainable}",
            flush=True,
        )
        return policy

    lerobot_train.make_policy = _make_trainable_peft_policy

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
