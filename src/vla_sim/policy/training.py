"""Linux-native, provenance-recording launcher for the local SmolVLA trainer."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from vla_sim.paths import project_root
from vla_sim.policy.runtime import resolve_checkpoint

ROOT = project_root()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--decay-lr", type=float, required=True)
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument(
        "--scheduler-decay-steps",
        type=int,
        help="Total scheduler horizon; defaults to --steps.",
    )
    parser.add_argument(
        "--continue-peft-adapter",
        action="store_true",
        help="Load one existing LoRA adapter as trainable without wrapping it again.",
    )
    parser.add_argument(
        "--task-sampling-proportions",
        default="",
        help="JSON mapping from exact task prompt to desired sampling probability.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Numbered checkpoint directory used for exact optimizer/scheduler/RNG resume.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="Enable all-linear LoRA adaptation at the given positive rank.",
    )
    return parser


def _git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _cuda_preflight() -> dict[str, str | int | bool]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA preflight failed; refusing to fall back to CPU")
    device = torch.device("cuda:0")
    left = torch.randn(32, 32, device=device, requires_grad=True)
    right = torch.randn(32, 32, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = (left @ right).square().mean()
    loss.backward()
    if left.grad is None or not torch.isfinite(left.grad).all():
        raise RuntimeError("CUDA AMP backward preflight produced invalid gradients")
    result = {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "amp_backward": True,
    }
    del left, right, loss
    torch.cuda.empty_cache()
    return result


def _input_hashes(model: Path, dataset: Path) -> dict[str, str]:
    sys.path.insert(0, str(ROOT / "src"))
    from vla_sim.simulation.artifacts import sha256_directory, sha256_file

    hashes = {"initial_checkpoint_sha256": sha256_directory(model)}
    for name in ("build_provenance.json", "info.json", "tasks.parquet"):
        path = dataset / "meta" / name
        if not path.is_file():
            raise FileNotFoundError(f"Dataset identity file is missing: {path}")
        hashes[f"dataset_{name}_sha256"] = sha256_file(path)
    return hashes


def _write_manifest(path: Path, payload: dict, *, resume: bool) -> None:
    if resume and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        history = list(existing.get("resume_history", []))
        history.append(payload)
        existing["resume_history"] = history
        payload = existing
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    args.model = resolve_checkpoint(args.model)
    scheduler_decay_steps = args.scheduler_decay_steps or args.steps
    if args.steps < 1 or args.batch_size < 1 or args.save_freq < 1 or args.lora_rank < 0:
        raise ValueError("steps, batch-size, and save-freq must be positive")
    if scheduler_decay_steps < args.steps:
        raise ValueError("scheduler-decay-steps must be greater than or equal to steps")
    if args.continue_peft_adapter and args.lora_rank:
        raise ValueError("Existing adapter continuation must not request a second LoRA wrapper")
    if args.resume_checkpoint and not args.continue_peft_adapter:
        raise ValueError("This resume path requires --continue-peft-adapter")
    if args.resume_checkpoint and not args.resume_checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_checkpoint}")
    if args.output.exists() and not args.resume_checkpoint:
        raise FileExistsError(f"Refusing to overwrite training output: {args.output}")
    if not args.model.is_dir() or not args.dataset.is_dir():
        raise FileNotFoundError("model and dataset must both exist before training")
    task_sampling_proportions: dict[str, float] = {}
    if args.task_sampling_proportions:
        task_sampling_proportions = {
            str(prompt): float(value)
            for prompt, value in json.loads(args.task_sampling_proportions).items()
        }
        if not task_sampling_proportions:
            raise ValueError("task-sampling-proportions must not be empty")

    cuda_preflight = _cuda_preflight()
    input_hashes = _input_hashes(args.model, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output.parent / ".sim2real_v2_formal_training.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another Sim2Real-v2 formal training launcher is active") from error

    runtime = ROOT / ".runtime"
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": str(runtime / "hf"),
            "HF_DATASETS_CACHE": str(runtime / "hf_datasets"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MUJOCO_GL": "egl",
            "NUMBA_CACHE_DIR": str(Path(tempfile.gettempdir()) / "vla_sim_numba"),
            "TOKENIZERS_PARALLELISM": "false",
            "VLA_FAST_PARQUET_LOADER": "0",
            "VLA_LAZY_PARQUET_LOADER": "1",
            "VLA_LAZY_PARQUET_CACHE_FILES": "2",
            "VLA_XYZ_LOSS_WEIGHT": os.environ.get("VLA_XYZ_LOSS_WEIGHT", "2.0"),
            "VLA_ROTATION_LOSS_WEIGHT": os.environ.get("VLA_ROTATION_LOSS_WEIGHT", "0"),
            "VLA_GRIPPER_LOSS_WEIGHT": os.environ.get("VLA_GRIPPER_LOSS_WEIGHT", "2.5"),
            "VLA_SAMPLING_SEED": str(args.seed),
            "VLA_CONTINUE_PEFT_ADAPTER": "1" if args.continue_peft_adapter else "0",
            "VLA_TASK_SAMPLING_PROPORTIONS": (
                json.dumps(task_sampling_proportions) if task_sampling_proportions else ""
            ),
        }
    )
    if args.resume_checkpoint:
        environment["VLA_RESUME_CHECKPOINT"] = str(args.resume_checkpoint.resolve())
    command = [
        sys.executable,
        "-m", "vla_sim.policy.trainer_entrypoint",
        f"--policy.path={args.model}",
        "--policy.input_features=null",
        "--policy.device=cuda",
        "--policy.use_amp=true",
        "--policy.push_to_hub=false",
        "--policy.load_vlm_weights=false",
        "--policy.resize_imgs_with_padding=[256,256]",
        "--policy.chunk_size=16",
        "--policy.n_action_steps=8",
        "--policy.tokenizer_max_length=16",
        "--policy.num_steps=10",
        f"--policy.optimizer_lr={args.learning_rate}",
        f"--policy.scheduler_warmup_steps={args.warmup_steps}",
        f"--policy.scheduler_decay_steps={scheduler_decay_steps}",
        f"--policy.scheduler_decay_lr={args.decay_lr}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={args.dataset}",
        "--dataset.use_imagenet_stats=false",
        f"--batch_size={args.batch_size}",
        "--num_workers=0",
        f"--steps={args.steps}",
        "--eval_freq=0",
        "--log_freq=20",
        f"--save_freq={args.save_freq}",
        f"--output_dir={args.output}",
        f"--seed={args.seed}",
        f"--job_name={args.output.name}",
        "--wandb.enable=false",
    ]
    if args.resume_checkpoint:
        command.append("--resume=true")
    if args.lora_rank:
        command.extend(
            [
                "--peft.method_type=LORA",
                f"--peft.r={args.lora_rank}",
                '--peft.target_modules=all-linear',
            ]
        )
    else:
        command.extend(
            [
                "--policy.freeze_vision_encoder=true",
                "--policy.train_expert_only=true",
            ]
        )
    manifest = {
        "schema_version": 1,
        "platform": "linux",
        "command": command,
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "repo_id": args.repo_id,
        "output": str(args.output.resolve()),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "scheduler_decay_steps": scheduler_decay_steps,
        "decay_lr": args.decay_lr,
        "lora_rank": args.lora_rank,
        "continue_peft_adapter": args.continue_peft_adapter,
        "task_sampling_proportions": task_sampling_proportions,
        "resume_checkpoint": (
            str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None
        ),
        "cuda_preflight": cuda_preflight,
        "input_hashes": input_hashes,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_status": subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines(),
    }
    _write_manifest(
        args.output.parent / f"{args.output.name}.run_manifest.json",
        manifest,
        resume=bool(args.resume_checkpoint),
    )
    try:
        return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
