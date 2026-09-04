"""Run staged three-task LoRA training and fixed development acceptance checks."""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/sim2real_v2/formal_lora_r32_seed1000"
STATUS = ROOT / "outputs/sim2real_v2/formal_training_status.json"
LOG = ROOT / "outputs/sim2real_v2/formal_training.log"
INITIAL = (
    ROOT
    / "outputs/multitask_nominal90/round9_lora_r32_aux6_seed1000"
    / "checkpoints/005500/pretrained_model"
)
DATASET = ROOT / "data/lerobot/multitask_sim2real_v2_4500"
SOURCE_DATASETS = {
    "push": ROOT / "data/lerobot/sim2real_v2_push_1500",
    "pick": ROOT / "data/lerobot/sim2real_v2_pick_place_1500",
    "color": ROOT / "data/lerobot/sim2real_v2_color_pick_1500",
}
TASK_PROPORTIONS = {
    "push the block into the red target circle": 1 / 3,
    "place the red cube in the blue storage bin": 1 / 3,
    "pick up the blue cube": 1 / 9,
    "pick up the green cube": 1 / 9,
    "pick up the red cube": 1 / 9,
}


def _status(stage: str, **details) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "updated_unix": time.time(), **details}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS)
    print(f"formal_training_stage={stage} details={details}", flush=True)


def _run(command: list[str], *, allow_failure: bool = False) -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode and not allow_failure:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command[1]}")
    return result.returncode


def _checkpoint(step: int) -> Path:
    return OUTPUT / "checkpoints" / f"{step:06d}"


def _checkpoint_complete(path: Path) -> bool:
    return all(
        candidate.is_file()
        for candidate in (
            path / "pretrained_model/config.json",
            path / "pretrained_model/adapter_config.json",
            path / "pretrained_model/adapter_model.safetensors",
            path / "training_state/optimizer_state.safetensors",
            path / "training_state/scheduler_state.json",
            path / "training_state/rng_state.safetensors",
            path / "training_state/training_step.json",
        )
    )


def _latest_checkpoint(max_step: int) -> tuple[int, Path] | None:
    candidates: list[tuple[int, Path]] = []
    checkpoint_root = OUTPUT / "checkpoints"
    if checkpoint_root.is_dir():
        for path in checkpoint_root.iterdir():
            try:
                step = int(path.name)
            except ValueError:
                continue
            if step <= max_step and _checkpoint_complete(path):
                candidates.append((step, path))
    return max(candidates, default=None)


def _train_to(target_step: int) -> None:
    restart_count = 0
    while True:
        latest = _latest_checkpoint(target_step)
        if latest and latest[0] >= target_step:
            return
        if restart_count > 3:
            raise RuntimeError("Training failed after three automatic recovery attempts")
        model = INITIAL if latest is None else latest[1] / "pretrained_model"
        command = [
            sys.executable,
            "scripts/train_smolvla_linux.py",
            "--model", str(model),
            "--dataset", str(DATASET),
            "--repo-id", "local/multitask_sim2real_v2_4500",
            "--output", str(OUTPUT),
            "--steps", str(target_step),
            "--scheduler-decay-steps", "20000",
            "--batch-size", "4",
            "--seed", "1000",
            "--learning-rate", "5e-6",
            "--warmup-steps", "500",
            "--decay-lr", "1e-6",
            "--save-freq", "2000",
            "--continue-peft-adapter",
            "--task-sampling-proportions", json.dumps(TASK_PROPORTIONS),
        ]
        if latest is not None:
            command += ["--resume-checkpoint", str(latest[1])]
        elif OUTPUT.exists():
            raise RuntimeError("Partial output exists without a complete checkpoint; refusing overwrite")
        _status(
            "training" if latest is None else "training_recovery",
            target_step=target_step,
            from_step=0 if latest is None else latest[0],
            recovery=restart_count,
        )
        return_code = _run(command, allow_failure=True)
        if return_code == 0 and _checkpoint_complete(_checkpoint(target_step)):
            return
        restart_count += 1


def _evaluate_two_task_screen(step: int) -> bool:
    checkpoint = _checkpoint(step) / "pretrained_model"
    output = ROOT / f"outputs/sim2real_v2/eval_{step:06d}/screen"
    output.mkdir(parents=True, exist_ok=True)
    push_path = output / "push.json"
    pick_path = output / "pick.json"
    if not push_path.exists():
        _run([
            sys.executable, "scripts/run_push_vla_only_benchmark.py",
            "--checkpoint", str(checkpoint),
            "--dataset-root", str(SOURCE_DATASETS["push"]),
            "--repo-id", "local/sim2real_v2_push_1500",
            "--manifest", "configs/benchmarks/push_robust_development_nominal_v1_screen20.json",
            "--episodes", "20", "--horizon", "250", "--replan-steps", "4",
            "--temporal-decay", "0.75", "--policy-seed", "1000",
            "--samples-per-plan", "1", "--output", str(push_path),
        ])
    if not pick_path.exists():
        _run([
            sys.executable, "scripts/run_pick_place_vla_only.py",
            "--checkpoint", str(checkpoint),
            "--dataset-root", str(SOURCE_DATASETS["pick"]),
            "--repo-id", "local/sim2real_v2_pick_place_1500",
            "--manifest", "configs/benchmarks/pick_place_robust_development_nominal_v1_screen20.json",
            "--episodes", "20", "--horizon", "250", "--replan-steps", "4",
            "--temporal-ensemble-decay", "0.75", "--samples-per-plan", "2",
            "--control-mode", "vla_action_calibrated", "--closed-negative-y-gain", "1.8",
            "--policy-seed", "1000", "--output", str(pick_path),
        ])
    push = json.loads(push_path.read_text(encoding="utf-8"))
    pick = json.loads(pick_path.read_text(encoding="utf-8"))
    push_successes = int(push["summary"]["successes"])
    pick_successes = sum(bool(row["success"]) for row in pick)
    report = {
        "step": step,
        "push": push_successes,
        "pick_place": pick_successes,
        "required": 18,
        "passed": push_successes >= 18 and pick_successes >= 18,
    }
    (output / "screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return bool(report["passed"])


def _evaluate_full(step: int) -> bool:
    checkpoint = _checkpoint(step) / "pretrained_model"
    output = ROOT / f"outputs/sim2real_v2/eval_{step:06d}/nominal_full"
    output.mkdir(parents=True, exist_ok=True)
    push_path, pick_path, color_path = (
        output / "push.json", output / "pick.json", output / "color.json"
    )
    commands = [
        (push_path, [
            sys.executable, "scripts/run_push_vla_only_benchmark.py",
            "--checkpoint", str(checkpoint), "--dataset-root", str(SOURCE_DATASETS["push"]),
            "--repo-id", "local/sim2real_v2_push_1500", "--manifest",
            "configs/benchmarks/push_robust_development_nominal_v1.json", "--episodes", "50",
            "--horizon", "250", "--replan-steps", "4", "--temporal-decay", "0.75",
            "--policy-seed", "1000", "--samples-per-plan", "1", "--output", str(push_path),
        ]),
        (pick_path, [
            sys.executable, "scripts/run_pick_place_vla_only.py",
            "--checkpoint", str(checkpoint), "--dataset-root", str(SOURCE_DATASETS["pick"]),
            "--repo-id", "local/sim2real_v2_pick_place_1500", "--manifest",
            "configs/benchmarks/pick_place_robust_development_nominal_v1.json", "--episodes", "50",
            "--horizon", "250", "--replan-steps", "4", "--temporal-ensemble-decay", "0.75",
            "--samples-per-plan", "2", "--control-mode", "vla_action_calibrated",
            "--closed-negative-y-gain", "1.8", "--policy-seed", "1000", "--output", str(pick_path),
        ]),
        (color_path, [
            sys.executable, "scripts/run_color_pick_vla_only.py",
            "--checkpoint", str(checkpoint), "--dataset-root", str(SOURCE_DATASETS["color"]),
            "--repo-id", "local/sim2real_v2_color_pick_1500", "--manifest",
            "configs/benchmarks/color_pick_development_v1.json", "--episodes", "60",
            "--horizon", "200", "--replan-steps", "4", "--temporal-ensemble-decay", "0.75",
            "--samples-per-plan", "1", "--policy-seed", "1000", "--output", str(color_path),
        ]),
    ]
    for path, command in commands:
        if not path.exists():
            _run(command)
    gate = output / "nominal90_gate.json"
    return_code = _run([
        sys.executable, "scripts/verify_sim2real_v2_nominal90.py",
        "--checkpoint", str(checkpoint), "--push", str(push_path), "--pick", str(pick_path),
        "--color", str(color_path), "--output", str(gate),
    ], allow_failure=True)
    return return_code == 0


def _evaluate_available(max_step: int) -> int | None:
    for step in range(2000, max_step + 1, 2000):
        if not _checkpoint_complete(_checkpoint(step)):
            continue
        _status("screening", checkpoint_step=step)
        if _evaluate_two_task_screen(step):
            _status("nominal_full", checkpoint_step=step)
            if _evaluate_full(step):
                return step
    return None


def _evaluate_randomized(step: int) -> Path:
    checkpoint = _checkpoint(step) / "pretrained_model"
    output = ROOT / f"outputs/sim2real_v2/eval_{step:06d}/randomized_development"
    output.mkdir(parents=True, exist_ok=True)
    push_path, pick_path, color_path = (
        output / "push.json", output / "pick.json", output / "color.json"
    )
    commands = [
        (push_path, [
            sys.executable, "scripts/run_push_vla_only_benchmark.py",
            "--checkpoint", str(checkpoint), "--dataset-root", str(SOURCE_DATASETS["push"]),
            "--repo-id", "local/sim2real_v2_push_1500", "--manifest",
            "configs/benchmarks/push_robust_development_randomized_v1.json", "--episodes", "50",
            "--horizon", "250", "--replan-steps", "4", "--temporal-decay", "0.75",
            "--policy-seed", "1000", "--samples-per-plan", "1", "--output", str(push_path),
        ]),
        (pick_path, [
            sys.executable, "scripts/run_pick_place_vla_only.py",
            "--checkpoint", str(checkpoint), "--dataset-root", str(SOURCE_DATASETS["pick"]),
            "--repo-id", "local/sim2real_v2_pick_place_1500", "--manifest",
            "configs/benchmarks/pick_place_robust_development_randomized_v1.json", "--episodes", "50",
            "--horizon", "250", "--replan-steps", "4", "--temporal-ensemble-decay", "0.75",
            "--samples-per-plan", "2", "--control-mode", "vla_action_calibrated",
            "--closed-negative-y-gain", "1.8", "--policy-seed", "1000", "--output", str(pick_path),
        ]),
        (color_path, [
            sys.executable, "scripts/run_color_pick_vla_only.py",
            "--checkpoint", str(checkpoint), "--dataset-root", str(SOURCE_DATASETS["color"]),
            "--repo-id", "local/sim2real_v2_color_pick_1500", "--manifest",
            "configs/benchmarks/color_pick_development_randomized_v1.json", "--episodes", "60",
            "--horizon", "200", "--replan-steps", "4", "--temporal-ensemble-decay", "0.75",
            "--samples-per-plan", "1", "--policy-seed", "1000", "--output", str(color_path),
        ]),
    ]
    for path, command in commands:
        if not path.exists():
            _run(command)
    push = json.loads(push_path.read_text(encoding="utf-8"))
    pick = json.loads(pick_path.read_text(encoding="utf-8"))
    color = json.loads(color_path.read_text(encoding="utf-8"))
    report = {
        "checkpoint_step": step,
        "acceptance_effect": "report_only",
        "push": {"successes": int(push["summary"]["successes"]), "episodes": 50},
        "pick_place": {"successes": sum(bool(row["success"]) for row in pick), "episodes": 50},
        "color_pick": {
            "successes": sum(bool(row["success"]) for row in color),
            "episodes": 60,
            "wrong_color_grasps": sum(
                bool(row.get("ever_wrong_object_grasped")) for row in color
            ),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    lock_path = STATUS.parent / ".sim2real_v2_formal_supervisor.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _status("already_running")
        return 0
    try:
        if not torch.cuda.is_available():
            _status("waiting_for_cuda", error="CUDA is unavailable; CPU fallback is forbidden")
            return 75
        _train_to(12000)
        accepted = _evaluate_available(12000)
        if accepted is None:
            _status("extending", from_step=12000, target_step=20000)
            _train_to(20000)
            accepted = _evaluate_available(20000)
        if accepted is None:
            _status("failed_gate", max_step=20000)
            return 2
        _status("nominal_passed", checkpoint_step=accepted)
        _status("randomized_development", checkpoint_step=accepted)
        randomized_report = _evaluate_randomized(accepted)
        _status(
            "complete",
            checkpoint_step=accepted,
            randomized_development=str(randomized_report),
        )
        return 0
    except Exception as error:
        _status("failed", error=str(error))
        raise
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
