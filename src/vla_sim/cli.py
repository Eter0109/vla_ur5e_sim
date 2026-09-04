"""Unified command line for the three-task VLA project."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .paths import load_catalog, project_root, resolve_asset

TASKS = ("push", "pick_place", "color_pick")
PRESETS = ("screen", "nominal", "randomized_screen", "randomized", "blind")


def _run(module: str, arguments: list[str]) -> int:
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=project_root(),
        check=False,
    ).returncode


def _simulation_catalog() -> dict[str, Any]:
    return load_catalog("simulation")


def _policy_paths() -> tuple[Path, Path]:
    catalog = load_catalog("policy")
    return resolve_asset(catalog["current"]), resolve_asset(catalog["training_state"]).parent


def _collect(args: argparse.Namespace, extra: list[str]) -> int:
    catalog = _simulation_catalog()
    tasks = TASKS if args.task == "all" else (args.task,)
    if args.task == "all" and (args.manifest or args.root or args.repo_id):
        raise ValueError("Per-task path overrides cannot be combined with --task all")
    for task in tasks:
        spec = catalog["tasks"][task]
        command = [
            "--task", task,
            "--manifest", str(args.manifest or resolve_asset(spec["collection"])),
            "--root", str(args.root or resolve_asset(spec["dataset"])),
            "--repo-id", args.repo_id or spec["repo_id"],
        ]
        if args.resume:
            command.append("--resume")
        code = _run("vla_sim.simulation.collection", [*command, *extra])
        if code:
            return code
    return 0


def _evaluate(args: argparse.Namespace, extra: list[str], *, inference: bool) -> int:
    catalog = _simulation_catalog()
    policy_catalog = load_catalog("policy")
    checkpoint = args.checkpoint or resolve_asset(policy_catalog["current"])
    tasks = (args.task,) if inference or args.task != "all" else TASKS
    for task in tasks:
        spec = catalog["tasks"][task]
        manifest_value = spec["benchmarks"].get(args.preset)
        if manifest_value is None:
            raise ValueError(f"{task} has no {args.preset!r} benchmark")
        manifest = resolve_asset(manifest_value)
        with manifest.open(encoding="utf-8") as handle:
            episode_count = len(json.load(handle)["scenes"])
        episodes = args.episodes or (1 if inference else episode_count)
        output = args.output
        if output is None:
            mode = "inference" if inference else "evaluation"
            output = project_root() / "outputs" / mode / f"{task}-{args.preset}.json"
        elif len(tasks) > 1:
            output = output / f"{task}-{args.preset}.json"
        command = [
            "--checkpoint", str(checkpoint),
            "--dataset-root", str(resolve_asset(spec["dataset"])),
            "--repo-id", spec["repo_id"],
            "--manifest", str(manifest),
            "--episodes", str(episodes),
            "--horizon", str(spec["horizon"]),
            "--output", str(output),
        ]
        code = _run(f"vla_sim.evaluation.tasks.{task}", [*command, *extra])
        if code:
            return code
    return 0


def _train(args: argparse.Namespace, extra: list[str]) -> int:
    simulation = _simulation_catalog()
    policy = load_catalog("policy")
    default_model = policy["base"] if args.fresh else policy["current"]
    model = args.model or resolve_asset(default_model)
    dataset = args.dataset or resolve_asset(simulation["combined_dataset"]["path"])
    output = args.output or project_root() / "outputs" / "training" / "sim2real_v2"
    proportions = {
        "push the block into the red target circle": 1 / 3,
        "place the red cube in the blue storage bin": 1 / 3,
        "pick up the blue cube": 1 / 9,
        "pick up the green cube": 1 / 9,
        "pick up the red cube": 1 / 9,
    }
    command = [
        "--model", str(model),
        "--dataset", str(dataset),
        "--repo-id", simulation["combined_dataset"]["repo_id"],
        "--output", str(output),
        "--steps", str(args.steps),
        "--scheduler-decay-steps", "20000",
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--learning-rate", "5e-6",
        "--warmup-steps", "500",
        "--decay-lr", "1e-6",
        "--save-freq", "2000",
        "--task-sampling-proportions", json.dumps(proportions),
    ]
    if not args.fresh:
        command.extend(
            [
                "--continue-peft-adapter",
                "--resume-checkpoint",
                str(resolve_asset(policy["training_state"]).parent),
            ]
        )
    else:
        command.extend(["--lora-rank", "32"])
    return _run("vla_sim.policy.training", [*command, *extra])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vla-sim", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="Collect three-task demonstrations")
    collect.add_argument("--task", choices=(*TASKS, "all"), required=True)
    collect.add_argument("--manifest", type=Path)
    collect.add_argument("--root", type=Path)
    collect.add_argument("--repo-id")
    collect.add_argument("--resume", action="store_true")

    train = commands.add_parser("train", help="Train or resume the current policy")
    train.add_argument("--model", type=Path)
    train.add_argument("--dataset", type=Path)
    train.add_argument("--output", type=Path)
    train.add_argument("--steps", type=int, default=12000)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--seed", type=int, default=1000)
    train.add_argument("--fresh", action="store_true")

    for name in ("infer", "evaluate"):
        command = commands.add_parser(name, help=f"Run policy {name}")
        choices = TASKS if name == "infer" else (*TASKS, "all")
        command.add_argument("--task", choices=choices, required=True)
        command.add_argument("--preset", choices=PRESETS, default="screen")
        command.add_argument("--checkpoint", type=Path)
        command.add_argument("--episodes", type=int)
        command.add_argument("--output", type=Path)

    pipeline = commands.add_parser("pipeline", help="Run collection, build, audit, and smoke")
    pipeline.add_argument(
        "--from", dest="from_stage", choices=("collect", "build", "audit", "smoke"), default="collect"
    )
    pipeline.add_argument(
        "--through", choices=("collect", "build", "audit", "smoke"), default="smoke"
    )
    return parser


def main() -> int:
    parser = _parser()
    args, extra = parser.parse_known_args()
    try:
        if args.command == "collect":
            return _collect(args, extra)
        if args.command == "train":
            return _train(args, extra)
        if args.command in {"infer", "evaluate"}:
            return _evaluate(args, extra, inference=args.command == "infer")
        return _run(
            "vla_sim.pipeline",
            ["--from", args.from_stage, "--through", args.through, *extra],
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
