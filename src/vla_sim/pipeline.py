"""Resumable collection, dataset build, audit, and training-smoke pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Any

from .paths import load_catalog, project_root, resolve_asset

ROOT = project_root()
OUTPUT = ROOT / "outputs" / "pipeline"
STATUS = OUTPUT / "status.json"
STAGES = ("collect", "build", "audit", "smoke")


def _write_status(stage: str, **details: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "updated_unix": time.time(), **details}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS)
    print(f"pipeline_stage={stage} details={details}", flush=True)


def _run(module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-m", module, *arguments]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {command}")


def _selected(stage: str, first: str, last: str) -> bool:
    return STAGES.index(first) <= STAGES.index(stage) <= STAGES.index(last)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="first", choices=STAGES, default="collect")
    parser.add_argument("--through", dest="last", choices=STAGES, default="smoke")
    args = parser.parse_args()
    if STAGES.index(args.first) > STAGES.index(args.last):
        parser.error("--from must not come after --through")

    catalog = load_catalog("simulation")
    task_specs = catalog["tasks"]
    combined = resolve_asset(catalog["combined_dataset"]["path"])
    try:
        if _selected("collect", args.first, args.last):
            for task, spec in task_specs.items():
                root = resolve_asset(spec["dataset"])
                if (root / "collection.complete").exists():
                    _write_status("collection_already_complete", task=task)
                    continue
                command = [
                    "--task", task,
                    "--manifest", str(resolve_asset(spec["collection"])),
                    "--root", str(root),
                    "--repo-id", spec["repo_id"],
                ]
                if root.exists():
                    command.append("--resume")
                _write_status("collecting", task=task)
                _run("vla_sim.simulation.collection", command)

        if _selected("build", args.first, args.last) and not (
            combined / "collection.complete"
        ).exists():
            _write_status("building_combined")
            _run(
                "vla_sim.simulation.dataset_build",
                [
                    "--push-root", str(resolve_asset(task_specs["push"]["dataset"])),
                    "--pick-place-root",
                    str(resolve_asset(task_specs["pick_place"]["dataset"])),
                    "--color-pick-root",
                    str(resolve_asset(task_specs["color_pick"]["dataset"])),
                    "--output-root", str(combined),
                ],
            )

        audit = OUTPUT / "dataset_audit.json"
        if _selected("audit", args.first, args.last):
            _write_status("auditing")
            _run(
                "vla_sim.simulation.dataset_audit",
                [
                    "--push-root", str(resolve_asset(task_specs["push"]["dataset"])),
                    "--pick-place-root",
                    str(resolve_asset(task_specs["pick_place"]["dataset"])),
                    "--color-pick-root",
                    str(resolve_asset(task_specs["color_pick"]["dataset"])),
                    "--combined-root", str(combined),
                    "--output", str(audit),
                ],
            )

        if _selected("smoke", args.first, args.last):
            smoke = OUTPUT / "training_smoke"
            if smoke.exists():
                raise FileExistsError(f"Refusing to overwrite smoke output: {smoke}")
            policy = load_catalog("policy")
            _write_status("training_smoke")
            _run(
                "vla_sim.policy.training",
                [
                    "--model", str(resolve_asset(policy["base"])),
                    "--dataset", str(combined),
                    "--repo-id", catalog["combined_dataset"]["repo_id"],
                    "--output", str(smoke),
                    "--steps", "1",
                    "--batch-size", "1",
                    "--learning-rate", "5e-5",
                    "--warmup-steps", "1",
                    "--decay-lr", "1e-6",
                    "--save-freq", "1",
                ],
            )
        _write_status("complete", first=args.first, last=args.last)
        return 0
    except Exception as error:
        _write_status("failed", error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
