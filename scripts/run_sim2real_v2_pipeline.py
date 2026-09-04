"""Run the complete resumable Sim2Real-v2 collection, build, audit, and smoke pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "sim2real_v2"
STATUS = OUTPUT / "pipeline_status.json"

TASKS = (
    (
        "push",
        ROOT / "configs/benchmarks/push_sim2real_v2_collection.json",
        ROOT / "data/lerobot/sim2real_v2_push_1500",
    ),
    (
        "pick_place",
        ROOT / "configs/benchmarks/pick_place_sim2real_v2_collection.json",
        ROOT / "data/lerobot/sim2real_v2_pick_place_1500",
    ),
    (
        "color_pick",
        ROOT / "configs/benchmarks/color_pick_sim2real_v2_collection.json",
        ROOT / "data/lerobot/sim2real_v2_color_pick_1500",
    ),
)
COMBINED = ROOT / "data/lerobot/multitask_sim2real_v2_4500"


def _write_status(stage: str, **details: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "updated_unix": time.time(), **details}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS)
    print(f"pipeline_stage={stage} details={details}", flush=True)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {command}")


def _collect(task: str, manifest: Path, dataset_root: Path) -> None:
    if (dataset_root / "collection.complete").exists():
        _write_status("collection_already_complete", task=task)
        return
    for retry in range(4):
        command = [
            sys.executable,
            str(ROOT / "scripts/collect_sim2real_v2.py"),
            "--task",
            task,
            "--manifest",
            str(manifest),
            "--root",
            str(dataset_root),
            "--repo-id",
            f"local/sim2real_v2_{task}_1500",
        ]
        if dataset_root.exists():
            command.append("--resume")
        _write_status("collecting", task=task, retry=retry)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode == 0 and (dataset_root / "collection.complete").exists():
            _write_status("collection_complete", task=task)
            return
        progress_path = dataset_root / "collection_progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            gate = progress.get("gate")
            if gate is not None and not gate.get("passed", False):
                raise RuntimeError(f"{task} first-attempt gate failed: {gate}")
        if retry == 3:
            raise RuntimeError(f"{task} collection failed after four launches")
        _write_status("collection_restart_wait", task=task, retry=retry + 1)
        time.sleep(60)


def main() -> int:
    try:
        for task, manifest, dataset_root in TASKS:
            _collect(task, manifest, dataset_root)
        if not (COMBINED / "collection.complete").exists():
            _write_status("building_combined")
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts/build_multitask_sim2real_v2_dataset.py"),
                    "--push-root",
                    str(TASKS[0][2]),
                    "--pick-place-root",
                    str(TASKS[1][2]),
                    "--color-pick-root",
                    str(TASKS[2][2]),
                    "--output-root",
                    str(COMBINED),
                ]
            )
        audit = OUTPUT / "audit.json"
        _write_status("auditing")
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/audit_sim2real_v2_dataset.py"),
                "--push-root",
                str(TASKS[0][2]),
                "--pick-place-root",
                str(TASKS[1][2]),
                "--color-pick-root",
                str(TASKS[2][2]),
                "--combined-root",
                str(COMBINED),
                "--output",
                str(audit),
            ]
        )
        smoke = OUTPUT / "train_smoke20"
        if not smoke.exists():
            _write_status("training_smoke20")
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts/train_smolvla_linux.py"),
                    "--model",
                    str(ROOT / ".runtime/models/smolvla_base"),
                    "--dataset",
                    str(COMBINED),
                    "--repo-id",
                    "local/multitask_sim2real_v2_4500",
                    "--output",
                    str(smoke),
                    "--steps",
                    "20",
                    "--batch-size",
                    "1",
                    "--learning-rate",
                    "5e-5",
                    "--warmup-steps",
                    "2",
                    "--decay-lr",
                    "1e-6",
                    "--save-freq",
                    "20",
                ]
            )
        _write_status("complete", audit=str(audit), smoke=str(smoke))
        return 0
    except Exception as error:
        _write_status("failed", error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
