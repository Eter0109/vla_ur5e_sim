"""Generate fixed manifests for robust Push and PickPlace multitask experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import (  # noqa: E402
    attach_domain_randomization,
    generate_pick_place_scenes,
    generate_push_scenes,
    save_manifest,
)


def _write(task: str, role: str, count: int, seed: int, tier_counts: dict[str, int]) -> None:
    benchmark_id = f"{task}_robust_{role}_v1"
    generator = generate_push_scenes if task == "push" else generate_pick_place_scenes
    scenes = generator(benchmark_id, count, seed)
    if sum(tier_counts.values()) != count:
        raise ValueError("tier counts must match manifest size")
    scenes = attach_domain_randomization(scenes, tier_counts=tier_counts, seed=seed + 50_000)
    path = save_manifest(
        ROOT / "configs" / "benchmarks" / f"{benchmark_id}.json",
        scenes,
        benchmark_id=benchmark_id,
        role=role,
        generator_seed=seed,
        environment_preset=f"{task}_robust_v1",
    )
    print(f"manifest_ok task={task} role={role} path={path} scenes={count} tiers={tier_counts}")


def main() -> int:
    # 1,500 demonstrations per task: 20% nominal, 40% light, 40% medium.
    for task, seed in (("push", 210_000), ("pick_place", 220_000)):
        _write(task, "collection", 1500, seed, {"nominal": 300, "light": 600, "medium": 600})
        _write(task, "development_nominal", 50, seed + 1_000, {"nominal": 50})
        _write(task, "development_randomized", 50, seed + 2_000, {"light": 25, "medium": 25})
        _write(task, "blind", 100, seed + 3_000, {"blind": 100})
        _write(task, "stress", 30, seed + 4_000, {"stress": 30})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
