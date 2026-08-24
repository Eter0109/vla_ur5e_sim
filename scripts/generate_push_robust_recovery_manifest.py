"""Generate fresh Push recovery scenes for the longer-contact expert."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import attach_domain_randomization, generate_push_scenes, save_manifest  # noqa: E402


def main() -> int:
    # The extra 300 scenes are deterministic reserves for any unrecoverable
    # simulator episode. The collector stops after 1,500 accepted episodes.
    benchmark_id = "push_robust_recovery_collection_v1"
    scenes = generate_push_scenes(benchmark_id, 1800, 212_000)
    scenes = attach_domain_randomization(
        scenes,
        tier_counts={"nominal": 180, "light": 540, "medium": 1080},
        seed=262_000,
    )
    path = save_manifest(
        ROOT / "configs" / "benchmarks" / f"{benchmark_id}.json",
        scenes,
        benchmark_id=benchmark_id,
        role="collection",
        generator_seed=212_000,
        environment_preset="push_robust_v1",
    )
    print(f"manifest_ok path={path} scenes={len(scenes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
