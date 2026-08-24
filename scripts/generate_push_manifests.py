"""Generate balanced, deterministic development and held-out Push manifests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import generate_push_scenes, save_manifest  # noqa: E402


def main() -> int:
    specs = (
        ("push_dev_v3", "development", 112000, 20),
        ("push_blind_v5", "blind", 122000, 20),
        ("push_blind_v6", "blind", 132000, 20),
        ("push_collect_v4", "collection", 150000, 1000),
    )
    for benchmark_id, role, seed, count in specs:
        path = ROOT / "configs" / "benchmarks" / f"{benchmark_id}.json"
        save_manifest(
            path,
            generate_push_scenes(benchmark_id, count, seed),
            benchmark_id=benchmark_id,
            role=role,
            generator_seed=seed,
            environment_preset="push_forward_v1",
        )
        print(f"manifest_ok path={path} role={role} scenes={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
