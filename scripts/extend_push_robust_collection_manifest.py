"""Append deterministic reserve scenes after expert failures during Push collection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import (  # noqa: E402
    attach_domain_randomization,
    generate_push_scenes,
    load_manifest,
    save_manifest,
)


def main() -> int:
    source = ROOT / "configs" / "benchmarks" / "push_robust_collection_v1.json"
    destination = ROOT / "configs" / "benchmarks" / "push_robust_collection_v2.json"
    original = load_manifest(source)
    reserve = generate_push_scenes("push_robust_reserve_v1", 600, 211_500)
    reserve = attach_domain_randomization(
        reserve,
        tier_counts={"nominal": 120, "light": 240, "medium": 240},
        seed=261_500,
    )
    save_manifest(
        destination,
        [*original, *reserve],
        benchmark_id="push_robust_collection_v2",
        role="collection",
        generator_seed=210_000,
        environment_preset="push_robust_v1",
    )
    print(f"manifest_ok path={destination} scenes={len(original) + len(reserve)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
