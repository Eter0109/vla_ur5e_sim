"""Create small stratified development subsets for checkpoint early stopping."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.scenes import (  # noqa: E402
    SceneSpec,
    load_manifest,
    load_manifest_metadata,
    select_stratified_scenes,
)

SOURCES = {
    "push_nominal": "push_robust_development_nominal_v1.json",
    "push_randomized": "push_robust_development_randomized_v1.json",
    "pick_nominal": "pick_place_robust_development_nominal_v1.json",
    "pick_randomized": "pick_place_robust_development_randomized_v1.json",
}
REFERENCE_CHECKPOINT_SHA256 = "DFFBFCD07911EBCC0658B853ED5031855741DAE26C3241B9AA67AB56C76DD7B7"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stratum(kind: str, scene: SceneSpec) -> tuple:
    overrides = scene.overrides
    tier = overrides.get("domain_randomization", {}).get("tier", "nominal")
    if kind.startswith("push"):
        return (tier, int(overrides["angle_bin"]), int(overrides["distance_bin"]))
    target_y_sign = "negative" if float(overrides["target_y_m"]) < 0 else "positive"
    return (tier, int(overrides["distance_bin"]), target_y_sign)


def main() -> int:
    destination = ROOT / "configs" / "benchmarks"
    for kind, filename in SOURCES.items():
        source = destination / filename
        scenes = load_manifest(source)
        metadata = load_manifest_metadata(source)
        selected = select_stratified_scenes(
            scenes,
            count=20,
            stratum=lambda scene, selected_kind=kind: _stratum(selected_kind, scene),
        )
        benchmark_id = f"{metadata['benchmark_id']}_screen20"
        payload = {
            "schema_version": 2,
            "benchmark_id": benchmark_id,
            "role": "development",
            "generator_seed": metadata.get("generator_seed"),
            "environment_preset": metadata.get("environment_preset"),
            "source_manifest": str(source.relative_to(ROOT)),
            "source_manifest_sha256": _sha256(source),
            "selection": "deterministic_stratified_round_robin_v1",
            "reference_checkpoint_sha256": REFERENCE_CHECKPOINT_SHA256,
            "scenes": [asdict(scene) for scene in selected],
        }
        output = destination / f"{benchmark_id}.json"
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        strata = {_stratum(kind, scene) for scene in selected}
        print(f"screening_manifest={output} scenes={len(selected)} strata={len(strata)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
