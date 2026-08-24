"""Build auditable screen20 reference rates from the canonical full results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.artifact_identity import sha256_directory  # noqa: E402

SPLITS = {
    "push_nominal": (
        "configs/benchmarks/push_robust_development_nominal_v1_screen20.json",
        "outputs/multitask_robust/eval_orig30k_push_nominal_d075.json",
    ),
    "push_randomized": (
        "configs/benchmarks/push_robust_development_randomized_v1_screen20.json",
        "outputs/multitask_robust/eval_orig30k_push_randomized_d075.json",
    ),
    "pick_nominal": (
        "configs/benchmarks/pick_place_robust_development_nominal_v1_screen20.json",
        "outputs/multitask_robust/ablation_orig30k_pick_nominal_gain18_samples2.json",
    ),
    "pick_randomized": (
        "configs/benchmarks/pick_place_robust_development_randomized_v1_screen20.json",
        "outputs/multitask_robust/ablation_orig30k_pick_randomized_gain18_samples2.json",
    ),
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_sha256 = sha256_directory(args.checkpoint)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "scope": "screening_reference_only_not_final_evidence",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "splits": {},
    }
    for split, (manifest_name, result_name) in SPLITS.items():
        manifest_path = ROOT / manifest_name
        result_path = ROOT / result_name
        ids = {scene["scene_id"] for scene in _json(manifest_path)["scenes"]}
        payload = _json(result_path)
        rows = payload["results"] if isinstance(payload, dict) else payload
        selected = [row for row in rows if row["scene_id"] in ids]
        if len(selected) != 20:
            raise ValueError(f"{split} reference result does not cover all screen scenes")
        if split.startswith("push"):
            checkpoint = Path(payload["checkpoint"])
            actual_sha256 = sha256_directory(checkpoint)
        else:
            metadata = _json(result_path.with_suffix(result_path.suffix + ".meta.json"))
            actual_sha256 = str(metadata["checkpoint_sha256"])
        if actual_sha256.lower() != checkpoint_sha256.lower():
            raise ValueError(f"{split} reference checkpoint SHA does not match")
        successes = sum(bool(row["success"]) for row in selected)
        summary["splits"][split] = {
            "episodes": 20,
            "successes": successes,
            "success_rate": successes / 20,
            "screen_manifest_sha256": _sha256(manifest_path),
            "source_result": str(result_path.resolve()),
            "source_result_sha256": _sha256(result_path),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
