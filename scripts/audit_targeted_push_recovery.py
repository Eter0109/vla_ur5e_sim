"""Audit a finalized target-80 Push recovery dataset before joint training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vla_sim.contracts import ACTION_KEY, IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY  # noqa: E402
from vla_sim.provenance import (  # noqa: E402
    scene_seed_overlap,
    targeted_push_distribution_errors,
)

PROMPT = "push the block into the red target circle"
REQUIRED_FEATURES = {IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY, ACTION_KEY}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, action="append", default=[])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (args.dataset_root / "collection.complete").exists():
        raise ValueError("targeted Push dataset is not finalized")
    info = _json(args.dataset_root / "meta" / "info.json")
    base_info = _json(args.base_root / "meta" / "info.json")
    provenance = _json(args.dataset_root / "meta" / "collection_provenance.json")
    manifest = _json(args.manifest)
    if int(info["total_episodes"]) != args.episodes:
        raise ValueError("targeted Push episode count does not match the training contract")
    if info["features"] != base_info["features"] or int(info["fps"]) != int(base_info["fps"]):
        raise ValueError("targeted Push features or FPS differ from the multitask base")
    if REQUIRED_FEATURES - set(info["features"]):
        raise ValueError("targeted Push dataset is missing required features")
    if provenance.get("manifest_sha256") != _sha256(args.manifest):
        raise ValueError("collection provenance does not match the targeted manifest")
    successful_ids = [
        result["scene_id"] for result in provenance["results"] if result.get("success")
    ]
    if len(successful_ids) != args.episodes or len(set(successful_ids)) != args.episodes:
        raise ValueError("collection provenance must contain unique successful scenes")
    scenes_by_id = {scene["scene_id"]: scene for scene in manifest["scenes"]}
    try:
        accepted_scenes = [scenes_by_id[scene_id] for scene_id in successful_ids]
    except KeyError as exc:
        raise ValueError(f"accepted scene is absent from the manifest: {exc.args[0]}") from exc
    distribution_errors = targeted_push_distribution_errors(
        accepted_scenes,
        expected_count=args.episodes,
        # The collector skips irrecoverable expert scenes. Allow at most a 2%
        # difference between the two targeted angle bins while still rejecting
        # a materially one-sided recovery dataset.
        max_angle_imbalance=10,
    )
    if distribution_errors:
        raise ValueError("targeted Push distribution mismatch: " + ", ".join(distribution_errors))

    overlap = {}
    for path in args.evaluation_manifest:
        reference = _json(path)
        report = scene_seed_overlap(manifest["scenes"], reference["scenes"])
        if any(report.values()):
            raise ValueError(f"targeted Push seeds overlap evaluation manifest {path}: {report}")
        overlap[str(path.resolve())] = report

    import pyarrow.parquet as pq

    prompts = set(
        map(
            str,
            pq.read_table(args.dataset_root / "meta" / "tasks.parquet")
            .column("__index_level_0__")
            .to_pylist(),
        )
    )
    if prompts != {PROMPT}:
        raise ValueError(f"targeted Push prompts are invalid: {sorted(prompts)}")
    report = {
        "status": "ok",
        "episodes": args.episodes,
        "frames": int(info["total_frames"]),
        "prompt": PROMPT,
        "manifest_sha256": _sha256(args.manifest),
        "evaluation_seed_overlap": overlap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
