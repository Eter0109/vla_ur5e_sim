"""Verify one checkpoint against the fixed three-task nominal acceptance gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from vla_sim.paths import load_catalog, project_root, resolve_asset

ROOT = project_root()

from vla_sim.simulation.artifacts import sha256_directory, sha256_file

_TASKS = load_catalog("simulation")["tasks"]
MANIFESTS = {
    task: resolve_asset(_TASKS[task]["benchmarks"]["nominal"])
    for task in ("push", "pick_place", "color_pick")
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--push", type=Path, required=True)
    parser.add_argument("--pick", type=Path, required=True)
    parser.add_argument("--color", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_hash = sha256_directory(args.checkpoint)
    push = _json(args.push)
    pick = _json(args.pick)
    pick_meta = _json(args.pick.with_suffix(args.pick.suffix + ".meta.json"))
    color = _json(args.color)
    color_meta = _json(args.color.with_suffix(args.color.suffix + ".meta.json"))
    successes_by_color = Counter(
        str(row["target_color"]) for row in color if bool(row.get("success"))
    )
    totals_by_color = Counter(str(row["target_color"]) for row in color)
    wrong_color_grasps = sum(
        bool(row.get("ever_wrong_object_grasped")) for row in color
    )
    entries = {
        "push": {
            "episodes": int(push["summary"]["episodes"]),
            "successes": int(push["summary"]["successes"]),
        },
        "pick_place": {
            "episodes": len(pick),
            "successes": sum(bool(row.get("success")) for row in pick),
        },
        "color_pick": {
            "episodes": len(color),
            "successes": sum(bool(row.get("success")) for row in color),
            "totals_by_color": dict(sorted(totals_by_color.items())),
            "successes_by_color": dict(sorted(successes_by_color.items())),
            "wrong_color_grasps": wrong_color_grasps,
        },
    }
    errors: list[str] = []
    for name in ("push", "pick_place"):
        entry = entries[name]
        if entry["episodes"] != 50 or entry["successes"] < 45:
            errors.append(
                f"{name}: required >=45/50, got {entry['successes']}/{entry['episodes']}"
            )
    color_entry = entries["color_pick"]
    if color_entry["episodes"] != 60 or color_entry["successes"] < 54:
        errors.append(
            "color_pick: required >=54/60, "
            f"got {color_entry['successes']}/{color_entry['episodes']}"
        )
    for color_name in ("red", "green", "blue"):
        total = totals_by_color[color_name]
        successes = successes_by_color[color_name]
        if total != 20 or successes < 18:
            errors.append(
                f"color_pick/{color_name}: required >=18/20, got {successes}/{total}"
            )
    if wrong_color_grasps:
        errors.append(f"color_pick: required zero wrong-color grasps, got {wrong_color_grasps}")

    identities = {
        "push": (push["checkpoint_sha256"], push["manifest_sha256"]),
        "pick_place": (pick_meta["checkpoint_sha256"], pick_meta["manifest_sha256"]),
        "color_pick": (color_meta["checkpoint_sha256"], color_meta["manifest_sha256"]),
    }
    for name, (actual_checkpoint, actual_manifest) in identities.items():
        if str(actual_checkpoint).lower() != checkpoint_hash.lower():
            errors.append(f"{name}: checkpoint sha256 mismatch")
        if str(actual_manifest).lower() != sha256_file(MANIFESTS[name]).lower():
            errors.append(f"{name}: manifest sha256 mismatch")

    report = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_hash,
        "thresholds": {
            "push": "45/50",
            "pick_place": "45/50",
            "color_pick": "54/60 and 18/20 per color",
            "wrong_color_grasps": 0,
        },
        "entries": entries,
        "errors": errors,
        "passed": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
