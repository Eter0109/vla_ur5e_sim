"""Strict one-checkpoint gate for the four target-80 development capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TARGET80_SPLITS = (
    "push_nominal",
    "push_randomized",
    "pick_nominal",
    "pick_randomized",
)


def key_screen_promoted(
    *,
    push_successes: int,
    pick_successes: int,
    push_reference: int = 14,
    pick_reference: int = 13,
) -> bool:
    """Promote only if one bottleneck improves and the other does not regress."""

    values = (push_successes, pick_successes, push_reference, pick_reference)
    if any(not 0 <= value <= 20 for value in values):
        raise ValueError("key-screen successes must be in [0, 20]")
    return (
        push_successes >= push_reference
        and pick_successes >= pick_reference
        and (push_successes > push_reference or pick_successes > pick_reference)
    )


def target80_gate_report(
    entries: Mapping[str, Mapping[str, Any]],
    *,
    checkpoint_sha256: str,
    manifest_sha256: Mapping[str, str],
    episodes: int = 50,
    threshold: float = 0.80,
) -> dict[str, Any]:
    """Return a strict report; every split must use one model and full manifest."""

    if set(entries) != set(TARGET80_SPLITS):
        raise ValueError("target-80 entries must contain exactly the four required splits")
    if set(manifest_sha256) != set(TARGET80_SPLITS):
        raise ValueError("target-80 manifest hashes must contain exactly the four splits")
    if episodes < 1 or not 0 < threshold <= 1:
        raise ValueError("episodes and threshold are invalid")
    required_successes = int(episodes * threshold + 0.999999999)
    errors: list[str] = []
    normalized: dict[str, dict[str, Any]] = {}
    for split in TARGET80_SPLITS:
        entry = entries[split]
        actual_episodes = int(entry["episodes"])
        successes = int(entry["successes"])
        actual_checkpoint = str(entry.get("checkpoint_sha256", "")).lower()
        actual_manifest = str(entry.get("manifest_sha256", "")).lower()
        if actual_episodes != episodes:
            errors.append(f"{split}: episodes={actual_episodes} expected={episodes}")
        if successes < required_successes:
            errors.append(
                f"{split}: successes={successes}/{actual_episodes} required={required_successes}/{episodes}"
            )
        if actual_checkpoint != checkpoint_sha256.lower():
            errors.append(f"{split}: checkpoint_sha256 mismatch")
        if actual_manifest != manifest_sha256[split].lower():
            errors.append(f"{split}: manifest_sha256 mismatch")
        normalized[split] = {
            "episodes": actual_episodes,
            "successes": successes,
            "success_rate": successes / actual_episodes if actual_episodes else None,
            "checkpoint_sha256": actual_checkpoint,
            "manifest_sha256": actual_manifest,
        }
    return {
        "schema_version": 1,
        "target": threshold,
        "required_successes": required_successes,
        "checkpoint_sha256": checkpoint_sha256.lower(),
        "splits": normalized,
        "errors": errors,
        "passed": not errors,
    }
