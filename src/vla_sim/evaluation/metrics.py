"""Statistics and summaries for deterministic rollout benchmark results."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson confidence interval for a binomial rate."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("successes must be within a positive total")
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return center - half, center + half


def exact_mcnemar_pvalue(candidate_only: int, baseline_only: int) -> float:
    """Two-sided exact McNemar p-value for discordant paired outcomes."""
    if candidate_only < 0 or baseline_only < 0:
        raise ValueError("discordant counts must be non-negative")
    total = candidate_only + baseline_only
    if total == 0:
        return 1.0
    lower_tail = sum(math.comb(total, index) for index in range(min(candidate_only, baseline_only) + 1))
    return min(1.0, 2 * lower_tail / 2**total)


def summarize_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(results)
    successes = sum(bool(value.get("success")) for value in values)
    total = len(values)
    lower, upper = wilson_interval(successes, total) if total else (None, None)
    failures = Counter(
        "success"
        if bool(value.get("success"))
        else str(value.get("failure_stage", "legacy_unclassified"))
        for value in values
    )
    def numeric_summary(field: str) -> dict[str, float] | None:
        numbers = [float(value[field]) for value in values if value.get(field) is not None]
        if not numbers:
            return None
        return {
            "mean": sum(numbers) / len(numbers),
            "p50": _percentile(numbers, 50),
            "p95": _percentile(numbers, 95),
            "max": max(numbers),
        }

    def rate(predicate: Any) -> float | None:
        return sum(bool(predicate(value)) for value in values) / total if total else None

    def reached_phase(value: Mapping[str, Any], phase: str) -> bool:
        return phase in value.get("phase_trace", [])

    task_results = {}
    for task in sorted({str(value["task"]) for value in values if value.get("task")}):
        selected = [value for value in values if value.get("task") == task]
        task_results[task] = {
            "episodes": len(selected),
            "successes": sum(bool(value.get("success")) for value in selected),
            "success_rate": sum(bool(value.get("success")) for value in selected) / len(selected),
        }

    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else None,
        "wilson_95": {"lower": lower, "upper": upper},
        "failure_stages": dict(sorted(failures.items())),
        "by_task": task_results,
        "stage_funnel": {
            "approach": rate(lambda value: value.get("approach_success", False)),
            "grasp": rate(lambda value: value.get("ever_grasped", False)),
            "lift": rate(lambda value: reached_phase(value, "transport")),
            "target_reached": rate(lambda value: reached_phase(value, "place")),
            "release": rate(lambda value: reached_phase(value, "verify")),
            "stable_success": successes / total if total else None,
        },
        "gripper_transitions": {
            "system": numeric_summary("gripper_transition_count"),
            "raw_policy": numeric_summary("raw_gripper_transition_count"),
        },
        "latency_s": {
            "episode_p50": numeric_summary("policy_inference_p50_s"),
            "episode_p95": numeric_summary("policy_inference_p95_s"),
        },
        "episode_wall_time_s": numeric_summary("episode_wall_time_s"),
        "peak_vram_mb": numeric_summary("peak_vram_mb"),
    }


def _percentile(values: list[float], percentile: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def paired_comparison(
    candidate: Iterable[Mapping[str, Any]], baseline: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare two result sets keyed by scene ID and policy seed when present."""
    def key(value: Mapping[str, Any]) -> tuple[str, int | None]:
        return str(value["scene_id"]), value.get("policy_seed")

    candidate_by_key = {key(value): value for value in candidate}
    baseline_by_key = {key(value): value for value in baseline}
    common_keys = sorted(candidate_by_key.keys() & baseline_by_key.keys())
    candidate_only = baseline_only = both_success = both_failure = 0
    changed: list[dict[str, Any]] = []
    for result_key in common_keys:
        candidate_success = bool(candidate_by_key[result_key].get("success"))
        baseline_success = bool(baseline_by_key[result_key].get("success"))
        if candidate_success and baseline_success:
            both_success += 1
        elif not candidate_success and not baseline_success:
            both_failure += 1
        elif candidate_success:
            candidate_only += 1
            changed.append({"scene_id": result_key[0], "outcome": "candidate_gain"})
        else:
            baseline_only += 1
            changed.append({"scene_id": result_key[0], "outcome": "candidate_loss"})
    return {
        "paired_episodes": len(common_keys),
        "candidate_only_success": candidate_only,
        "baseline_only_success": baseline_only,
        "both_success": both_success,
        "both_failure": both_failure,
        "absolute_delta": (candidate_only - baseline_only) / len(common_keys) if common_keys else None,
        "mcnemar_exact_pvalue": exact_mcnemar_pvalue(candidate_only, baseline_only),
        "changed_scenes": changed,
    }
