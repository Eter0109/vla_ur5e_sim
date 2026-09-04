from __future__ import annotations

import pytest

from vla_sim.evaluation.metrics import (
    exact_mcnemar_pvalue,
    paired_comparison,
    summarize_results,
    wilson_interval,
)


def test_wilson_interval_and_summary_capture_success_and_failures() -> None:
    lower, upper = wilson_interval(36, 50)
    assert lower == pytest.approx(0.583, abs=0.001)
    assert upper == pytest.approx(0.825, abs=0.001)
    summary = summarize_results(
        [
            {"success": True, "failure_stage": "success"},
            {"success": False, "failure_stage": "no_grasp"},
        ]
    )
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["failure_stages"] == {"no_grasp": 1, "success": 1}
    assert summary["stage_funnel"]["grasp"] == pytest.approx(0.0)
    assert summary["episode_wall_time_s"] is None


def test_paired_comparison_uses_scene_and_policy_seed() -> None:
    candidate = [
        {"scene_id": "a", "policy_seed": 1, "success": True},
        {"scene_id": "b", "policy_seed": 1, "success": False},
    ]
    baseline = [
        {"scene_id": "a", "policy_seed": 1, "success": False},
        {"scene_id": "b", "policy_seed": 1, "success": True},
    ]
    result = paired_comparison(candidate, baseline)
    assert result["candidate_only_success"] == 1
    assert result["baseline_only_success"] == 1
    assert result["mcnemar_exact_pvalue"] == 1.0
    assert exact_mcnemar_pvalue(3, 2) == 1.0
