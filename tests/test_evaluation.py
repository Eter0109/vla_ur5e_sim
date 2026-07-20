from __future__ import annotations

import pytest

from vla_sim.evaluation import (
    exact_mcnemar_pvalue,
    paired_comparison,
    seed_promotion_gate,
    select_ablation_winner,
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


def test_ablation_tie_break_prefers_no_oversampling_then_lower_lr() -> None:
    summaries = {"A": {"successes": 30}, "B": {"successes": 31}, "C": {"successes": 30}, "D": {"successes": 29}}
    specs = {
        "A": {"peak_lr": 2e-5, "transition_oversample_factor": 1},
        "B": {"peak_lr": 2e-5, "transition_oversample_factor": 3},
        "C": {"peak_lr": 5e-5, "transition_oversample_factor": 1},
        "D": {"peak_lr": 5e-5, "transition_oversample_factor": 3},
    }
    assert select_ablation_winner(summaries, specs)["winner"] == "A"


def test_seed_promotion_gate_requires_all_three_thresholds() -> None:
    passed = seed_promotion_gate(
        [{"success_rate": 0.70}, {"success_rate": 0.75}, {"success_rate": 0.80}]
    )
    assert passed["passed"] is True
    failed = seed_promotion_gate(
        [{"success_rate": 0.59}, {"success_rate": 0.75}, {"success_rate": 0.80}]
    )
    assert failed["passed"] is False
