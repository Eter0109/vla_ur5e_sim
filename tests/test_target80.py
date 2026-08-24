from __future__ import annotations

from pathlib import Path

import pytest

from vla_sim.target80 import TARGET80_SPLITS, key_screen_promoted, target80_gate_report


ROOT = Path(__file__).resolve().parents[1]


def _entries(successes: int = 40) -> dict:
    return {
        split: {
            "episodes": 50,
            "successes": successes,
            "checkpoint_sha256": "model",
            "manifest_sha256": f"manifest-{split}",
        }
        for split in TARGET80_SPLITS
    }


def _manifests() -> dict:
    return {split: f"manifest-{split}" for split in TARGET80_SPLITS}


def test_target80_gate_requires_all_four_full_splits_from_one_checkpoint() -> None:
    report = target80_gate_report(
        _entries(), checkpoint_sha256="model", manifest_sha256=_manifests()
    )

    assert report["passed"] is True
    assert report["required_successes"] == 40


def test_target80_gate_reports_rate_checkpoint_manifest_and_episode_failures() -> None:
    entries = _entries()
    entries["push_randomized"]["successes"] = 39
    entries["pick_nominal"]["checkpoint_sha256"] = "other"
    entries["pick_randomized"]["manifest_sha256"] = "other"
    entries["push_nominal"]["episodes"] = 20

    report = target80_gate_report(
        entries, checkpoint_sha256="model", manifest_sha256=_manifests()
    )

    assert report["passed"] is False
    assert any("push_randomized: successes=39/50" in error for error in report["errors"])
    assert any("pick_nominal: checkpoint_sha256 mismatch" in error for error in report["errors"])
    assert any("pick_randomized: manifest_sha256 mismatch" in error for error in report["errors"])
    assert any("push_nominal: episodes=20 expected=50" in error for error in report["errors"])


def test_target80_gate_rejects_missing_split() -> None:
    entries = _entries()
    del entries["pick_randomized"]

    with pytest.raises(ValueError, match="exactly the four"):
        target80_gate_report(
            entries, checkpoint_sha256="model", manifest_sha256=_manifests()
        )


def test_blind_launcher_is_locked_by_strict_target80_gate_and_best_controls() -> None:
    source = (ROOT / "scripts" / "run_multitask_robust_blind_evaluation.ps1").read_text(
        encoding="utf-8"
    )

    assert "verify_target80_development.py" in source
    assert "--temporal-decay 0.75" in source
    assert "--samples-per-plan 2" in source
    assert "--closed-negative-y-gain 1.8" in source
    assert "$Summary.push -ge 0.80" in source
    assert "$Summary.pick_place -ge 0.80" in source


def test_full_development_launcher_uses_target80_controls_and_verifier() -> None:
    source = (
        ROOT / "scripts" / "run_multitask_robust_development_evaluation.ps1"
    ).read_text(encoding="utf-8")

    assert "verify_target80_development.py" in source
    assert '"--temporal-decay", "0.75"' in source
    assert '"--samples-per-plan", "2"' in source
    assert '"--closed-negative-y-gain", "1.8"' in source


def test_key_checkpoint_screen_targets_both_current_bottlenecks() -> None:
    source = (ROOT / "scripts" / "run_target80_key_checkpoint_screen.ps1").read_text(
        encoding="utf-8"
    )

    assert "push_robust_development_randomized_v1_screen20.json" in source
    assert "pick_place_robust_development_nominal_v1_screen20.json" in source
    assert "key_screening_only_not_final_evidence" in source
    assert "target80_screen_reference_lazy_bs2_final.json" in source
    assert "$PushReference" in source
    assert "$PickReference" in source
    assert "foreach ($Row in $Pick)" in source


def test_screen_and_development_launchers_enumerate_pick_json_arrays_on_windows() -> None:
    screen = (ROOT / "scripts" / "run_target80_checkpoint_screen.ps1").read_text(
        encoding="utf-8"
    )
    development = (
        ROOT / "scripts" / "run_multitask_robust_development_evaluation.ps1"
    ).read_text(encoding="utf-8")

    assert "foreach ($Row in $PickNominal)" in screen
    assert "foreach ($Row in $PickRandomized)" in screen
    assert "foreach ($Row in $Rows)" in development


def test_joint_training_requires_a_successful_targeted_dataset_audit() -> None:
    source = (
        ROOT / "scripts" / "train_multitask_target80_joint_recovery.ps1"
    ).read_text(encoding="utf-8")

    assert "audit_targeted_push_recovery_v2_500.json" in source
    assert '$Audit.status -ne "ok"' in source
    assert "[int]$Audit.episodes -ne 500" in source
    assert "$Audit.manifest_sha256" in source


@pytest.mark.parametrize(
    ("push", "pick", "expected"),
    [(15, 13, True), (14, 14, True), (15, 12, False), (13, 14, False), (14, 13, False)],
)
def test_key_screen_promotion_requires_improvement_without_regression(
    push: int, pick: int, expected: bool
) -> None:
    assert key_screen_promoted(push_successes=push, pick_successes=pick) is expected
