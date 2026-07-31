import numpy as np
import pytest

from vla_sim.stack_control import (
    ObjectPoseEstimate,
    StackPhase,
    StackSupervisor,
    StackSupervisorConfig,
)


def estimate(pick=(0.0, 0.0, 0.82), target=(0.1, 0.0, 0.82), confidence=0.9):
    return ObjectPoseEstimate(np.asarray(pick), np.asarray(target), confidence)


# ---------------------------------------------------------------------------
# Existing tests (updated where new phase changes behaviour)
# ---------------------------------------------------------------------------


def test_unreliable_pose_freezes_xyz_and_opens_gripper():
    supervisor = StackSupervisor()
    action = supervisor.filter_action(
        np.ones(7), eef_xyz=np.asarray([0.0, 0.0, 0.9]), estimate=None, gripper_opening=0.0
    )
    assert np.array_equal(action[:6], np.zeros(6))
    assert action[6] == -1


def test_task_and_phase_prompt_tracks_supervisor_phase():
    supervisor = StackSupervisor(task="blue_on_red")
    assert supervisor.prompt == "move above blue block"
    supervisor.phase = StackPhase.GRASP
    assert supervisor.prompt == "move down and grasp blue block"
    supervisor.phase = StackPhase.LIFT
    assert supervisor.prompt == "lift blue block"
    supervisor.phase = StackPhase.PLACE
    assert supervisor.prompt == "place blue block on red and release"


def test_invalid_task_is_rejected():
    with pytest.raises(ValueError, match="task"):
        StackSupervisor(task="green_on_red")


def test_object_pose_allows_one_visible_object():
    pose = ObjectPoseEstimate(None, np.asarray([0.1, 0.0, 0.82]), 0.0, 0.9)
    assert pose.pick_xyz is None
    assert pose.target_confidence == pytest.approx(0.9)


def test_gripper_requires_distance_and_two_confident_frames():
    supervisor = StackSupervisor()
    close = np.zeros(7)
    close[6] = 1
    pose = estimate()
    # First confident frame at grasp distance -> phase GRASP
    action = supervisor.filter_action(
        close, eef_xyz=np.asarray([0.0, 0.0, 0.83]), estimate=pose, gripper_opening=1.0
    )
    assert supervisor.phase is StackPhase.GRASP
    assert action[6] == -1
    # Second confident frame -> phase GRIPPER_CLOSE (dwell + confirmation)
    action = supervisor.filter_action(
        close, eef_xyz=np.asarray([0.0, 0.0, 0.83]), estimate=pose, gripper_opening=1.0
    )
    assert supervisor.phase is StackPhase.GRIPPER_CLOSE
    assert action[6] == 1  # gripper closed
    assert np.array_equal(action[:3], np.zeros(3))  # XYZ frozen


def test_calibrated_grasp_gate_accepts_observed_approach_band():
    supervisor = StackSupervisor(StackSupervisorConfig(confidence_frames=1))
    pose = estimate()
    supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.847]),
        estimate=pose,
        gripper_opening=1.0,
    )
    assert supervisor.phase is StackPhase.GRIPPER_CLOSE


def test_calibrated_grasp_gate_rejects_beyond_thirty_mm():
    supervisor = StackSupervisor(StackSupervisorConfig(confidence_frames=1))
    pose = estimate()
    supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.851]),
        estimate=pose,
        gripper_opening=1.0,
    )
    assert supervisor.phase is StackPhase.GRASP


def test_gripper_close_survives_pick_object_occlusion():
    config = StackSupervisorConfig(confidence_frames=1, gripper_dwell_steps=3)
    supervisor = StackSupervisor(config)
    supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=estimate(),
        gripper_opening=0.5,
    )
    action = supervisor.filter_action(
        np.ones(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=None,
        gripper_opening=0.5,
    )
    assert supervisor.phase is StackPhase.GRIPPER_CLOSE
    assert supervisor.last_perception_reliable
    assert np.array_equal(action[:3], np.zeros(3))


def test_supervisor_preserves_xyz_but_zeros_rotation_and_bounds_workspace():
    supervisor = StackSupervisor()
    action = supervisor.filter_action(
        np.asarray([1.0, -0.5, 0.25, 0.8, -0.8, 0.4, 1.0]),
        eef_xyz=np.asarray([0.24, 0.0, 0.9]),
        estimate=estimate(pick=(0.2, 0.1, 0.82)),
        gripper_opening=1.0,
    )
    assert np.allclose(action[:3], [0.2, -0.5, 0.25])
    assert np.array_equal(action[3:6], np.zeros(3))


def test_target_pose_ema_rejects_single_frame_drift():
    config = StackSupervisorConfig(target_pose_ema_alpha=0.2)
    supervisor = StackSupervisor(config)
    supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([-0.2, 0.0, 0.9]),
        estimate=estimate(target=(0.1, 0.0, 0.82)),
        gripper_opening=1.0,
    )
    supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([-0.2, 0.0, 0.9]),
        estimate=estimate(target=(0.2, 0.0, 0.82)),
        gripper_opening=1.0,
    )
    assert supervisor._last_target_xyz is not None
    assert supervisor._last_target_xyz[0] == pytest.approx(0.12)


def test_place_reacquires_transport_after_xy_drift():
    supervisor = StackSupervisor()
    supervisor.phase = StackPhase.PLACE
    supervisor._attachment_confirmed = True
    supervisor._eef_to_pick_offset = np.zeros(3)
    action = supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([0.05, 0.0, 0.9]),
        estimate=estimate(target=(0.0, 0.0, 0.82)),
        gripper_opening=0.5,
    )
    assert supervisor.phase is StackPhase.TRANSPORT
    assert action[6] == supervisor.config.close_command


def test_transport_visual_servo_overrides_biased_vla_xy_action():
    config = StackSupervisorConfig(
        transport_visual_servo_gain=1.0,
        transport_visual_servo_max_action=0.8,
    )
    supervisor = StackSupervisor(config)
    supervisor.phase = StackPhase.TRANSPORT
    supervisor._attachment_confirmed = True
    supervisor._eef_to_pick_offset = np.zeros(3)
    action = supervisor.filter_action(
        np.asarray([-0.4, 0.4, 0.1, 0.0, 0.0, 0.0, 1.0]),
        eef_xyz=np.asarray([0.0, 0.0, 0.9]),
        estimate=estimate(pick=(0.0, 0.0, 0.82), target=(0.1, -0.05, 0.82)),
        gripper_opening=0.5,
    )
    assert action[:3] == pytest.approx([0.8, -0.8, 0.1])
    assert action[6] == supervisor.config.close_command


@pytest.mark.parametrize("phase", [StackPhase.APPROACH, StackPhase.GRASP])
def test_pick_visual_servo_centers_approach_and_grasp(phase):
    config = StackSupervisorConfig(
        pick_visual_servo_gain=1.0,
        pick_visual_servo_max_action=0.8,
    )
    supervisor = StackSupervisor(config)
    supervisor.phase = phase
    action = supervisor.filter_action(
        np.asarray([-0.4, 0.4, -0.2, 0.0, 0.0, 0.0, -1.0]),
        eef_xyz=np.asarray([0.1, -0.1, 0.9]),
        estimate=estimate(pick=(0.0, 0.0, 0.82)),
        gripper_opening=0.5,
    )
    assert action[:3] == pytest.approx([-0.8, 0.8, -0.2])


def test_object_pose_validation_rejects_bad_confidence():
    try:
        estimate(confidence=1.1)
    except ValueError as error:
        assert "confidence" in str(error)
    else:
        raise AssertionError("Expected invalid confidence to fail")


def test_unreliable_pose_retries_twice_then_fails_safely():
    supervisor = StackSupervisor(StackSupervisorConfig(unreliable_retry_steps=1))
    for expected_retries in (1, 2):
        action = supervisor.filter_action(
            np.ones(7),
            eef_xyz=np.asarray([0.0, 0.0, 0.9]),
            estimate=None,
            gripper_opening=0.0,
        )
        assert supervisor.retries == expected_retries
        assert supervisor.phase is StackPhase.APPROACH
        assert np.array_equal(action[:6], np.zeros(6))
    supervisor.filter_action(
        np.ones(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.9]),
        estimate=None,
        gripper_opening=0.0,
    )
    assert supervisor.phase is StackPhase.FAILED


# ---------------------------------------------------------------------------
# New tests: dwell, grasp confirmation, retry, timeouts
# ---------------------------------------------------------------------------


def test_gripper_close_dwell_waits_required_steps():
    """Phase stays GRIPPER_CLOSE until dwell is complete."""
    config = StackSupervisorConfig(gripper_dwell_steps=3, confidence_frames=1)
    supervisor = StackSupervisor(config)
    pose = estimate()
    # Enter GRIPPER_CLOSE with one confident frame
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=1.0,
    )
    assert supervisor.phase is StackPhase.GRIPPER_CLOSE
    # Dwell steps 0,1,2 — still in GRIPPER_CLOSE
    for _ in range(2):
        action = supervisor.filter_action(
            np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
            estimate=pose, gripper_opening=1.0,
        )
        assert supervisor.phase is StackPhase.GRIPPER_CLOSE
        assert np.array_equal(action[:3], np.zeros(3))
    # Third step (dwell complete) -> LIFT (gripper_opening=1.0 >= threshold)
    action = supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=1.0,
    )
    assert supervisor.phase is StackPhase.LIFT
    assert action[6] == 1  # stays closed


def test_grasp_confirmation_success_transitions_to_lift():
    """When gripper_opening >= threshold after dwell, transition to LIFT."""
    config = StackSupervisorConfig(gripper_dwell_steps=1, confidence_frames=1)
    supervisor = StackSupervisor(config)
    pose = estimate()
    # Enter GRIPPER_CLOSE
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.5,
    )
    assert supervisor.phase is StackPhase.GRIPPER_CLOSE
    # Dwell complete, opening=0.3 >= 0.15 -> LIFT
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.3,
    )
    assert supervisor.phase is StackPhase.LIFT


def test_grasp_confirmation_failure_triggers_retry():
    """When gripper_opening < threshold after dwell, retry to APPROACH."""
    config = StackSupervisorConfig(gripper_dwell_steps=1, confidence_frames=1)
    supervisor = StackSupervisor(config)
    pose = estimate()
    # Enter GRIPPER_CLOSE
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.5,
    )
    assert supervisor.phase is StackPhase.GRIPPER_CLOSE
    # Dwell complete, opening=0.0 < 0.15 -> retry to APPROACH
    action = supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.0,
    )
    assert supervisor.phase is StackPhase.APPROACH
    assert supervisor._grasp_retries == 1
    assert action[6] == -1  # gripper open for re-approach


def test_lift_requires_visual_attachment_motion():
    config = StackSupervisorConfig(
        gripper_dwell_steps=1,
        confidence_frames=1,
        attachment_confirmation_lift_m=0.01,
    )
    supervisor = StackSupervisor(config)
    supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=estimate(),
        gripper_opening=1.0,
    )
    supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=estimate(),
        gripper_opening=1.0,
    )
    assert supervisor.phase is StackPhase.LIFT
    assert not supervisor._attachment_confirmed
    supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.845]),
        estimate=estimate(pick=(0.0, 0.0, 0.835)),
        gripper_opening=1.0,
    )
    assert supervisor._attachment_confirmed
    assert supervisor.phase is StackPhase.LIFT


def test_attachment_window_can_be_extended_for_slow_visual_lift():
    config = StackSupervisorConfig(
        confidence_frames=1,
        gripper_dwell_steps=0,
        attachment_confirmation_steps=90,
    )
    supervisor = StackSupervisor(config)
    supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=estimate(),
        gripper_opening=1.0,
    )
    assert supervisor.phase is StackPhase.LIFT
    for _ in range(60):
        supervisor.filter_action(
            np.zeros(7),
            eef_xyz=np.asarray([0.0, 0.0, 0.83]),
            estimate=estimate(),
            gripper_opening=1.0,
        )
    assert supervisor.phase is StackPhase.LIFT
    assert supervisor._grasp_retries == 0


def test_release_waits_for_open_gripper_state():
    supervisor = StackSupervisor()
    supervisor.phase = StackPhase.RELEASE
    supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.9]),
        estimate=estimate(),
        gripper_opening=1.0,
    )
    assert supervisor.phase is StackPhase.RELEASE
    supervisor.filter_action(
        np.zeros(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.9]),
        estimate=estimate(),
        gripper_opening=0.2,
    )
    assert supervisor.phase is StackPhase.VERIFY


def test_grasp_retry_exhaustion_causes_failure():
    """After grasp_retry_max retries, phase becomes FAILED."""
    config = StackSupervisorConfig(
        gripper_dwell_steps=1, confidence_frames=1, grasp_retry_max=2,
    )
    supervisor = StackSupervisor(config)
    pose = estimate()
    # First retry cycle: approach from far, descend, close, fail
    for attempt in range(2):
        # Enter GRASP from APPROACH (use close EEF so both APPROACH→GRASP
        # and GRASP→GRIPPER_CLOSE fire in this call with confidence_frames=1)
        supervisor.filter_action(
            np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
            estimate=pose, gripper_opening=0.5,
        )
        assert supervisor.phase is StackPhase.GRIPPER_CLOSE
        # Fail confirmation -> retry to APPROACH
        supervisor.filter_action(
            np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
            estimate=pose, gripper_opening=0.0,
        )
        assert supervisor.phase is StackPhase.APPROACH
    # Third attempt enters GRIPPER_CLOSE then fails -> FAILED
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.5,
    )
    assert supervisor.phase is StackPhase.GRIPPER_CLOSE
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.0,
    )
    assert supervisor.phase is StackPhase.FAILED


def test_grasp_retry_resets_initial_pick_z():
    """After retry to APPROACH, _initial_pick_z is re-captured."""
    config = StackSupervisorConfig(gripper_dwell_steps=1, confidence_frames=1)
    supervisor = StackSupervisor(config)
    # Provide a pick at z=0.82 to seed _initial_pick_z
    pose1 = estimate(pick=(0.0, 0.0, 0.82))
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose1, gripper_opening=0.5,
    )
    assert supervisor._initial_pick_z == 0.82
    # Fail confirmation -> retry
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose1, gripper_opening=0.0,
    )
    assert supervisor.phase is StackPhase.APPROACH
    assert supervisor._initial_pick_z is None
    # New estimate with different pick z
    pose2 = estimate(pick=(0.0, 0.0, 0.84))
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.85]),
        estimate=pose2, gripper_opening=0.5,
    )
    assert supervisor._initial_pick_z == 0.84


def test_xyz_frozen_during_gripper_close():
    """VLA XYZ action is zeroed during GRIPPER_CLOSE phase."""
    config = StackSupervisorConfig(confidence_frames=1)
    supervisor = StackSupervisor(config)
    # Enter GRIPPER_CLOSE
    supervisor.filter_action(
        np.ones(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=estimate(), gripper_opening=1.0,
    )
    assert supervisor.phase is StackPhase.GRIPPER_CLOSE
    action = supervisor.filter_action(
        np.asarray([0.5, -0.3, 0.2, 0.0, 0.0, 0.0, 1.0]),
        eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=estimate(), gripper_opening=0.5,
    )
    assert np.array_equal(action[:3], np.zeros(3))


def test_gripper_close_command_during_gripper_close():
    """Gripper is commanded closed during GRIPPER_CLOSE."""
    config = StackSupervisorConfig(confidence_frames=1)
    supervisor = StackSupervisor(config)
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=estimate(), gripper_opening=1.0,
    )
    assert supervisor.phase is StackPhase.GRIPPER_CLOSE
    action = supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=estimate(), gripper_opening=0.5,
    )
    assert action[6] == 1  # close_command


def test_phase_timeout_triggers_failure():
    """After phase_timeout_steps without transition, phase becomes FAILED.

    _phase_step is checked against timeout before incrementing each call,
    so timeout=3 fires on the 4th call (when _phase_step reaches 3).
    """
    config = StackSupervisorConfig(phase_timeout_steps=3, grasp_retry_max=0)
    supervisor = StackSupervisor(config)
    pose = estimate()
    # Stay in APPROACH without meeting transition condition
    for _ in range(4):
        supervisor.filter_action(
            np.zeros(7), eef_xyz=np.asarray([0.5, 0.0, 0.9]),
            estimate=pose, gripper_opening=1.0,
        )
    assert supervisor.phase is StackPhase.FAILED


def test_per_phase_timeout_overrides_default():
    """phase_timeouts dict overrides phase_timeout_steps per phase.

    timeout=2 fires on the 3rd call (_phase_step reaches 2).
    """
    config = StackSupervisorConfig(
        phase_timeout_steps=60,
        phase_timeouts={"approach": 2},
        grasp_retry_max=0,
    )
    supervisor = StackSupervisor(config)
    pose = estimate()
    # APPROACH times out after 3 calls (timeout=2, _phase_step reaches 2)
    for _ in range(3):
        supervisor.filter_action(
            np.zeros(7), eef_xyz=np.asarray([0.5, 0.0, 0.9]),
            estimate=pose, gripper_opening=1.0,
        )
    assert supervisor.phase is StackPhase.FAILED


def test_approach_timeout_retries_before_failure():
    config = StackSupervisorConfig(phase_timeout_steps=2, grasp_retry_max=1)
    supervisor = StackSupervisor(config)
    pose = estimate()
    for _ in range(3):
        supervisor.filter_action(
            np.zeros(7),
            eef_xyz=np.asarray([0.5, 0.0, 0.9]),
            estimate=pose,
            gripper_opening=1.0,
        )
    assert supervisor.phase is StackPhase.APPROACH
    assert supervisor._grasp_retries == 1
    assert supervisor.retry_epoch == 1


def test_failed_phase_freezes_xyz_and_opens_gripper():
    supervisor = StackSupervisor()
    supervisor.phase = StackPhase.FAILED
    action = supervisor.filter_action(
        np.ones(7),
        eef_xyz=np.asarray([0.0, 0.0, 0.9]),
        estimate=estimate(),
        gripper_opening=1.0,
    )
    assert np.array_equal(action[:6], np.zeros(6))
    assert action[6] == -1


def test_grasp_retry_independent_of_perception_retry():
    """Grasp retry counter and perception retry counter don't interfere."""
    config = StackSupervisorConfig(
        gripper_dwell_steps=1, confidence_frames=1, grasp_retry_max=1,
    )
    supervisor = StackSupervisor(config)
    # Exhaust grasp retries
    pose = estimate()
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.5,
    )
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.0,
    )
    # Second attempt enters close then fails
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.5,
    )
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.0,
    )
    assert supervisor.phase is StackPhase.FAILED
    # Perception retries should still be 0
    assert supervisor.retries == 0
    assert supervisor._grasp_retries == 1


def test_full_grasp_retry_cycle():
    """End-to-end: APPROACH->GRASP->CLOSE->(fail)->APPROACH->GRASP->CLOSE->(succeed)->LIFT."""
    config = StackSupervisorConfig(
        confidence_frames=1, gripper_dwell_steps=1, grasp_retry_max=2,
    )
    supervisor = StackSupervisor(config)
    pose = estimate()
    # --- First attempt: approach then fail grasp ---
    # Enter GRASP then GRIPPER_CLOSE (both fire with confidence_frames=1
    # and a close EEF)
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.5,
    )
    assert supervisor.phase is StackPhase.GRIPPER_CLOSE
    # Dwell complete, fail confirmation -> retry
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.0,
    )
    assert supervisor.phase is StackPhase.APPROACH
    assert supervisor._grasp_retries == 1

    # --- Second attempt: approach from far, then close, succeed ---
    # Step back far to only enter GRASP
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.90]),
        estimate=pose, gripper_opening=0.5,
    )
    assert supervisor.phase is StackPhase.GRASP
    # Move close to enter GRIPPER_CLOSE
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.5,
    )
    assert supervisor.phase is StackPhase.GRIPPER_CLOSE
    # Dwell complete, confirm success
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.3,
    )
    assert supervisor.phase is StackPhase.LIFT
    assert supervisor._grasp_retries == 1  # counter stays at 1


def test_grasp_confirmation_with_zero_dwell():
    """With dwell_steps=0, dwell check fires in the same call that enters GRIPPER_CLOSE."""
    config = StackSupervisorConfig(
        confidence_frames=1, gripper_dwell_steps=0,
    )
    supervisor = StackSupervisor(config)
    pose = estimate()
    # With confidence_frames=1, dwell=0, and a close EEF, a single call
    # fires APPROACH→GRASP→GRIPPER_CLOSE and immediately (dwell=0) checks
    # confirmation. Since gripper_opening=0.5 >= threshold, goes to LIFT.
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.5,
    )
    assert supervisor.phase is StackPhase.LIFT


def test_grasp_confirmation_respects_threshold_boundary():
    """Confirmation uses >= threshold for success, < threshold for retry."""
    config = StackSupervisorConfig(
        confidence_frames=1, gripper_dwell_steps=1,
        grasp_confirmation_threshold=0.2,
    )
    pose = estimate()

    # --- At threshold: success ---
    supervisor = StackSupervisor(config)
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.5,
    )
    supervisor.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.2,
    )
    assert supervisor.phase is StackPhase.LIFT

    # --- Below threshold: retry ---
    supervisor2 = StackSupervisor(config)
    supervisor2.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.5,
    )
    supervisor2.filter_action(
        np.zeros(7), eef_xyz=np.asarray([0.0, 0.0, 0.83]),
        estimate=pose, gripper_opening=0.19,
    )
    assert supervisor2.phase is StackPhase.APPROACH


def test_perception_retry_does_not_affect_grasp_counter():
    """Perception exhaustion leaves _grasp_retries at 0."""
    config = StackSupervisorConfig(
        max_retries=1, unreliable_retry_steps=1,
    )
    supervisor = StackSupervisor(config)
    # Exhaust perception retries
    supervisor.filter_action(
        np.ones(7), eef_xyz=np.asarray([0.0, 0.0, 0.9]),
        estimate=None, gripper_opening=0.0,
    )
    assert supervisor.retries == 1
    supervisor.filter_action(
        np.ones(7), eef_xyz=np.asarray([0.0, 0.0, 0.9]),
        estimate=None, gripper_opening=0.0,
    )
    assert supervisor.phase is StackPhase.FAILED
    # Grasp retries untouched
    assert supervisor._grasp_retries == 0


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


def test_config_rejects_negative_dwell():
    with pytest.raises(ValueError, match="gripper_dwell_steps"):
        StackSupervisorConfig(gripper_dwell_steps=-1)


def test_config_rejects_invalid_place_hysteresis():
    with pytest.raises(ValueError, match="place_reacquire_xy_m"):
        StackSupervisorConfig(place_reacquire_xy_m=0.025)


def test_config_rejects_invalid_target_pose_ema_alpha():
    with pytest.raises(ValueError, match="target_pose_ema_alpha"):
        StackSupervisorConfig(target_pose_ema_alpha=0.0)


def test_config_rejects_out_of_range_confirmation_threshold():
    with pytest.raises(ValueError, match="grasp_confirmation_threshold"):
        StackSupervisorConfig(grasp_confirmation_threshold=1.1)
    with pytest.raises(ValueError, match="grasp_confirmation_threshold"):
        StackSupervisorConfig(grasp_confirmation_threshold=0.0)


def test_config_rejects_negative_grasp_retry_max():
    with pytest.raises(ValueError, match="grasp_retry_max"):
        StackSupervisorConfig(grasp_retry_max=-1)


def test_config_rejects_zero_phase_timeout():
    with pytest.raises(ValueError, match="phase_timeout_steps"):
        StackSupervisorConfig(phase_timeout_steps=0)


def test_config_rejects_invalid_per_phase_timeout():
    with pytest.raises(ValueError, match="phase_timeouts"):
        StackSupervisorConfig(phase_timeouts={"approach": 0})
