import numpy as np

from vla_sim.sim.expert import (
    HeuristicExpertConfig,
    HeuristicPickPlaceExpert,
    PickPlacePhase,
)
from vla_sim.stack_control import task_phase_prompt


def test_pick_place_expert_uses_shared_phase_prompts() -> None:
    expert = HeuristicPickPlaceExpert()
    expected_groups = {
        PickPlacePhase.APPROACH: "approach",
        PickPlacePhase.DESCEND: "grasp",
        PickPlacePhase.CLOSE: "grasp",
        PickPlacePhase.LIFT: "lift",
        PickPlacePhase.TRANSPORT: "transport",
        PickPlacePhase.PLACE: "place_release",
        PickPlacePhase.OPEN: "place_release",
        PickPlacePhase.DONE: "failed",
    }
    for phase, group in expected_groups.items():
        expert.phase = phase
        assert expert.prompt == task_phase_prompt("red_to_storage_bin", group)


def test_pick_place_expert_can_use_strict_xy_and_reachable_z_release_tolerances() -> None:
    expert = HeuristicPickPlaceExpert(
        HeuristicExpertConfig(
            place_release_tolerance_m=0.020,
            place_release_xy_tolerance_m=0.008,
        )
    )
    expert.phase = PickPlacePhase.PLACE
    observation = {
        "cube_pos": np.asarray([0.0, 0.0, 0.825]),
        "target_zone_pos": np.asarray([0.150, -0.120, 0.802]),
        "robot0_eef_pos": np.asarray([0.156, -0.120, 0.838]),
    }
    expert.act(observation)
    assert expert.phase is PickPlacePhase.OPEN


def test_pick_place_expert_strict_xy_does_not_release_when_laterally_misaligned() -> None:
    expert = HeuristicPickPlaceExpert(
        HeuristicExpertConfig(
            place_release_tolerance_m=0.020,
            place_release_xy_tolerance_m=0.008,
        )
    )
    expert.phase = PickPlacePhase.PLACE
    observation = {
        "cube_pos": np.asarray([0.0, 0.0, 0.825]),
        "target_zone_pos": np.asarray([0.150, -0.120, 0.802]),
        "robot0_eef_pos": np.asarray([0.160, -0.120, 0.838]),
    }
    expert.act(observation)
    assert expert.phase is PickPlacePhase.PLACE
