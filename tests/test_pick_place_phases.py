import numpy as np

from vla_sim.pick_place_phases import pick_place_phase_group, pick_place_phase_prompt


def _raw(*, cube=(0.0, 0.0, 0.825), eef=(0.0, 0.0, 0.90), target=(0.15, -0.12, 0.802)):
    return {
        "cube_pos": np.asarray(cube),
        "robot0_eef_pos": np.asarray(eef),
        "target_zone_pos": np.asarray(target),
    }


def _action(gripper: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[6] = gripper
    return action


def test_pick_place_phase_labels_cover_collection_funnel() -> None:
    assert pick_place_phase_group(_raw(eef=(0.08, 0.0, 0.92)), _action(-1), {}) == "approach"
    assert pick_place_phase_group(_raw(eef=(0.0, 0.0, 0.84)), _action(-1), {}) == "grasp"
    assert pick_place_phase_group(_raw(eef=(0.0, 0.0, 0.84)), _action(1), {}) == "grasp"
    assert pick_place_phase_group(_raw(), _action(1), {"grasped": True}) == "lift"
    assert pick_place_phase_group(
        _raw(cube=(0.04, 0.0, 0.94)), _action(1), {"ever_lifted": True}
    ) == "transport"
    assert pick_place_phase_group(
        _raw(cube=(0.145, -0.118, 0.86)), _action(1), {"ever_lifted": True}
    ) == "place_release"


def test_pick_place_phase_prompt_is_not_the_global_prompt() -> None:
    prompt = pick_place_phase_prompt(_raw(eef=(0.08, 0.0, 0.92)), _action(-1), {})
    assert prompt == "move above the red cube"
    assert prompt != "place the red cube in the blue storage bin"
