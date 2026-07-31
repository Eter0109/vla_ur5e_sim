import numpy as np

from vla_sim.sim.expert import HeuristicExpertConfig, HeuristicStackExpert, StackPhase


def observation(cube_a=(0.0, 0.0, 0.82), cube_b=(0.1, 0.0, 0.82), eef=(0.0, 0.0, 0.83)):
    return {
        "cubeA_pos": np.asarray(cube_a),
        "cubeB_pos": np.asarray(cube_b),
        "robot0_eef_pos": np.asarray(eef),
    }


def test_expert_retries_when_close_geometry_does_not_confirm_grasp():
    expert = HeuristicStackExpert(HeuristicExpertConfig(close_steps=1))
    expert.phase = StackPhase.CLOSE
    expert.act(observation(eef=(0.2, 0.2, 1.0)))
    assert expert.phase is StackPhase.RETRY


def test_expert_phase_prompts_fit_training_phase_groups():
    expert = HeuristicStackExpert()
    prompts = set()
    for phase in StackPhase:
        expert.phase = phase
        prompts.add(expert.prompt)
    assert "move above the grasp object" in prompts
    assert "hold position for release" in prompts
