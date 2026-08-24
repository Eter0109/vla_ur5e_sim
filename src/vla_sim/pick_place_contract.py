"""Canonical dataset and runtime contract for native-bin PickPlace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vla_sim.envs import UR5ePickPlaceConfig
from vla_sim.stack_control import task_phase_prompt

PICK_PLACE_ENVIRONMENT_PRESET = "pick_place_v1"
PICK_PLACE_TASK = "red_to_storage_bin"
PICK_PLACE_GLOBAL_PROMPT = "place the red cube in the blue storage bin"
PICK_PLACE_FPS = 10
PICK_PLACE_PROMPT_GROUPS = (
    "approach",
    "grasp",
    "lift",
    "transport",
    "place_release",
    "failed",
)
PICK_PLACE_CONTRACT_FILE = "pick_place_environment.json"


def pick_place_prompts() -> tuple[str, ...]:
    return tuple(task_phase_prompt(PICK_PLACE_TASK, group) for group in PICK_PLACE_PROMPT_GROUPS)


def build_pick_place_contract(config: UR5ePickPlaceConfig) -> dict[str, Any]:
    camera = config.camera
    obj = config.object
    return {
        "schema_version": 1,
        "environment_preset": PICK_PLACE_ENVIRONMENT_PRESET,
        "task": PICK_PLACE_TASK,
        "fps": PICK_PLACE_FPS,
        "horizon": config.horizon,
        "control_frequency_hz": config.control_frequency_hz,
        "cameras": {
            "front": {
                "key": "observation.images.front",
                "name": camera.third_person.name,
                "shape": [camera.third_person.height, camera.third_person.width, 3],
                "position_m": list(camera.third_person_position_m),
                "look_at_m": list(camera.third_person_look_at_m),
                "fovy_deg": camera.third_person_fovy_deg,
                "flip_vertical": camera.third_person.flip_vertical,
            },
            "wrist": {
                "key": "observation.images.wrist",
                "name": camera.wrist_name,
                "shape": [camera.wrist_height, camera.wrist_width, 3],
                "fovy_deg": camera.wrist_fovy_deg,
                "forward_offset_m": camera.wrist_forward_offset_m,
                "flip_vertical": camera.wrist_flip_vertical,
            },
        },
        "object": {
            "shape": obj.shape,
            "dimensions_m": list(obj.dimensions_m),
            "rgba": list(obj.rgba),
            "density_kg_m3": obj.density_kg_m3,
            "friction": list(obj.friction),
        },
        "storage_bin": {"size_m": config.target_size_m, "height_m": 0.040},
        "state_shape": [10],
        "action_shape": [7],
        "success": {
            "hold_steps": config.success_hold_steps,
            "placement_tolerance_m": config.placement_tolerance_m,
            "table_height_tolerance_m": config.table_height_tolerance_m,
            "required_lift_m": config.required_lift_m,
            "max_linear_speed_m_s": config.max_linear_speed_m_s,
            "max_angular_speed_rad_s": config.max_angular_speed_rad_s,
        },
        "prompts": list(pick_place_prompts()),
    }


def write_pick_place_contract(dataset_root: Path, config: UR5ePickPlaceConfig) -> Path:
    path = dataset_root / "meta" / PICK_PLACE_CONTRACT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_pick_place_contract(config), indent=2), encoding="utf-8")
    return path


def validate_pick_place_contract(dataset_root: Path, config: UR5ePickPlaceConfig) -> None:
    path = dataset_root / "meta" / PICK_PLACE_CONTRACT_FILE
    if not path.exists():
        raise RuntimeError(f"Dataset is missing PickPlace environment contract: {path}")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    expected = build_pick_place_contract(config)
    if recorded != expected:
        raise RuntimeError("Dataset PickPlace environment contract does not match rollout config.")

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("fps") != PICK_PLACE_FPS:
        raise RuntimeError(f"Dataset FPS mismatch: expected {PICK_PLACE_FPS}, got {info.get('fps')}")
    for camera in expected["cameras"].values():
        feature = info.get("features", {}).get(camera["key"], {})
        if feature.get("shape") != camera["shape"]:
            raise RuntimeError(f"Dataset camera shape mismatch for {camera['key']}")
    for key, shape in (("observation.state", [10]), ("action", [7])):
        if info.get("features", {}).get(key, {}).get("shape") != shape:
            raise RuntimeError(f"Dataset feature shape mismatch for {key}")

    import pyarrow.parquet as pq

    task_table = pq.read_table(dataset_root / "meta" / "tasks.parquet").to_pydict()
    recorded_prompts = set(task_table.get("__index_level_0__", ()))
    supported_prompt_sets = (
        set(expected["prompts"]),
        {PICK_PLACE_GLOBAL_PROMPT},
    )
    if recorded_prompts not in supported_prompt_sets:
        raise RuntimeError("Dataset task prompts do not match a supported PickPlace instruction mode.")
