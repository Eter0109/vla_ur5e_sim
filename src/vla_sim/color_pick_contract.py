"""Dataset and prompt contract for the three-color selection task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vla_sim.envs import UR5eColorPickConfig
from vla_sim.scenes import COLOR_PICK_COLORS, COLOR_PICK_TASK

COLOR_PICK_ENVIRONMENT_PRESET = "color_pick_v1"
COLOR_PICK_FPS = 10
COLOR_PICK_CONTRACT_FILE = "color_pick_environment.json"


def color_pick_prompt(color: str) -> str:
    if color not in COLOR_PICK_COLORS:
        raise ValueError(f"Unsupported ColorPick target color: {color!r}")
    return f"pick up the {color} cube"


def color_pick_prompts() -> tuple[str, ...]:
    return tuple(color_pick_prompt(color) for color in COLOR_PICK_COLORS)


def build_color_pick_contract(config: UR5eColorPickConfig) -> dict[str, Any]:
    camera = config.camera
    return {
        "schema_version": 1,
        "environment_preset": COLOR_PICK_ENVIRONMENT_PRESET,
        "task": COLOR_PICK_TASK,
        "fps": COLOR_PICK_FPS,
        "horizon": config.horizon,
        "control_frequency_hz": config.control_frequency_hz,
        "target_colors": list(COLOR_PICK_COLORS),
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
        "objects": {
            color: {
                "shape": obj.shape,
                "dimensions_m": list(obj.dimensions_m),
                "rgba": list(obj.rgba),
                "density_kg_m3": obj.density_kg_m3,
                "friction": list(obj.friction),
            }
            for color, obj in config.object_configs.items()
        },
        "state_shape": [10],
        "action_shape": [7],
        "success": {
            "required_lift_m": config.required_lift_m,
            "hold_steps": config.success_hold_steps,
            "wrong_grasp_terminates": config.terminate_on_wrong_grasp,
        },
        "prompts": list(color_pick_prompts()),
        "policy_visible_target_signal": "language_only",
    }


def write_color_pick_contract(dataset_root: Path, config: UR5eColorPickConfig) -> Path:
    path = dataset_root / "meta" / COLOR_PICK_CONTRACT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_color_pick_contract(config), indent=2), encoding="utf-8")
    return path


def validate_color_pick_contract(dataset_root: Path, config: UR5eColorPickConfig) -> None:
    path = dataset_root / "meta" / COLOR_PICK_CONTRACT_FILE
    if not path.exists():
        raise RuntimeError(f"Dataset is missing ColorPick environment contract: {path}")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    expected = build_color_pick_contract(config)
    if recorded != expected:
        raise RuntimeError("Dataset ColorPick environment contract does not match rollout config.")

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("fps") != COLOR_PICK_FPS:
        raise RuntimeError(
            f"Dataset FPS mismatch: expected {COLOR_PICK_FPS}, got {info.get('fps')}"
        )
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
    if recorded_prompts != set(color_pick_prompts()):
        raise RuntimeError("Dataset must contain exactly one ColorPick prompt per target color.")
