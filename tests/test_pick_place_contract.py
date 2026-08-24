from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from vla_sim.envs import UR5ePickPlaceConfig
from vla_sim.pick_place_contract import (
    PICK_PLACE_GLOBAL_PROMPT,
    build_pick_place_contract,
    validate_pick_place_contract,
)


def test_contract_accepts_single_global_task_prompt(tmp_path) -> None:
    config = UR5ePickPlaceConfig()
    contract = build_pick_place_contract(config)
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "pick_place_environment.json").write_text(json.dumps(contract), encoding="utf-8")
    info = {
        "fps": 10,
        "features": {
            "observation.images.front": {"shape": [256, 256, 3]},
            "observation.images.wrist": {"shape": [256, 256, 3]},
            "observation.state": {"shape": [10]},
            "action": {"shape": [7]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pa.table({"task_index": [0], "__index_level_0__": [PICK_PLACE_GLOBAL_PROMPT]}),
        meta / "tasks.parquet",
    )

    validate_pick_place_contract(tmp_path, config)
