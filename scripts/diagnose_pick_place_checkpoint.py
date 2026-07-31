"""Compare a PickPlace checkpoint prediction with an exact stored training frame."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
RUNTIME = ROOT / ".runtime"
os.environ.update(
    {
        "HF_HOME": str(RUNTIME / "hf"),
        "HF_DATASETS_CACHE": str(RUNTIME / "hf_datasets"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "USE_TF": "0",
        "NUMBA_CACHE_DIR": str(Path(tempfile.gettempdir()) / "vla_sim_numba"),
        "NUMBA_DISABLE_JIT": "1" if os.name == "nt" else "0",
    }
)

from lerobot.datasets.utils import load_stats  # noqa: E402
from vla_sim.contracts import IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY  # noqa: E402
from vla_sim.policy_runtime import load_policy, predict_ensemble_chunk  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    info = json.loads((args.dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    metadata = SimpleNamespace(
        features=info["features"],
        stats=load_stats(args.dataset_root),
    )
    dataset = SimpleNamespace(meta=metadata)

    first_file = next((args.dataset_root / "data").rglob("*.parquet"))
    table = pq.ParquetFile(first_file).read_row_group(
        0,
        columns=[IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY, "action", "task_index"],
    )
    prompts = {
        3: "move above the red cube",
        2: "move down and grasp the red cube",
        0: "lift the red cube",
        4: "move the red cube above the blue storage bin",
        1: "place the red cube in the blue storage bin and release",
    }
    task_indices = table["task_index"].to_pylist()

    config, policy, preprocessor, postprocessor = load_policy(args.checkpoint, dataset, None)
    results = []
    for task_index, prompt in prompts.items():
        frame = task_indices.index(task_index)
        row = table.slice(frame, 1).to_pydict()
        observation = {
            IMAGE_KEY: _decode_image(row[IMAGE_KEY][0]),
            WRIST_IMAGE_KEY: _decode_image(row[WRIST_IMAGE_KEY][0]),
            STATE_KEY: np.asarray(row[STATE_KEY][0], dtype=np.float32),
        }
        expert = np.asarray(row["action"][0], dtype=np.float32)
        chunk = predict_ensemble_chunk(
            observation,
            policy,
            preprocessor,
            postprocessor,
            torch.device("cuda"),
            config.use_amp,
            args.samples,
            task_prompt=prompt,
        )
        results.append(
            {
                "frame": frame,
                "task_index": task_index,
                "prompt": prompt,
                "expert_action": expert.tolist(),
                "predicted_first_action": chunk[0].tolist(),
                "predicted_chunk_mean": chunk.mean(axis=0).tolist(),
                "xyz_l2_first": float(np.linalg.norm(chunk[0, :3] - expert[:3])),
            }
        )
    print(json.dumps(results, indent=2))
    return 0


def _decode_image(value: dict[str, bytes | str | None]) -> np.ndarray:
    payload = value.get("bytes")
    if not isinstance(payload, bytes):
        raise ValueError("Expected embedded image bytes in training parquet")
    return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB")).copy()


if __name__ == "__main__":
    raise SystemExit(main())
