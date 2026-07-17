"""Average compatible safetensors checkpoints from the same training basin."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--second-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.second_weight <= 1.0:
        raise ValueError("second-weight must be in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    first_model = args.first / "model.safetensors"
    second_model = args.second / "model.safetensors"
    first = load_file(first_model, device="cpu")
    second = load_file(second_model, device="cpu")
    if first.keys() != second.keys():
        raise ValueError("Checkpoint tensor keys do not match")

    averaged: dict[str, torch.Tensor] = {}
    for key, first_tensor in first.items():
        second_tensor = second[key]
        if first_tensor.shape != second_tensor.shape or first_tensor.dtype != second_tensor.dtype:
            raise ValueError(f"Incompatible tensor: {key}")
        if first_tensor.is_floating_point():
            averaged[key] = torch.lerp(first_tensor, second_tensor, args.second_weight)
        else:
            averaged[key] = first_tensor.clone()

    shutil.copytree(args.first, args.output, ignore=shutil.ignore_patterns("model.safetensors"))
    save_file(averaged, args.output / "model.safetensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
