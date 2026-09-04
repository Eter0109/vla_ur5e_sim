"""Repository asset discovery that is independent of the caller's CWD."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the checkout root, allowing an explicit installed-runtime override."""

    override = os.environ.get("VLA_SIM_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[2]
    if not (root / "assets" / "simulation" / "catalog.json").is_file():
        raise RuntimeError(
            "Unable to locate VLA assets. Set VLA_SIM_ROOT to the repository checkout."
        )
    return root


def load_catalog(name: str) -> dict[str, Any]:
    path = project_root() / "assets" / name / "catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_asset(relative_path: str) -> Path:
    """Resolve and constrain a catalog path to this checkout."""

    root = project_root()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Asset path escapes the checkout: {relative_path}")
    return candidate
