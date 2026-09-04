"""Optional simulator dependency loading with Windows-oriented diagnostics."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

ROBOSUITE_INSTALL_HINT = """robosuite is required to create the UR5e simulator.

Activate the project environment and install the simulator extras:
  conda activate vla_sim_gpu
  python -m pip install -e ".[sim,vla,dev]"

On Windows, follow robosuite's WGL / mujoco.dll troubleshooting section if the
viewer or off-screen camera cannot start:
  https://robosuite.ai/docs/installation.html#installing-on-windows

The contract tests do not require robosuite and can still be run with pytest.
"""


class RobosuiteUnavailableError(RuntimeError):
    """Raised when a caller requests a simulator without its optional package."""


def configure_runtime_directories() -> Path:
    """Route Numba and temporary files to a project-owned writable directory.

    robosuite decorates several transforms with ``numba.jit(cache=True)`` at
    import time. In restricted Windows environments Numba can otherwise spin
    while probing an unwritable package or user temp directory.
    """

    project_root = Path(__file__).resolve().parents[3]
    runtime_root = Path(
        os.environ.get("VLA_SIM_RUNTIME_DIR", project_root / ".runtime")
    ).resolve()
    temp_dir = runtime_root / "tmp"
    numba_dir = runtime_root / "numba"
    temp_dir.mkdir(parents=True, exist_ok=True)
    numba_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_dir)
    # On this Windows runtime, robosuite's import-time Numba compilation can
    # block indefinitely on a stale cache lock. Keep the safe interpreter path
    # as the default; callers with a healthy cache may opt back in explicitly.
    if os.environ.get("VLA_SIM_NUMBA_DISABLE_JIT", "1") == "1":
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    return runtime_root


def require_robosuite() -> ModuleType:
    """Import and return robosuite, translating common load failures to a hint."""

    configure_runtime_directories()
    try:
        return importlib.import_module("robosuite")
    except (ImportError, OSError) as exc:
        raise RobosuiteUnavailableError(ROBOSUITE_INSTALL_HINT) from exc
