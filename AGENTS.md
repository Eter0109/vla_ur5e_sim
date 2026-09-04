# Repository Guidelines

## Supported scope

Only `push`, `pick_place`, and `color_pick` are supported. Do not add Lift, Stack, Target80,
historical experiment launchers, or compatibility wrappers. Executable Python belongs under
`src/vla_sim/`; do not recreate a top-level `scripts/` directory.

## Organization

- Simulation environments, experts, collection, scenes, and datasets: `src/vla_sim/simulation/`.
- Policy training and inference: `src/vla_sim/policy/`.
- Simulation evaluation and gates: `src/vla_sim/evaluation/`.
- Task datasets and manifests: `assets/simulation/`.
- Runnable checkpoints: `assets/policy/`.
- Tests: `tests/`, named `test_*.py` with fixed seeds.

Catalog paths are repository-relative and must be resolved through `vla_sim.paths`; never store a
developer-specific absolute path in code or active metadata. Large assets are ignored and must not
be copied, committed, or deleted without an explicit user request.

## Commands

```bash
python -m pip install -e ".[sim,vla,dev]"
vla-sim --help
PYTHONPYCACHEPREFIX=/tmp/vla_sim_pycache python -m pytest -q
PYTHONPYCACHEPREFIX=/tmp/vla_sim_pycache python -m ruff check src tests
vla-sim pipeline --from audit --through audit
```

Use four-space indentation, Python 3.10-compatible type hints, and a 100-character line limit.
Keep CLI parsing at package entry points and reusable behavior in importable modules. Changes to
environment behavior require focused tests and a one-episode headless smoke for each affected task.
