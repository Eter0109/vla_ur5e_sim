# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/vla_sim/`. Put robot environments and object definitions in
`envs/`, simulator adapters and heuristic experts in `sim/`, and shared policy logic in the
top-level modules such as `temporal.py`, `losses.py`, and `evaluation.py`. Operational entry
points for data collection, training, rollout, and analysis belong in `scripts/`. Tests live in
`tests/`; name files `test_*.py` and functions `test_<behavior>`.
`configs/` contains reference YAML settings; note that the current smoke and training scripts do
not automatically load them. Project documentation is indexed from `docs/README.md`.

`data/`, `outputs/`, and `.runtime/` hold ignored local datasets, checkpoints, models, and caches.
They may be expensive to recreate; do not delete or commit them.

## Build, Test, and Development Commands

Run commands from the repository root in PowerShell:

```powershell
conda activate vla_sim_gpu
python -m pip install -e ".[sim,vla,dev]"       # editable install and toolchains
python -m pytest -q                            # unit tests
python -m ruff check src tests scripts          # lint
python scripts/smoke_sim.py --episodes 1 --steps 300
.\scripts\train_smolvla.ps1 -Steps 20 -Dataset data\lerobot\expert_gate10 -Output outputs\smoke
```

Use the 20-step training command only as an environment smoke test. Add `--render` for visual
inspection; headless execution is the default.

## Coding Style & Naming Conventions

Use four-space indentation, type hints for public interfaces, and a 100-character line limit.
Ruff targets Python 3.10. Name modules, functions, and variables `snake_case`; classes
`PascalCase`; constants `UPPER_SNAKE_CASE`. Keep CLI parsing in scripts and reusable behavior in
`src/vla_sim/`. Preserve explicit action, observation, and scene contracts rather than passing
unvalidated dictionaries between layers.

## Testing Guidelines

Use pytest and fixed seeds. Prefer fake backends for contract and policy logic; reserve
MuJoCo/robosuite for smoke or rollout validation. Every behavior change needs a focused regression
test. For simulator changes, run both `pytest` and one `smoke_sim.py` episode. No coverage threshold
is configured, so review affected branches directly.

## Commit & Pull Request Guidelines

History generally uses `<type>(<optional-scope>): <imperative summary>`, for example
`feat(phase4): add stack-task rollout`; common types are `feat`, `docs`, and `chore`. Keep commits
focused. Pull requests should summarize behavior and experiment impact, list exact validation
commands, link relevant issues, and include screenshots or short clips for rendered changes.
Record reproducibility metadata for training or benchmark changes and update
`docs/reference/EXPERIMENT_REGISTRY.md` when canonical experiment status changes.
