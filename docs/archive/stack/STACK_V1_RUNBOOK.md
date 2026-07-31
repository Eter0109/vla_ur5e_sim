# Stack v1 Runbook

> Historical runbook. PickPlace v2 is the current default task; use
> `docs/PICK_PLACE_V2_RUNBOOK.md` for current commands.

Stack v1 uses a 10-D robot state, phase prompts, RGB-D object poses, supervised rotation and
gripper timing, and strict 10-step stable-stack scoring. Run all commands from the repository
root after activating `vla_sim_gpu`.

## 1. Expert and data gates

```powershell
python scripts/evaluate_stack_expert.py `
  --manifest configs/benchmarks/stack_dev_v1.json `
  --output outputs/stack_v1/expert_dev.json
python scripts/collect_demos.py `
  --manifest configs/benchmarks/stack_collect_v1.json `
  --root data/lerobot/stack_v1_3000 --episodes 3000
python scripts/audit_stack_dataset.py `
  --root data/lerobot/stack_v1_3000 `
  --manifest configs/benchmarks/stack_collect_v1.json `
  --tokenizer outputs/smolvla_ablation_c_15k_seed1000/checkpoints/015000/pretrained_model
```

Do not collect unless the expert report passes 98% overall and 95% in every task/distance cell.
Collection refuses to overwrite an existing dataset and accepts exactly 500 successful episodes
per task/distance cell.

## 2. Train and select

Run `scripts/train_stack_v1.ps1` once for each seed `1000`, `1001`, and `1002`. Screen every
2,000-step checkpoint on `stack_screen_v1`, rank with `select_stack_checkpoints.py`, then evaluate
the top three per seed on all 120 `stack_dev_v1` scenes. Rollouts must use:

```powershell
--experiment-config configs/stack_v1.json --episodes 120 --benchmark-role development
```

Create the promotion record from the three winning development result files:

```powershell
python scripts/check_stack_promotion.py seed1000.json seed1001.json seed1002.json `
  --output outputs/stack_v1/promotion.json
```

## 3. One-time blind evaluation

Only a passing promotion record unlocks `stack_blind_v1`. Run all 100 scenes once with the frozen
checkpoint, dataset, `configs/stack_v1.json`, and `--promotion-record`. Use `--resume` only to
continue that same interrupted run. Report strict successes, Wilson 95% interval, task breakdown,
and the VLA-only development ablation; never interpret the Wilson lower bound as the point target.
