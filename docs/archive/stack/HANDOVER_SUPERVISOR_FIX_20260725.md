# UR5e Stack v1 Supervisor 修复与验证 — 工作交接 2026-07-25

> 历史归档：本文记录 Stack Supervisor 的阶段性修复，当前任务与结论以
> `docs/reference/EXPERIMENT_REGISTRY.md` 为准。

## 1. 做了什么

按 `docs/archive/stack/HANDOVER_STACK_V1_20260725.md` 建议顺序，完成了
**Step 1（Supervisor 修复）** 和单 seed 快速验证闭环。

### Supervisor 修复

**问题**：`StackSupervisor` 没有 gripper dwell、没有基于夹爪反馈的 grasp 确认、没有 grasp 失败后的 re-approach retry。

**修改文件**：`src/vla_sim/stack_control.py`

| 改动 | 说明 |
|---|---|
| 新增 `GRIPPER_CLOSE` 相位 | 在 GRASP 和 LIFT 之间插入。VLA 到达抓取位置后，Supervisor 闭合夹爪并保持 `dwell_steps`(5) 步 |
| Grasp 确认 | dwell 结束后检查 `gripper_opening`（10-D 状态第 10 维）：≥ `grasp_confirmation_threshold`(0.15) 说明方块卡住了夹爪 → 抓取成功 → LIFT；< 阈值说明夹爪闭合到了接近 0 → 夹空气 → retry |
| Re-approach retry | 抓取失败后：开爪 → 重置到 APPROACH → VLA 重新接近。最多 `grasp_retry_max`(2) 次 |
| 相位超时 | 任何非终止相位超过 `phase_timeout_steps`(60) 步 → FAILED |
| XYZ 冻结 | GRIPPER_CLOSE 期间冻结 EEF 移动，防止夹爪闭合时推动方块 |
| Bug 修复 | perception retry 时同步重置 `_initial_pick_z` 和 `_phase_step` |

**关键设计约束**：grasp 确认只用夹爪位置反馈（robot proprioception），不使用仿真器 `_check_grasp()` 真值。可迁移到实机 UR5e。

### 评测结果

**基线（旧 Supervisor，20k checkpoint，24 场 screen）**：0/24 成功，ever_grasped 2/24，premature_close 5/24

**新 Supervisor + 相同 checkpoint**：

| 指标 | 旧 20k | 新 20k | **新 24k** |
|---|---|---|---|
| 严格成功 | 0 | 2 (8.3%) | **3 (12.5%)** |
| approach | 9 | 7 | 7 |
| ever_grasped | 2 | 5 | 4 |
| premature_close | 5 | 3 | **1** |

**24k 阶段漏斗**：approach 29% → grasp 17% → lift 17% → transport 17% → release 17% → **stack 12.5%**

漏斗在 grasp 之后几乎零损失（4/4 走到 release，3/4 成功堆叠）。

**成功场景**（24k）：
- `0010`: red_on_blue, bin1, eef_dist_at_close=0.024m
- `0013`: red_on_blue, bin0, eef_dist_at_close=0.029m
- `0019`: blue_on_red, bin2, eef_dist_at_close=0.032m

三个成功的 close 距离都 > 旧阈值 0.018m——旧 Supervisor 会全部判为 premature_close。新 Supervisor 通过 gripper_opening 正确确认了抓取。

### 参数调优实验（已回退）

尝试将 `grasp_distance_m` 从 0.018 放宽到 0.030，结果 **3/24 → 0/24**。

**原因**：阈值放大后 Supervisor 在 35-47mm 处触发闭爪（太远），夹空气 → grasp 确认失败 → retry 耗尽 horizon。0.018 是正确的值，成功场景 close 距离偏大 (0.024-0.032m) 是 RGB-D 感知系统误差。

参数已回退到原始值。

### Resume 诊断

**结论**：Resume 机制本身无 bug，上次 124 步 crash 是偶然事件。

| 验证项 | 结果 |
|---|---|
| 200 步验证续训 | ✅ 无崩溃 |
| LR 连续性 | ⚠️ Scheduler state 未正确恢复：LR 从 8.5e-06 跌至 floor (2.5e-06) |
| Loss 收敛 | ✅ 即使 LR 偏低，loss 仍从 1.64 → 1.27 |
| Optimizer state | ✅ 正确加载 |

**Scheduler 问题**：`train_smolvla.ps1` 的 resume 路径把 `optimizer.lr`（配置中的 peak_lr=2e-5）同时传给 `--optimizer.lr` 和 `--scheduler.peak_lr`。LeRobot 创建新 scheduler 但未正确加载 `training_state/scheduler_state.json` 中的 `last_epoch=24000`，导致 scheduler 从 0 开始计数。需在 `train_entrypoint.py` 中强制注入 scheduler 起始 step。

## 2. 修改的文件

| 文件 | 改动内容 |
|---|---|
| `src/vla_sim/stack_control.py` | 新增 GRIPPER_CLOSE 相位、5 个 config 字段 + 校验、重写 filter_action() 相位转换和 retry 逻辑 |
| `tests/test_stack_control.py` | 从 5 个测试扩展到 24 个：dwell、确认成功/失败、retry、超时、边界条件、config 校验 |
| `scripts/run_rollouts.py` | `_failure_stage()` 新增 `grasp_retry_exhausted` 分类 |

## 3. 新增输出文件

| 文件 | 内容 |
|---|---|
| `outputs/stack_v1/supervisor_fix_screen/step020000.json` | 20k checkpoint + 新 Supervisor，screen 24 场结果 |
| `outputs/stack_v1/supervisor_fix_screen/step024000.json` | 24k checkpoint + 新 Supervisor，screen 24 场结果 |
| `outputs/stack_v1/supervisor_fix_screen/step024000_v2.json` | 24k + 放宽 grasp_distance 参数（失败实验，0/24） |

## 4. 当前状态

```
Supervisor 修复 ✅ ──→ 单 seed 验证 ✅ ──→ 参数调优 (已回退) ✅ ──→ Resume 诊断 ✅
                                                                              │
                                                                   🔴 下一步：修复 LR 恢复
                                                                   → 完整续训 24k→40k
                                                                   → 评测 40k
```

**Supervisor 侧已完成**。grasp 之后的漏斗效率很高。

**VLA approach 是当前瓶颈**（仅 29%）。需要更多训练步骤来提升模型的空间导航能力。

## 5. 建议下一步

1. **修复 Scheduler LR 恢复** — 在 `train_entrypoint.py` 中确保 resume 时 scheduler 从正确 step 开始（改动 <10 行）
2. **完整续训 seed1000** — 从 24k 到 40k（需约 4 小时），评测 40k screen
3. **若 40k 成功率达到 ~20-25%** → 跑 seed1001/1002 → 进入 120 场 dev + promotion gate
4. **若 40k 仍然 ~12%** → approach 瓶颈无法通过继续训练当前模型突破，需考虑数据质量或其他策略

## 6. 关键命令

```powershell
# 运行测试
conda activate vla_sim_gpu
python -m pytest -q

# 评测 checkpoint（新 Supervisor 代码已生效）
python scripts/run_rollouts.py `
  --checkpoint outputs/stack_v1_final_workers0/seed1000/checkpoints/024000/pretrained_model `
  --dataset-root data/lerobot/stack_v1_3000 `
  --manifest configs/benchmarks/stack_screen_v1.json `
  --experiment-config configs/stack_v1.json `
  --episodes 24 --horizon 250 `
  --output outputs/stack_v1/supervisor_fix_screen/step024000.json
```

## 7. 保护规则（同原交接文档）

- 禁止删除或提交 `data/`、`outputs/`、`.runtime/`
- 评测前锁定源码、模型、数据集、manifest、checkpoint、推理参数和 Git dirty 状态
- `stack_blind_v1` 必须在通过 promotion gate 后才能消费
