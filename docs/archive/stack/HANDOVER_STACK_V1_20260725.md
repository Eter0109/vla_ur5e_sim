# UR5e Stack v1 工作交接 — 2026-07-25

> 历史归档：本文记录 2026-07-25 的 Stack 状态，当前任务与结论以
> `docs/reference/EXPERIMENT_REGISTRY.md` 为准。

## 1. 当前结论

目标是在仿真中将双任务堆叠严格成功率提升至 `>=70/100`，同时保持控制接口可迁移到
实机 UR5e。当前工程、专家、数据集和基准设施已基本完成，但模型路线尚未达标：

- 训练当前**已停止**，仅运行过 `seed1000`，checkpoint 保存至 `024000`。
- `seed1001/1002` 尚未启动；120 场开发集晋级评测尚未运行。
- 20k checkpoint 的有效 24 场筛选结果为 `0/24`。
- `stack_blind_v1` 尚未消费，必须继续保持冻结，达到开发集门槛后才能运行一次。
- 不应直接继续到 40k：恢复训练后的 loss 出现异常跳变，且 20k 失败的主要原因包含
  Supervisor 闭爪/阶段切换问题，仅增加训练步数无法解决。

## 2. 已完成工作

### 基准、环境与控制

- 建立版本化 manifest：
  `configs/benchmarks/stack_screen_v1.json`（24 场）、
  `stack_dev_v1.json`（120 场）和 `stack_blind_v1.json`（100 场）。
- 场景生成支持红/蓝双任务、三个距离分层、`75–140 mm` 中心距和非重叠检查。
- 观测状态统一为 10 维：6 个关节角、EEF XYZ、夹爪开度。
- 严格成功判定包含接触、XY/高度误差、释放、速度限制及连续 10 步稳定保持。
- 实现 RGB-D 颜色位姿提供器、Stack Supervisor、阶段漏斗、失败分类和 provenance。
- SmolVLA 只负责 XYZ；旋转固定，夹爪由 Supervisor 门控。正式评测不读取仿真物体真值。

### 专家与数据

- 反馈式专家在 `stack_dev_v1` 达到 `120/120`，六个任务×距离分层均为 `20/20`。
  证据：`outputs/stack_v1/expert_dev_direction_safe_final.json`。
- 已采集并审计 `data/lerobot/stack_v1_3000`：
  3,000 episodes、285,185 frames、六个分层各 500 条、状态维度 10、phase 完整。
- 数据 provenance 位于
  `data/lerobot/stack_v1_3000/collection_provenance.json`；旧数据保持只读。
- 训练配置使用 Lift 15k warm-start、batch 8、chunk 16、action steps 8、
  LR `2e-5`、warmup 500、cosine floor `2.5e-6`、phase-balanced sampler，
  rotation/gripper loss 权重均为 0。

## 3. 20k 筛选结果与失败诊断

唯一有效结果：
`outputs/stack_v1_final_workers0/evaluations/manual_screen_step020000_retry2.json`。
前两个同名前缀结果因 rollout 的 CUDA/状态形状运行时问题无效，不得用于统计。

| 指标 | 结果 |
| --- | ---: |
| 严格成功 | 0/24 |
| approach | 9/24 |
| grasp attempt | 5/24 |
| 瞬时 ever_grasped | 2/24 |
| no_grasp | 19/24 |
| premature_close | 5/24 |
| 平均最小 EEF–方块距离 | 51.9 mm |

18k–20k、20k–22k、22k–24k 的平均训练 loss 分别约为
`0.1901`、`0.1877`、`0.1851`。这说明离线 expert-state 动作回归趋于平台，但不代表
闭环任务收敛。当前主要问题是：

1. approach 本身仍弱，小动作误差在闭环中累积。
2. Supervisor 根据视觉估计距离 `<=18 mm` 即闭爪并立刻进入 LIFT，没有闭爪驻留和
   可迁移的抓取确认。
3. 五次闭爪均被真实几何统计为 `premature_close`，提示视觉距离与门限可能存在系统偏差；
   这是诊断推断，尚未完成标定测量。
4. 抓取失败后没有有效的重新定位—再抓取流程；现有 retry 主要处理低置信度感知。
5. gripper loss 为 0，因此原始夹爪输出噪声较大，但运行时已被 Supervisor 屏蔽。

## 4. 训练中断与恢复风险

原训练在 24k 后暂停。首次 resume 因历史 W&B run ID 不可用而失败；脚本随后改为
`WandbMode=disabled`。第二次 resume 只运行了剩余 16k 中的 124 steps，日志无 traceback
但进程已退出，且 loss 从暂停前约 `0.184` 跳至约 `1.65`，之后仍在 `1.19–1.33`。

相关文件：

- `scripts/resume_stack_v1_seed.ps1`
- `outputs/stack_v1_final_workers0/seed1000.run_manifest.json`
- `outputs/stack_v1_final_workers0/resume2_seed1000.stdout.log`
- `outputs/stack_v1_final_workers0/resume2_seed1000.stderr.log`
- `outputs/stack_v1_final_workers0/seed1000.train.log`

在确认 optimizer、scheduler、sampler 和 step 状态能连续恢复前，**不要再次执行 resume**。
当前命令仅作记录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\resume_stack_v1_seed.ps1 `
  -Python C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe `
  -OutputRoot outputs\stack_v1_final_workers0 `
  -Seed 1000 `
  -CheckpointPath outputs\stack_v1_final_workers0\seed1000\checkpoints\024000
```

## 5. 建议接手顺序

1. 修复 Supervisor：加入闭爪 dwell、基于夹爪位置/电流或反馈的抓取确认接口，以及失败后
   最多两次重新 approach。决策逻辑不得使用仿真物体真值。
2. 增加阶段切换、确认超时、重试上限和实机反馈适配的单元测试。
3. 在相同 24 场 screen 上复测 20k/24k；先确认 grasp/lift 漏斗显著改善，再投入训练。
4. 单独诊断 resume loss 跳变。用短程受控恢复验证首批 batch、学习率、optimizer state
   和 sampler state 与中断前连续，再决定续训或重新启动 seed1000。
5. 完成三个 seeds 后，按 `docs/archive/stack/STACK_V1_RUNBOOK.md` 执行 checkpoint 筛选和
   120 场开发集。
   晋级要求：三 seed 中位数 `>=75%`、最低 `>=65%`、任务差距 `<=10` 个百分点。
6. 只有晋级后才运行一次 `stack_blind_v1`；严格成功需 `>=70/100`，并报告 Wilson 95%
   区间、分任务结果和 VLA-only 开发集消融。

## 6. 验证与保护规则

接手前先运行：

```powershell
conda activate vla_sim_gpu
python -m pytest -q
python -m ruff check src tests scripts
```

最近一次 rollout 修复后的测试结果为 `54 passed`。工作树当前包含大量未提交改动及本地
数据/输出；先检查 `git status`，不要重置或覆盖不相关修改。禁止删除或提交
`data/`、`outputs/`、`.runtime/`。不同配置的 resume 结果不得合并，任何训练或评测必须
锁定源码、模型、数据集、manifest、checkpoint、推理参数和 Git dirty 状态。
