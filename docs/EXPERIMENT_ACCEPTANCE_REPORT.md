# UR5e SmolVLA 训练优化实验与验收报告

- 报告日期：2026-07-17
- 数据集：`data/lerobot/expert_500demos`（24,993 frames / 496 episodes）
- 任务：抓取红色方块并抬升至少 10 cm
- 目标：验证集成功率至少 60%，跨 seed 稳定，并在一次性 held-out 50 局测试达到 60%

## 1. 结论

本轮完成了训练可复现性、动作维度加权损失、转折帧采样、时序动作集成、离散抓手滤波、逐局诊断和评测元数据等工程改造。

最终冻结的推理配置在 validation 上达到 **19/30（63.3%）**，seed=1001 复训达到 **18/30（60.0%）**；但一次性 held-out test 仅达到 **25/50（50.0%）**。因此：

- 工程质量门禁：通过。
- validation 60% 门槛：通过。
- 跨 seed 稳定性门槛：通过。
- held-out 60% 总验收：**未通过**。
- 项目状态：代码可合并、实验可复现，但不能宣称已达到 60% 泛化成功率。

## 2. 验收矩阵

| 验收项 | 门槛 | 实际结果 | 状态 |
| --- | ---: | ---: | --- |
| validation（seed=1000） | >=18/30 | 19/30（63.3%） | 通过 |
| seed 稳定性（seed=1001） | >=16/30 | 18/30（60.0%） | 通过 |
| 一次性 held-out test | >=30/50 | 25/50（50.0%） | 未通过 |
| 单元测试 | 全部通过 | 29 passed | 通过 |
| Ruff | 无告警 | All checks passed | 通过 |
| PowerShell 脚本语法 | 无错误 | OK | 通过 |

held-out 结果中有 33/50 局达到接近阈值，但只有 25/50 局完成成功条件，主要损失发生在接近物体后的稳定闭爪与持续抬升阶段。

## 3. 冻结候选

### Checkpoint

`outputs/smolvla_full_finetune_from_baseline_3k/checkpoints/003000/pretrained_model`

### 推理配置

```text
temporal_ensemble=true
replan_steps=4
temporal_ensemble_decay=0.5
samples_per_plan=1
gripper_mode=confirm
gripper_confirm_steps=2
gripper_close_threshold=0.5
horizon=200
```

`confirm` 模式要求连续两步预测闭爪后才执行闭爪，但不会永久锁存，因此能过滤单步抓手抖动，同时允许后续重新张开。它将相同 checkpoint 的 validation 成功率从 16/30 提高到 19/30。

## 4. 主要实验记录

| 实验 | split | 完成局数 | 成功 | 接近 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 3k baseline + latest | validation | 30 | 16 | 20 | 原最佳基线 |
| 3k baseline + latch | validation | 30 | 15 | 19 | 永久锁存有损失 |
| 3k baseline + debounce2 | validation | 27 | 14 | 18 | 提前淘汰；新增场景但丢失原成功场景 |
| rotation mask, gripper weight 1 | validation | 30 | 13 | 17 | 退化 |
| rotation mask, gripper weight 2 | validation | 30 | 13 | 18 | 退化 |
| transition-focus continuation | validation | 26 | 13 | 16 | 理论不可达后停止 |
| 3k baseline + confirm2 | validation | 30 | 19 | 24 | 最佳冻结候选 |
| seed=1001 复训 + confirm2 | validation | 30 | 18 | 22 | 稳定性通过 |
| 冻结候选一次性测试 | test | 50 | 25 | 33 | 50%，总验收未通过 |
| 4k checkpoint + confirm2 | validation | 30 | 17 | 20 | 长训退化 |
| seed1000/1001 50:50 soup | validation | 30 | 19 | 22 | 成功数无净增 |
| batch4×3000 | validation | 2 | 0 | 1 | 用户要求收尾后中止，不可用于验收 |

完整逐局结果保存在 `outputs/*.json`，相应 `*.json.meta.json` 保存 checkpoint、manifest SHA-256、命令参数和 Git 来源。`outputs/` 已被 `.gitignore` 排除，不随代码提交。

## 5. 最后一次 batch4 训练

为了提高样本覆盖，训练脚本新增可配置 `BatchSize`。最后一次实验采用 FullExpert、batch=4、3000 updates，累计约 12,000 个采样帧：

- 输出：`outputs/smolvla_full_batch4_seed1000`
- 耗时：31:08
- 速度：约 1.61 step/s
- 结果：3000/3000 正常完成
- 产物：step 1000/2000/3000 checkpoint、优化器/调度器/RNG 状态、`train.log`、`run_manifest.json`、离线 W&B、`source.patch`
- validation：仅执行 2/30 局后按项目收尾要求停止，不能用于模型优劣判断

## 6. 工程改动

- `src/vla_sim/losses.py`
  - 真实动作维度裁剪、padding mask、按动作维度加权并按有效权重归一化。
- `src/vla_sim/sampling.py`
  - 按 episode 检测抓手转折帧并生成可复现的过采样权重。
- `src/vla_sim/temporal.py`
  - 连续动作指数时序集成；抓手支持 `latest`、`confirm`、`debounce`、`latch`、`hold`、`hysteresis`、`average`。
- `scripts/train_entrypoint.py`
  - 安装加权损失与转折帧 sampler 兼容补丁。
- `scripts/train_smolvla.ps1`
  - seed、batch size、损失权重、过采样、离线 W&B、文本日志、运行 manifest、dirty source patch 和 resume 支持。
- `scripts/run_rollouts.py`
  - 逐局持久化、原子写入及受限 Windows 回退、抓手诊断、manifest hash 和 Git 元数据。
- `scripts/average_checkpoints.py`
  - 对兼容 safetensors checkpoint 做权重平均。
- `tests/`
  - 覆盖加权损失、padding、转折帧采样和全部抓手时序策略。

## 7. 可复现命令

冻结候选 validation：

```powershell
python scripts/run_rollouts.py `
  --checkpoint outputs/smolvla_full_finetune_from_baseline_3k/checkpoints/003000/pretrained_model `
  --dataset-root data/lerobot/expert_500demos `
  --repo-id local/ur5e_custom_lift `
  --manifest data/manifests/validation.json `
  --episodes 30 --horizon 200 `
  --temporal-ensemble --replan-steps 4 --temporal-ensemble-decay 0.5 `
  --samples-per-plan 1 --gripper-mode confirm --gripper-confirm-steps 2 `
  --output outputs/eval_reproduced_confirm2_val30.json
```

质量门禁：

```powershell
python -m pytest -q
python -m ruff check .
```

## 8. 后续建议

如果后续重新开启优化，建议不要复用本报告中的 `test.json` 做选择。应只基于 train/validation 改进，再生成全新、未见过的 seed manifest 做一次性 50 局测试。优先方向是提高接近后的抓取保持和抬升成功率，而不是继续增加闭爪锁存时间。
