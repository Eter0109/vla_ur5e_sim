# UR5e 双任务 SmolVLA 与 Sim-to-Real 鲁棒训练交接

> 状态日期：2026-08-14（本文件名保留为 `20260812`，内容已更新）  
> 当前目标：同一个 SmolVLA checkpoint 在四个固定开发分片上均达到 **>=80%（>=40/50）**。  
> 当前结论：目标尚未达成；冻结盲测尚未运行。500 条定向 Push 恢复数据已完成并通过隔离审计，6k 联合恢复训练已于 2026-08-14 00:18 启动，训练完成后执行分级筛选。

## 1. 必须先知道的结论

1. 历史最佳四项结果来自同一个 checkpoint，但旧交接文档把它错误归到了
   `smolvla_full_retrain_30k_corrected_push`。真实来源是：

   ```text
   outputs/multitask_robust/smolvla_30k_lazy_bs2_final/seed1000/checkpoints/030000/pretrained_model
   ```

2. 真实历史最佳结果为：Push nominal 37/50、Push randomized 33/50、
   PickPlace nominal 36/50、PickPlace randomized 40/50。按当前 80% 门槛，只有最后一项达标。
3. `smolvla_full_retrain_30k_corrected_push` 不是上述最佳结果的来源。它在默认控制配置下的已知结果为
   Push 31/50、PickPlace nominal 29/50、PickPlace randomized 21/50，不能作为本轮 warm start。
4. Push 的 `samples_per_plan=2` 已在 randomized 开发集运行到 35 场时停止：24/35（68.6%）。
   即使剩余 15 场全成功，最高也只有 39/50，无法达到 80%，因此不再继续这条控制器分支。
5. 目前最主要的学习瓶颈是 Push randomized 的困难横向角度/接触恢复，同时必须防止 PickPlace 遗忘。

## 2. 验收规则

必须由同一个 checkpoint、固定控制参数和原始 50 场开发 manifest 得到以下结果：

| 分片 | 固定 manifest | 门槛 |
|---|---|---:|
| Push nominal | `push_robust_development_nominal_v1.json` | >=40/50 |
| Push randomized | `push_robust_development_randomized_v1.json` | >=40/50 |
| PickPlace nominal | `pick_place_robust_development_nominal_v1.json` | >=40/50 |
| PickPlace randomized | `pick_place_robust_development_randomized_v1.json` | >=40/50 |

严格验证脚本：

```powershell
python scripts\verify_target80_development.py --help
```

只有四项完整 50 场结果全部通过，才能运行：

```powershell
.\scripts\run_multitask_robust_blind_evaluation.ps1
```

筛选用的 20 场子集只用于选择 checkpoint，不能替代正式 50 场证据。开发集结果和盲测结果不得混用；
盲测文件不得覆盖或提前查看。

## 3. 已核实的模型来源和哈希

历史最佳 checkpoint：

```text
outputs/multitask_robust/smolvla_30k_lazy_bs2_final/seed1000/checkpoints/030000/pretrained_model
```

- `model.safetensors` SHA-256：
  `DFFBFCD07911EBCC0658B853ED5031855741DAE26C3241B9AA67AB56C76DD7B7`
- checkpoint 目录 SHA-256：
  `6e73fb8aea72c2fe2e18311773c79554b5879af2bcced410fd156a72acf6e38d`
- 本轮联合恢复训练脚本会在启动前校验模型文件哈希，避免再次选错 warm start。

## 4. 历史最佳开发结果

| 分片 | 成功数 | 成功率 | 相对 80% 门槛 |
|---|---:|---:|---:|
| Push nominal | 37/50 | 74% | -3 |
| Push randomized | 33/50 | 66% | -7 |
| PickPlace nominal | 36/50 | 72% | -4 |
| PickPlace randomized | 40/50 | 80% | 已达标 |

固定控制参数：

- Push：`replan_steps=4`、`temporal_decay=0.75`、`samples_per_plan=1`、`policy_seed=1000`。
- PickPlace：`replan_steps=4`、`temporal_decay=0.75`、`samples_per_plan=2`、
  `control_mode=vla_action_calibrated`、`closed_negative_y_gain=1.8`、`policy_seed=1000`。

对应完整结果文件：

```text
outputs/multitask_robust/eval_orig30k_push_nominal_d075.json
outputs/multitask_robust/eval_orig30k_push_randomized_d075.json
outputs/multitask_robust/ablation_orig30k_pick_nominal_gain18_samples2.json
outputs/multitask_robust/ablation_orig30k_pick_randomized_gain18_samples2.json
```

## 5. 当前正在执行的改进方案

### 5.1 Push 定向恢复数据

目标数据集：

```text
data/lerobot/push_robust_targeted_recovery_v2_500
```

来源 manifest：

```text
configs/benchmarks/push_robust_targeted_recovery_collection_v2.json
```

设计原则：

- 只加强当前薄弱的中距离、横向困难角度单元（angle bins 1 和 4，distance bin 1）。
- 使用独立的新场景种子；已检查与两个 Push 开发 manifest 和冻结盲测 manifest 无种子重叠。
- 正式采集前的专家 smoke 为 20/20。
- 已得到 500 个成功 episode（另有 6 个失败场景记录），共 33,070 帧。
- 成功角度分布为 253/247，差值 6，小于审计上限 10。
- 已存在 `collection.complete`，独立审计状态为 `ok`。
- 与两个 Push 开发 manifest 和冻结盲测 manifest 的环境/域随机化种子重叠均为 0。

审计命令：

```powershell
python scripts\audit_targeted_push_recovery.py `
  --dataset-root data\lerobot\push_robust_targeted_recovery_v2_500 `
  --manifest configs\benchmarks\push_robust_targeted_recovery_collection_v2.json `
  --base-root data\lerobot\multitask_robust_3000 `
  --evaluation-manifest configs\benchmarks\push_robust_development_nominal_v1.json `
  --evaluation-manifest configs\benchmarks\push_robust_development_randomized_v1.json `
  --evaluation-manifest configs\benchmarks\push_robust_blind_v1.json `
  --episodes 500 `
  --output outputs\multitask_robust\audit_targeted_push_recovery_v2_500.json
```

### 5.2 6k 联合恢复训练

训练入口：

```powershell
.\scripts\train_multitask_target80_joint_recovery.ps1
```

输出目录：

```text
outputs/multitask_robust/smolvla_target80_joint_recovery_6k/seed1000
```

运行状态：训练进程 PID 27308，已进入 6,000-step 主循环；速度约 1.9 step/s。
首个 `001000` checkpoint 已于 00:28 正常保存，模型文件约 1.20 GB，SHA-256 为
`284EC5DC6C7B80CDD1DCD4DF7762CD41075640188BAD2815D587675D97A426FD`。
运行日志确认基础帧数 216,932、辅助帧数 135,742，辅助数据有效采样占比 14.0%。

关键配置：

- 从已核实的 `smolvla_30k_lazy_bs2_final` warm start，而不是错误的 corrected-push 模型。
- 训练 6,000 steps，每 1,000 steps 保存一次。
- batch size 2，lazy parquet loader，避免此前的内存/OOM问题。
- 学习率 `1.5e-6`，衰减到 `3e-7`，降低灾难性遗忘风险。
- 原始 3,000 条双任务数据保持主体。
- 辅助数据包含定向 Push 恢复和三组 PickPlace 修正数据，辅助有效占比约 10%-15%。
- Push/PickPlace 辅助数据分别使用正确任务提示词。
- 启动前强制检查 500 条 Push 审计结果、所有辅助采集数据的 `collection.complete`，以及原始 3,000 条合并数据的构建来源和精确 episodes/frames/tasks 合约。

## 6. Checkpoint 分级筛选

已生成可审计的历史 20 场筛选基线：

```text
outputs/multitask_robust/target80_screen_reference_lazy_bs2_final.json
```

| 分片 | 历史筛选基线 |
|---|---:|
| Push nominal | 17/20 |
| Push randomized | 14/20 |
| PickPlace nominal | 13/20 |
| PickPlace randomized | 16/20 |

筛选顺序：

1. 优先检查 1k、3k、6k checkpoint 的两个关键瓶颈：Push randomized 和 PickPlace nominal。
2. 候选至少保证两个关键分片都不低于历史基线，并且其中一个有提升，才进入四分片 20 场筛选。
3. 若趋势不清楚，再补查 2k、4k、5k，避免无差别跑满全部评测。
4. 选出的最优候选才运行四个原始 50 场开发分片。
5. 严格验证四项均 >=40/50 后，才解锁冻结盲测。

关键筛选入口：

```powershell
.\scripts\run_target80_key_checkpoint_screen.ps1 `
  -Checkpoint <checkpoint_path> `
  -Output <fresh_output_directory>
```

四分片筛选入口：

```powershell
.\scripts\run_target80_checkpoint_screen.ps1 `
  -Checkpoint <checkpoint_path> `
  -Output <fresh_output_directory>
```

完整开发评测入口：

```powershell
.\scripts\run_multitask_robust_development_evaluation.ps1 `
  -Checkpoint <checkpoint_path> `
  -Output <fresh_output_directory>
```

## 7. 代码与验证状态

本轮新增或修改的关键能力：

- `scripts/collect_push_demos_v2.py`：安全续采，按最后处理的 `source_index` 恢复。
- `scripts/audit_targeted_push_recovery.py`：训练前数据、分布、manifest 和种子隔离审计。
- `scripts/train_multitask_target80_joint_recovery.ps1`：小学习率联合恢复训练与 warm-start 哈希门禁。
- `scripts/build_target80_screen_reference.py`：从完整历史结果构建可追溯筛选基线。
- `scripts/run_target80_key_checkpoint_screen.ps1`：关键瓶颈快速筛选。
- `scripts/run_target80_checkpoint_screen.ps1`：四分片 20 场筛选。
- `scripts/verify_target80_development.py`：同 checkpoint、原始 manifest、固定控制和 80% 的严格门禁。
- `src/vla_sim/sampling.py`：保持原始多任务提示词，给辅助数据应用正确任务提示词。
- `src/vla_sim/provenance.py`：数据/场景隔离和定向分布检查。
- `src/vla_sim/target80.py`：筛选晋级与最终门禁逻辑。

截至 2026-08-14 的仓库验证：

- `pytest`：165 项通过。
- `ruff check src tests scripts`：通过。
- 新增 PowerShell 筛选脚本语法解析：通过。

## 8. 机械臂、夹具与 Sim-to-Real 边界

- 当前两个仿真任务都使用 `UR5e + Robotiq85Gripper` 两指夹爪和 7 维 `OSC_POSE` 动作。
- 用户真实设备使用夹板/推板时，不能把两指夹爪仿真成功率直接等价为实机成功率。
- Push 的定向数据会提高视觉和接触扰动下的鲁棒性，但末端几何仍存在 sim-to-real gap。
- 在实机部署前应增加真实夹板几何、接触面宽度/摩擦随机化、相机外参偏移、曝光/光照、桌面纹理和背景随机化。
- 首次实机验证必须低速、有限位和急停，先做观察/空载轨迹，再做少量推/抓测试；实机成功率单独统计。

## 9. 接手时的正确执行顺序

1. 确认 `push_robust_targeted_recovery_v2_500` 的审计报告仍为 `status=ok`。
2. 启动 6k 联合恢复训练，监控 step、loss、GPU、错误日志和磁盘空间。
3. 若训练启动门禁失败，核对数据合约，不要直接删除或伪造采集标记。
5. 对 1k/3k/6k 做关键 20 场筛选，必要时补中间 checkpoint。
6. 晋级候选做四分片 20 场筛选。
7. 最优候选做四项各 50 场完整开发评测。
8. 四项均 >=40/50 后运行严格验证器；只有验证器通过才运行盲测。
9. 若未达标，按失败单元再做一轮小规模定向修正，不要直接重复全量 30k 训练。

## 10. 禁止事项

- 不要把 `smolvla_full_retrain_30k_corrected_push` 当成 74/66/72/80 结果的来源。
- 不要把 20 场筛选结果、35 场中断结果或局部成功率写成正式 50 场结论。
- 不要在四项开发门禁通过前运行盲测。
- 不要覆盖既有评测输出、盲测输出、用户数据或 checkpoint。
- 不要为了短期提高 Push 而只用 Push 数据高权重微调；这会显著增加 PickPlace 遗忘风险。

**当前最合理的动作：完成并审计 500 条定向 Push 恢复数据，随后从已核实的
`smolvla_30k_lazy_bs2_final` 启动 6k 小学习率联合恢复训练。**
