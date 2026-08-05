# PickPlace 纯 VLA Teacher Distillation v4 交接文档

> **历史文档：已被 v5.3 结果替代。** 当前 development-24 候选与后续边界见
> [v5.3 实验报告](PICK_PLACE_TEACHER_DISTILL_V5_3_SUCCESS_20260804.md)。本文仅保留 v4
> 数据、根因和实现过程，文中的“当前结果”不再代表项目最新状态。

日期：2026-08-03  
工作区：`D:\vla_ur5e_sim`  
运行环境：`C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe`  
任务：UR5e 双相机 SmolVLA，将红色方块放入蓝色收纳盒  
目标：raw pure-VLA 端到端成功率严格超过 90%

## 1. 当前结论

本轮解决了 teacher 数据阶段标签全部错分的问题，并证明校准 teacher 的 transport
修正可以部分蒸馏进 raw pure-VLA。当前最佳配置为：

- checkpoint：`teacher_distill_transport_v4_3k/seed1000/checkpoints/002000`
- 推理：`samples_per_plan=2`、`replan_steps=8`、`policy_seed=1000`
- 控制：`vla_raw_safety`
- 动作校准：无
- object/target pose、phase supervisor、oracle：无
- development 结果：`19/24 (79.2%)`
- 抓取：`24/24`
- 失败：`5` 次，全部为 `xy_miss`

因此当前模型仍未达到严格大于 90% 的目标。24 场景至少需要 `22/24` 才能超过 90%。

本轮没有运行、读取或修改 test/blind benchmark。当前没有活动训练或评估进程，也没有
活动监控 automation。

## 2. pure-VLA 边界

本轮最终候选符合以下边界：

- 模型输入为双 RGB 图像、机器人状态和静态全局任务 prompt；
- 部署时不读取 cube pose、target pose 或任务阶段；
- 允许固定末端姿态和 workspace clamp，二者仅作为机器人安全约束；
- `samples_per_plan=2` 是模型 flow 的多样本平均，仍属于 raw pure-VLA 推理；
- 不允许 `closed_negative_y_gain`、`transport_positive_x_gain` 等动作校准；
- 不允许 RGB-D supervisor、heuristic expert 或 oracle control。

teacher 采集期间使用的动作校准和 privileged simulator state 只用于生成训练标签，不进入
部署路径。

## 3. 本轮发现的根因

### 3.1 teacher v3 阶段标签错误

旧 calibrated-teacher v3 为所有帧保存同一个全局 prompt：

```text
place the red cube in the blue storage bin
```

训练入口根据 prompt 推断阶段。该全局 prompt 含有 `in the blue storage bin`，因此
36,364 帧全部被归入 `place_release`。transport 校准动作没有进入 transport replay 池，
导致此前 teacher distillation 和 Visual-LoRA 的负结果不能用于否定蒸馏方案本身。

### 3.2 transport-only 过滤顺序错误

v4 训练时，采样器先根据完整 base+teacher 数据计算阶段权重，再把 teacher 的非 transport
帧权重清零。名义比例和实际比例因此不同：

| 阶段 | 名义比例 | v4 实际比例 |
| --- | ---: | ---: |
| approach | 18.0% | 14.5% |
| grasp | 20.0% | 20.8% |
| lift | 20.0% | 17.2% |
| transport | 27.0% | 34.3% |
| place/release | 15.0% | 13.1% |

teacher 占所有训练样本约 `11.6%`，占 transport 样本约 `33.9%`。transport 被过度放大，
full action-expert 的共享参数产生抓取遗忘。

重要：采样器的精确归一化修复是在 v4 训练完成后实现的。当前 v4 checkpoint 没有受益于
该修复；修复只应用于未来 v5 训练。

## 4. 已实现的代码修改

### 4.1 正确的训练阶段标签

- `src/vla_sim/pick_place_phases.py`
  - 使用 simulator privileged state 为采集帧标注 `approach/grasp/lift/transport/place_release`；
  - 该信息只用于训练采样，不进入 policy observation。
- `scripts/collect_pick_place_calibrated_rollouts.py`
  - 保存 phase-specific task prompt；
  - provenance schema 升级为 2；
  - 明确记录 `phase_labels=privileged_sim_state_training_only` 和部署全局 prompt。
- `tests/test_pick_place_phases.py`
  - 覆盖五阶段漏斗；
  - 验证采集 prompt 与部署全局 prompt 分离。

### 4.2 source × phase replay

- `scripts/train_entrypoint.py`
  - 新增 `VLA_AUXILIARY_PHASE_GROUPS`；
  - teacher 可限定为仅在 transport 阶段参与 replay；
  - 支持多个 auxiliary 数据源。
- `scripts/train_smolvla.ps1`
  - 新增 `-AuxiliaryPhaseGroups`；
  - 将该字段写入 run manifest 和 resume provenance lock。

### 4.3 精确阶段质量归一化

- `src/vla_sim/sampling.py`
  - `phase_sampling_weights()` 新增 `sampling_multipliers`；
  - 先应用 source weight 和 phase filter，再计算每个阶段的有效质量；
  - 过滤后仍严格满足目标阶段比例。
- `tests/test_sampling.py`
  - 新增 source-phase 过滤后的质量分布回归测试；
  - 验证被过滤帧权重为零，五阶段归一化比例与目标一致。

### 4.4 可复现脚本

- `scripts/collect_pick_place_calibrated_teacher_v4.ps1`
- `scripts/train_pick_place_teacher_distill.ps1`
- `scripts/evaluate_pick_place_teacher_distill_transport_v4.ps1`
- `scripts/evaluate_pick_place_teacher_distill_transport_v4_diagnostics.ps1`
- `scripts/evaluate_pick_place_teacher_distill_transport_v4_dev24.ps1`

## 5. teacher v4 数据集

路径：

```text
data/lerobot/pick_place_calibrated_teacher_v4_400
```

采集配置：

| 字段 | 值 |
| --- | --- |
| base checkpoint | 原始 20k checkpoint `020000` |
| accepted episodes | 400 |
| distance bin 0 / 1 | 200 / 200 |
| frames | 36,392 |
| samples per plan | 2 |
| replan steps | 4 |
| negative-Y teacher gain | 1.3 |
| positive-X teacher gain | 0.95 |
| oracle control | false |

验证后的阶段帧数：

| 阶段 | 帧数 |
| --- | ---: |
| approach | 6,843 |
| grasp | 4,185 |
| lift | 4,327 |
| transport | 6,567 |
| place/release | 14,470 |

关键 provenance：

```text
data/lerobot/pick_place_calibrated_teacher_v4_400/meta/collection_provenance.json
```

## 6. v4 训练配置

输出：

```text
outputs/pick_place_v2_native_bin/teacher_distill_transport_v4_3k/seed1000
```

配置摘要：

| 字段 | 值 |
| --- | --- |
| initialization | 原始 20k checkpoint `020000` |
| steps | 3,000 |
| batch size | 8 |
| learning rate | `1.5e-6` |
| decay LR | `4e-7` |
| warmup | 200 |
| chunk/action steps | 16 / 8 |
| mode | full action-expert |
| XYZ loss weight | 2.0 |
| gripper loss weight | 2.5 |
| rotation loss weight | 0 |
| teacher phase | transport only |
| auxiliary sample weight | 1.0 |
| global deployment prompt | `place the red cube in the blue storage bin` |

完整 manifest：

```text
outputs/pick_place_v2_native_bin/teacher_distill_transport_v4_3k/seed1000/run_manifest.json
```

训练正常完成，保存 checkpoint `000500`、`001000`、`001500`、`002000`、`002500`、
`003000`。

## 7. development 结果

### 7.1 6 场景 checkpoint 筛选，samples=1

| checkpoint | success | grasp | xy_miss | no_grasp |
| --- | ---: | ---: | ---: | ---: |
| 001000 | 3/6 | 4/6 | 1 | 2 |
| 002000 | 4/6 | 4/6 | 0 | 2 |
| 003000 | 3/6 | 4/6 | 1 | 2 |

`002000` 在发生抓取的四个场景上放置 `4/4`，说明 transport 修正已经学入模型；主要副作用是
抓取遗忘。

### 7.2 多样本诊断

| checkpoint | samples | success | grasp | xy_miss | no_grasp |
| --- | ---: | ---: | ---: | ---: | ---: |
| 000500 | 1 | 3/6 | 4/6 | 1 | 2 |
| 000500 | 2 | 4/6 | 6/6 | 2 | 0 |
| 002000 | 2 | 5/6 | 6/6 | 1 | 0 |
| 原始 20k | 2 | 4/6 | 6/6 | 2 | 0 |

`samples=2` 恢复了抓取稳定性；`002000` 比原始 20k 的相同 6 场景 samples=2 快筛多成功
一次。

### 7.3 24 场景完整 development

结果文件：

```text
outputs/pick_place_v2_native_bin/teacher_distill_transport_v4_3k/seed1000/
  development/dev24_002000_samples2_replan8_seed1000.json
```

结果：

| 指标 | 结果 |
| --- | ---: |
| success | `19/24 (79.2%)` |
| ever grasped | `24/24 (100%)` |
| xy_miss | 5 |
| no_grasp | 0 |

失败场景：

| scene | bin | final XY error | XY error vector `(x, y)` |
| --- | ---: | ---: | --- |
| `pick_place_dev_v1-0000` | 1 | 34.27 mm | `(4.00, 34.04) mm` |
| `pick_place_dev_v1-0010` | 0 | 31.87 mm | `(2.37, -31.78) mm` |
| `pick_place_dev_v1-0011` | 1 | 35.94 mm | `(9.34, -34.71) mm` |
| `pick_place_dev_v1-0019` | 1 | 39.88 mm | `(10.73, 38.41) mm` |
| `pick_place_dev_v1-0020` | 1 | 30.43 mm | `(3.63, 30.21) mm` |

五次失败均为 Y 方向欠行程，且正负 Y 都存在。下一轮不应只针对 negative-Y；应做双向、
只在校准真正生效帧上的 correction replay。

历史原始 20k 的 development 参考为 `17/24`、抓取 `24/24`。当前结果多成功两次，但当前
使用 `samples=2/replan=8/policy_seed=1000`；如需发布级严格增益结论，应让原始 20k 在完全
相同的 `pick_place_dev_v1` 协议下重跑一次。不要用历史结果替代严格配对 A/B。

## 8. 验证状态

本轮执行过：

```powershell
& C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe `
  -m pytest -q tests\test_pick_place.py tests\test_pick_place_prompts.py `
  tests\test_pick_place_phases.py tests\test_sampling.py

& C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe `
  -m ruff check src\vla_sim\sampling.py scripts\train_entrypoint.py `
  tests\test_sampling.py

git diff --check
```

阶段标注相关 40 项测试通过；采样器修复后的 21 项相关测试通过；Ruff 通过；
`git diff --check` 仅报告 PowerShell 的 LF/CRLF 提示。

尚未在本轮重新执行全量 `python -m pytest -q`。

## 9. 下一轮 v5 建议

不要从 v4 checkpoint 继续训练，应重新从原始 20k checkpoint `020000` 开始，确保采样器
精确归一化修复生效。

建议首轮最小配置：

| 字段 | v5 建议 |
| --- | --- |
| initialization | 原始 20k `020000` |
| phase proportions | 严格 `18/20/20/27/15` |
| auxiliary phase | transport only |
| auxiliary sample weight | `0.35–0.50`，首选 `0.40` |
| teacher share within transport | 约 15–20%，weight=0.40 时约 17% |
| total teacher mass | 约 4–5% |
| learning rate | `5e-7–8e-7` |
| steps | 1,000 |
| checkpoints | 250 / 500 / 750 / 1000 |
| inference gate | samples=2、replan=8 |

在已有帧数下，auxiliary weight `0.40` 对应 teacher 占 transport 约 `17.0%`，占全部采样
约 `4.6%`。这比 v4 的 transport 内 `33.9%` 更适合保护原始抓取能力。

进一步的数据改造建议：

1. 采集时同时记录 raw action、calibrated action 和 correction magnitude；
2. 只提高 `|calibrated - raw| > epsilon` 附近连续 transport 窗口的权重；
3. positive-Y 和 negative-Y correction 分桶采样；
4. 不让未被校准修改的普通 transport 帧占用 teacher 预算；
5. targeted action-expert LoRA 尚未实现，不应在交接后直接假设已有该能力。

development gate：

1. 6 场景：grasp `6/6`、success 至少 `5/6`；
2. 24 场景：grasp `24/24`、success 至少 `22/24`；
3. 未达到上述门槛时不得运行 test/blind；
4. 配置冻结后，才允许一次独立 holdout/test 评估。

## 10. 推荐接手顺序

1. 先为原始 20k checkpoint 补跑完全相同的 `pick_place_dev_v1`、samples=2、replan=8、
   policy seed 1000，建立严格配对 A/B 基线。
2. 用修复后的采样器准备 v5 配置，将 auxiliary weight 改为 `0.40`，训练 1,000 步。
3. 依次筛选 250/500/750/1000；任何 checkpoint 出现 `no_grasp` 都不扩展。
4. 最佳候选先跑 24 development；达到 `22/24` 才冻结配置。
5. 在整个调参阶段保持 test/blind 不动。

## 11. 工作区注意事项

- 当前 worktree 很脏，包含多轮尚未提交的用户和实验修改；不要运行
  `git reset --hard`、`git checkout --` 或批量清理。
- `data/`、`outputs/`、`.runtime/` 是昂贵且被忽略的本地资产，不得删除。
- 当前改动尚未提交；接手人应先用 `git status --short` 和 `git diff` 分离本轮修改与更早的
  用户改动。
- 当前最佳 checkpoint 不应标记为 canonical 或“达到目标”；它只是 development 候选。
- test/blind 未被本轮消费或修改。

## 12. 关键路径汇总

```text
# 原始模型
outputs/pick_place_v2_native_bin/vla_only_global_20k/seed1000/checkpoints/020000/pretrained_model

# teacher v4 数据
data/lerobot/pick_place_calibrated_teacher_v4_400

# v4 训练
outputs/pick_place_v2_native_bin/teacher_distill_transport_v4_3k/seed1000

# 当前最佳 checkpoint
outputs/pick_place_v2_native_bin/teacher_distill_transport_v4_3k/seed1000/checkpoints/002000/pretrained_model

# 当前最佳 development 结果
outputs/pick_place_v2_native_bin/teacher_distill_transport_v4_3k/seed1000/development/
dev24_002000_samples2_replan8_seed1000.json

# 关键代码
src/vla_sim/pick_place_phases.py
src/vla_sim/sampling.py
scripts/collect_pick_place_calibrated_rollouts.py
scripts/train_entrypoint.py
scripts/train_pick_place_teacher_distill.ps1
```
