# PickPlace 纯 VLA Teacher Distillation v5.3：development-24 达到 91.7%

日期：2026-08-04  
工作区：`D:\vla_ur5e_sim`  
任务：UR5e 双相机 SmolVLA，将红色方块放入蓝色收纳盒

## 1. 结论与适用边界

Teacher Distillation v5.3 的 step 300 checkpoint 在 `pick_place_dev_v1` 的前 24 个
development 场景上取得：

- strict success：`22/24 (91.7%)`
- ever grasped：`24/24 (100.0%)`
- 失败：`2` 场，均为 `xy_miss`

这是当前 raw pure-VLA 部署链路首次通过 `22/24` 的 development-24 晋级门槛。它是从同一
development 子集上筛选多个 checkpoint 与配方后得到的单训练 seed、单 policy seed 候选，
不是完整 100 场 development 结果，不是 test/blind 结果，也尚未冻结为 canonical checkpoint。
该 24 场成功率的 Wilson 95% 区间约为 `74.2%–97.7%`，不能据此宣称总体泛化率稳定超过 90%。

本轮没有读取、运行或修改 test/blind benchmark。

## 2. 最佳 development 候选

| 字段 | 值 |
| --- | --- |
| Checkpoint | `outputs/pick_place_v2_native_bin/teacher_distill_transport_v5_3_600/seed1000/checkpoints/000300/pretrained_model` |
| Manifest | `configs/benchmarks/pick_place_dev_v1.json` 的前 `24/100` 场 |
| Manifest role | `development` |
| Manifest SHA-256 | `659b0f0a228c3039836cdda2a18d9e2450538efc9b1b344c22e309744057a5b9` |
| Control mode | `vla_raw_safety` |
| Prompt | `place the red cube in the blue storage bin` |
| Inference | `samples_per_plan=2`、`replan_steps=8`、`decay=0.5`、base policy seed `1000` |
| 动作校准 | 无；negative-Y/positive-X gain 均为 `1.0` |
| Oracle / RGB-D supervisor | 无 |
| 安全边界 | 固定旋转与工作空间裁剪 |
| 结果 | `22/24 (91.7%)`，抓取 `24/24` |
| 结果 SHA-256 | `30709c97886630bd7ea4326b1826454365555335f8f6035b87530a6113fc3501` |
| Evaluation fingerprint | `42140628139a952a89a67237cd1ed4cb05fafd22c9a6db151db300177612ee59` |

逐局 JSON 中的 policy seed 会与各场景 env seed 组合；表中的 `1000` 是评估入口使用的 base
policy seed。旧结果逐局字段曾固定写入 `transport_direction_lock=true`，但 raw 模式并未实例化
动作校准器；该元数据问题已在后续代码中修正，不影响本次执行动作或成功判定。

## 3. 配方与训练 lineage

训练从原始 global-prompt 20k checkpoint 继续：

`outputs/pick_place_v2_native_bin/vla_only_global_20k/seed1000/checkpoints/020000/pretrained_model`

核心配置：

| 字段 | 值 |
| --- | --- |
| Base dataset | `data/lerobot/pick_place_v2_native_bin_1000` |
| Teacher dataset | `data/lerobot/pick_place_calibrated_teacher_v4_400` |
| Teacher 使用范围 | 仅 `transport` 阶段 |
| Auxiliary sample weight | `0.45` |
| Phase target | approach `0.18` / grasp `0.20` / lift `0.20` / transport `0.27` / place-release `0.15` |
| Optimizer schedule | peak LR `6e-7`，warmup `100`，600 steps 衰减至 `2e-7` |
| Loss | XYZ `3.0` / rotation `0` / gripper `2.5` |
| Model update | FullExpert；VLM/vision 主干保持冻结 |
| Save frequency | 150 steps |

v5.3 同时将 auxiliary weight 从 v5.2 的 `0.50` 调至 `0.45`，并将 XYZ loss weight 从
`2.0` 调至 `3.0`。由于两个变量同时变化，当前证据只支持“v5.3 组合配方取得提升”，不能把
增益单独归因于 XYZ loss weight。

训练 `run_manifest.json` 记录基准 Git commit
`f4d4b988f6583dee7f66dd324b97d9f78b528c58`，且训练时工作区为 dirty；同目录保存的
`source.patch` SHA-256 为
`181c46739ff6138980725f118dce2150a586b86528b9bfe93e5d16cde0c7d2fb`，用于复原实际训练源码。

## 4. development 演进

所有下列 dev24 数字均使用同一 manifest 前 24 场、`samples=2`、`replan=8`、base policy
seed `1000`，除非另有说明。

| 候选 | Step | Success | Grasp | 失败 |
| --- | ---: | ---: | ---: | --- |
| 原始 global 20k | 20,000 | `20/24` | `24/24` | 4 × `xy_miss` |
| v5.1 | 500 | `21/24` | `24/24` | 3 × `xy_miss` |
| v5.2 | 400 | `21/24` | `24/24` | 3 × `xy_miss` |
| **v5.3** | **300** | **`22/24`** | **`24/24`** | **2 × `xy_miss`** |

与严格配对的原始 20k 双样本基线相比，v5.3 在这 24 场上净增 2 次成功。原始基线结果
SHA-256 为 `e7e0541e82657d7e8a445b38b4ad0e52eabefba8ad490993f06f0a9a112c5ce0`。

同一 v5.3 run 的 checkpoint 曲线为：

| Step | Success | Grasp |
| ---: | ---: | ---: |
| 150 | `20/24` | `24/24` |
| **300** | **`22/24`** | **`24/24`** |
| 450 | `20/24` | `24/24` |
| 600 | `20/24` | `24/24` |

这说明最佳点出现在短程微调中段，最终 step 600 不是推荐 checkpoint。

## 5. 剩余失败

| Scene | Distance bin | XY error | XY error vector |
| --- | ---: | ---: | --- |
| `pick_place_dev_v1-0000` | 1 | `57.45 mm` | `(+13.99, +55.72) mm` |
| `pick_place_dev_v1-0019` | 1 | `47.37 mm` | `(+16.47, +44.42) mm` |

两场均已完成抓取、抬升、释放并稳定落桌，只是最终 XY 超出目标区，因此剩余瓶颈仍是远距离
transport 末端定位。

## 6. 复现入口

- 训练：[train_pick_place_teacher_distill_v5_3.ps1](../../scripts/train_pick_place_teacher_distill_v5_3.ps1)
- development-24 评估：[evaluate_pick_place_teacher_distill_v5_3_dev24.ps1](../../scripts/evaluate_pick_place_teacher_distill_v5_3_dev24.ps1)
- 通用 raw runner：[run_pick_place_vla_only.py](../../scripts/run_pick_place_vla_only.py)
- 训练 manifest：`outputs/pick_place_v2_native_bin/teacher_distill_transport_v5_3_600/seed1000/run_manifest.json`
- 结果：`outputs/pick_place_v2_native_bin/teacher_distill_transport_v5_3_600/seed1000/development/dev24_000300_samples2_replan8_seed1000.json`
- 结果 metadata：同名 `.json.meta.json`

`outputs/` 与 `data/` 是忽略的本地证据资产，不提交到 Git；报告中的 hash 和 fingerprint 用于
确认本地文件身份。

## 7. 验证与下一步

晋级前仍需完成：

1. 在完整 `pick_place_dev_v1` 100 场上评估 step 300；
2. 使用预先固定的多个 policy seed 验证方差；
3. 若需要声明训练稳定性，再补独立训练 seed；
4. 仅在 development 协议与 checkpoint 全部冻结后，才考虑一次性 test/blind 评估。

test/blind 在上述条件满足前继续保持未触碰状态。
