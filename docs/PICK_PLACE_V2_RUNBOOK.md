# PickPlace v2 native bin Runbook

所有命令从仓库根目录运行，先激活 `vla_sim_gpu`。当前任务是使用第三视角 RGB 和腕部 RGB，
将红色方块放入蓝色 MuJoCo 原生收纳盒。

## 1. 当前冻结资产

| 资产 | 路径 |
| --- | --- |
| 数据集 | `data/lerobot/pick_place_v2_native_bin_1000` |
| 数据身份 | `local/ur5e_pick_place_v2_native_bin` |
| 20k checkpoint | `outputs/pick_place_v2_native_bin/maskfix_20k/seed1000/checkpoints/020000/pretrained_model` |
| 50 场 manifest | `configs/benchmarks/pick_place_test_v2_50.json` |
| 50 场结果 | `outputs/pick_place_v2_native_bin/maskfix_20k/test_50_pick_transport_servo.json` |

结果文件 SHA-256 为
`B3340831089B121BB721343F2E1A20B113007E2CA5A99FF0ADF2E2C2B581CA6C`。当前结果是
`50/50`；它衡量 SmolVLA、阶段 Supervisor 和有界 RGB-D 视觉伺服组成的完整系统。

另一个不使用物体真值位姿或 RGB-D 视觉伺服的冻结结果为：

| 资产 | 路径或配置 |
| --- | --- |
| 20k checkpoint | `outputs/pick_place_v2_native_bin/vla_only_global_20k/seed1000/checkpoints/020000/pretrained_model` |
| 最终留出集 | `configs/benchmarks/pick_place_holdout_v4_50.json`（seed 67000） |
| 推理 | `samples_per_plan=2`，negative-Y gain `1.3`，positive-X gain `0.95` |
| 结果 | `46/50 (92%)`，抓取 `49/50` |
| 结果文件 | `outputs/pick_place_v2_native_bin/vla_only_phase_calibration_samples2/best_gain_1p3_test50_seeded.json` |

该结果必须标为“SmolVLA + 非 oracle 固定动作校准”，不能简称为未经校准的 VLA-only。
训练和评测的完整实验记录见
[PickPlace VLA 训练与评测报告](reports/PICK_PLACE_VLA_EXPERIMENT_REPORT_20260731.md)。

## 2. 查看模型实际运行

同时打开 MuJoCo 场景和模型实际接收的双 RGB 窗口：

```powershell
python scripts/run_pick_place_rollouts.py `
  --checkpoint outputs\pick_place_v2_native_bin\maskfix_20k\seed1000\checkpoints\020000\pretrained_model `
  --dataset-root data\lerobot\pick_place_v2_native_bin_1000 `
  --repo-id local/ur5e_pick_place_v2_native_bin `
  --manifest configs\benchmarks\pick_place_test_v2_50.json `
  --episodes 1 --rgb-window --render `
  --output outputs\pick_place_v2_native_bin\maskfix_20k\visual_check.json
```

只看 VLA 输入时保留 `--rgb-window`、去掉 `--render`；只看 MuJoCo 场景时反之。输出请使用新文件名，
不要覆盖冻结的 50 场结果。

## 3. 生成场景并验证专家

```powershell
python scripts/generate_pick_place_manifests.py

python scripts/evaluate_pick_place_expert.py `
  --manifest configs\benchmarks\pick_place_test_v2_50.json `
  --episodes 50 `
  --output outputs\pick_place_v2_native_bin\expert_test_50.json
```

manifest 生成是确定性的，但正式实验前仍应记录文件 SHA-256。内部
`environment_preset="pick_place_v1"` 是兼容保留的契约标识，不代表旧绿色平面目标；环境实现和
数据契约均为原生收纳盒版本。

## 4. 重新采集数据

采集脚本拒绝覆盖已存在的数据目录。要重新采集，必须使用一个新的空路径，不能删除当前冻结数据。

```powershell
python scripts/collect_pick_place_demos.py `
  --manifest configs\benchmarks\pick_place_collect_v1.json `
  --root data\lerobot\pick_place_v2_native_bin_1000_recollect `
  --repo-id local/ur5e_pick_place_v2_native_bin_recollect `
  --episodes 1000

python scripts/audit_pick_place_dataset_images.py `
  --root data\lerobot\pick_place_v2_native_bin_1000_recollect `
  --repo-id local/ur5e_pick_place_v2_native_bin_recollect `
  --output outputs\pick_place_v2_native_bin\recollect_audit.png
```

合格数据必须包含两个距离分层各 500 条成功 episode，并通过相机、几何、10-D 状态、7-D action
和六类任务 prompt 契约校验。

## 5. 训练与离线诊断

20-step 仅用于环境 smoke；正式当前配置为 seed 1000、20k steps：

```powershell
.\scripts\train_pick_place_v2_native_bin.ps1 `
  -Dataset data\lerobot\pick_place_v2_native_bin_1000 `
  -OutputRoot outputs\pick_place_v2_native_bin\new_run `
  -Seed 1000 -Steps 20

.\scripts\train_pick_place_v2_native_bin.ps1 `
  -Dataset data\lerobot\pick_place_v2_native_bin_1000 `
  -OutputRoot outputs\pick_place_v2_native_bin\new_run `
  -Seed 1000 -Steps 20000
```

训练使用 phase-balanced sampler。action chunk 越过当前 prompt 边界后，后续位置必须由
`action_is_pad` 屏蔽；否则阶段 prompt 与监督动作会错位。

```powershell
python scripts/diagnose_pick_place_checkpoint.py `
  --checkpoint outputs\pick_place_v2_native_bin\new_run\seed1000\checkpoints\020000\pretrained_model `
  --dataset-root data\lerobot\pick_place_v2_native_bin_1000 `
  --samples 4
```

## 6. 评测与报告

先运行少量场景做 smoke，再决定是否运行完整 50 场。每次都写入新的结果文件，并记录 checkpoint、
数据集、manifest、命令参数、源码状态和结果 SHA-256。

```powershell
python scripts/run_pick_place_rollouts.py `
  --checkpoint <checkpoint> `
  --dataset-root data\lerobot\pick_place_v2_native_bin_1000 `
  --repo-id local/ur5e_pick_place_v2_native_bin `
  --manifest configs\benchmarks\pick_place_test_v2_50.json `
  --episodes 5 `
  --output outputs\pick_place_v2_native_bin\<run_name>\smoke_5.json
```

只有完整运行全部 50 个固定场景，才能报告该 test 的成功率。不得把 1 场可视化或 5 场 smoke
写成正式成功率，也不得把系统结果写成纯端到端 SmolVLA 结果。

不使用 RGB-D 视觉伺服的固定动作校准评测入口为：

```powershell
python scripts/run_pick_place_vla_only.py `
  --checkpoint outputs\pick_place_v2_native_bin\vla_only_global_20k\seed1000\checkpoints\020000\pretrained_model `
  --dataset-root data\lerobot\pick_place_v2_native_bin_1000 `
  --repo-id local/ur5e_pick_place_v2_native_bin `
  --manifest configs\benchmarks\pick_place_holdout_v4_50.json `
  --episodes 50 --samples-per-plan 2 `
  --closed-negative-y-gain 1.3 --transport-positive-x-gain 0.95 `
  --output outputs\pick_place_v2_native_bin\<run_name>\test50.json
```

`pick_place_holdout_v4_50` 已用于当前正式结果。复现实验可以重跑，但不能根据其结果继续调参；
新候选必须先走 development screen，并生成未参与筛选的新 test manifest。

## 7. 保护规则

- 不删除或提交 `data/`、`outputs/`、`.runtime/`。
- 不覆盖冻结数据集、checkpoint、manifest 或结果文件。
- 历史绿色平面 PickPlace v1 和失败微调批处理已从当前源码树移除；需要追溯时使用 Git 历史和
  实验目录中的 `source.patch`。
- Stack 资料位于 `docs/archive/stack/`，不应混入当前 PickPlace 结论。
