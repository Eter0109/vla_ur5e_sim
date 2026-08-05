# PickPlace v2 native bin Runbook

所有命令从仓库根目录运行，先激活 `vla_sim_gpu`。当前任务是使用第三视角 RGB 和腕部 RGB，
将红色方块放入蓝色 MuJoCo 原生收纳盒。

## 1. 当前 development 候选与冻结基线

当前 raw pure-VLA development 候选：

| 资产 | 路径或配置 |
| --- | --- |
| Checkpoint | `outputs/pick_place_v2_native_bin/teacher_distill_transport_v5_3_600/seed1000/checkpoints/000300/pretrained_model` |
| 训练入口 | `scripts/train_pick_place_teacher_distill_v5_3.ps1` |
| 评估入口 | `scripts/evaluate_pick_place_teacher_distill_v5_3_dev24.ps1` |
| 评测范围 | `pick_place_dev_v1` 前 24/100 场；单 policy seed 1000 |
| 推理 | `vla_raw_safety`；`samples=2`；`replan=8`；无动作 gain 或 RGB-D Supervisor |
| 结果 | strict success `22/24 (91.7%)`；抓取 `24/24` |

它已经通过 development-24 晋级门槛，但尚未完成 100 场 development、多 seed 或 test/blind，
因此不是冻结 canonical checkpoint。详情见
[Teacher Distillation v5.3 报告](reports/PICK_PLACE_TEACHER_DISTILL_V5_3_SUCCESS_20260804.md)。

以下为历史冻结的系统/校准基线，不应与 raw pure-VLA 候选混称：

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

## 2. 查看 raw pure-VLA 候选实际运行

同时打开 MuJoCo 场景和模型实际接收的双 RGB 窗口：

```powershell
python scripts/run_pick_place_vla_only.py `
  --checkpoint outputs\pick_place_v2_native_bin\teacher_distill_transport_v5_3_600\seed1000\checkpoints\000300\pretrained_model `
  --dataset-root data\lerobot\pick_place_v2_native_bin_1000 `
  --repo-id local/ur5e_pick_place_v2_native_bin `
  --manifest configs\benchmarks\pick_place_dev_v1.json `
  --episodes 1 --scene-index 0 --samples-per-plan 2 --replan-steps 8 `
  --policy-seed 1000 --control-mode vla_raw_safety --rgb-window --render `
  --output outputs\pick_place_v2_native_bin\manual_checks\v5_3_scene0000.json
```

只看 VLA 输入时保留 `--rgb-window`、去掉 `--render`；只看 MuJoCo 场景时反之。输出请使用新文件名，
不要覆盖冻结的 50 场结果。

## 3. 生成场景并验证专家

```powershell
python scripts/generate_pick_place_manifests.py

python scripts/evaluate_pick_place_expert.py `
  --manifest configs\benchmarks\pick_place_screen_v1.json `
  --episodes 24 `
  --output outputs\pick_place_v2_native_bin\expert_development_24.json
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

## 6. development 评测与晋级

所有筛选和 smoke 只使用 role=`development` 的 manifest。每次写入新的结果文件，并保留 runner
生成的 `.meta.json`、evaluation fingerprint、checkpoint、manifest、命令参数和结果 SHA-256。

```powershell
python scripts/run_pick_place_vla_only.py `
  --checkpoint <checkpoint> `
  --dataset-root data\lerobot\pick_place_v2_native_bin_1000 `
  --repo-id local/ur5e_pick_place_v2_native_bin `
  --manifest configs\benchmarks\pick_place_screen_v1.json `
  --episodes 6 --samples-per-plan 2 --replan-steps 8 `
  --policy-seed 1000 --control-mode vla_raw_safety `
  --output outputs\pick_place_v2_native_bin\<run_name>\development\screen6.json
```

当前 v5.3 候选的可复现 development-24 入口：

```powershell
.\scripts\evaluate_pick_place_teacher_distill_v5_3_dev24.ps1 -Step 000300
```

该脚本只运行 `pick_place_dev_v1` 的前 24/100 场。完整 development 晋级应在 checkpoint、推理参数
和 seed 列表预先固定后运行全部 100 场；不要把 24 场筛选结果写成完整 development 成功率。

test/blind runner 只允许一次性全量运行且拒绝覆盖。配置未冻结时不得运行 test/blind，也不得使用
test 的前缀子集做 smoke。以下固定动作校准结果只作为已经完成的历史证据保留：

`pick_place_holdout_v4_50` 已用于 `46/50` 结果，不再用于 v5.3 调参或重复评测。新候选必须先完成
development 协议，再生成未参与筛选的新 test manifest。

## 7. 保护规则

- 不删除或提交 `data/`、`outputs/`、`.runtime/`。
- 不覆盖冻结数据集、checkpoint、manifest 或结果文件。
- 历史绿色平面 PickPlace v1 和失败微调批处理已从当前源码树移除；需要追溯时使用 Git 历史和
  实验目录中的 `source.patch`。
- Stack 资料位于 `docs/archive/stack/`，不应混入当前 PickPlace 结论。
