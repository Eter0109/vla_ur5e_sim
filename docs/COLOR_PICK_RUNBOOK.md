# UR5e 三色目标抓取任务操作手册

## 1. 任务定义

桌面上同时随机放置红、绿、蓝三个同尺寸方块。每个 episode 的语言指令从以下三种之一
均衡选择：

```text
pick up the red cube
pick up the green cube
pick up the blue cube
```

策略输入仍然只有 front/wrist 双 RGB 相机、10 维机器人状态和语言指令；目标颜色、目标位置、
其他方块位置均不会写入策略 observation。特权位置只供专家采集和离线诊断使用。

成功条件：夹爪抓住指定颜色方块，将其相对初始高度抬升至少 0.08 m，并连续保持 5 个控制步。
抓住任意非目标颜色方块立即记为失败。该定义要求模型完成“根据语言选择正确物体”，而不是
随机抓起任意一个方块。

## 2. 固定场景

| Manifest | 数量 | 颜色分布 | 用途 |
| --- | ---: | --- | --- |
| `configs/benchmarks/color_pick_smoke_v1.json` | 3 | 每色 1 | 环境 smoke |
| `configs/benchmarks/color_pick_collection_v1.json` | 1,500 | 每色 500 | 采集；300 nominal / 600 light / 600 medium |
| `configs/benchmarks/color_pick_development_v1.json` | 60 | 每色 20 | 固定 nominal 开发验收 |
| `configs/benchmarks/color_pick_development_randomized_v1.json` | 60 | 每色 20 | 30 light / 30 medium 鲁棒开发验收 |
| `configs/benchmarks/color_pick_blind_v1.json` | 60 | 每色 20 | 全 blind，训练期不要消费 |
| `configs/benchmarks/color_pick_stress_v1.json` | 30 | 每色 10 | 全 stress，边界诊断 |

三个方块在固定工作区内随机采样，方块之间额外保留 0.035 m 夹爪间隙。所有 manifest 都由
`scripts/generate_color_pick_manifests.py` 确定性生成。

域随机化与 Push/PickPlace 共用同一套实现，包括前视/腕部相机位姿与视场角、
灯光、桌面颜色与摩擦系数，以及策略 RGB 图像的亮度、对比度、饱和度、色偏、
噪声和模糊。每个样本写入 manifest，同一场景重跑时扰动完全一致。

相机扰动上限（每个轴独立采样）：

| Tier | 前视相机 XY / Z | 前视朝向目标 XY / Z | 前视 FOV | 腕部位置 | 腕部旋转 | 腕部 FOV |
| --- | --- | --- | ---: | --- | --- | ---: |
| light | ±3 cm / ±2 cm | ±2 cm / ±2 cm | ±3° | ±2 mm | ±1° | ±1° |
| medium | ±8 cm / ±5 cm | ±4 cm / ±2 cm | ±7° | ±5 mm | ±3° | ±3° |
| blind | 每个 XY 轴 6–10 cm，Z 4–7 cm | XY 3–5 cm，Z 1–3 cm | 6–9° | 每轴 4–7 mm | 每轴 2–4° | 2–4° |
| stress | ±15 cm / ±10 cm | ±7 cm / ±4 cm | ±11° | ±10 mm | ±6° | ±6° |

## 3. 当前实现状态

- 三自由方块 MuJoCo/robosuite 环境已完成；
- 目标颜色语言契约和错误颜色终止规则已完成；
- 相机、光照、桌面摩擦和策略图像域随机化已接入；
- 特权专家、LeRobot 数据采集、VLA rollout 和 90% 门禁已完成；
- 真实仿真固定开发集专家验收：60/60 成功，红/绿/蓝各 20/20，误抓 0；
- 端到端临时数据集 smoke：3 episode、126 frame，三种 prompt 和环境合同验证通过；
- 正式 1,500 episode 数据尚未采集；
- 现有 checkpoint 未见过该任务数据，不能把它当作 ColorPick 模型。

## 4. 专家验证

```bash
conda run -n vla_sim_gpu env MUJOCO_GL=egl \
  python scripts/evaluate_color_pick_expert.py \
  --manifest configs/benchmarks/color_pick_smoke_v1.json \
  --episodes 3 \
  --output outputs/color_pick/expert_smoke_v1.json
```

完整 60 场 development 专家证据已保存为
`outputs/color_pick/expert_development60.json`，结果为 60/60 且误抓数为 0。

## 5. 正式数据采集

```bash
conda run -n vla_sim_gpu env MUJOCO_GL=egl \
  python scripts/collect_color_pick_demos.py \
  --manifest configs/benchmarks/color_pick_collection_v1.json \
  --episodes 1500 \
  --root data/lerobot/color_pick_1500 \
  --repo-id local/color_pick_1500
```

采集器拒绝覆盖已有目录，只保存成功且从未误抓的 episode，并要求红/绿/蓝目标各恰好 500 条。
完成后会生成：

- `meta/color_pick_environment.json`
- `meta/collection_manifest.json`
- `meta/collection_provenance.json`
- `collection.complete`

数据特征与现有 Push/PickPlace 一致，因此后续可将当前 3,000 episode 双任务联合集作为基础数据，
把 ColorPick 作为辅助重放源加入同一个 SmolVLA。不要把目标颜色编码到 state 向量中。

## 6. VLA 评测

单场开发 smoke：

```bash
conda run -n vla_sim_gpu env MUJOCO_GL=egl \
  python scripts/run_color_pick_vla_only.py \
  --checkpoint <checkpoint>/pretrained_model \
  --dataset-root data/lerobot/color_pick_1500 \
  --repo-id local/color_pick_1500 \
  --manifest configs/benchmarks/color_pick_development_v1.json \
  --episodes 1 \
  --replan-steps 4 --temporal-ensemble-decay 0.75 \
  --samples-per-plan 1 --policy-seed 1000 \
  --output outputs/color_pick/<candidate>_screen1.json
```

固定 60 场正式开发验收：

```bash
conda run -n vla_sim_gpu env MUJOCO_GL=egl \
  python scripts/run_color_pick_vla_only.py \
  --checkpoint <checkpoint>/pretrained_model \
  --dataset-root data/lerobot/color_pick_1500 \
  --repo-id local/color_pick_1500 \
  --manifest configs/benchmarks/color_pick_development_v1.json \
  --episodes 60 \
  --replan-steps 4 --temporal-ensemble-decay 0.75 \
  --samples-per-plan 1 --policy-seed 1000 \
  --output outputs/color_pick/<candidate>_development60.json

python scripts/verify_color_pick_development.py \
  --checkpoint <checkpoint>/pretrained_model \
  --results outputs/color_pick/<candidate>_development60.json \
  --output outputs/color_pick/<candidate>_development60_gate.json
```

门禁要求红、绿、蓝各至少 18/20；因此总成功数也至少为 54/60。结果 metadata 还会校验
manifest 哈希、checkpoint 哈希、语言-only 目标信号和固定推理参数。

nominal 通过后，将 manifest 替换为
`configs/benchmarks/color_pick_development_randomized_v1.json` 跑同样的 60 场，用于评估
light/medium 鲁棒性；blind 清单只在候选模型冻结后使用。

## 7. 三任务训练建议

1. 先完成 ColorPick development 专家 60/60 和正式 1,500 条数据采集。
2. 审计三种 prompt、颜色数量、帧数、图像/state/action 特征和 collection manifest 哈希。
3. 使用当前 `multitask_robust_3000` 作为 base，ColorPick 作为辅助数据；首轮避免同时修改
   LoRA rank、学习率和多个采样比例。
4. 显式控制三个任务的抽样比例，建议从 Push/PickPlace/ColorPick = 25%/40%/35% 起步，
   根据旧任务遗忘情况再调整。
5. 每个候选必须分别复测原 Push 50 场、PickPlace 50 场和 ColorPick 60 场；最终交付仍是
   一个共享 checkpoint。
6. ColorPick 的正确颜色选择率和误抓率必须单独报告，不能只看总体抬升成功率。
