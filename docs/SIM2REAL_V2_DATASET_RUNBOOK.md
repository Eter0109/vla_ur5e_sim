# 三任务 Sim2Real-v2 数据集 Runbook

> 实施日期：2026-09-02（Asia/Shanghai）  
> 数据状态的机器可读事实源：`outputs/sim2real_v2/pipeline_status.json`

## 1. 交付物与边界

本阶段重新采集 Push、PickPlace、ColorPick 各 1,500 条成功轨迹，旧数据只保留为对照，
不进入新联合集。三个源集与联合集分别位于：

- `data/lerobot/sim2real_v2_push_1500`
- `data/lerobot/sim2real_v2_pick_place_1500`
- `data/lerobot/sim2real_v2_color_pick_1500`
- `data/lerobot/multitask_sim2real_v2_4500`

统一数据合同为 10 FPS、front/wrist 双 RGB `256x256`、10 维机器人状态和 7 维控制动作。
专家从未受扰的特权状态决策；数据保存策略看到的受扰图像/状态以及专家请求的原始动作。
动作增益和一帧动作延迟仅作用于 MuJoCo 实际执行链路。

## 2. 随机化合同

每个几何/颜色分层严格使用 nominal/light/medium = 20%/50%/30%。所有实际采样值都写入
collection manifest，因此同一 seed 可完全复现。

| 参数 | light | medium |
| --- | --- | --- |
| 前视相机位置 | XY +/-3 cm，Z +/-2 cm | XY +/-8 cm，Z +/-5 cm |
| 腕部相机位置 | 各轴 +/-2 mm | 各轴 +/-5 mm |
| 物体质量 | 0.85--1.15 倍 | 0.70--1.30 倍 |
| 物体/指垫摩擦 | 0.90--1.10 倍 | 0.75--1.25 倍 |
| 平移/旋转动作增益 | 0.97--1.03 | 0.93--1.07 |
| 关节/末端/夹爪噪声 sigma | 0.1 deg / 1 mm / 0.005 | 0.3 deg / 3 mm / 0.015 |
| gamma | 0.9--1.1 | 0.8--1.2 |
| RGB 白平衡 | 0.95--1.05 | 0.88--1.12 |
| 径向畸变 k1 | +/-0.03 | +/-0.08 |
| 时序偏差 | 无 | 50% 无、25% 图像 1 帧、25% 动作 1 帧 |

视觉随机化还包括中性墙面/地面、灯光、桌面颜色、曝光、对比度、饱和度、噪声与模糊。
ColorPick 使用收紧的色偏范围，红绿蓝参考色在全部训练扰动后不得交换主色通道。没有黑屏、
大面积遮挡或超过一个控制步的延迟。

## 3. Manifest 与配额

生成入口：

```bash
conda run -n vla_sim_gpu python scripts/generate_sim2real_v2_manifests.py
```

每任务有 1,500 个主场景和 300 个同分布储备场景。Push 为 10 个角度/距离单元，每单元
30/75/45；PickPlace 两个距离层各 150/375/225；ColorPick 每种颜色各 100/250/150。
储备场景只能补同任务、同几何/颜色分层和同扰动档位的缺口。

正式采集在处理完 1,500 个主场景时检查首次成功率：总体必须至少 95%，每个 tier+分层
单元必须至少 90%。不通过时停止，不使用储备场景掩盖过强扰动。每个场景最多尝试两次；
ColorPick 还要求整个轨迹从未误抓其他颜色。

## 4. 运行、恢复和监管

完整流水线入口：

```bash
conda run -n vla_sim_gpu python scripts/run_sim2real_v2_pipeline.py
```

流水线按 Push、PickPlace、ColorPick 串行运行。已有未完成目录会自动使用 `--resume`；异常
退出最多自动重启三次，每次等待 60 秒。正式门槛失败不会自动重启。采集器逐 episode 原子化
保存进度，每 20 个成功 episode 重建环境，只在整百和阶段边界输出进度。

当前有一个 30 分钟周期的 Codex 心跳监管任务 `vla-sim2real-v2`。它只在新增整百、进程退出
或重启、门槛失败、源集完成、联合构建/审计/smoke 完成时通知；无新事件不发消息。

单任务手动恢复示例：

```bash
conda run -n vla_sim_gpu python scripts/collect_sim2real_v2.py \
  --task push \
  --manifest configs/benchmarks/push_sim2real_v2_collection.json \
  --root data/lerobot/sim2real_v2_push_1500 \
  --repo-id local/sim2real_v2_push_1500 \
  --resume
```

## 5. 预检与测试证据

三份分层 30 场 MuJoCo 预检均已完成：

| 任务 | 结果 | 分层覆盖 |
| --- | ---: | --- |
| Push | 30/30 | 10 个角度/距离单元，三档各 10 |
| PickPlace | 30/30 | 两个距离层各 15，三档各 10 |
| ColorPick | 30/30 | 红/绿/蓝各 10，误抓 0，三档各 10 |

原始结果位于 `outputs/sim2real_v2/preflight_*.json`。自动测试覆盖随机化复现、nominal 字节
兼容、严格一帧延迟、reset 恢复、物理参数写入、颜色语义保持以及三源联合索引/语言/统计。

## 6. 联合构建、审计与训练加载 smoke

三个源集完成后流水线自动执行：

```bash
conda run -n vla_sim_gpu python scripts/build_multitask_sim2real_v2_dataset.py \
  --push-root data/lerobot/sim2real_v2_push_1500 \
  --pick-place-root data/lerobot/sim2real_v2_pick_place_1500 \
  --color-pick-root data/lerobot/sim2real_v2_color_pick_1500 \
  --output-root data/lerobot/multitask_sim2real_v2_4500

conda run -n vla_sim_gpu python scripts/audit_sim2real_v2_dataset.py \
  --push-root data/lerobot/sim2real_v2_push_1500 \
  --pick-place-root data/lerobot/sim2real_v2_pick_place_1500 \
  --color-pick-root data/lerobot/sim2real_v2_color_pick_1500 \
  --combined-root data/lerobot/multitask_sim2real_v2_4500 \
  --output outputs/sim2real_v2/audit.json
```

联合构建器验证三源 FPS 与 feature 合同，连续重映射 frame/episode/task index，并保存 Push、
PickPlace、抓红、抓绿、抓蓝五条语言任务。首源数据文件使用硬链接，其余源重写全局索引。
`build_provenance.json` 保存源路径、manifest/环境合同/来源哈希、Git commit、采集命令、失败
统计和联合映射。

最终流水线使用 `.runtime/models/smolvla_base` 和 batch 1 执行 20 步加载 smoke，输出到
`outputs/sim2real_v2/train_smoke20`。该 smoke 不属于正式模型训练。

## 7. 完成判定

只有 `pipeline_status.json` 的 stage 为 `complete`，且 `outputs/sim2real_v2/audit.json` 的
`passed` 为 true，才可宣布完成。三个源目录和联合目录都必须有 `collection.complete`；联合
集必须为 4,500 episode、五个 prompt，三个源集的精确配额、有限 state/action、连续索引、
零 seed 重叠和 ColorPick 误抓 0 必须全部通过审计。
