# 实验注册表

本文件保存可提交的实验摘要和 benchmark 身份；逐局 rollout、checkpoint 与训练日志仍位于本地 `outputs/`。

## 当前任务：UR5e 双任务 SmolVLA Sim-to-Real 鲁棒训练（2026-08-12 ～）

双任务鲁棒训练目标：单一 SmolVLA checkpoint 同时满足 Push 与 PickPlace 两类任务在 nominal 和 randomized 场景下各 50 场的成功率门禁（≥80%）。

### 2026-08-24：当前最优开发集结果

完整实验历程、参数消融矩阵和接手执行顺序见
[双任务 SmolVLA 交接文档（2026-08-24）](../reports/MULTITASK_ROBUST_HANDOVER_20260824.md)。

| 字段 | 值 |
| --- | --- |
| 状态 | 开发进行中；四项 ≥70% 门禁 3/4 达标（Push Randomized 未达标）；四项 ≥80% 门禁 1/4 达标（PickPlace Randomized） |
| Checkpoint（最优基线） | `outputs/multitask_robust/smolvla_30k_lazy_bs2_final/seed1000/checkpoints/030000/pretrained_model` |
| checkpoint SHA-256 | `DFFBFCD07911EBCC0658B853ED5031855741DAE26C3241B9AA67AB56C76DD7B7`（model.safetensors） |
| 训练 | 30k steps；seed=1000；batch=2；双任务 1500+1500 episodes |
| Push 控制参数 | `replan=4, decay=0.75, seed=1000, samples_per_plan=1` |
| PickPlace 控制参数 | `replan=4, decay=0.75, gain=1.8, samples_per_plan=2, mode=vla_action_calibrated` |
| Push Nominal（50场）| 37/50 = **74.0%**（≥70% 达标，≥80% 未达标）|
| Push Randomized（50场）| 33/50 = **66.0%**（≥70% 未达标）|
| PickPlace Nominal（50场）| 36/50 = **72.0%**（≥70% 达标，≥80% 未达标）|
| PickPlace Randomized（50场）| 40/50 = **80.0%**（≥70% 和 ≥80% 均达标）|
| LoRA 最优筛选点 | `key_screen_lora1500`：Push Randomized 20场 16/20 (80%)，PickPlace Nominal 20场 11/20 (55%) —— 50场全量验证待完成 |
| 盲测状态 | push_robust_blind_v1 / pick_place_robust_blind_v1 均未消费 |
| 后续行动 | 确认 lora1500 checkpoint 路径 → 50场全量四项评测 → 视结果选择联合微调或解锁盲测 |

---

## 历史任务：PickPlace v2 双相机原生收纳盒放置

当前场景使用第三视角 RGB 与腕部 RGB 共同观测红色方块和 MuJoCo 原生蓝色收纳盒。任务要求抓取
红色方块、运输到盒内、释放，并通过连续稳定性检查。训练与评测入口会校验数据集中的环境契约，
确保相机、方块、收纳盒、状态、动作和文字 prompt 一致。

### 2026-08-05：Teacher Distillation v5.3 (step 150) 全物理实态收纳盒 22/24 (91.7%) 晋级

完整证据、失败场景和适用边界见
[Teacher Distillation v5.3 实验报告](../reports/PICK_PLACE_TEACHER_DISTILL_V5_3_SUCCESS_20260804.md)。

| 字段 | 值 |
| --- | --- |
| 状态 | full-physics solid bin development promotion candidate；前 24/100 场 `22/24 (91.7%)` |
| Checkpoint | `outputs/pick_place_v2_native_bin/teacher_distill_transport_v5_3_600/seed1000/checkpoints/000150/pretrained_model` |
| Base checkpoint | `outputs/pick_place_v2_native_bin/vla_only_global_20k/seed1000/checkpoints/020000/pretrained_model` |
| 环境配置 | 全刚体物理碰撞开启（未遮罩 contype/conaffinity），视觉组 geom_group=1 包含蓝盒子 |
| 数据 | base 1,000 episodes；teacher v4 400 episodes，仅在 `transport` 阶段参与 replay |
| 训练 | seed 1000；step 150；auxiliary weight `0.45`；LR `6e-7`；XYZ loss weight `3.0` |
| 评测集 | `pick_place_dev_v1` 的前 24/100 场；role=`development`；单 policy seed 1000 |
| 推理 | static prompt；`samples_per_plan=2`；`replan_steps=8`；`decay=0.5` |
| 部署边界 | `vla_raw_safety`；无动作校准、无 RGB-D Supervisor、无物体/目标位姿；仅固定旋转和工作空间裁剪 |
| 结果 | strict success `22/24 (91.7%)`；ever grasped `24/24`；2 个失败均为 `xy_miss` |
| 严格配对 step300 | 在相同全物理实态 24 场和推理协议下为 `21/24 (87.5%)` |
| Manifest SHA-256 | `659b0f0a228c3039836cdda2a18d9e2450538efc9b1b344c22e309744057a5b9` |
| 可用结论 | 在环境恢复真实物理碰撞与视觉网格后，step 150 部署策略达成 `22/24 (91.7%)`，达到 goal 门槛 |

下一晋级条件是完整 100 场 development 与预先固定的多 policy seed 验证。由于该 24 场已用于
checkpoint/超参数筛选，它不是 held-out 集；test/blind 在配置冻结前继续保持不动。

### 2026-07-31：SmolVLA 双样本 + 固定动作校准，留出集 92%

完整训练配置、失败微调、动作校准迭代和失败场景分析见
[PickPlace VLA 训练与评测报告](../reports/PICK_PLACE_VLA_EXPERIMENT_REPORT_20260731.md)。

| 字段 | 值 |
| --- | --- |
| 状态 | 新固定 50 场 test 完成；strict success `46/50 (92%)` |
| Checkpoint | `outputs/pick_place_v2_native_bin/vla_only_global_20k/seed1000/checkpoints/020000/pretrained_model` |
| 数据集 | `data/lerobot/pick_place_v2_native_bin_1000`，1,000 episodes / 84,819 frames |
| VLA 推理 | static prompt；每次规划平均 2 个 flow 样本；temporal ensemble `replan=4`, `decay=0.5` |
| 固定校准 | 仅依据模型 7-D 动作历史识别水平搬运；3 步 Y 方向投票，负 Y gain `1.3`，正 X gain `0.95` |
| 非 oracle 边界 | 不读取方块/收纳盒位姿，不使用 RGB-D 视觉伺服；仅保留固定姿态和工作空间安全裁剪 |
| 开发集 | `pick_place_screen_v1`：`23/24`，抓取 `24/24` |
| 最终留出集 | `pick_place_holdout_v4_50`，role=`test`，generator seed=`67000`，运行前未参与筛选 |
| 留出结果 | `46/50`；distance bin 0 为 `23/25`，bin 1 为 `23/25`；抓取 `49/50` |
| 失败漏斗 | 未抓取 1 场；抓取后放置失败 3 场 |
| 成功场放置误差 | 最终 XY 平均 `15.94 mm`，最大 `30.09 mm` |
| Manifest SHA-256 | `6f5140a77d0b48f3db48c0d55542caf3d03d4a9ad7f8ff3e3ebcc931b2718f70` |
| 结果文件 | `outputs/pick_place_v2_native_bin/vla_only_phase_calibration_samples2/best_gain_1p3_test50_seeded.json` |
| 结果 SHA-256 | `08e630e73ae4e2d0ded45146b3aa5b58133f39b3155a096f8648a1d36ff77832` |
| 可用结论 | 固定新留出集达到 92%；应表述为“SmolVLA + 非 oracle 固定动作校准”，不应表述为未经校准的纯 VLA |

### 2026-07-30：prompt-aligned 20k + PickPlace 视觉伺服

| 字段 | 值 |
| --- | --- |
| 状态 | 50 场 test 完成；strict success `50/50 (100%)` |
| Checkpoint | `outputs/pick_place_v2_native_bin/maskfix_20k/seed1000/checkpoints/020000/pretrained_model` |
| 数据集 | `data/lerobot/pick_place_v2_native_bin_1000`，1,000 episodes / 84,819 frames |
| 训练修复 | 保留所有五类 prompt 起点；action chunk 越过当前 prompt 后使用 `action_is_pad` 屏蔽 loss |
| 系统推理 | temporal ensemble；PickPlace approach/grasp 与 transport 使用有界 RGB-D XY 视觉伺服 |
| 评测集 | `pick_place_test_v2_50`，50 个固定 test 场景，manifest SHA-256 `760c837806a08ce77d81032434de46d779ff8ea954cdfbb679bb7e6082f681a7` |
| 结果 | 50/50；distance bin 0 为 25/25，bin 1 为 25/25；平均 71.42 步，范围 67–80 |
| 放置误差 | 最终 XY 平均 6.71 mm，最大 14.13 mm |
| 结果文件 | `outputs/pick_place_v2_native_bin/maskfix_20k/test_50_pick_transport_servo.json` |
| 结果 SHA-256 | `B3340831089B121BB721343F2E1A20B113007E2CA5A99FF0ADF2E2C2B581CA6C` |
| 可用结论 | 当前固定 50 场 test 达到 100%；这是 VLA + 受限视觉伺服的系统结果，不应表述为纯端到端 SmolVLA 成功率 |

| Benchmark | 角色 | 场景数 | 状态 |
| --- | --- | ---: | --- |
| `pick_place_test_v2_50` | test | 50 | 已完成一次，50/50 |
| `pick_place_holdout_v4_50` | test | 50 | 已完成一次，46/50；当前非 oracle 固定动作校准结果 |
| `pick_place_screen_v1` | development screen | 24 | checkpoint 快速筛选 |
| `pick_place_dev_v1` | development | 100 | 前 24 场已用于 v5.3 单 seed 筛选；完整 100 场与多 seed 尚未完成 |
| `pick_place_blind_v1` | blind | 100 | 一次性盲测，尚未消费 |
| `pick_place_collect_v1` | collection | 1,200 | 采集 1,000 个成功 episode 的候选场景 |

## 历史任务：Stack v1 双色方块堆叠

Stack v1 为已降级的进阶任务，保留其 `red_on_blue` 与 `blue_on_red` 的实验追溯资料。
严格成功要求源方块已释放在目标方块上，并通过接触、位置、高度、速度和连续 10 步稳定性检查。
盲测目标为 `stack_blind_v1` 的严格成功率 `>= 70/100`。

### 2026-07-27：Stack V4 开发筛选结果

| 字段 | 值 |
| --- | --- |
| 状态 | 开发筛选结果；未达到盲测解锁条件，`stack_blind_v1` 尚未消费 |
| Checkpoint | `outputs/stack_v4_40k/seed1000/checkpoints/020000/pretrained_model` |
| 训练 | Stack V4 全参数微调；20k checkpoint 为当前已记录的最佳筛选点 |
| 评测集 | `stack_screen_v1`，24 场 development screen |
| 结果 | strict success `4/24 (16.7%)`；ever grasped `5/24`；抓取到严格成功转化率 `4/5 (80.0%)` |
| 对照 | 同次 40k checkpoint 为 `0/24`，不作为候选 |
| 可用结论 | 放置、释放和退避的物理控制已可稳定转化已发生的抓取；接近与抓取发生率仍是主要瓶颈，不能据此宣称盲测达标 |

### Stack v1 benchmark 状态

| Benchmark | 角色 | 场景数 | 状态 |
| --- | --- | ---: | --- |
| `stack_screen_v1` | development screen | 24 | 用于 checkpoint 快速筛选，可重复运行 |
| `stack_dev_v1` | development | 120 | 用于三 seed 候选的完整开发评测，可重复运行 |
| `stack_blind_v1` | blind | 100 | 未消费；仅在三 seed 晋级检查通过并提供 promotion record 后运行一次 |
| `stack_collect_v1` | collection | 3,000 | 用于受分层约束的专家数据采集，不是评测集 |

规则：不得用 `stack_blind_v1` 调参或重复评测。每次 Stack 结果必须记录 checkpoint、数据集、
`configs/stack_v1.json`、manifest、命令参数和 Git commit。具体晋级门槛见
`docs/archive/stack/STACK_V1_RUNBOOK.md`。

## 历史任务：Lift v2（仅追溯）

## 2026-07-20：P3 canonical 与最终盲测

| 字段 | 值 |
| --- | --- |
| 状态 | canonical 已冻结；`test_v2` 已完整消费一次 |
| Checkpoint | `outputs/smolvla_ablation_c_15k_seed1000/checkpoints/015000/pretrained_model` |
| 训练 | seed=1000，15k steps，peak LR=`5e-5`，无 transition oversampling |
| 系统推理 | temporal ensemble (`replan=4`, `decay=0.5`) + `confirm2` |
| 三 seed validation_v2 | `36/40`、`34/40`、`34/40`；中位数 85%，极差 5 个百分点 |
| test_v2 | `81/100 (81.0%)`，Wilson 95%=`72.2%–87.5%` |
| 峰值显存 | 1190.3 MiB（1.16 GiB） |
| 可用结论 | 成功数、Wilson 下界和显存门槛通过；延迟退化门槛缺少同负载基线，未正式判定 |

## 2026-07-19：历史冷启动 15k 候选

| 字段 | 值 |
| --- | --- |
| 状态 | 历史开发候选，已被 2026-07-20 P3 canonical 替代 |
| Checkpoint | `outputs/smolvla_fullexpert_cosine_15k/checkpoints/015000/pretrained_model` |
| 训练 | seed=1000，15k steps，batch=8，peak LR=`5e-5`，floor=`1e-6`，3x transition oversampling |
| 系统推理 | temporal ensemble (`replan=4`, `decay=0.5`) + `confirm2` |
| 旧 validation | `25/30 (83.3%)` |
| 旧 test（已降级为开发回归集） | `36/50 (72.0%)`，Wilson 95%=`58.3%–82.5%` |
| 对 35/50 历史候选的配对结果 | 新增成功 3 局、退化 2 局、McNemar `p=1.0` |
| 可用结论 | 冷启动存在高分可行解；尚不可作统计显著 SOTA 声明 |

## 历史 Lift v2 benchmark 状态

| Benchmark | 角色 | 场景数 | SHA-256 | 状态 |
| --- | --- | ---: | --- | --- |
| `validation_v2` | development | 40 | `009b72e8fa6eb688a35086d6c43bea47a467999f3c1a407dae2a126fef67e4f6` | 已用于 P3 选择；保留为历史回归集 |
| `test_v2` | blind | 100 | `d23c21b9ba0214e1edaf2d7fc6aa8833a39f8629975aa7778d27c819c2f8c640` | 2026-07-20 已消费一次，81/100；永久锁定 |
| `ood_v1` | diagnostic | 40 | `be9661c5501cd273edcfb5730079e596e42b240edeeab746dfb466827a19b46d` | 已生成，未消费；物理/相机分层需待环境 override 支持后启用 |

规则：`test_v2` 不得再次运行或参与调参。该表仅记录已完成的 Lift 阶段；旧 `test.json` 不得再标记为 held-out 或 blind，也不能替代 Stack 的 benchmark 身份。
