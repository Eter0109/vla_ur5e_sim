# UR5e 双任务 SmolVLA 90% 目标交接文档

> 快照日期：2026-08-31（Asia/Shanghai）  
> 仓库：`/home/eter/桌面/workspace/vla_ur5e_sim`  
> Git 分支：`main`  
> 基线提交：`f16671a0a88e2da04e2b3541bca3b196caf06f52`  
> 当前结论：环境、数据和三轮训练均已落盘；双任务同一模型达到 90% 的目标尚未完成。

## 1. 目标与完成定义

目标是训练出**同一个 SmolVLA checkpoint**，在以下两个固定开发集上分别达到至少
`45/50`（90%）成功：

| 任务 | 固定场景 | 合格线 |
| --- | --- | --- |
| Push Nominal | `configs/benchmarks/push_robust_development_nominal_v1.json` | ≥45/50 |
| PickPlace Nominal | `configs/benchmarks/pick_place_robust_development_nominal_v1.json` | ≥45/50 |

正式证据必须由 `scripts/run_multitask_nominal90.py` 生成，并通过
`scripts/verify_multitask_nominal90.py`。验证器同时检查：场次数、推理参数、控制模式、
checkpoint 哈希和场景 manifest 哈希；20 场筛选或中途结果不能代替正式门禁。

固定推理合约：

| 任务 | 参数 |
| --- | --- |
| Push | 50 场，horizon 250，replan 4，temporal decay 0.75，seed 1000，samples 1，raw safety control |
| PickPlace | 50 场，horizon 250，replan 4，temporal decay 0.75，seed 1000，samples 2，`vla_action_calibrated`，closed negative-y gain 1.8 |

不要为了获得更高分修改上述门禁、场景、seed 或控制参数。

## 2. 当前状态摘要

- 新 Linux 机器环境已搭建并可使用 CUDA。
- 三套正式训练数据完整存在，共 3,000 条联合 episode。
- 第 1、2 轮没有使用 LoRA，均为冻结视觉编码器、仅训练 action expert。
- 第 3 轮使用 rank 16、`all-linear` LoRA，已正常完成 4,000 步。
- 第 3 轮 2,000 步候选未通过；4,000 步候选已生成但尚未开始固定验收。
- 快照时没有训练或评测进程在运行。
- 当前最好且证据完整的 Push 成绩为第 1 轮 25,000 步的 `50/50`，但同一模型的
  PickPlace 只有 `21/50`，因此不能作为最终模型。
- 当前目标仍未达成，不应将任何 checkpoint 标记为 canonical success。

## 3. 环境与硬件

Conda 环境名：`vla_sim_gpu`

| 组件 | 当前版本/状态 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER，16,376 MiB |
| Python | 3.12.14 |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime | 12.8，`torch.cuda.is_available() == True` |
| LeRobot | 0.4.4 |
| MuJoCo | 3.3.7 |
| robosuite | 1.5.2 |
| PEFT | 0.17.1 |
| Transformers | 4.57.6 |

重新搭建环境：

```bash
./scripts/setup_linux_env.sh vla_sim_gpu
./scripts/download_smolvla_base.sh
```

基础模型为 `lerobot/smolvla_base`，固定 revision：
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`。下载脚本会校验
`model.safetensors` SHA-256：
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`。

robosuite 启动时会提示缺少私有 `macros.py`、`robosuite_models` 和 Mink IK；当前 UR5e
任务、单元测试和已有训练/评测不受影响。

## 4. 数据与本地资产

| 数据集 | Episode | Frame | 磁盘占用（约） |
| --- | ---: | ---: | ---: |
| `data/lerobot/multitask_robust_push_1500` | 1,500 | 92,757 | 13 GiB |
| `data/lerobot/multitask_robust_pick_place_1500` | 1,500 | 124,296 | 20 GiB |
| `data/lerobot/multitask_robust_3000` | 3,000 | 217,053 | 19 GiB |

三套数据均为 UR5e、10 FPS，包含 256×256 front/wrist 双相机图像。联合数据是两项任务的
基础训练集；PickPlace 单任务集用于第 2、3 轮的辅助重放。

以下目录昂贵且被 Git 忽略，禁止清理或覆盖：

- `data/`
- `outputs/`
- `.runtime/`

当前主要占用：基础模型约 873 MiB，第 1 轮约 12 GiB，第 2 轮约 6 GiB，第 3 轮约
249 MiB。

## 5. 三轮训练记录

### 第 1 轮：cold start

- 输出：`outputs/multitask_nominal90/round1_coldstart_seed1000`
- 起点：`.runtime/models/smolvla_base`
- 数据：联合 3,000 episode
- 配置：40,000 steps，batch 4，LR `5e-5`，warmup 1,500，decay LR `1e-6`
- 保存频率：5,000 steps
- 训练范围：`freeze_vision_encoder=true`、`train_expert_only=true`
- **未使用 LoRA**

最强候选为 25,000 步：Push `50/50`，PickPlace `21/50`；正式 gate 为 `passed=false`。
其他候选的 PickPlace 大多在达到不可能通过门槛后保留为 `.partial` 证据；10,000 步候选
则因 Push 已经失败而无需继续完整 PickPlace 测试。

### 第 2 轮：PickPlace transition replay

- 输出：`outputs/multitask_nominal90/round2_pick_transition_replay_seed1000`
- 起点：第 1 轮 25,000 步
- 基础数据：联合 3,000 episode
- 辅助数据：PickPlace 1,500 episode，sample weight `2.0`
- 配置：20,000 steps，batch 4，LR `1e-5`，warmup 1,000，decay LR `1e-6`
- 保存频率：5,000 steps
- 训练范围：`freeze_vision_encoder=true`、`train_expert_only=true`
- **未使用 LoRA**

20,000 步候选：Push `48/50`；PickPlace 在 18 场时为 `10/18`，已有 8 次失败，数学上
无法再达到 `45/50`，因此停止剩余测试。

失败样本中经常已经抓取和抬升，但运输/放置阶段出现 XY 偏差。这是启用视觉/跨模态
适配的主要依据，而不是对门禁控制参数做调整。

### 第 3 轮：visual LoRA + PickPlace replay

- 输出：`outputs/multitask_nominal90/round3_visual_lora_pick_replay_seed1000`
- 起点：第 2 轮 20,000 步
- 基础数据：联合 3,000 episode
- 辅助数据：PickPlace 1,500 episode，sample weight `4.0`
- 有效采样中 PickPlace 辅助重放占比约 69.6%
- 配置：4,000 steps，batch 4，LR `2e-6`，warmup 200，decay LR `4e-7`
- 保存频率：2,000 steps
- PEFT：LoRA rank 16，target modules 为 `all-linear`
- 状态：训练正常完成，2,000 和 4,000 步 checkpoint 均存在

2,000 步候选：Push `48/50`；PickPlace 在停止时为 `14/21`，已有 7 次失败，未通过。

当前待验收候选：

```text
outputs/multitask_nominal90/round3_visual_lora_pick_replay_seed1000/checkpoints/004000/pretrained_model
```

该目录只保存约 46 MiB 的 LoRA adapter。其 `adapter_config.json` 指向：

```text
outputs/multitask_nominal90/round2_pick_transition_replay_seed1000/checkpoints/020000/pretrained_model
```

因此第 3 轮 checkpoint **不能脱离第 2 轮基础 checkpoint 单独迁移或删除**。

## 6. 验收结果矩阵

`partial` 表示 PickPlace 未完成 50 场：通常是累计至少 6 次失败后提前停止；第 1 轮
10,000 步候选则是因为 Push 已失败而提前结束。

| 候选 | Push | PickPlace | 结论 |
| --- | ---: | ---: | --- |
| Round 1 / 5k | 46/50 | 19/40 partial | 失败 |
| Round 1 / 10k | 30/50 | 4/5 partial | Push 已失败 |
| Round 1 / 15k | 49/50 | 2/11 partial | 失败 |
| Round 1 / 20k | 47/50 | 17/23 partial | 失败 |
| Round 1 / 25k | 50/50 | 21/50 | 正式 gate 失败 |
| Round 1 / 30k | 47/50 | 22/31 partial | 失败 |
| Round 1 / 35k | 49/50 | 18/26 partial | 失败 |
| Round 1 / 40k | 49/50 | 20/28 partial | 失败 |
| Round 2 / 5k | 49/50 | 8/15 partial | 失败 |
| Round 2 / 10k | 49/50 | 22/32 partial | 失败 |
| Round 2 / 15k | 49/50 | 18/24 partial | 失败 |
| Round 2 / 20k | 48/50 | 10/18 partial | 失败 |
| Round 3 / 2k | 48/50 | 14/21 partial | 失败 |
| Round 3 / 4k | 未测 | 未测 | **下一候选** |

证据目录统一位于 `outputs/multitask_nominal90/eval_round*`。第 1 轮 25,000 步的正式
gate 文件为 `outputs/multitask_nominal90/eval_round1_025000/nominal90_gate.json`。

## 7. 接手后的第一条命令

先对第 3 轮 4,000 步候选运行固定 50×2 验收：

```bash
conda run -n vla_sim_gpu env MUJOCO_GL=egl \
  python scripts/run_multitask_nominal90.py \
  --checkpoint outputs/multitask_nominal90/round3_visual_lora_pick_replay_seed1000/checkpoints/004000/pretrained_model \
  --push-dataset data/lerobot/multitask_robust_push_1500 \
  --pick-dataset data/lerobot/multitask_robust_pick_place_1500 \
  --output outputs/multitask_nominal90/eval_round3_004000
```

执行规则：

1. 先完成 Push 50 场；Push 少于 45 即可判定候选失败。
2. PickPlace 在未完成时一旦出现第 6 次失败，即不可能达到 45/50，可停止该候选的剩余测试。
3. 两项均完成后必须让验证器生成 `nominal90_gate.json`；只有其中 `passed: true` 才算完成目标。
4. 输出目录已存在时入口会拒绝覆盖；不要删除历史证据，应换新目录或先核验既有内容。

建议低频检查：根据当前机器速度，完整 Push 或 PickPlace 50 场通常按分钟级估算；不要按秒轮询。

## 8. 关键入口与代码改动

| 文件 | 作用 |
| --- | --- |
| `scripts/setup_linux_env.sh` | 在新 Linux 机器建立固定 GPU 环境 |
| `scripts/download_smolvla_base.sh` | 下载并校验 SmolVLA base |
| `scripts/train_smolvla_linux.py` | Linux 训练启动器、运行清单、LoRA 开关 |
| `scripts/train_entrypoint.py` | 加权损失、快速 parquet、transition/phase 采样和辅助数据重放 |
| `scripts/run_multitask_nominal90.py` | 固定 Push/PickPlace 50 场串行验收入口 |
| `scripts/verify_multitask_nominal90.py` | 90% 严格门禁与哈希验证 |
| `scripts/collect_push_demos_v2.py` | Push 数据安全续采相关修改 |
| `scripts/extend_push_collection_manifest.py` | 扩展 Push 采集 manifest |

训练启动器默认保持原来的 expert-only 训练；只有传入正数 `--lora-rank` 才启用 LoRA。

第 3 轮的等价训练命令：

```bash
conda run -n vla_sim_gpu env MUJOCO_GL=egl \
  python scripts/train_smolvla_linux.py \
  --model outputs/multitask_nominal90/round2_pick_transition_replay_seed1000/checkpoints/020000/pretrained_model \
  --dataset data/lerobot/multitask_robust_3000 \
  --repo-id local/multitask_robust_3000 \
  --auxiliary-dataset data/lerobot/multitask_robust_pick_place_1500 \
  --auxiliary-repo-id local/multitask_robust_pick_place_1500 \
  --auxiliary-sample-weight 4.0 \
  --auxiliary-task-prompts 'place the red cube in the blue storage bin' \
  --output outputs/multitask_nominal90/round3_visual_lora_pick_replay_seed1000 \
  --steps 4000 --batch-size 4 --learning-rate 2e-6 \
  --warmup-steps 200 --decay-lr 4e-7 --save-freq 2000 \
  --lora-rank 16
```

启动器拒绝覆盖已有输出目录，因此该命令只用于复现说明，不能直接在当前目录重跑。

## 9. 代码质量与工作树状态

快照时验证结果：

```text
python -m py_compile ... 通过
python -m pytest -q      166 passed
```

当前工作树未提交，必须保留这些修改：

```text
 M docs/README.md
 M pyproject.toml
 M scripts/collect_push_demos_v2.py
 M scripts/train_entrypoint.py
?? docs/reports/MULTITASK_NOMINAL90_HANDOFF_20260831.md
?? configs/benchmarks/push_robust_collection_v1_with_reserve.json
?? scripts/download_smolvla_base.sh
?? scripts/extend_push_collection_manifest.py
?? scripts/run_multitask_nominal90.py
?? scripts/setup_linux_env.sh
?? scripts/train_smolvla_linux.py
?? scripts/verify_multitask_nominal90.py
```

不要用 `git checkout`、`git reset --hard` 或清理命令覆盖上述用户工作和本轮实现。

## 10. 交接决策边界

- 最终交付必须是一个共享模型，不接受两个任务各自一个模型。
- 只认固定 50 场×2 的 `>=45/50` 结果，不用短筛选结果冒充正式成功。
- 不修改验收控制器来掩盖模型能力问题。
- 训练/评测产物不得删除；失败候选的 partial 结果也是决策证据。
- 监控应低频进行；出现新 checkpoint、训练退出、候选达到第 6 次失败或正式 gate 完成时再通知。
- 如果第 3 轮 4,000 步仍失败，先按失败阶段与 XY 误差统计诊断，再决定下一轮数据或适配策略；
  不应未经分析简单扩大训练步数。

## 11. 后续改进计划

以下顺序按优先级执行。每一阶段都保留原始输出，只有得到当前阶段的证据后才进入下一阶段。

### P0：验收第 3 轮 4,000 步候选

第一优先级是执行第 7 节的固定 50×2 命令，不再启动新训练。结果按以下规则分流：

| 条件 | 动作 |
| --- | --- |
| Push ≥45 且 PickPlace ≥45 | 运行/核验严格 gate，进入 P4 完成交付 |
| 任一任务出现第 6 次失败 | 停止该候选剩余测试，保留 `.partial`，进入 P1 |
| 评测进程异常退出 | 保留日志，先定位环境或加载错误；不能把异常退出记为模型失败 |

验收输出固定为 `outputs/multitask_nominal90/eval_round3_004000`。不要覆盖已有输出，也不要
修改场景、seed、控制模式或增益。

### P1：失败归因与候选选择

如果 4,000 步未通过，在训练前生成一份 Round 2 / 20k、Round 3 / 2k、Round 3 / 4k 的
同口径对比。至少统计：

- Push/PickPlace 成功数和失败数；
- `ever_grasped`、`ever_lifted` 比例；
- `failure_stage` 分布；
- 最终 `xy_error_vector_m`、XY 误差均值/P90 和正负方向偏差；
- 已进入目标区但释放或稳定性失败的数量；
- 推理延迟是否异常，排除性能问题造成的假失败。

建议将机器可读结果保存为：

```text
outputs/multitask_nominal90/diagnosis_round3_004000.json
```

按主导故障决定训练方向：

| 主导现象 | 判断 | 下一动作 |
| --- | --- | --- |
| 已抓取、已抬升，但 `xy_miss` 多 | 运输/放置视觉映射不足 | P2-A：定向 transport/place 数据 + LoRA |
| `ever_grasped` 明显偏低 | 接近/抓取映射不足 | P2-B：补 approach/grasp 数据并恢复相应 phase 权重 |
| 已进入目标区但释放/稳定失败 | release 时序不足 | P2-C：增加 place/release 尾段样本 |
| PickPlace 提升但 Push <45 | 多任务遗忘 | P2-D：提高 Push 显式重放占比、降低 LR |
| 4k 比 2k 全面退化 | 过拟合或 adapter 漂移 | 不从 4k 继续；回到 Round 2 / 20k 重新做受控 LoRA |

历史证据表明 Round 2 的主要瓶颈更接近第一行，因此 P2-A 是当前默认路线，但必须由
Round 3 / 4k 的实际失败统计确认。

### P2：下一轮受控改进

#### P2-A：运输/放置定向数据（默认方案）

1. 从与固定开发 manifest **种子隔离**的训练场景采集或筛选定向 episode，禁止把开发集
   rollout 回灌训练。
2. 优先保留完整成功轨迹以及能覆盖正/负 XY 偏差、远近距离和不同运输方向的轨迹。
3. 先制作 300～500 episode 的小规模定向集；通过数据审计后再决定是否扩充，避免直接
   生成大规模低价值重复数据。
4. phase 采样建议从以下起点做一次受控实验：approach 15%、grasp 10%、lift 15%、
   transport 35%、place/release 25%。所有比例和实际抽样计数必须写入 run manifest。
5. 每个 batch 保留 25%～35% Push 样本，防止视觉 LoRA 只优化 PickPlace 而破坏已稳定的 Push。

`train_entrypoint.py` 已支持 phase/transition/auxiliary replay 环境参数；在下一轮开始前，应把
实际需要的参数暴露到 `train_smolvla_linux.py` 的 CLI，并写入运行清单，避免依赖未记录的
临时环境变量。

#### P2-B/P2-C：按阶段补数据

- 抓取失败主导时，提高 approach/grasp/lift 的联合占比，并检查 wrist/front 图像与动作时序
  是否对齐；不应只增加总 episode 数。
- 释放失败主导时，增加目标区内减速、开爪和连续稳定尾段；采集验收要检查完整的成功尾段，
  不能把“方块短暂进入目标区”标成成功。

#### P2-D：防止 Push 遗忘

- 使用显式 Push replay，而不是只依赖联合数据中隐含的 Push 帧。
- LoRA rank 先保持 16，不同时改 rank、数据配比和学习率；一次实验只改变一个主因素。
- 学习率从 `5e-7`～`1e-6` 的低区间开始，warmup 取总步数约 5%，先运行短 smoke，再运行
  1,000～2,000 步候选。
- 若需要从现有 LoRA adapter 继续训练，必须先做 20-step resume smoke，证明不是重复挂载
  adapter；否则从 Round 2 / 20k 基础 checkpoint 重新生成新的 LoRA run。

### P3：候选筛选与正式晋级

为控制 GPU 时间和 Codex 额度，使用分层验收：

1. **代码 smoke**：`py_compile`、单元测试、20-step 训练加载/保存测试。
2. **训练期检查点**：建议每 1,000 步保存一次；只在新 checkpoint 出现后评测，不轮询 step 日志。
3. **快速否决**：可以用与训练隔离的小场景集发现明显退化，但结果只能用于淘汰，不能宣布成功。
4. **正式验收**：候选晋级后必须重新执行固定 50×2；任一任务第 6 次失败即可提前终止。
5. **最终 gate**：只有 `nominal90_gate.json` 中 `passed: true`，且两项 checkpoint 哈希一致，
   才能进入 P4。

同一训练 run 不建议同时评测大量相邻 checkpoint。优先评测训练中段和最终 checkpoint；只有
曲线显示临界变化时才补测相邻候选。

### P4：成功后的交付与冻结

门禁通过后执行完整收尾：

1. 记录共享 checkpoint 路径、目录 SHA-256、基础模型/adapter 依赖和 Git commit。
2. 保留 Push、PickPlace、metadata 和 `nominal90_gate.json` 四类原始证据。
3. 将成功实验登记到 `docs/reference/EXPERIMENT_REGISTRY.md`，并更新本交接文档的“当前结论”。
4. 再运行一次 `python -m pytest -q`，确保训练支持代码没有破坏仓库测试。
5. 停止已不再需要的训练监控自动化，避免任务完成后继续消耗额度。
6. 不删除失败实验；将最终模型标为 canonical 时，只更新文档指针，不移动或复制大体积目录。

### 执行清单

- [ ] Round 3 / 4k 固定 Push 50 场
- [ ] Round 3 / 4k 固定 PickPlace 50 场或第 6 次失败提前停止
- [ ] 严格 gate 或失败诊断报告
- [ ] 如有必要，完成定向数据审计与下一轮受控 LoRA
- [ ] 同一 checkpoint 的双任务均达到 ≥45/50
- [ ] 更新实验注册、哈希和最终交接结论

## 12. 2026-09-01 新增第三任务：ColorPick

已新增一个独立、可训练和可验收的第三任务：红、绿、蓝三个方块随机摆放，语言指令指定
目标颜色，机器人必须抓住并抬升正确方块。任务详情与命令见
`docs/COLOR_PICK_RUNBOOK.md`。

当前边界：

- 环境、场景、专家、采集、VLA rollout 和按颜色 90% 门禁均已实现；
- 固定开发集专家验收为 60/60（每色 20/20、误抓 0），临时 LeRobot 采集 smoke 为
  3 episode / 126 frame；
- 正式 1,500 episode ColorPick 数据尚未采集，当前双任务 checkpoint 未训练该能力；
- 原双任务 90% 目标和历史证据保持不变；后续若转为三任务共享模型，应同时复测
  Push 50、PickPlace 50 和 ColorPick 60 场，不能牺牲已有任务换取新任务成绩。
