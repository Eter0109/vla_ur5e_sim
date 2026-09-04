# 三任务运行手册 (RUNBOOK)

本手册面向 UR5e 三任务（`push`、`pick_place`、`color_pick`）VLA 系统的完整生命周期操作，包括环境就绪、数据采集与审计、策略微调、推理诊断、全量评测与门禁验收。

---

## 1. 资产规范与布局

### 目录与入口
- `assets/simulation/catalog.json`：仿真任务定义、Prompt、Horizon、Manifest 路径及各任务数据集存储路径的唯一标准入口。
- `assets/policy/catalog.json`：指向基础模型（`base`）、当前可运行 LoRA 检查点（`current`）与续训优化器状态（`training_state`）。

### 数据集规模与特征

| 数据集 | 任务 | Episodes | Frames | Horizon | 默认 Prompt |
| --- | --- | ---: | ---: | ---: | --- |
| `sim2real_v2_push_1500` | Push | 1500 | 94,806 | 250 | `push the block into the red target circle` |
| `sim2real_v2_pick_place_1500` | PickPlace | 1500 | 126,182 | 250 | `place the red cube in the blue storage bin` |
| `sim2real_v2_color_pick_1500` | ColorPick | 1500 | 66,788 | 200 | `pick up the requested color cube` (按红绿蓝各 500 集) |
| `multitask_sim2real_v2_4500` | Combined | 4500 | 287,776 | - | 包含上述 3 任务全部 5 个 Prompt |

---

## 2. 仿真数据采集与审计

### 数据采集安全设计
- 采集脚本默认**严格防覆盖**：若目标目录已存在且未传入 `--resume`，会抛出 `FileExistsError`。
- 标杆数据集位于 `assets/simulation/datasets/` 下，切勿直接以默认路径覆写。

### 常用采集命令
```bash
# 1. 体验或采集新数据（指定输出路径与采集数量）
vla-sim collect --task push --root outputs/datasets/push_demo --stop-after 20

# 2. 中断恢复继续采集
vla-sim collect --task push --root outputs/datasets/push_demo --resume

# 3. 运行三任务数据与元数据审计
vla-sim pipeline --from audit --through audit
```

### 合并数据集生成
若需从重新采集的三任务源数据构建 combined 多任务数据集：
```bash
vla-sim pipeline --from build --through audit
```

---

## 3. 策略训练与微调

### 采样与优化参数
训练过程使用冻结的 5 个 Prompt 任务采样概率配比：
- `push the block into the red target circle`: 1/3 (≈33.33%)
- `place the red cube in the blue storage bin`: 1/3 (≈33.33%)
- `pick up the red cube`: 1/9 (≈11.11%)
- `pick up the green cube`: 1/9 (≈11.11%)
- `pick up the blue cube`: 1/9 (≈11.11%)

### 训练启动命令
```bash
# 1. 默认微调续训（从 assets/policy/current 的 004000 状态继续训练至 12000 步）
vla-sim train --steps 12000 --batch-size 4

# 2. 从基础模型从头微调（Fresh LoRA Rank 32，输出必须为新目录）
vla-sim train --fresh \
  --model assets/policy/base/pretrained_model \
  --output outputs/training/my_experiment_v1 \
  --steps 10000 \
  --batch-size 4

# 3. 训练冒烟验证（1 step 快速测试）
vla-sim pipeline --from audit --through smoke
```

> [!TIP]
> 训练启动时会自动加进程文件锁 `.sim2real_v2_formal_training.lock`，防止并发训练导致显存或权重写坏。

---

## 4. 推理与基准评测

### Presets 矩阵

- `screen`：开发快速筛选子集（1~6 集不等）。
- `nominal`：标杆基准评测集（Push: 50 集, PickPlace: 50 集, ColorPick: 60 集）。
- `randomized_screen`：扰动开发子集（注意：ColorPick 无此 preset）。
- `randomized`：域随机化评测集。
- `blind`：盲测集（严禁开发排查参数覆盖，必须全量执行）。

### 推理命令（Inference）
```bash
# 单集调试，支持对开发集输出进行覆盖
vla-sim infer --task push --preset screen --episodes 1 --overwrite-development
vla-sim infer --task pick_place --preset screen --episodes 1 --overwrite-development
vla-sim infer --task color_pick --preset screen --episodes 1 --overwrite-development
```

### 评测命令（Evaluation）
```bash
# 评测单个任务
vla-sim evaluate --task color_pick --preset nominal

# 评测全部三个任务（nominal 预设）
vla-sim evaluate --task all --preset nominal
```

---

## 5. 质量门槛验收 (Acceptance Gates)

完成 nominal 评测后，运行独立门禁校验脚本，自动对比 Checkpoint 哈希与清单指纹：

```bash
python -m vla_sim.evaluation.gates \
  --checkpoint assets/policy/current/pretrained_model \
  --push outputs/evaluation/push-nominal.json \
  --pick outputs/evaluation/pick_place-nominal.json \
  --color outputs/evaluation/color_pick-nominal.json \
  --output outputs/evaluation/gate_report.json
```

**验收门槛指标**：
1. **Push**：成功率 ≥ 45/50 (90%)。
2. **PickPlace**：成功率 ≥ 45/50 (90%)。
3. **ColorPick**：总成功率 ≥ 54/60 (90%)，红/绿/蓝各子类均 ≥ 18/20，且抓错颜色次数为 0。

---

## 6. 运行机制与排错指引

### PEFT Checkpoint 解析机制
PEFT LoRA adapter 模型需要绑定基础模型路径。系统在运行时通过 `vla_sim.policy.runtime.resolve_checkpoint()` 在 `.runtime/resolved_checkpoints/` 中构建硬链接视图，使 CLI 可以在任意工作目录下稳定运行，无需担心相对路径偏移。设置环境变量 `VLA_SIM_ROOT` 可指定 checkout 根目录。

### 常见问题与对策

1. **`FileExistsError: refusing to overwrite dataset root`**
   - 原因：目标数据集路径已存在。
   - 解决：新采集时指定 `--root outputs/datasets/<新路径>`，断点续采时追加 `--resume`。

2. **`FileExistsError: Refusing to overwrite smoke output`**
   - 原因：`outputs/pipeline/training_smoke` 目录已存在。
   - 解决：清理临时目录 `rm -rf outputs/pipeline/training_smoke` 后重新运行。

3. **`ValueError: color_pick has no 'randomized_screen' benchmark`**
   - 原因：`color_pick` 任务仅提供 `screen`, `nominal`, `randomized`, `blind` 四种 preset。
   - 解决：使用 `--task all` 时请指定 `nominal` 或 `screen`。

4. **`AttributeError: module 'ctypes' has no attribute 'WinDLL'`**
   - 原因：代码已修复，现已兼容 Linux 环境自动读取 `/proc/self/status`。

5. **`FileNotFoundError: config.json`**
   - 原因：代码已修复，现已自动兼容 PEFT LoRA adapter 的 `adapter_config.json`。
