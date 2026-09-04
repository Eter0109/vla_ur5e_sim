# UR5e 三任务 VLA

本项目基于 SmolVLA（VLM 骨干网络 + Flow Matching 动作分块）与 MuJoCo / robosuite 仿真环境，面向 UR5e 机械臂实现端到端具身操作。

仓库**仅支持三个核心仿真任务**：
- `push`：推物料块至红色目标圆心（Prompt: `push the block into the red target circle`，Horizon: 250）
- `pick_place`：抓取红色方块并放入蓝色收纳盒（Prompt: `place the red cube in the blue storage bin`，Horizon: 250）
- `color_pick`：依指定颜色抓取方块（Prompt: `pick up the [red/green/blue] cube`，Horizon: 200）

代码统一位于 `src/vla_sim/`，仿真元数据与数据集位于 `assets/simulation/`，策略模型位于 `assets/policy/`。

---

## 环境准备

本项目要求 Python 3.10 ~ 3.12 及支持 CUDA 的 PyTorch 环境。

```bash
# 1. 激活已配置环境（或新建 conda 环境）
conda activate vla_sim_gpu

# 2. 可编辑模式安装依赖（包含仿真、VLA 与开发工具）
python -m pip install -e ".[sim,vla,dev]"

# 3. 验证 CLI 安装与环境就绪
vla-sim --help
```

> [!TIP]
> 无头服务器（Headless）环境下若出现渲染问题，建议预设环境变量：`export MUJOCO_GL=egl`。

---

## 极速上手验证（Smoke Loop）

为快速核验环境、权重与仿真管线是否完好，推荐按以下三步执行快速冒烟闭环：

### 1. 数据集与元数据审计
核验数据集合约、任务分布与校验和（约 2 秒完成）：
```bash
vla-sim pipeline --from audit --through audit
```

### 2. 策略训练冒烟测试
执行 1 个 step 的极简训练，验证模型加载、数据采样流水线与 CUDA 反向传播：
```bash
vla-sim pipeline --from audit --through smoke
```
> [!NOTE]
> 冒烟测试输出目录为 `outputs/pipeline/training_smoke`。如需重新运行，请先清理该目录以触发防覆盖保护。

### 3. 单 Episode 推理测试
验证三任务策略在仿真环境中的闭环执行：
```bash
# Push 任务
vla-sim infer --task push --preset screen --episodes 1 --overwrite-development

# Pick & Place 任务
vla-sim infer --task pick_place --preset screen --episodes 1 --overwrite-development

# Color Pick 任务
vla-sim infer --task color_pick --preset screen --episodes 1 --overwrite-development
```

---

## 统一操作指南

CLI 入口为 `vla-sim`，所有默认路径均通过 `assets/*/catalog.json` 自动解析，支持在任意目录下调用。

### 1. 数据采集 (`vla-sim collect`)

```bash
# 查看帮助
vla-sim collect --help

# 采集新数据（建议显式指定未存在的 --root，并用 --stop-after 控制采集集数）
vla-sim collect --task push --root outputs/datasets/push_demo --stop-after 10

# 断点续采（若上次采集因中断未完成，传入 --resume 继续填充配额）
vla-sim collect --task push --root outputs/datasets/push_demo --resume
```

> [!IMPORTANT]
> **防覆盖保护说明**：`assets/simulation/datasets/` 下已包含三任务各 1500 集的完整标杆数据集。直接执行 `vla-sim collect --task push` 会触发防覆盖保护并抛出 `FileExistsError`。体验或生成新数据时，请务必传入自定义的 `--root <路径>`。

### 2. 策略训练 (`vla-sim train`)

```bash
# 查看帮助
vla-sim train --help

# 默认模式：从当前 assets/policy/current (Step 4000) 状态继续微调
vla-sim train --steps 12000 --batch-size 4

# 从头训练（Fresh LoRA 微调基础模型，输出必须为新目录）
vla-sim train --fresh --model assets/policy/base/pretrained_model --output outputs/training/fresh_v1 --steps 10000
```
- 默认使用合并数据集 `assets/simulation/datasets/multitask_sim2real_v2_4500`。
- 自动按照冻结的任务采样比例加载（Push 1/3, PickPlace 1/3, ColorPick 1/9 * 3）。

### 3. 策略推理 (`vla-sim infer`)

用于对指定任务进行单集或小批量快速推理与诊断：

```bash
# 查看帮助
vla-sim infer --help

# 执行 Push 单集推理
vla-sim infer --task push --preset screen --episodes 1 --overwrite-development

# 执行 PickPlace 推理并保存自定义路径
vla-sim infer --task pick_place --preset screen --episodes 5 --output outputs/inference/pick_custom.json --overwrite-development
```
- 输出默认保存至 `outputs/inference/{task}-{preset}.json`。

### 4. 基准评测 (`vla-sim evaluate`)

在指定 Benchmark 场景清单下执行全量或抽样评测：

```bash
# 查看帮助
vla-sim evaluate --help

# 单任务 Nominal 评测
vla-sim evaluate --task color_pick --preset nominal

# 全任务评测
vla-sim evaluate --task all --preset nominal
```

**Preset 支持矩阵**：

| Preset | Push | PickPlace | ColorPick | 说明 |
| --- | :---: | :---: | :---: | --- |
| `screen` | ✅ | ✅ | ✅ | 开发筛选子集（支持 `--overwrite-development` 覆盖） |
| `nominal` | ✅ | ✅ | ✅ | 标准基准评测集（Push: 50集, Pick: 50集, Color: 60集） |
| `randomized_screen` | ✅ | ✅ | ❌ | 带随机扰动的开发子集（ColorPick 无此项） |
| `randomized` | ✅ | ✅ | ✅ | 全域域随机化基准集 |
| `blind` | ✅ | ✅ | ✅ | 盲测集（严禁覆盖与抽样，必须全量执行） |

> [!WARNING]
> 当使用 `--task all` 时，不能使用 `randomized_screen`，因为 `color_pick` 不包含该 preset。

### 5. 流水线阶段执行 (`vla-sim pipeline`)

流水线按固定阶段编排：`collect` -> `build` -> `audit` -> `smoke`。

```bash
# 仅执行审计
vla-sim pipeline --from audit --through audit

# 执行审计并跑通训练冒烟
vla-sim pipeline --from audit --through smoke
```

### 6. 质量门槛验收 (`gates`)

当三任务 nominal 评测完成后，运行验收门禁：

```bash
python -m vla_sim.evaluation.gates \
  --checkpoint assets/policy/current/pretrained_model \
  --push outputs/evaluation/push-nominal.json \
  --pick outputs/evaluation/pick_place-nominal.json \
  --color outputs/evaluation/color_pick-nominal.json \
  --output outputs/evaluation/gate_report.json
```
- **门禁标准**：
  - Push: >= 45/50 成功率 (90%)
  - PickPlace: >= 45/50 成功率 (90%)
  - ColorPick: >= 54/60 成功率 (90%) 且每种颜色 >= 18/20，零错误抓取

---

## 目录结构

```
vla_ur5e_sim/
├── src/vla_sim/            # 核心 Python 源码
│   ├── cli.py              # vla-sim 统一命令行入口
│   ├── pipeline.py         # 流水线编排
│   ├── paths.py            # 仓库内资产解析
│   ├── simulation/         # 仿真环境、专家、采集、合约与数据审计
│   ├── policy/             # 训练器、LoRA 适配器、损失与推理运行时
│   └── evaluation/         # 三任务评测适配器、统计指标与门禁
├── assets/
│   ├── simulation/         # 仿真清单 (manifests) 与四个 1500/4500 规整数据集
│   └── policy/             # SmolVLA 基础模型与当前 004000 LoRA checkpoint
├── outputs/                # 训练、评测与流水线产物目录（已被 gitignore）
├── tests/                  # 单元测试与契约测试
└── docs/
    └── RUNBOOK.md          # 详细运维与运行手册
```

---

## 质量与测试

```bash
# 运行单元测试套件（89 个固定种子测试）
PYTHONPYCACHEPREFIX=/tmp/vla_sim_pycache python -m pytest -q

# 静态代码分析与格式检查
PYTHONPYCACHEPREFIX=/tmp/vla_sim_pycache python -m ruff check src tests
```

详细运维操作规范见 [docs/RUNBOOK.md](docs/RUNBOOK.md)。
