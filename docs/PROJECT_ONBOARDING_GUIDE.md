# UR5e SmolVLA 机械臂方块抬升项目 - 从零上手指南 (Onboarding Guide)

本指南旨在帮助新加入项目的开发者或研究人员“开箱即用”地理解本项目的全貌、技术架构、算法细节及核心避坑经验。

---

## 1. 项目概览 (Project Overview)

本项目构建在 **NVIDIA Isaac Lab** 和 **Robosuite** 仿真引擎之上，核心任务是利用纯视觉（Vision-Based）端到端大模型，控制 **UR5e 机械臂** 及其端部的 **Robotiq 2F-85 双指夹爪**，准确识别、抓取桌面上的红色方块并将其抬升至少 10 厘米。

- **核心算法**: 使用了视觉-语言-动作 (VLA) 架构的开源模型 —— **SmolVLA**。
- **算力适配**: 为兼容消费级单卡 GPU 的显存瓶颈，我们在代码层面将 SmolVLA 的 Transformer 层数暴力裁剪到了 **16 层**，以此作为我们微调训练的基础（Base Model）。
- **任务验收指标**: 在未见过（Held-out）的 50 局随机测试集中，闭环抓取成功率必须 $\ge 60.0\%$（最终已达标至 $70.0\%$）。

---

## 2. 核心架构与目录结构 (Codebase Structure)

整个代码仓库（Workspace）的功能划分如下：

```text
D:\vla_ur5e_sim\
├── configs/            # 模型/环境的静态 JSON 配置
├── data/               
│   ├── manifests/      # 定义每局（Episode）初始条件的清单文件 (train/val/test)
│   └── lerobot/        # LeRobotDataset 格式存储的 500 局专家演示录像
├── docs/               # 存放验收报告、学习总结及本文档
├── outputs/            # 训练中途产出的 Checkpoint 权重及评估结果的 JSON 报告
├── scripts/            # 核心流水线启动脚本（重点，下文详解）
├── src/vla_sim/        # Python 核心业务逻辑层 (仿真环境、损失重写、时序推理等)
└── tests/              # 针对时序算法、损失权重修改的 pytest 单元测试
```

---

## 3. 从零打通全链路 (End-to-End Pipeline)

本项目分为三个核心步骤：**数据收集 $\rightarrow$ 模型微调 $\rightarrow$ 闭环仿真评估**。

### 3.1 第一步：数据生成 (Data Collection)
- **数据来源**：项目的训练数据（`expert_500demos`）并不是人工佩戴 VR 设备手把手录制的！
- **实现原理**：我们在 `scripts/collect_demos.py` 和 `vla_sim.sim.HeuristicLiftExpert` 中，利用运动学启发式算法（硬编码了如何接近、夹紧、抬升）在仿真器中自动跑出 500 局完美操作，并以此作为供 VLA 模型模仿的“标准教材”。
- **运行命令**:
  ```powershell
  python scripts/collect_demos.py --manifest data/manifests/train.json --root data/lerobot/expert_500demos
  ```

### 3.2 第二步：大模型微调 (Model Fine-Tuning)
早期的“3K 基线模型”因训练步数不够和 Loss 权重不当，导致经常在半空中“瞎闭爪”。经过我们的魔改，以下是**最终能达到 70% 成功率的训练最佳实践**：
- **微调超参数**：Batch Size 扩大至 **8**，迭代步数跑满 **10,000 步**，学习率压低至 **2e-5** 以确保稳定收敛。
- **Loss 维度干预（核心）**：在 `src/vla_sim/losses.py` 中，我们剥夺了模型猜测“方块旋转角度”的误差权重（从 $1.0$ 降至 $0.01$），同时把“爪子该开还是该合”的损失权重加倍（从 $1.0$ 提至 $2.0$）。这迫使模型放弃去拟合对称物体的无用角度，把全部算力聚焦在“位置”和“夹紧”上。
- **启动训练命令**:
  ```powershell
  pwsh scripts/train_smolvla.ps1 -BatchSize 8 -Steps 10000 -Seed 1000 -OutDir "outputs/smolvla_efficiency_fix_10k"
  ```

### 3.3 第三步：闭环推理评估 (Simulation Rollout)
把模型挂在仿真环境里跑闭环，是检验抓取能力的唯一标准。
- **启动评估及 3D 可视化命令**:
  ```powershell
  python scripts/run_rollouts.py `
    --checkpoint outputs/smolvla_efficiency_fix_10k/checkpoints/010000/pretrained_model `
    --dataset-root data/lerobot/expert_500demos `
    --repo-id local/ur5e_custom_lift `
    --manifest data/manifests/test.json `
    --episodes 5 `
    --horizon 200 `
    --temporal-ensemble `
    --replan-steps 4 `
    --temporal-ensemble-decay 0.5 `
    --samples-per-plan 1 `
    --gripper-mode confirm `
    --gripper-confirm-steps 2 `
    --render   # 加上此参数会弹框显示 3D 可视化动画，方便直观 debug
  ```
- **Sim2Real 推理侧的“防抖”黑科技（详见 `src/vla_sim/temporal.py`）**:
  为了应对神经网络单帧计算可能产生的“抽风闪烁”，我们加了两个物理层面的软拦截：
  1. **Temporal Ensemble (时序衰减融合)**：模型一帧会输出未来多步轨迹。我们取“过去预测的轨迹”与“现在预测的轨迹”做指数衰减叠加求均值，强行抹平了移动过程中的抖动。
  2. **Gripper Confirm (连续确信机制)**：强行要求 `gripper-confirm-steps = 2`。模型必须**连续 2 帧**输出闭爪指令，物理引擎才会真正闭合。这就如同给爪子装了“防误触弹簧”，彻底根治了早期模型还没碰上物体就乱夹空气的毛病。

---

## 4. 最佳实践与踩坑总结 (Takeaways)

1. **别被无关的损失函数（Loss）绑架**：处理对称或旋转无关物体时，如果任由模型去拟合所有的 Action 维度，微小的旋转误差梯度可能像海啸一样毁掉核心抓取通道。及时修改代码加 Mask 或调小权重。
2. **算力即正义，Epochs 要管够**：VLA 模型的微调很吃数据分布，初期不收敛不要立刻怀疑代码写错了，先看看显存是不是太小导致 Batch Size 太低，或者 3000 步是不是根本没让它“背熟” 500 个 Demo。
3. **永远不裸控，中间要滤波**：实机部署或高仿真中，模型原始输出和物理执行机构之间**必须有一层类似 Temporal Ensemble / Debounce 的软性缓冲网**，这是提高成功率最低成本的方法。
