# UR5e SmolVLA 项目改进计划（评测优先）

- 日期：2026-07-19
- 依据：`../../reports/archive/EXPERIMENT_REPORT_20260719.md`、历史实验/验收报告及本地训练与 rollout 产物
- 当前状态：已完成；最终结论见 `../../reports/P3_FINAL_EXPERIMENT_REPORT_20260720.md`
- 本轮预算：约 24 GPU 小时（RTX 3060 Laptop，6 GiB）

## 1. 执行摘要

2026-07-19 的冷启动 15k checkpoint 在现有固定场景上得到 Validation `25/30 (83.3%)`、Test `36/50 (72.0%)`，原始 JSON、训练 manifest 与报告数值一致。该结果证明：从纯净裁剪版 `smolvla_base` 冷启动存在可达到较高闭环成功率的配置。

但是，该模型相对历史最佳只分别多成功 1 局（Validation `24 -> 25`、Test `35 -> 36`）。在同一 50 个场景上，最新模型新增成功 3 局、退化 2 局；配对 McNemar 精确检验 `p=1.0`。`36/50` 的 Wilson 95% 区间约为 `58.3%–82.5%`，尚不能以 95% 置信下界证明真实成功率超过 60%。

此外，当前 `test.json` 已在多轮候选模型评估中反复使用，不能继续视为严格的 held-out 测试集。因此，本计划首先建立可复现、可比较、可一次性解封的评测协议，再以等因素消融和跨 seed 复现选择模型。当前 15k checkpoint 的正确定位是：**复用开发基准上的最佳候选**，而非“无争议 SOTA”。

## 2. 当前实验审计

### 2.1 已确认的事实

| 项目 | 已确认结果 |
| --- | --- |
| 训练 | 从 `.runtime/models/smolvla_base` 冷启动，seed=1000，FullExpert，15,000 steps，batch=8 |
| 训练参数 | peak LR `5e-5`、warmup=1,500、decay floor=`1e-6`、rotation/gripper loss=`0.01/2.0` |
| 采样 | 转折帧 3x 过采样、window=5 |
| 推理 | Temporal Ensemble：`replan_steps=4`、`decay=0.5`；Gripper Confirm：2 步 |
| 指标 | Validation `25/30`；复用 test `36/50`；10k checkpoint validation `23/30` |
| 训练来源 | run manifest 记录了 commit、参数与 source patch，但运行时工作树为 dirty |

### 2.2 证据不足的报告表述

1. **“无争议 SOTA”**：最新和历史模型在同一 50 局上的净增仅 1 局，置信区间高度重叠，未达到统计显著。
2. **“test 为 held-out”**：该清单已至少被 6 个完整评测复用；历史验收报告也要求后续不要使用它做模型选择。
3. **“过采样起决定性作用”**：此前 12k 冷启动已使用相同的 3x/window=5 采样。最新方案同时改变了峰值学习率、warmup、退火下限与训练时长，缺少等因素对照。
4. **“15k 排除过拟合”**：10k 到 15k 在同一 validation 上仅由 3 个转好、1 个转坏场景组成；这只能说明观察到小幅净增，不能证明不存在过拟合。

### 2.3 当前失败模式

最新 test 的 14 个失败局可按现有产物分为：

| 失败阶段 | 数量 | 现象 |
| --- | ---: | --- |
| 未接近阈值 | 10 | 提前闭爪、末端偏移，或接近过程未完成 |
| 已接近但未抬升 | 2 | 出现接近/闭爪，但最大抬升不足 1 cm |
| 抬升未保持 | 2 | 超过 10 cm，但仅保持 9 步或 2 步 |

成功局平均夹爪切换约 1.83 次，失败局约 17.86 次。当前 `confirm` 模式会在单次 open 预测后立即重新打开夹爪，可能导致失败轨迹中的高频开合。与此同时，现有 `approach_success` 使用三维距离小于 3 cm 的启发式定义，存在成功局仍被记为“未接近”的情况；后续诊断必须依赖真实抓取和阶段状态，而不是单一距离阈值。

## 3. 本轮目标与范围

### 3.1 目标

1. 将评测集角色、模型选择和一次性盲测严格分离。
2. 让训练、推理、环境与评测配置拥有单一真实来源，并保留 `legacy_20260719` 的精确复现能力。
3. 用学习率峰值与转折帧采样的 2x2 消融，确定当前改进真正来自哪里。
4. 验证胜出配置在 3 个训练 seed 上的稳定性。
5. 用新的 100 局盲测给出可信的发布结论。
6. 将失败诊断升级为“对准—抓取—抬升—保持”的阶段漏斗，用结果指导下一轮数据或控制改进。

### 3.2 本轮不包含

- 多相机或 wrist camera 训练。
- 实机部署、接触力传感器、阻抗控制的大规模改造。
- 重新定义历史基线的仿真动力学。
- 使用新盲测集进行模型或控制器调参。

多视角和力控仅在单视角数据与控制改造后仍无法达到目标，或 OOD 诊断明确显示相机遮挡/摩擦是主导问题时启动。

## 4. 评测协议 v2

### 4.1 实验配置

建立 schema v2 的版本化实验配置，覆盖以下段落：

- `experiment_id`、`schema_version`、`code_preset`
- 基座模型、checkpoint、数据集与内容哈希
- 训练参数、scheduler 语义、采样策略与训练 seed
- 环境 preset、推理控制器、policy seed 策略
- benchmark 角色、manifest、比较基线与验收阈值

训练 PowerShell 包装脚本和 rollout 脚本均应支持 `--config`/`-Config`。显式 CLI 参数允许覆盖配置，但必须写入最终 resolved manifest。现有命令行参数保持兼容。

必须保留 `legacy_20260719` preset：它冻结当前有效的环境默认值、scheduler 语义和推理参数。后续修正 scheduler 或让旧 YAML 生效时，只能以新 preset 发布，不能静默改变历史结果。

### 4.2 Benchmark Manifest v2

新 manifest 使用带元数据的对象格式，旧的列表格式继续只读兼容。新格式至少包含：

```json
{
  "schema_version": 2,
  "benchmark_id": "validation_v2",
  "role": "development",
  "generator_seed": 45000,
  "environment_preset": "legacy_20260719",
  "scenes": [
    {
      "scene_id": "validation-v2-0000",
      "env_seed": 45000,
      "x_m": 0.01,
      "y_m": -0.02,
      "yaw_rad": 0.0,
      "overrides": {}
    }
  ]
}
```

场景几何、环境随机性和 policy 随机性必须解耦：几何写入 manifest，`env_seed` 只控制环境，`policy_seed` 由评测配置提供。盲测中每个场景只使用预先派生的一个 policy seed；开发集的固定子集额外使用多个 policy seed 估计采样方差。

### 4.3 Rollout Result v2 与 provenance

每局结果除现有字段外，必须记录：

- `ever_grasped`、`first_grasp_step`、`max_consecutive_grasp_steps`
- XY 对准误差、Z 偏差、首次闭爪时距离
- 原始模型夹爪信号和执行夹爪信号的切换次数
- `max_lift_m`、`max_success_hold_steps`
- `failure_stage`：`no_grasp`、`grasp_no_lift`、`lift_no_hold`、`success`
- 实际模型调用的延迟分位数、episode 墙钟时间与峰值显存

评测开始前立即原子写入 metadata，状态为 `running`；每局完成后更新进度；正常结束写为 `completed`，异常结束写为 `interrupted`。恢复评测时只运行尚未完成的场景，不可重复或替换已记录场景。metadata 必须包含 checkpoint、预后处理器、数据集、配置、manifest、Git source/patch、Python/CUDA/依赖/硬件的内容指纹。

### 4.4 统计输出

新增汇总工具自动生成 JSON 和 Markdown 摘要，至少输出：

- 成功率、Wilson 95% 区间
- 相对冻结基线的逐场景配对差值、McNemar 精确检验
- 训练 seed 的均值、中位数、最差值和极差
- 对准、抓取、抬升、保持阶段的转化率
- 按位置半径、象限与 OOD 条件的分桶表现
- 模型调用 p50/p95 延迟和显存

实验注册表保存每个正式 run 的配置、哈希、指标、置信区间、benchmark 角色和结论状态；原始逐局数据仍保留在 `outputs/`。

## 5. 新评测集与治理规则

| 集合 | 场景数 | 角色 | 使用规则 |
| --- | ---: | --- | --- |
| 现有 validation/test | 30 / 50 | development-regression | 可用于回归和控制器筛选，不得再称 blind/held-out |
| `validation_v2` | 40 | development | 可重复用于配置选择与跨 seed 评测 |
| `test_v2` | 100 | blind | 候选、checkpoint、控制器冻结后，仅运行一次 |
| `ood_v1` | 40 | diagnostic | 只诊断泛化，不决定本轮晋级 |

`validation_v2` 使用 generator seed 45000；`test_v2` 使用 seed 46000；`ood_v1` 使用 seed 47000。创建后记录 SHA-256。`test_v2` 不得被用于训练时长、学习率、采样、控制器或 checkpoint 的选择。

`ood_v1` 包含 4 个各 10 局的分层：工作区外沿、摩擦缩放 0.7/1.3、方块尺寸缩放 0.8/1.2、相机位姿平移 ±2 cm/旋转 ±5°。本轮仅在 ID 盲测完成后运行。

## 6. 工程实施阶段

### P0：冻结历史行为并修订实验定位

1. 将现有 15k checkpoint 标记为 `candidate_dev_20260719`。
2. 固定其系统配置：`replan_steps=4`、`temporal_ensemble_decay=0.5`、`gripper_mode=confirm`、`gripper_confirm_steps=2`。
3. 以 `legacy_20260719` 保存当前实际环境与 scheduler 行为。
4. 更新实验报告，加入 Wilson 区间、配对比较、test 复用说明、单 seed 限制与 dirty source 说明。
5. 将“无争议 SOTA”改为“当前复用开发基准的最高点估计”。

完成条件：能从单一 preset 重建当前命令；历史 test 不再被标为 held-out。

### P1：配置、manifest、结果与来源证明

1. 实现实验配置 schema v2 及 legacy 兼容加载。
2. 实现 manifest v2 和生成器；旧 manifest 保持可读。
3. 为 rollout 增加独立 policy seed、状态化 metadata 与无重复 resume。
4. 实现 rollout summary 与实验注册表。
5. 同步 README，删除指向旧 checkpoint、旧数据集和未生效 YAML 的“最佳实践”描述。

完成条件：训练和评测的 resolved config、完整命令、环境指纹、数据/模型/manifest 哈希均被保存；中断后的结果仍可追溯。

### P2：物理阶段诊断与控制器

1. 环境每步暴露 `grasped`、`lifted`、当前抬升高度等状态；fake backend 保持兼容。
2. rollout 从真实抓取状态生成失败阶段，而不再以 3 cm 距离作为唯一判据。
3. 增加 `confirm_then_hold` 控制器：连续 2 次闭爪确认后，强制保持 4 或 8 步，再允许重新打开。
4. 保留 `confirm`、`latch` 等既有模式和行为，避免破坏历史复现。

控制器选择流程：先在旧 development-regression 集筛选 `confirm2`、`confirm_then_hold4`、`confirm_then_hold8`；再用 `validation_v2` 验证。新控制器仅在成功数至少增加 `2/40`、`no_grasp` 不增加且失败局夹爪切换下降至少 50% 时替换 `confirm2`。

### P3：学习率峰值与转折采样消融

所有配置固定为：FullExpert、batch=8、15,000 steps、warmup=1,500、decay floor=`1e-6`、rotation/gripper loss=`0.01/2.0`、seed=1000、legacy scheduler 语义和固定推理配置。

| 组别 | Peak LR | 转折帧采样 | 状态 |
| --- | ---: | --- | --- |
| A | `2e-5` | 1x | 新训练 |
| B | `2e-5` | 3x / window=5 | 新训练 |
| C | `5e-5` | 1x | 新训练 |
| D | `5e-5` | 3x / window=5 | 复用当前 15k checkpoint |

选择规则：

1. 四组只能通过 `validation_v2` 选择。
2. 若最高组与其他组相差不超过 `1/40`，视为持平。
3. 持平时优先无过采样，再优先较低 peak LR，减少训练分布扭曲和优化风险。
4. 同时报告转折帧样本比例、阶段转化率和失败分类，不允许只报告总成功率。
5. 现有 5k/10k/15k checkpoint 补齐同一 development 集上的学习曲线；该曲线仅为观察证据，不用于声称“排除过拟合”。

### P4：跨训练 seed 复现

对 P3 胜出配置追加 seed=1001 和 seed=1002。三 seed 都在 `validation_v2` 上评估，固定相同场景与 policy seed。另在 10 个固定场景上运行 3 个 policy seeds，以区分训练波动与推理采样波动。

晋级条件：

- 三 seed 中位数至少 70%。
- 最差 seed 至少 60%。
- 三 seed 极差不超过 10 个百分点。

未通过时不将新配置送入盲测；保留 `candidate_dev_20260719`，并在报告中记录“不稳定，不晋级”。

### P5：一次性盲测与发布结论

冻结以下项目后，才解封 `test_v2`：训练配置、canonical seed=1000 checkpoint、推理参数、控制器版本、代码提交和环境 preset。

对当前 15k 基线与新候选使用同一个 100 局 `test_v2`。若新候选就是当前 checkpoint，则只运行一次。技术故障可以按原配置恢复未完成场景，但不得更换模型、参数、seed 或 benchmark。

主验收：

- 至少 `70/100` 成功。
- Wilson 95% 下界不低于 60%。
- p95 模型调用延迟相对冻结基线退化不超过 10%。
- 峰值显存不超过 6 GiB。

只有同时满足下列条件才能使用“SOTA”措辞：

- 相对当前 15k 冻结基线绝对提升至少 5 个百分点。
- 同场景 McNemar 精确检验 `p < 0.05`。

满足主验收但不满足 SOTA 条件时，结论只能为“通过盲测的发布候选”。

### P6：根据盲测失败阶段决定下一轮数据方向

若 `no_grasp` 仍占失败的一半以上，下一轮新增 160 条只来自新 train seeds 的演示：

- 80 条工作区外沿场景。
- 40 条与 development 集失败类型匹配的全新场景。
- 40 条均匀分布控制场景。

采集必须保存 scene ID、环境参数、专家成功/失败、重试次数及失败原因。训练集只使用成功轨迹，但失败尝试不得再静默丢失。

若 `lift_no_hold` 成为主要失败阶段，再优先研究较长的非永久夹爪保持、抓取反馈和阻抗控制。多视角仅在上述单视角数据方案后仍无法在盲测达到 75%，且 OOD 显示视觉遮挡为主导问题时启动。

## 7. 测试计划

### 7.1 单元测试

- 配置解析、CLI 覆盖优先级及 resolved config。
- manifest v1/v2 兼容和 schema 校验。
- 几何、环境、策略随机种子解耦。
- legacy 与新 scheduler 的端点和 warmup 行为。
- 转折帧阈值、窗口边界与采样占比。
- `confirm_then_hold` 的关闭、保持、重新打开状态转换。
- 真实“抬升 10 cm 且保持 10 步”的成功正例。
- Wilson 区间、McNemar 与配对统计。
- metadata 启动写入、中断状态和无重复 resume。
- 失败阶段自动分类。

### 7.2 集成测试

- 固定 5 个场景重复运行，逐局成功结果和执行动作一致。
- 中断后续跑不重复已完成场景。
- checkpoint、配置、数据集和 manifest 哈希完全匹配。
- 真实 robosuite smoke 增加初始化、首帧和首步超时诊断。
- 原始策略夹爪输出与滤波后的系统输出可分别汇总。

### 7.3 工程门禁

- pytest 全部通过。
- Ruff 无告警。
- PowerShell AST 语法通过。
- 官方实验使用干净源码快照；如有 patch，必须保存可重放 patch 和完整哈希。

## 8. GPU 时间预算

| 工作项 | 预计 GPU 时间 |
| --- | ---: |
| P3 三个缺失 15k 消融训练 | 约 12.5 小时 |
| P4 胜出配置的两个复现 seed | 约 8.5 小时 |
| validation、控制器筛选与 policy-seed 稳定性检查 | 约 1–2 小时 |
| 基线/候选 100 局一次性盲测 | 约 2 小时 |
| OOD 诊断 | 使用剩余时间；必要时顺延，不影响主验收 |

若预算不足，优先级为：P0/P1/P2 -> P3 -> P4 -> P5 -> OOD。不能为了节省时间跳过跨 seed 稳定性和盲测。

## 9. 风险与回退策略

| 风险 | 控制措施 | 回退策略 |
| --- | --- | --- |
| 修正 scheduler 改变历史结果 | 保留 `legacy_20260719` | 新 scheduler 仅作为显式新 preset |
| 陈旧 YAML 与真实环境不一致 | 先固化有效默认值 | 不让旧 `sim.yaml` 直接改变历史 benchmark |
| 新控制器掩盖模型错误 | 同时报 raw-policy 和 system 指标 | 未满足阶段指标则保留 confirm2 |
| 盲测泄漏 | 解封前冻结配置和 SHA | 泄漏后废弃并生成新的 test_v3 |
| 新配置跨 seed 不稳定 | 三 seed 门槛 | 不晋级，不触发盲测 |
| 仿真初始化过慢/超时 | 分阶段超时和状态化 metadata | 从未完成场景恢复，不改变实验参数 |
| 预算超限 | 分阶段 gate | OOD 顺延，主验收不降级 |
| 当前 checkpoint 来自 dirty source | 保存 source patch、hash 与 legacy preset | 正式发布前从干净源码快照复现 |

## 10. 交付物清单

1. 评测协议和实验配置 schema v2。
2. `legacy_20260719` 与新候选 preset。
3. `validation_v2`、冻结的 `test_v2`、`ood_v1` manifest。
4. rollout 物理阶段诊断、状态化 metadata 和统计汇总工具。
5. 控制器筛选结果与 2x2 消融报告。
6. 三 seed 稳定性报告。
7. 盲测报告，明确标记为“发布候选”“未晋级”或“SOTA”。
8. 更新后的 README、实验注册表和测试覆盖说明。

## 11. 完成定义

本计划完成并不等同于“模型已经 SOTA”。完成是指：评测协议可执行、历史结果已正确定位、消融和跨 seed 结论可复现、新盲测只被使用一次、报告措辞与统计证据一致，并且下一轮数据或控制改进能由阶段诊断直接驱动。
