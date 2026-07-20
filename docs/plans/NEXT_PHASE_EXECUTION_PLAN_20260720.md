# UR5e SmolVLA 下一阶段详细执行计划

- 制定日期：2026-07-20
- 当前基线：P3 canonical C，seed=1000，15k，peak LR=`5e-5`，无转折帧过采样
- 当前结果：`test_v2` 81/100，Wilson 95% 下界 72.2%，峰值显存 1.16 GiB
- 计划状态：待执行
- 优先级：评测流程可靠性 > 失败阶段数据改进 > 新候选训练 > sim-to-real 准备

## 1. 目标与边界

下一阶段不再围绕已消费的 `test_v2` 做局部调参，而是建立更严格的评测基础设施，使用新的 development 场景验证针对性改进，并在完整冻结后使用新生成的盲测集做一次性验收。同时开始把仿真控制接口、观测和安全约束整理成可迁移到真实 UR5e 的形式。

量化目标如下：

1. 所有评估在统计前自动验证场景数、唯一 scene ID、policy seed、manifest 指纹、checkpoint 指纹和 metadata 完成状态。
2. 建立独立的 `latency_benchmark_v1`，在同硬件、同进程模式、同场景负载下保留 canonical baseline，后续候选 p95 退化不得超过 10%。
3. 在新的 `validation_v3` 上把三 seed 中位成功率保持在至少 80%，最差 seed 至少 75%，极差不超过 10 个百分点。
4. 相比冻结 canonical，新候选重点降低 `no_grasp` 与 `grasp_no_lift`，且不得以显著增加夹爪抖动换取成功率。
5. 保持模型规模不超过 3B、峰值显存不超过 6 GiB，并继续支持 RTX 3060 Laptop 单 GPU 串行训练。

本阶段不包含未经安全评审的真实机械臂自主闭环抓取，不把历史 `test.json` 或已经消费的 `test_v2` 重新标为盲测，也不根据 `test_v2` 单独场景做 scene-specific 调参。

## 2. 不可变基线与数据治理

- canonical checkpoint、控制器、配置与代码指纹以 `outputs/p3_ablation_20260719/canonical_freeze_20260720.json` 为准。
- `test_v2` 已于 2026-07-20 完整消费一次，后续只可用于审计，不得重跑、续作开发集或参与模型选择。
- `validation_v2` 已参与 A/B/C/D 与三 seed 选择，降级为历史回归集；新一轮主要选择必须使用新生成并冻结的 `validation_v3`。
- `ood_v1` 继续作为诊断集；只有环境 override、物理和相机分层语义完成后才启用，不参与主模型选择。
- 新的 `test_v3` 必须在候选、控制器、延迟基线、代码和环境 preset 全部冻结后才生成或解封；只允许一次完整运行。
- `test_v2` 的失败分布只用于确定总体改进方向，不针对其具体 scene ID 增加训练样本或规则。

## 3. 阶段 N0：文档和实验状态收口

状态：本轮已完成。

交付物：

- 当前最终报告、机器可读验收 JSON 和 canonical freeze 清单。
- 实验注册表中的 `test_v2` 消费状态与 canonical 记录。
- 当前、归档、计划、参考四类文档目录和统一索引。

退出条件：当前结论只有一个权威入口，旧报告明确归档，临时交接和状态说明已清理。

## 4. 阶段 N1：评测流程加固

这是下一步的最高优先级，完成前不启动新的训练矩阵。

### N1.1 产物完整性检查器

新增统一检查命令，至少验证：

- rollout 条目数等于 manifest 预期场景数；scene ID 唯一且集合完全一致。
- 每局 env seed、policy seed 与 manifest/固定派生规则一致。
- `.meta.json` 状态为 `completed`，completed scene ID 与结果 JSON 一致。
- checkpoint、dataset、manifest、experiment config 的 SHA-256 与启动时记录一致。
- 同一输出中不存在重复 scene、越界 scene 或不同 checkpoint/config 混跑。

晋级脚本必须先调用完整性检查器；任何一组不是完整的 40/40 或未来规定场景数时，直接拒绝计算 promotion gate。特别要覆盖“metadata 已推进但结果 JSON 尚未落盘”和技术中断后的安全 `--resume`。

### N1.2 原子写入与恢复

- rollout 结果先写临时文件，完成 fsync/关闭后再原子替换正式 JSON。
- metadata 明确区分 `initializing`、`running`、`interrupted`、`completed`、`failed`。
- 当前场景开始不应提前计入 completed scene ID；只有结果成功持久化后才更新。
- 恢复只运行缺失 scene，且启动前再次校验所有不可变输入指纹。
- 添加 2–4 场景的故障注入测试：正常完成、评估中断、结果写入中断、错误 checkpoint 恢复。

### N1.3 统计与报告统一

单次命令生成原始 JSON、metadata、summary JSON、summary Markdown 和可选配对报告。统计固定包含：成功数、Wilson 95% 区间、阶段漏斗、失败阶段、夹爪切换、episode wall time、模型推理 p50/p95、峰值显存和配对 McNemar。

退出门槛：单元测试与小规模集成测试全部通过，同一 checkpoint/config/manifest 重跑时 scene ID、seed 和布尔结果一致；不完整结果无法进入模型选择。

## 5. 阶段 N2：建立正式延迟基线

创建与任务成功率 benchmark 分离的 `latency_benchmark_v1`，避免场景难度和 episode 长度改变延迟结论。

固定条件：

- RTX 3060 Laptop GPU、电源模式、CUDA/Python/torch 版本与图像尺寸固定。
- 先进行不少于 20 次 warm-up，再测不少于 200 次真实 policy forward。
- 输入张量形状、batch、chunk、action steps、temporal ensemble 和后处理固定。
- 同时记录纯模型 forward、完整 policy 调用和控制循环端到端延迟。
- 每次运行记录 GPU 型号、显存、温度/功耗可用信息、进程启动模式和依赖版本。

canonical seed=1000 先运行三次建立 baseline，报告中位 p95 和最差 p95。未来候选用同一工具比较，门槛为中位 p95 退化≤10%、峰值显存≤6 GiB。

退出门槛：同一 canonical 三次 p95 极差≤5%；若超过 5%，先处理系统噪声，不进入训练阶段。

## 6. 阶段 N3：新开发集和失败阶段诊断

根据盲测聚合结果，当前 19 个失败由 `no_grasp=9` 和 `grasp_no_lift=10` 构成。下一轮分别建立两个诊断切片，但不复制 `test_v2` 的具体场景。

### N3.1 `validation_v3`

建议至少 60 局：

- 20 局常规分布，用于保持总体能力。
- 20 局 approach/grasp 边界分布，覆盖横向偏移、接近角度、视觉边界和初始末端位姿变化。
- 20 局 grasp/lift 稳定性分布，覆盖摩擦、质量、尺寸和夹取位置的小范围随机化。

manifest 创建后记录 SHA-256、generator seed 和环境 preset；在训练启动前冻结。若环境暂不支持物理 override，应先实现并测试 override，不用控制器硬编码模拟。

### N3.2 诊断输出

新增接近误差、首次闭合距离、连续抓持步数、首次稳定抓持至抬升延迟、抬升速度、滑落时刻和夹爪命令抖动等指标。目标是区分视觉/接近问题、闭合时机问题、抓持稳定性问题和抬升轨迹问题。

退出门槛：canonical 在 `validation_v3` 完成完整基线评估，所有切片有独立统计，不使用 `test_v2` 做验证。

## 7. 阶段 N4：数据与控制器改进实验

训练数据优先于扩大模型。模型仍保持当前 SmolVLA 规模和 15k 训练预算，第一轮只比较最多四个候选：

| 候选 | 数据 | 控制器 | 目的 |
| --- | --- | --- | --- |
| R0 | 当前数据 | canonical | `validation_v3` 冻结基线 |
| R1 | 增加 approach/grasp 边界示范 | canonical | 降低 `no_grasp` |
| R2 | 增加稳定抓持与抬升示范 | canonical | 降低 `grasp_no_lift` |
| R3 | R1+R2 数据 | canonical | 检查联合收益 |

数据要求：

- 新示范按失败阶段和场景桶标注来源，保持 train/development 场景隔离。
- 对 grasp/lift 示例增加闭合后稳定持握和持续抬升片段，避免只学习短暂抓住。
- 审计动作范围、图像同步、末端位姿和夹爪状态；异常轨迹不得进入训练集。
- 记录新旧样本比例，不使用无界过采样；若要改变采样，只能作为下一轮独立变量。

选择规则：首先比较 `validation_v3` 总成功数，其次比较两个失败切片；差距≤1/60 时优先数据更少、控制器不变、推理更快的方案。不得同时改变学习率、采样、数据和控制器后把收益归因于单一因素。

## 8. 阶段 N5：跨 seed、冻结与新盲测

胜出配置追加 seed=1001、1002，三 seed 均在 `validation_v3` 评估。晋级门槛：

- 三 seed 中位成功率≥80%。
- 最差 seed≥75%。
- 极差≤10 个百分点。
- 两个失败切片均不得比 R0 明显退化；夹爪切换 p95 不得增加超过 20%。
- 正式延迟 p95 相对 canonical baseline 退化≤10%，显存≤6 GiB。

通过后冻结 checkpoint、训练 manifest、source patch、controller/config、代码提交、环境 preset、`validation_v3` 汇总和延迟报告。随后才允许运行一次新 `test_v3`。建议主验收仍使用至少 100 局，成功率目标≥80%、Wilson 95% 下界≥70%，并保持延迟和显存门槛。

若未通过，保留为 development candidate，不生成或不解封 `test_v3`，也不修改已归档的 `test_v2` 结论。

## 9. 阶段 N6：sim-to-real 准备

在仿真新候选晋级后开始，先做接口与安全验证，不直接开放全自主抓取。

- 固定真实 UR5e 控制频率、关节/笛卡尔速度和加速度上限、工作空间边界、急停和 watchdog。
- 建立仿真动作到真实控制命令的单位、坐标系、姿态表示和夹爪方向映射测试。
- 完成相机内外参标定、时间同步、图像裁剪和归一化一致性检查。
- 先离线回放真实图像与状态，再 shadow mode，再低速空载动作，最后进行受限物体抓取。
- 建立真实任务的独立 manifest 和人工安全检查表；真实结果不得与仿真成功率直接合并。

实机准入门槛：100 次无物体低速控制循环无越界命令，watchdog/急停验证通过，坐标变换误差在预设阈值内，操作者能够随时接管。

## 10. 预计资源与顺序

| 阶段 | 预计耗时 | GPU 预算 |
| --- | ---: | ---: |
| N1 评测流程加固 | 1–2 天 | <1 小时 |
| N2 延迟基线 | 0.5 天 | <1 小时 |
| N3 新开发集与基线 | 1–2 天 | 1–2 小时 |
| N4 四候选训练评估 | 2–4 天 | 18–28 小时，单 GPU 串行 |
| N5 跨 seed 与一次盲测 | 2–3 天 | 12–20 小时 |
| N6 sim-to-real 准备 | 3–7 天 | 以实机安全验证为主 |

严格执行顺序为 N1→N2→N3→N4→N5→N6。N1/N2 未完成前不启动新训练；N5 未通过前不运行新盲测；仿真和接口安全门槛未通过前不进行真实物体自主抓取。

## 11. 立即执行清单

1. 为 rollout 与 promotion 增加完整性检查器和自动化测试。
2. 修复 metadata/result 原子写入及安全 resume 的状态语义。
3. 实现 `latency_benchmark_v1` 并对 canonical 运行三次冻结 baseline。
4. 定义 `validation_v3` 场景分层与环境 override schema。
5. 完成 canonical 的 `validation_v3` 基线后，再决定新增示范数量和 R1/R2/R3 的训练启动时间。

每完成一个阶段，更新实验注册表和本计划状态；不要再创建独立的临时状态、说明或交接文档。
