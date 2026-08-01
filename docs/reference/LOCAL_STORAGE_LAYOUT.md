# 本地保存结构与归档规则

本仓库将可版本化的代码、配置和说明文档与昂贵的本地实验资产分开保存。Git 仅追踪前者；
数据集、模型、checkpoint、rollout、日志和运行时缓存保留在工作区，不能因为整理文档而删除、
移动或提交。

## 目录职责

| 路径 | 内容 | Git 状态 | 保存规则 |
| --- | --- | --- | --- |
| `src/vla_sim/` | 环境、控制、策略、评测与数据契约 | 追踪 | 复用逻辑放在包内；脚本不复制实现。 |
| `scripts/` | 当前 PickPlace 的采集、训练、诊断、评测入口 | 追踪 | 根目录只保留当前入口；历史 Stack 脚本在 `scripts/archive/stack/`。 |
| `configs/` | 可复现实验配置和 benchmark manifest | 追踪 | manifest 不覆盖；新增 benchmark 使用新文件和明确角色。 |
| `tests/` | 固定种子的契约与回归测试 | 追踪 | 行为变化须添加或更新聚焦测试。 |
| `docs/` | 当前指南、实验登记、报告与历史归档 | 追踪 | 当前结论更新到注册表和 Runbook；历史证据移入 `archive/`。 |
| `data/` | 本地 LeRobot 数据集、采集产物与 manifest 副本 | 忽略 | 不删除、不重命名冻结数据集；新数据集使用任务和版本后缀。 |
| `outputs/` | checkpoint、rollout、日志、可视化和运行元数据 | 忽略 | 每次训练/评测使用新的运行目录，不覆盖冻结证据。 |
| `.runtime/` | Hugging Face、模型、Numba 和临时缓存 | 忽略 | 可按环境重建，但不纳入 Git。 |

## 推荐命名与保存单元

训练运行保存在 `outputs/<task>/<experiment>/<seed>/`，例如
`outputs/pick_place_v2_native_bin/vla_only_global_20k/seed1000/`。评测结果应写入同一实验目录
下的新文件，或新建 `outputs/<task>/<experiment>_evaluations/`；不要覆盖已登记结果。

每一个可报告实验至少保留以下可追溯信息：

- checkpoint 或模型路径；
- 数据集和 benchmark manifest 路径及 SHA-256；
- 完整命令、seed、代码提交和必要的 `source.patch`；
- 原始 rollout/日志与机器可读结果；
- 在 `EXPERIMENT_REGISTRY.md` 中的结论、角色和盲测消费状态。

## 文档流转

1. 先在 [实验注册表](EXPERIMENT_REGISTRY.md) 更新 benchmark、canonical checkpoint 和结论。
2. 将可重复操作更新到 [PickPlace v2 Runbook](../PICK_PLACE_V2_RUNBOOK.md)。
3. 将完整实验叙述记录在 `docs/reports/`，文件名包含日期。
4. 已不再代表当前入口的材料移入对应 `archive/`，并在 [文档索引](../README.md) 保留链接。

归档仅改变文档和脚本的导航，不改变任何 `data/`、`outputs/` 或 `.runtime/` 中的资产。
