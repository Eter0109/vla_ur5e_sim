# 项目文档索引

当前项目是 UR5e 双相机 PickPlace v2：按视觉完成红色方块抓取、移动和放入蓝色原生收纳盒，
并通过连续 10 步稳定放置验证。训练 checkpoint、逐局 rollout、运行日志与机器可读统计保存在
`outputs/`。

## 当前有效文档

| 类别 | 文档 | 用途 |
| --- | --- | --- |
| 实验注册 | [实验注册表](reference/EXPERIMENT_REGISTRY.md) | PickPlace benchmark 身份、盲测消费状态和治理规则 |
| 实验报告 | [PickPlace VLA 训练与评测报告](reports/PICK_PLACE_VLA_EXPERIMENT_REPORT_20260731.md) | 20k 训练配置、失败实验、校准迭代和 50 场结果 |
| 上手资料 | [项目上手指南](reference/PROJECT_ONBOARDING_GUIDE.md) | 双相机任务、系统架构和最小可复现路径 |
| 保存结构 | [本地保存结构与归档规则](reference/LOCAL_STORAGE_LAYOUT.md) | 代码、文档、数据、checkpoint、rollout 和缓存的保存边界 |
| 操作手册 | [PickPlace v2 Runbook](PICK_PLACE_V2_RUNBOOK.md) | 数据、训练、诊断、可视化和评测的标准命令 |

## 历史资料

- [`archive/stack/`](archive/stack/)：Stack Runbook、阶段交接和优化记录，仅用于追溯。
- `../scripts/archive/stack/`：保留的 Stack v1 历史脚本；不是当前入口，部分命令依赖旧仓库布局。
- `reports/archive/`：2026-07-17、2026-07-19 的 lift 阶段报告和旧验收报告，只用于历史追溯。
- `plans/archive/`：已经执行完成或被后续计划替代的改进计划。

归档文档中的旧 lift benchmark、checkpoint、数据集和路径不代表当前结论。

## 文档生命周期规则

1. 当前结论应优先更新至实验注册表和主操作文档；历史报告保留可追溯证据。
2. benchmark、canonical checkpoint 和盲测消费状态只在实验注册表中维护唯一事实源。
3. 报告不得替代原始证据；原始 rollout、metadata、manifest、日志和 source patch 必须保留。
4. `data/`、`outputs/` 和 `.runtime/` 是昂贵的本地资产，项目整理时不得删除。
5. `scripts/` 根目录只保留当前 PickPlace 入口；历史脚本必须放入明确的 `archive/`。
