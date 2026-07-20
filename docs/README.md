# 项目文档索引

本目录只保存当前有效文档和需要追溯的正式归档。训练 checkpoint、逐局 rollout、运行日志与机器可读统计保存在 `outputs/`，不与报告文档混放。

## 当前有效文档

| 类别 | 文档 | 用途 |
| --- | --- | --- |
| 当前最终报告 | [P3 消融与盲测最终实验报告](reports/P3_FINAL_EXPERIMENT_REPORT_20260720.md) | 当前 canonical、P3 消融、跨 seed 与 `test_v2` 最终结论 |
| 后续计划 | [下一阶段详细执行计划](plans/NEXT_PHASE_EXECUTION_PLAN_20260720.md) | 评测加固、数据改进、新候选验证与实机准备 |
| 实验注册 | [实验注册表](reference/EXPERIMENT_REGISTRY.md) | benchmark 身份、消费状态、canonical 和治理规则 |
| 上手资料 | [项目上手指南](reference/PROJECT_ONBOARDING_GUIDE.md) | 数据、训练和仿真 rollout 的基础流程 |

## 历史归档

- `reports/archive/`：2026-07-17、2026-07-19 的阶段实验报告和旧验收报告，只用于历史追溯，不代表当前结论。
- `plans/archive/`：已经执行完成或被后续计划替代的改进计划。

归档文档中的旧 benchmark、旧 checkpoint 和旧路径可能只具备历史意义。引用项目当前结论时，应以当前最终报告和实验注册表为准。

## 机器可读证据

- `outputs/p3_ablation_20260719/canonical_freeze_20260720.json`
- `outputs/p3_ablation_20260719/matrix_summary.json`
- `outputs/p3_ablation_20260719/p4_promotion.json`
- `outputs/p3_ablation_20260719/test_v2_C_canonical_seed1000.json`
- `outputs/p3_ablation_20260719/final_acceptance_20260720.json`

## 文档生命周期规则

1. 当前结论只保留一份主报告；新报告替代旧报告后，将旧报告移动到 `reports/archive/`。
2. 执行中的计划放在 `plans/`；完成或被替代后移动到 `plans/archive/`，并在标题附近写明状态。
3. benchmark、canonical checkpoint 和盲测消费状态只在实验注册表中维护唯一事实源。
4. 交接、临时说明和阶段状态文件在任务完成后删除，不进入正式归档。
5. 报告不得替代原始证据；原始 rollout、metadata、manifest、日志和 source patch 必须保留。
