# P3 消融与盲测最终实验报告

日期：2026-07-20  
状态：已完成 P3 矩阵、跨 seed 晋级与唯一一次 `test_v2` 盲测。

## 结论

胜出配置为 C：peak LR=`5e-5`、transition oversample=`1x`、window=`0`，即无转折帧过采样。它在开发集四组矩阵中取得最高 `36/40` 成功，并通过三训练 seed 的稳定性门槛。冻结 seed=1000 的 15k checkpoint 后，唯一一次 100 局盲测取得 `81/100` 成功，Wilson 95% 下界为 `72.2%`，成功率和统计下界均超过主验收要求。

## 协议与冻结

- 模型选择只使用 `validation_v2`（40 局 development）；`test_v2` 在冻结前未被使用。
- canonical：`outputs/smolvla_ablation_c_15k_seed1000/checkpoints/015000/pretrained_model`。
- 训练：seed=1000，15k steps，peak LR=`5e-5`，full expert，transition oversample=`1x/window=0`。
- 控制器：temporal ensemble（replan=4、decay=0.5）与 confirm gripper（confirm=2、hold=4）。
- 冻结清单：`outputs/p3_ablation_20260719/canonical_freeze_20260720.json`；其中记录 checkpoint、配置、source patch、代码快照和 `test_v2` manifest 的 SHA-256。

## 开发集消融矩阵

| 组别 | peak LR | 过采样 | 成功 | 成功率 | Wilson 95% |
| --- | ---: | --- | ---: | ---: | --- |
| A | 2e-5 | 1x / 0 | 16/40 | 40.0% | [26.4%, 55.4%] |
| B | 2e-5 | 3x / 5 | 15/40 | 37.5% | [24.2%, 53.0%] |
| C | 5e-5 | 1x / 0 | 36/40 | 90.0% | [76.9%, 96.0%] |
| D | 5e-5 | 3x / 5 | 31/40 | 77.5% | [62.5%, 87.7%] |

按预先规则，C 是唯一距最高成功数不超过 1/40 的候选。C 相对 D 的配对结果为：C-only=7、D-only=2、双方成功=29、双方失败=2，精确 McNemar `p=0.1797`。完整矩阵统计见 `outputs/p3_ablation_20260719/matrix_summary.json`。

## 跨 seed 晋级

| 训练 seed | validation_v2 成功 | 成功率 | Wilson 95% 下界 |
| --- | ---: | ---: | ---: |
| 1000 | 36/40 | 90.0% | 76.9% |
| 1001 | 34/40 | 85.0% | 70.9% |
| 1002 | 34/40 | 85.0% | 70.9% |

三 seed 中位成功率=85.0%，最差=85.0%，极差=5.0 个百分点；均满足门槛（中位数≥70%、最差≥60%、极差≤10 个百分点）。报告见 `outputs/p3_ablation_20260719/p4_promotion.json`。

## `test_v2` 盲测与主验收

盲测 manifest 为固定的 100 局 `test_v2`，角色为 blind。逐局 rollout、metadata 与完成状态已保存，记录为 100 个唯一 scene ID。

| 指标 | 结果 | 门槛 | 判定 |
| --- | ---: | ---: | --- |
| 成功数 | 81/100 | ≥70/100 | 通过 |
| 成功率 Wilson 95% 下界 | 72.2% | ≥60% | 通过 |
| 峰值显存 | 1190.3 MiB（1.16 GiB） | ≤6 GiB | 通过 |
| 系统夹爪切换 | mean=4.83，p95=23.15 | 诊断项 | 记录 |
| 阶段漏斗 | approach=83%，grasp=91%，lift=81%，hold=81% | 诊断项 | 记录 |

失败漏斗：no-grasp=9、grasp-no-lift=10。原始盲测结果为 `outputs/p3_ablation_20260719/test_v2_C_canonical_seed1000.json`。

### 延迟门槛说明

盲测中 episode-level inference p95 的 p95 为 `0.755 s`。冻结时的模型与控制器就是被测 baseline，且协议只允许一次 `test_v2`，因此没有保留独立、同工作负载的冻结前 p95 样本可用于正式计算“相对冻结基线”的退化率。`validation_v2` 的 p95 不是同一场景分布，不能替代该对照；以它作非正式代理会得到约 +15.2%，不应作为盲测验收结论。

因此，本实验已通过成功数、Wilson 下界和显存三项可验证主验收，但延迟退化≤10% 这一项应标为“未能正式比较”，而不是声称已通过。不得重新运行或重复使用 `test_v2` 来补测。

## 可复现与治理

- 每个候选保留原始 rollout、`.meta.json`、训练 manifest、日志与 source patch。
- `test_v2` 已消费且完整完成；今后不得用于调参、模型选择或二次主验收。
- 没有使用 SOTA 表述。
- 机器峰值显存显著低于 6 GiB 上限，适合既定 RTX 3060 Laptop 单 GPU 串行流程。

## 产物索引

- 冻结清单：`outputs/p3_ablation_20260719/canonical_freeze_20260720.json`
- 矩阵汇总：`outputs/p3_ablation_20260719/matrix_summary.json` 与 `matrix_summary.md`
- 跨 seed 报告：`outputs/p3_ablation_20260719/p4_promotion.json`
- 盲测 rollout：`outputs/p3_ablation_20260719/test_v2_C_canonical_seed1000.json` 与对应 `.meta.json`
- 最终验收 JSON：`outputs/p3_ablation_20260719/final_acceptance_20260720.json`
