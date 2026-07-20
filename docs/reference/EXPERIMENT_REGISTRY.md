# 实验注册表

本文件保存可提交的实验摘要和 benchmark 身份；逐局 rollout、checkpoint 与训练日志仍位于本地 `outputs/`。

## 2026-07-20：P3 canonical 与最终盲测

| 字段 | 值 |
| --- | --- |
| 状态 | canonical 已冻结；`test_v2` 已完整消费一次 |
| Checkpoint | `outputs/smolvla_ablation_c_15k_seed1000/checkpoints/015000/pretrained_model` |
| 训练 | seed=1000，15k steps，peak LR=`5e-5`，无 transition oversampling |
| 系统推理 | temporal ensemble (`replan=4`, `decay=0.5`) + `confirm2` |
| 三 seed validation_v2 | `36/40`、`34/40`、`34/40`；中位数 85%，极差 5 个百分点 |
| test_v2 | `81/100 (81.0%)`，Wilson 95%=`72.2%–87.5%` |
| 峰值显存 | 1190.3 MiB（1.16 GiB） |
| 冻结清单 | `outputs/p3_ablation_20260719/canonical_freeze_20260720.json` |
| 最终报告 | `docs/reports/P3_FINAL_EXPERIMENT_REPORT_20260720.md` |
| 可用结论 | 成功数、Wilson 下界和显存门槛通过；延迟退化门槛缺少同负载基线，未正式判定 |

## 2026-07-19：历史冷启动 15k 候选

| 字段 | 值 |
| --- | --- |
| 状态 | 历史开发候选，已被 2026-07-20 P3 canonical 替代 |
| Checkpoint | `outputs/smolvla_fullexpert_cosine_15k/checkpoints/015000/pretrained_model` |
| 训练 | seed=1000，15k steps，batch=8，peak LR=`5e-5`，floor=`1e-6`，3x transition oversampling |
| 系统推理 | temporal ensemble (`replan=4`, `decay=0.5`) + `confirm2` |
| 旧 validation | `25/30 (83.3%)` |
| 旧 test（已降级为开发回归集） | `36/50 (72.0%)`，Wilson 95%=`58.3%–82.5%` |
| 对 35/50 历史候选的配对结果 | 新增成功 3 局、退化 2 局、McNemar `p=1.0` |
| 可用结论 | 冷启动存在高分可行解；尚不可作统计显著 SOTA 声明 |

## Benchmark v2 状态

| Benchmark | 角色 | 场景数 | SHA-256 | 状态 |
| --- | --- | ---: | --- | --- |
| `validation_v2` | development | 40 | `009b72e8fa6eb688a35086d6c43bea47a467999f3c1a407dae2a126fef67e4f6` | 已用于 P3 选择；保留为历史回归集 |
| `test_v2` | blind | 100 | `d23c21b9ba0214e1edaf2d7fc6aa8833a39f8629975aa7778d27c819c2f8c640` | 2026-07-20 已消费一次，81/100；永久锁定 |
| `ood_v1` | diagnostic | 40 | `be9661c5501cd273edcfb5730079e596e42b240edeeab746dfb466827a19b46d` | 已生成，未消费；物理/相机分层需待环境 override 支持后启用 |

规则：`test_v2` 不得再次运行或参与调参。旧 `test.json` 不得再标记为 held-out 或 blind；下一轮必须使用新冻结的 development manifest，并在全量冻结后启用新的盲测版本。
