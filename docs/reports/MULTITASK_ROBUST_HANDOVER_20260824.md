# UR5e 双任务 SmolVLA 多轮实验交接文档（更新版）

> **文档版本**: 2026-08-24
> **前序文档**: `MULTITASK_ROBUST_HANDOVER_20260812.md`（已归档，见 `docs/reports/archive/`）
> **实验周期**: 2026-08-12 ～ 2026-08-14
> **当前状态**: 四项 ≥70% 门禁：3/4 已达标；四项 ≥80% 门禁：1/4 已达标（PickPlace Randomized）

---

## 1. 必须先知道的关键结论

1. **历史最佳四项结果来自同一个 checkpoint**，真实来源是：
   ```
   outputs/multitask_robust/smolvla_30k_lazy_bs2_final/seed1000/checkpoints/030000/pretrained_model
   ```
   - `model.safetensors` SHA-256: `DFFBFCD07911EBCC0658B853ED5031855741DAE26C3241B9AA67AB56C76DD7B7`

2. **`smolvla_full_retrain_30k_corrected_push` 不是最优结果来源**，不能用作 warm-start。

3. **两套门禁并存**：
   - 交接文档原始目标为 **≥70%（≥35/50）**；
   - 2026-08-12 版交接文档已将目标提升至 **≥80%（≥40/50）**；
   - 当前 Push Randomized 在两套门禁下均未达标（50 场最优 66%）。

4. **LoRA lora1500 在 20 场筛选中取得 Push Randomized 80%（16/20）**，但尚未进行 50 场全量验证，且同一 checkpoint 下 PickPlace Nominal 仅 55%（11/20），存在遗忘迹象。

5. **冻结盲测尚未运行**，`push_robust_blind_v1.json` 和 `pick_place_robust_blind_v1.json` 均未消费。

---

## 2. 验收规则（两套门禁）

### 当前原始门禁（≥70%）

| 分片 | Manifest | 门禁 |
|---|---|---|
| Push Nominal | `push_robust_development_nominal_v1.json` | ≥35/50 (70%) |
| Push Randomized | `push_robust_development_randomized_v1.json` | ≥35/50 (70%) |
| PickPlace Nominal | `pick_place_robust_development_nominal_v1.json` | ≥35/50 (70%) |
| PickPlace Randomized | `pick_place_robust_development_randomized_v1.json` | ≥35/50 (70%) |

### 升级门禁（≥80%，2026-08-12 更新）

| 分片 | 门禁 |
|---|---|
| Push Nominal | ≥40/50 (80%) |
| Push Randomized | ≥40/50 (80%) |
| PickPlace Nominal | ≥40/50 (80%) |
| PickPlace Randomized | ≥40/50 (80%) |

> **重要**：筛选用 20 场子集只用于选择 checkpoint，不能替代正式 50 场证据。盲测文件不得覆盖或提前查看。

验证脚本：
```powershell
python scripts\verify_target80_development.py --help
```

---

## 3. 模型与数据集

### 3.1 当前最优基线 Checkpoint

```
outputs\multitask_robust\smolvla_30k_lazy_bs2_final\seed1000\checkpoints\030000\pretrained_model
```

| 属性 | 值 |
|---|---|
| 训练步数 | 30,000 steps |
| 训练种子 | seed=1000 |
| `model.safetensors` SHA-256 | `DFFBFCD07911EBCC0658B853ED5031855741DAE26C3241B9AA67AB56C76DD7B7` |
| checkpoint 目录 SHA-256 | `6e73fb8aea72c2fe2e18311773c79554b5879af2bcced410fd156a72acf6e38d` |

### 3.2 数据集路径

| 数据集 | 路径 | 说明 |
|---|---|---|
| Push 主数据 | `data\lerobot\multitask_robust_push_1500` | Push 任务训练 |
| PickPlace 主数据 | `data\lerobot\multitask_robust_pick_place_1500` | PickPlace 任务训练 |
| 联合双任务数据 | `data\lerobot\multitask_robust_3000` | 联合训练合并集 |
| Push 定向恢复数据 | `data\lerobot\push_robust_targeted_recovery_v2_500` | Push Randomized 增强（500 条，已审计） |

### 3.3 场景清单（Manifest）路径

| 类型 | 路径 |
|---|---|
| Push 开发 Nominal（50场）| `configs\benchmarks\push_robust_development_nominal_v1.json` |
| Push 开发 Randomized（50场）| `configs\benchmarks\push_robust_development_randomized_v1.json` |
| Push 筛选 Nominal（20场）| `configs\benchmarks\push_robust_development_nominal_v1_screen20.json` |
| Push 筛选 Randomized（20场）| `configs\benchmarks\push_robust_development_randomized_v1_screen20.json` |
| Push 盲测（100场，未消费）| `configs\benchmarks\push_robust_blind_v1.json` |
| PickPlace 开发 Nominal（50场）| `configs\benchmarks\pick_place_robust_development_nominal_v1.json` |
| PickPlace 开发 Randomized（50场）| `configs\benchmarks\pick_place_robust_development_randomized_v1.json` |
| PickPlace 筛选 Nominal（20场）| `configs\benchmarks\pick_place_robust_development_nominal_v1_screen20.json` |
| PickPlace 盲测（100场，未消费）| `configs\benchmarks\pick_place_robust_blind_v1.json` |

---

## 4. 完整实验结果矩阵

### 4.1 控制器消融（基于 smolvla_30k_lazy_bs2_final，50 场全量）

| replan | decay | samples | Push Nominal | Push Randomized | PickPlace Nominal | PickPlace Randomized |
|---|---|---|---|---|---|---|
| 4 | 0.50 | 1 | 62.0% (31/50) | 60.0% (30/50) | 58.0% (29/50) | 42.0% (21/50) |
| 2 | 0.25 | 1 | 50.0% (25/50) | — | — | — |
| 2 | 0.50 | 1 | 54.0% (27/50) | — | — | — |
| 4 | 0.72 | 1 | 58.0% (29/50) | — | — | — |
| **4** | **0.75** | **1** | **74.0% (37/50)** | **66.0% (33/50)** | **62.0% (31/50)** | **46.0% (23/50)** |
| 4 | 0.75 | 1 (h=300) | 64.0% (32/50) | — | — | — |
| 4 | 0.75 | 1 (seed=2000) | 54.0% (27/50) | — | — | — |
| 4 | 0.85 | 1 | 62.0% (31/50) | — | — | — |
| 4 | 0.77 | 1 | — | 62.0% (31/50) | — | — |
| 6 | 0.75 | 1 | 64.0% (32/50) | — | — | — |
| 4 | 0.75 | 2 | 68.0% (34/50) | — | — | — |

### 4.2 PickPlace 增益消融（50 场全量）

| gain | samples | PickPlace Nominal | PickPlace Randomized |
|---|---|---|---|
| 1.3 | 1 | 58.0% (29/50) | 42.0% (21/50) |
| 1.4 | 1 | ~64% | 64.0% (32/50) |
| 1.6 | 1 | — | 64.0% (32/50) |
| 1.8 | 1 | — | 68.0% (34/50) |
| **1.8** | **2** | **72.0% (36/50)** | **80.0% (40/50)** |

### 4.3 LoRA 微调实验（20 场筛选，08-14）

| 实验名 | 步数 | Push Random(20场) | Push Nominal(20场) | Pick Nominal(20场) |
|---|---|---|---|---|
| `key_screen_1k` | 1k | 70.0% (14/20) | — | — |
| `key_screen_2k` | 2k | 70.0% (14/20) | — | — |
| `key_screen_3k` | 3k | 70.0% (14/20) | — | — |
| `key_screen_4k` | 4k | 65.0% (13/20) | — | — |
| `key_screen_5k` | 5k | 65.0% (13/20) | — | — |
| `key_screen_lora500` | 500 | 75.0% (15/20) | — | — |
| `key_screen_lora1000` | 1k | 60.0% (12/20) | — | — |
| **`key_screen_lora1500`** | **1500** | **80.0% (16/20)** | — | **55.0% (11/20)** |
| `key_screen_lora2000` | 2k | 65.0% (13/20) | — | — |
| `key_screen_lora2500` | 2.5k | 65.0% (13/20) | — | — |
| `key_screen_lora3000` | 3k | 65.0% (13/20) | — | — |
| `key_screen_balanced250` | 250 | 70.0% (14/20) | — | — |
| `four_screen_lora500` | 500 | 70.0% (14/20) | 70.0% (14/20) | — |

> **lora1500 是 Push Randomized 最优（20场80%），但 PickPlace 遗忘严重，50 场全量验证尚未完成。**

### 4.4 当前最终成绩（50 场全量）

| 评测 | 最优结果 | 达标(70%) | 达标(80%) | 结果文件 |
|---|---|---|---|---|
| **Push Nominal** | **74.0% (37/50)** | YES | NO | `eval_orig30k_push_nominal_d075.json` |
| **Push Randomized** | **66.0% (33/50)** | NO | NO | `eval_orig30k_push_randomized_d075.json` |
| **PickPlace Nominal** | **72.0% (36/50)** | YES | NO | `ablation_orig30k_pick_nominal_gain18_samples2.json` |
| **PickPlace Randomized** | **80.0% (40/50)** | YES | YES | `ablation_orig30k_pick_randomized_gain18_samples2.json` |

---

## 5. 关键参数配置

### 5.1 Push 评测模板

```powershell
conda activate vla_sim_gpu
python scripts\run_push_vla_only_benchmark.py `
  --checkpoint outputs\multitask_robust\smolvla_30k_lazy_bs2_final\seed1000\checkpoints\030000\pretrained_model `
  --dataset-root data\lerobot\multitask_robust_push_1500 `
  --repo-id local/multitask_robust_push_1500 `
  --manifest configs\benchmarks\push_robust_development_nominal_v1.json `
  --episodes 50 --replan-steps 4 --temporal-decay 0.75 --policy-seed 1000 `
  --output outputs\multitask_robust\eval_push_nominal_FINAL.json --overwrite-development
```

### 5.2 PickPlace 评测模板

```powershell
python scripts\run_pick_place_vla_only.py `
  --checkpoint outputs\multitask_robust\smolvla_30k_lazy_bs2_final\seed1000\checkpoints\030000\pretrained_model `
  --dataset-root data\lerobot\multitask_robust_pick_place_1500 `
  --repo-id local/multitask_robust_pick_place_1500 `
  --manifest configs\benchmarks\pick_place_robust_development_nominal_v1.json `
  --episodes 50 --closed-negative-y-gain 1.8 --samples-per-plan 2 `
  --output outputs\multitask_robust\eval_pick_nominal_FINAL.json --overwrite-development
```

### 5.3 Checkpoint 筛选模板

```powershell
# 关键瓶颈 20 场快速筛选（Push Randomized + PickPlace Nominal）
.\scripts\run_target80_key_checkpoint_screen.ps1 -Checkpoint <ckpt_path> -Output <output_dir>

# 四分片 20 场筛选
.\scripts\run_target80_checkpoint_screen.ps1 -Checkpoint <ckpt_path> -Output <output_dir>

# 完整 50 场开发评测
.\scripts\run_multitask_robust_development_evaluation.ps1 -Checkpoint <ckpt_path> -Output <output_dir>
```

---

## 6. 新增代码模块说明

### 6.1 本轮新增脚本（`scripts/`）

| 文件 | 说明 |
|---|---|
| `run_push_vla_only_benchmark.py` | Push 评测主脚本（含 `--samples-per-plan`）|
| `run_push_vla_only.py` | Push 单次推理脚本 |
| `collect_push_demos.py` / `_v2.py` | Push 数据采集（v2 支持安全续采）|
| `generate_push_manifests.py` | Push manifest 生成器 |
| `generate_push_robust_recovery_manifest.py` | Push 恢复 manifest 生成器 |
| `generate_push_targeted_recovery_manifest.py` | 定向角度/距离恢复 manifest 生成器 |
| `generate_multitask_robust_manifests.py` | 双任务 manifest 批量生成器 |
| `generate_target80_screening_manifests.py` | 20 场筛选 manifest 生成器 |
| `audit_targeted_push_recovery.py` | Push 恢复数据分布 + 种子隔离审计 |
| `audit_multitask_datasets.py` | 双任务数据集合约审计 |
| `build_multitask_robust_dataset.py` | 多任务数据集合并构建器 |
| `build_target80_screen_reference.py` | 可追溯 20 场筛选基线 |
| `verify_target80_development.py` | 严格 80% 门禁验证器 |
| `run_multitask_robust_development_evaluation.ps1` | 四分片完整开发评测入口 |
| `run_multitask_robust_blind_evaluation.ps1` | 冻结盲测入口（谨慎执行）|
| `run_target80_key_checkpoint_screen.ps1` | 关键瓶颈快速筛选 |
| `run_target80_checkpoint_screen.ps1` | 四分片筛选 |
| `train_multitask_robust_full_retrain.ps1` | 30k 全量训练 |
| `train_multitask_robust_recovery.ps1` | 联合恢复训练 |
| `train_multitask_target80_joint_recovery.ps1` | 6k 联合恢复训练（带哈希门禁）|
| `train_multitask_target80_lora_recovery.ps1` | LoRA 微调训练 |
| `smoke_push.py` | Push 环境 smoke 测试 |
| `evaluate_push_expert.py` | Push 专家策略评测 |

### 6.2 新增/修改的核心模块（`src/vla_sim/`）

| 文件 | 说明 |
|---|---|
| `envs/ur5e_push.py` | Push 环境定义（新增）|
| `domain_randomization.py` | 域随机化辅助模块（新增）|
| `target80.py` | 筛选晋级与门禁逻辑（新增）|
| `sampling.py` | 多任务提示词与辅助数据采样修复 |
| `provenance.py` | 数据/场景隔离与分布检查 |
| `scenes.py` | Manifest 加载与场景管理（更新）|
| `sim/expert.py` | Push 专家策略（更新）|

---

## 7. 接手后的执行顺序

### 步骤 1：确认 lora1500 的实际 checkpoint 路径

```powershell
python -c "import json; d=json.load(open('outputs/multitask_robust/target80_key_screen_lora1500_20260814/push_randomized.json')); print(d.get('checkpoint'))"
```

### 步骤 2：运行 lora1500 的 50 场全量评测（四项）

```powershell
# Push Randomized（最关键）
python scripts\run_push_vla_only_benchmark.py --checkpoint <lora1500_ckpt> `
  --manifest configs\benchmarks\push_robust_development_randomized_v1.json `
  --episodes 50 --replan-steps 4 --temporal-decay 0.75 --policy-seed 1000 `
  --output outputs\multitask_robust\eval_lora1500_push_randomized_50ep.json --overwrite-development

# PickPlace Nominal（检测遗忘）
python scripts\run_pick_place_vla_only.py --checkpoint <lora1500_ckpt> `
  --manifest configs\benchmarks\pick_place_robust_development_nominal_v1.json `
  --episodes 50 --closed-negative-y-gain 1.8 --samples-per-plan 2 `
  --output outputs\multitask_robust\eval_lora1500_pick_nominal_50ep.json --overwrite-development
```

### 步骤 3：根据结果决策

- **情形 A（理想）**: 四项全量均达标 → 运行严格验证器 → 解锁盲测
- **情形 B（遗忘）**: Push 达标但 PickPlace 遗忘 → 在 lora1500 基础上联合微调或走 6k joint recovery 路线
- **情形 C（不达标）**: Push Randomized 50 场 <35/50 → 20 场方差大，尝试其他 checkpoint 或增加数据

### 步骤 4：通过全量评测后解锁盲测

```powershell
# 严格验证
python scripts\verify_target80_development.py `
  --push-nominal <json> --push-randomized <json> `
  --pick-nominal <json> --pick-randomized <json>

# 确认通过后运行冻结盲测（不可覆盖！）
.\scripts\run_multitask_robust_blind_evaluation.ps1
```

---

## 8. 禁止事项

- 不要把 `smolvla_full_retrain_30k_corrected_push` 当成 74/66/72/80 结果的来源。
- 不要把 20 场筛选结果作为正式 50 场结论。
- 不要在四项开发门禁通过前运行盲测。
- 不要覆盖既有评测输出、盲测输出或 checkpoint。
- 不要为了短期提高 Push 而只用 Push 数据高权重微调（会导致 PickPlace 遗忘）。

---

*文档生成时间: 2026-08-24 20:52 CST*
