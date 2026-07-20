# 2026-07-19 冷启动消融矩阵

所有实验使用 `legacy_20260719.json` 的环境、损失、训练时长和推理设置；只改变 peak LR 与转折帧采样。

| ID | Peak LR | Oversample factor | Window | 状态 |
| --- | ---: | ---: | ---: | --- |
| A | 2e-5 | 1 | 0 | 训练中（seed=1000，输出 `smolvla_ablation_a_15k_seed1000`，已保存 5k） |
| B | 2e-5 | 3 | 5 | 待训练 |
| C | 5e-5 | 1 | 0 | 待训练 |
| D | 5e-5 | 3 | 5 | 已有 `smolvla_fullexpert_cosine_15k` |

模型选择只使用 `validation_v2`。胜出配置再运行 training seed 1001、1002；只有三 seed 中位数至少 70%、最差 seed 至少 60%、极差不超过 10 个百分点时，才可以进入 `test_v2`。
