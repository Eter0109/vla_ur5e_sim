# UR5e SmolVLA 双相机 PickPlace 仿真闭环

当前主任务是 **PickPlace v2 native bin**：UR5e 使用正前方约 45° 的第三视角 RGB 和腕部 RGB，
抓取红色方块并放入远处的蓝色 MuJoCo 原生收纳盒。收纳盒有真实底面和四侧壁，参与 RGB 渲染、
深度和遮挡；任务通过抓取、抬升、释放、盒内位置和连续 10 步稳定性判定成功。

当前有两个冻结的 50 场结果：**SmolVLA + 有界 RGB-D 视觉伺服**为 `50/50 (100%)`；
不读取物体真值位姿的 **SmolVLA + 固定动作校准**在全新留出集上为 `46/50 (92%)`。后者仍含
手工推理校准，不应表述为未经校准的纯端到端 VLA。Stack 和绿色平面目标 PickPlace v1 均为
历史任务，不再作为默认入口。

## 环境

```powershell
conda activate vla_sim_gpu
python -m pip install -e ".[sim,vla,dev]"
python -m pytest -q
python -m ruff check src tests scripts
```

PickPlace 使用两个 `256×256` RGB 输入：`agentview` 提供全局场景，`robot0_eye_in_hand` 提供
近距离抓取和放置视角。深度仅供 Supervisor 定位红色方块和蓝色收纳盒，不输入 VLA。

## 当前标准入口

```powershell
python scripts/run_pick_place_rollouts.py `
  --checkpoint outputs\pick_place_v2_native_bin\maskfix_20k\seed1000\checkpoints\020000\pretrained_model `
  --dataset-root data\lerobot\pick_place_v2_native_bin_1000 `
  --repo-id local/ur5e_pick_place_v2_native_bin `
  --manifest configs\benchmarks\pick_place_test_v2_50.json `
  --episodes 1 --rgb-window --render `
  --output outputs\pick_place_v2_native_bin\maskfix_20k\visual_check.json
```

这条命令同时显示 MuJoCo 场景窗口和模型实际接收的双 RGB 画面。完整的数据采集、20k 训练、
离线诊断和 50 场评测命令见 [PickPlace v2 Runbook](docs/PICK_PLACE_V2_RUNBOOK.md)，实验身份和
结果见 [实验注册表](docs/reference/EXPERIMENT_REGISTRY.md)，完整训练迭代见
[PickPlace VLA 实验报告](docs/reports/PICK_PLACE_VLA_EXPERIMENT_REPORT_20260731.md)。

## 资产目录

- `.runtime`：本地模型与 Hugging Face 缓存。
- `configs/benchmarks`：版本化 benchmark manifest。
- `data`：LeRobotDataset 和采集契约。
- `outputs`：checkpoint、rollout 与日志。
- `scripts`：当前 PickPlace 数据、训练、评测和可视化入口；历史 Stack 脚本位于
  `scripts/archive/stack`。
- `docs`：当前 Runbook、实验注册、详细实验报告和历史归档。

`data/`、`outputs/` 和 `.runtime/` 都是昂贵的本地资产，不应作为普通清理对象删除或提交。
