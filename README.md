# UR5e SmolVLA 双相机 PickPlace 仿真闭环

当前主任务是 **PickPlace v2 native bin**：UR5e 使用正前方约 45° 的第三视角 RGB 和腕部 RGB，
抓取红色方块并放入远处的蓝色 MuJoCo 原生收纳盒。收纳盒有真实底面和四侧壁，参与 RGB 渲染、
深度和遮挡；任务通过抓取、抬升、释放、盒内位置和连续 10 步稳定性判定成功。

最新 raw pure-VLA development 候选是 Teacher Distillation v5.3 step 300：在
`pick_place_dev_v1` 前 24/100 场、`samples=2`、`replan=8`、单 policy seed 1000 下取得
`22/24 (91.7%)`，抓取 `24/24`。该结果不使用动作校准、物体/目标位姿或 RGB-D Supervisor，
但仍只是经过筛选的 development-24 里程碑，不是完整 development、test/blind 或冻结部署结果。

项目另有两个历史冻结的 50 场系统结果：**SmolVLA + 有界 RGB-D 视觉伺服**为
`50/50 (100%)`；**SmolVLA + 固定动作校准**为 `46/50 (92%)`。三种协议不能混称。
Stack 和绿色平面目标 PickPlace v1 均为历史任务，不再作为默认入口。

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
python scripts/run_pick_place_vla_only.py `
  --checkpoint outputs\pick_place_v2_native_bin\teacher_distill_transport_v5_3_600\seed1000\checkpoints\000300\pretrained_model `
  --dataset-root data\lerobot\pick_place_v2_native_bin_1000 `
  --repo-id local/ur5e_pick_place_v2_native_bin `
  --manifest configs\benchmarks\pick_place_dev_v1.json `
  --episodes 1 --scene-index 0 --samples-per-plan 2 --replan-steps 8 `
  --policy-seed 1000 --control-mode vla_raw_safety --rgb-window --render `
  --output outputs\pick_place_v2_native_bin\manual_checks\v5_3_scene0000.json
```

这条命令只使用 development 场景，同时显示 MuJoCo 场景窗口和模型实际接收的双 RGB 画面。
完整的数据采集、训练和评测规则见 [PickPlace v2 Runbook](docs/PICK_PLACE_V2_RUNBOOK.md)，
实验身份见 [实验注册表](docs/reference/EXPERIMENT_REGISTRY.md)，最新结果见
[Teacher Distillation v5.3 报告](docs/reports/PICK_PLACE_TEACHER_DISTILL_V5_3_SUCCESS_20260804.md)。
项目文档导航与本地数据、checkpoint、rollout 的保存规则见
[文档索引](docs/README.md) 和 [本地保存结构](docs/reference/LOCAL_STORAGE_LAYOUT.md)。

## 资产目录

- `.runtime`：本地模型与 Hugging Face 缓存。
- `configs/benchmarks`：版本化 benchmark manifest。
- `data`：LeRobotDataset 和采集契约。
- `outputs`：checkpoint、rollout 与日志。
- `scripts`：当前 PickPlace 数据、训练、评测和可视化入口；历史 Stack 脚本位于
  `scripts/archive/stack`。
- `docs`：当前 Runbook、实验注册、详细实验报告和历史归档。

`data/`、`outputs/` 和 `.runtime/` 都是昂贵的本地资产，不应作为普通清理对象删除或提交。
