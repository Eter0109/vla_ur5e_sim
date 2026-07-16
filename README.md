# 本机 VLA 抓取仿真闭环

当前项目使用 Windows、RTX 3060 Laptop 6 GiB、MuJoCo/robosuite、UR5e 和
SmolVLA-450M，覆盖仿真、专家数据采集、训练和闭环 rollout。

## 环境安装

项目统一使用已有的 `vla_sim_gpu` Conda 环境。它包含 Python 3.11、CUDA 版
PyTorch 2.7 和 LeRobot 0.4.4；首次使用或依赖变化后，在项目根目录运行：

```powershell
conda activate vla_sim_gpu
python -m pip install -e ".[sim,vla,dev]"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
```

后续命令均默认已位于 `D:\vla_ur5e_sim` 并激活 `vla_sim_gpu`。

## 测试与仿真

```powershell
python -m pytest -q

# 默认无界面；添加 --render 可打开窗口
python scripts/smoke_sim.py --episodes 1 --steps 300

# 快速检查随机动作和相机观测
python scripts/smoke_sim.py --episodes 1 --steps 10 --random
```

`smoke_sim.py` 不读取 `configs/sim.yaml`，也没有 `--config` 或 `--headless`
参数；无界面运行是默认行为。

## 生成场景与采集数据

```powershell
python scripts/generate_manifests.py

python scripts/collect_demos.py `
  --manifest data\manifests\train.json `
  --root data\lerobot\expert_gate10 `
  --repo-id local/ur5e_custom_lift `
  --overwrite
```

`--overwrite` 会替换目标数据集；如需保留现有数据，请改用新的 `--root`。

## 训练 SmolVLA

基础模型位于 `.runtime\models\smolvla_base`。先用少量步数验证环境，再开始正式训练：

```powershell
# 20 步 LoRA smoke run
.\scripts\train_smolvla.ps1 `
  -Steps 20 `
  -Dataset data\lerobot\expert_gate10 `
  -Output outputs\smolvla_lora_smoke

# 2,000 步 LoRA 起始实验
.\scripts\train_smolvla.ps1 `
  -Steps 2000 `
  -Rank 4 `
  -Dataset data\lerobot\expert_gate10 `
  -Output outputs\smolvla_lora_2000
```

训练参数以 PowerShell 参数为准；`configs/train_smolvla_lora.yaml` 当前是配置参考，
不会被训练脚本自动读取。使用 `-FullExpert` 可关闭 LoRA，训练完整 expert。

## Rollout

以下命令使用项目中已有的 10,000 步 checkpoint 运行 5 个测试场景：

```powershell
python scripts/run_rollouts.py `
  --checkpoint outputs\smolvla_full_expert_200demos_10000\checkpoints\010000\pretrained_model `
  --dataset-root data\lerobot\expert_gate10 `
  --repo-id local/ur5e_custom_lift `
  --manifest data\manifests\test.json `
  --episodes 5 `
  --output outputs\rollout_current.json
```

要查看自训练策略控制机械臂的 MuJoCo 窗口，使用单个场景并添加 `--render`；
关闭窗口或按 `Ctrl+C` 可结束运行。

## 本地目录

- `.runtime`：基础模型、Hugging Face 离线缓存、Numba 缓存和临时文件。
- `data`：场景清单和 LeRobotDataset。
- `outputs`：训练 checkpoint 和 rollout 结果。

这些目录包含模型或实验资产，不属于可随意删除的 Python 缓存。
