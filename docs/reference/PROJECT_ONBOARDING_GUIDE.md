# PickPlace v2 native bin 上手指南

项目使用 SmolVLA（500M 级）控制 UR5e 和 Robotiq 2F-85，在 MuJoCo/robosuite 中完成“红色方块
放入蓝色收纳盒”。它是 Stack 的降阶任务：保留视觉抓取、移动与释放，同时将目标改为更大的
固定收纳盒，降低毫米级堆叠和碰撞敏感放置的难度。

## 观测与控制

- 第三视角 `agentview` 位于正前方约 45° 俯视位置，用于全局定位红色方块与蓝色收纳盒。
- 腕部视角 `robot0_eye_in_hand` 用于抓取与下放的近距离观察。
- VLA 接收两路 RGB 和 10-D 本体状态，预测 7-D action；当前 raw 链路执行 XYZ 与 gripper，
  只保留固定旋转和工作空间安全裁剪。
- 历史 Supervisor 系统会确认夹爪抓取并使用第三视角 RGB-D 执行有界 XY 视觉伺服；它与
  当前 `vla_raw_safety` 评估是不同协议。
- 收纳盒使用 MuJoCo 原生 `Bin`，有底面和四侧壁并参与 RGB、深度与遮挡。

## 场景与成功条件

红色方块为 5 cm 立方体；蓝色收纳盒平面尺寸为 12 cm、高 4 cm。方块位于工作区左侧，盒子在
右侧小范围随机化，并按近/远两档运输距离平衡采样。数据集、采集和 rollout 共享
`meta/pick_place_environment.json` 契约，任何相机、几何、状态、动作或 prompt 不一致都会拒绝运行。

成功要求：本局曾抓取、曾抬升至少 4 cm、方块已释放、方块中心位于盒内 3 cm 轴向容差内、回到
盒底高度且低速稳定，并连续保持 10 个仿真步。

## 当前模型与结果

- 数据集：`data/lerobot/pick_place_v2_native_bin_1000`，1,000 episodes / 84,819 frames。
- raw pure-VLA development 候选：
  `outputs/pick_place_v2_native_bin/teacher_distill_transport_v5_3_600/seed1000/checkpoints/000300/pretrained_model`。
- 当前开发结果：`pick_place_dev_v1` 前 24/100 场为 `22/24 (91.7%)`，抓取 `24/24`；
  使用 `samples=2/replan=8/seed=1000`，无动作 gain、phase Supervisor 或物体/目标位姿。
- 该结果是单 seed development 候选，完整 100 场 development 和多 seed 验证尚未完成。
- 系统 checkpoint：`outputs/pick_place_v2_native_bin/maskfix_20k/seed1000/checkpoints/020000/pretrained_model`。
- 有界 RGB-D 视觉伺服系统：固定 50 场 `50/50`，最终 XY 平均误差 6.71 mm。
- 非 oracle checkpoint：`outputs/pick_place_v2_native_bin/vla_only_global_20k/seed1000/checkpoints/020000/pretrained_model`。
- SmolVLA + 固定动作校准：全新固定留出集 `46/50 (92%)`，抓取 `49/50`。
- 结论边界：50/50 是完整视觉伺服系统结果，46/50 包含手工动作校准；只有 v5.3 的
  `vla_raw_safety` 链路属于当前未经推理校准的 pure-VLA 候选，但它尚非 test/blind 结论。

完整命令见 [`docs/PICK_PLACE_V2_RUNBOOK.md`](../PICK_PLACE_V2_RUNBOOK.md)。当前可执行入口只放在
根 `scripts/`；旧 Stack 数据、benchmark 和 `scripts/archive/stack/` 仅用于历史追溯。
