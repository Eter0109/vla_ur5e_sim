"""Real-time multi-track training and evaluation monitor."""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRACKS = [
    {
        "id": "Track 0 (Baseline)",
        "name": "formal_lora_r32_seed1000",
        "output": ROOT / "outputs/sim2real_v2/formal_lora_r32_seed1000",
        "status": ROOT / "outputs/sim2real_v2/formal_training_status.json",
        "log": ROOT / "outputs/sim2real_v2/formal_training.log",
    },
    {
        "id": "Track 1 (Color+Grip)",
        "name": "exp1_color_focus_grip30",
        "output": ROOT / "outputs/sim2real_v2_exp1/formal_lora_r32_grip30",
        "status": ROOT / "outputs/sim2real_v2_exp1/formal_training_status.json",
        "log": ROOT / "outputs/sim2real_v2_exp1/formal_training.log",
    },
    {
        "id": "Track 2 (LR8e-6+Seed2000)",
        "name": "exp2_lr8e6_seed2000",
        "output": ROOT / "outputs/sim2real_v2_exp2/formal_lora_r32_lr8e6",
        "status": ROOT / "outputs/sim2real_v2_exp2/formal_training_status.json",
        "log": ROOT / "outputs/sim2real_v2_exp2/formal_training.log",
    },
]

def parse_log(log_path: Path):
    if not log_path.exists():
        return None, None, None, None
    try:
        with open(log_path, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - 8192))
            lines = f.read().decode("utf-8", errors="ignore").splitlines()

        step, loss, lr, speed = None, None, None, None
        for line in reversed(lines):
            if "ot_train.py" in line and "step:" in line:
                m_step = re.search(r"step:(\d+)", line)
                m_loss = re.search(r"loss:([0-9.]+)", line)
                m_lr = re.search(r"lr:([0-9.e+-]+)", line)
                if m_step and not step:
                    step = int(m_step.group(1))
                if m_loss and not loss:
                    loss = float(m_loss.group(1))
                if m_lr and not lr:
                    lr = m_lr.group(1)
            m_bar = re.search(r"(\d+)/12000.*,\s+([0-9.]+)step/s", line)
            if m_bar and not step:
                step = int(m_bar.group(1))
            if m_bar and not speed:
                speed = f"{m_bar.group(2)} step/s"
            if step and loss and speed:
                break
        return step, loss, lr, speed
    except Exception:
        return None, None, None, None

def print_summary():
    print("=" * 88)
    print(f"  三轨并行训练与评测实时看板 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 88)
    header = f"{'轨道名称':<25} | {'当前阶段':<12} | {'步数进度':<12} | {'最新Loss':<10} | {'学习率':<10} | {'速度':<12}"
    print(header)
    print("-" * 88)
    for t in TRACKS:
        stage = "未启动"
        if t["status"].exists():
            try:
                s_data = json.loads(t["status"].read_text())
                stage = s_data.get("stage", "unknown")
            except Exception:
                stage = "读取中"
        step, loss, lr, speed = parse_log(t["log"])
        step_str = f"{step}/12000" if step is not None else "--"
        loss_str = f"{loss:.4f}" if loss is not None else "--"
        lr_str = str(lr) if lr else "--"
        speed_str = str(speed) if speed else "--"
        print(f"{t['id']:<23} | {stage:<10} | {step_str:<12} | {loss_str:<10} | {lr_str:<10} | {speed_str:<12}")
    print("=" * 88)

if __name__ == '__main__':
    print_summary()
