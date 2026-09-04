#!/usr/bin/env bash
# Create the reproducible Linux training environment for this repository.
set -euo pipefail

ENV_NAME="${1:-vla_sim_gpu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.12 pip
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0
python -m pip install -e "${ROOT}[sim,vla,dev]"

python - <<'PY'
import torch
import lerobot
import mujoco
import peft
import robosuite
import transformers

assert torch.cuda.is_available(), "CUDA is not available in this environment"
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "lerobot": lerobot.__version__,
    "mujoco": mujoco.__version__,
    "robosuite": robosuite.__version__,
    "peft": peft.__version__,
    "transformers": transformers.__version__,
})
PY
