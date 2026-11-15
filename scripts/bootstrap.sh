#!/usr/bin/env bash
# 一次性环境准备。可重复执行。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 0. 系统信息 =="
uname -a
python3 -V
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || \
  echo "!! 没有 nvidia-smi，先装驱动再回来"
df -h / | tail -1

echo "== 1. venv =="
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel >/dev/null

echo "== 2. 依赖 =="
pip install numpy pyyaml
# 4070 Ti SUPER 是 Ada(sm_89)，cu121 及以上都行；cu126 装不上就往下退
pip install torch --index-url https://download.pytorch.org/whl/cu126 \
  || pip install torch --index-url https://download.pytorch.org/whl/cu124 \
  || pip install torch

echo "== 3. 自检 =="
python - <<'PY'
import numpy, torch
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
PY

python tests/test_schema.py
python tests/test_env.py
python tests/test_models.py
python tests/test_pipeline.py
python tests/test_adapter.py
echo "== bootstrap ok =="
