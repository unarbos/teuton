#!/usr/bin/env bash
# Idempotent setup for a Lium GPU box: python3-venv, rsync, .venv with
# torch (cpu wheel, ~200MB), boto3, python-dotenv. Locus source itself is
# rsynced separately.
set -euo pipefail

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv rsync curl

if [ ! -d /root/.venv ]; then
    python3 -m venv /root/.venv
fi

# shellcheck disable=SC1091
. /root/.venv/bin/activate
python -m pip install --quiet --upgrade pip wheel

# CPU-only torch keeps the wheel small and downloads fast; v2 task math is
# 16x16 matmul, GPU would add launch overhead.
python -m pip install --quiet \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    'torch>=2.1' 'numpy<2'
python -m pip install --quiet boto3 'python-dotenv>=1.0'

echo "=== verify ==="
python -c "import torch, boto3, dotenv; print('torch', torch.__version__); print('boto3', boto3.__version__)"
echo "=== gpu visibility ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=== done ==="
