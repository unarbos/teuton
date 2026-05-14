#!/usr/bin/env bash
# Quick setup for daturaai/pytorch pods. Most have torch+cuda+pip already.
# Just need: boto3, python-dotenv, rsync. Then sync source.
set -euo pipefail

# Use the system python (not a venv) since the daturaai image puts everything there.
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq rsync 2>&1 | tail -1
pip install --quiet --break-system-packages 'boto3' 'python-dotenv>=1.0' 'numpy<2' 2>&1 | tail -3 || pip install --quiet 'boto3' 'python-dotenv>=1.0' 'numpy<2' 2>&1 | tail -3

echo "=== verify ==="
python3 -c "import torch, boto3, dotenv; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'GPUs'); print('boto3', boto3.__version__)"
echo "=== gpu ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -10
echo "=== done ==="
