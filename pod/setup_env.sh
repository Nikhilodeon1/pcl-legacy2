#!/usr/bin/env bash
# Build the venv in /tmp (ephemeral) -- rerun this on every new pod (a few minutes).
#   bash pod/setup_env.sh
# GPU: V100 (sm_70). PyTorch wheels built against CUDA >= 12.8 dropped Volta, so default to the cu126 index.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash pod/setup_env.sh      # if the driver is older
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
PYVER="${PCL_PYTHON_VERSION:-3.11}"

echo "--- disk/memory ---"; df -h "$HOME" /tmp | sed 's/^/  /'; free -g | sed 's/^/  /' || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARNING: nvidia-smi not found"

# a crash mid-install leaves a venv that looks fine but is incomplete: only trust it if the ready-marker was written
if [ -d "$PCL_VENV" ] && [ ! -f "$PCL_VENV/.pcl_ready" ]; then echo "--- removing incomplete venv from an interrupted setup ---"; rm -rf "$PCL_VENV"; fi
if [ ! -x "$PCL_VENV/bin/python" ]; then
  if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    echo "--- creating venv with system python3 ($(python3 --version)) ---"
    python3 -m venv "$PCL_VENV"
  else
    echo "--- system python3 is too old ($(python3 --version 2>&1)); fetching Python $PYVER with uv into /tmp ---"
    if ! command -v uv >/dev/null 2>&1; then
      python3 -m pip install --quiet --target /tmp/uvbin uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/tmp/uvbin sh
      export PATH="/tmp/uvbin/bin:/tmp/uvbin:$PATH"
    fi
    uv venv --python "$PYVER" "$PCL_VENV"
  fi
fi
. "$PCL_VENV/bin/activate"
python -m pip install --quiet --upgrade pip
echo "--- torch (CUDA wheels from $TORCH_INDEX; torch>=2.3 is required: torch.amp.GradScaler('cuda')) ---"
python -m pip install "torch>=2.3,<2.9" --index-url "$TORCH_INDEX"
python -m pip install -r "$PCL_REPO/pod/requirements-pod.txt"
touch "$PCL_VENV/.pcl_ready"
echo "--- preflight ---"
python "$PCL_REPO/pod/preflight.py" || true
