#!/usr/bin/env bash
# Fast Google Colab setup: uses Colab's native CUDA PyTorch, unpacks the bundled 25,160-stay PhysioNet 2019 cache,
# automatically purges any truncated (<10,381 Site A) runs/caches, and runs preflight.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"

echo "=== 1. Hardware & Python check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARNING: nvidia-smi not found (check Runtime -> Change runtime type)"
free -h || true
python3 --version

python3 -c "import torch, numpy, scipy, pandas, sklearn" 2>/dev/null || {
  echo "Installing missing Python packages..."
  python3 -m pip install -q -r "$PCL_REPO/pod/requirements-pod.txt"
}

mkdir -p "$PCL_LEGACY2_CACHE_DIR" "$PCL_LEGACY2_PRETRAIN_DIR" "$PCL_LEGACY2_CKPT_ROOT" "$PCL_REPO/results/revision/logs"

# Detect and purge any stale/truncated runs from an incomplete PhysioNet download (e.g. n_site_a=233 instead of 10381)
if grep -rq '"n_site_a": 233' "$PCL_COLAB_PERSIST" "$PCL_REPO/results/phase2" 2>/dev/null; then
  echo "=== Purging stale 233-stay outputs/checkpoints from $PCL_COLAB_PERSIST and results/phase2 ==="
  rm -rf "$PCL_COLAB_PERSIST/pretrained/phase2" "$PCL_COLAB_PERSIST/ckpt" "$PCL_COLAB_PERSIST/results_phase2" "$PCL_COLAB_PERSIST/physionet_frac1.0.pkl.gz"
  rm -rf "$PCL_REPO/results/phase2" "$PCL_LEGACY2_CACHE_DIR/physionet_frac1.0"*
fi

# Restore valid results from persistent storage if present
if [ -d "$PCL_COLAB_PERSIST/results_phase2" ]; then
  echo "=== Restoring previous results/phase2 from $PCL_COLAB_PERSIST/results_phase2 ==="
  mkdir -p "$PCL_REPO/results/phase2"
  cp -rn "$PCL_COLAB_PERSIST/results_phase2/"* "$PCL_REPO/results/phase2/" 2>/dev/null || true
fi

echo "=== 2. PhysioNet 2019 cache setup ==="
CACHE_PKL="$PCL_LEGACY2_CACHE_DIR/physionet_frac1.0.pkl"
CACHE_S42="$PCL_LEGACY2_CACHE_DIR/physionet_frac1.0_s42.pkl"

# Verify existing cache size (>= 200 MB); if smaller or missing, unpack from bundled colab/physionet_frac1.0.pkl.gz
NEED_UNPACK=1
if [ -s "$CACHE_PKL" ]; then
  SZ=$(stat -c%s "$(readlink -f "$CACHE_PKL")" 2>/dev/null || echo 0)
  if [ "$SZ" -ge 200000000 ]; then
    echo "[skip] Verified PhysioNet cache already present at $CACHE_PKL ($(( SZ / 1048576 )) MB)"
    NEED_UNPACK=0
  else
    echo "Removing incomplete cache ($(( SZ / 1048576 )) MB < 200 MB)..."
    rm -f "$CACHE_PKL" "$CACHE_S42"
  fi
fi

if [ "$NEED_UNPACK" = "1" ]; then
  GZ_FOUND=""
  for candidate in \
    "$PCL_REPO/colab/physionet_frac1.0.pkl.gz" \
    "$PCL_COLAB_PERSIST/physionet_frac1.0.pkl.gz" \
    "/content/physionet_frac1.0.pkl.gz" \
    "$HOME/pcl_cache_gz/physionet_frac1.0.pkl.gz"; do
    if [ -s "$candidate" ]; then
      GZ_SZ=$(stat -c%s "$candidate" 2>/dev/null || echo 0)
      if [ "$GZ_SZ" -ge 24000000 ]; then
        GZ_FOUND="$candidate"
        break
      fi
    fi
  done

  if [ -n "$GZ_FOUND" ]; then
    echo "Unpacking full PhysioNet cache (25,160 stays) from $GZ_FOUND ..."
    gzip -dc "$GZ_FOUND" > "$CACHE_S42"
    ln -sf "$CACHE_S42" "$CACHE_PKL"
  else
    echo "ERROR: Full PhysioNet cache ($PCL_REPO/colab/physionet_frac1.0.pkl.gz) not found. Make sure you ran git pull."
    exit 1
  fi
fi

echo "=== 3. Preflight check ==="
python3 "$PCL_REPO/pod/preflight.py"
