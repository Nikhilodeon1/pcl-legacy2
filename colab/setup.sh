#!/usr/bin/env bash
# Fast Google Colab setup: uses Colab's native CUDA PyTorch, restores or downloads PhysioNet 2019 via zip, and runs preflight.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"

echo "=== 1. Hardware & Python check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARNING: nvidia-smi not found (check Runtime -> Change runtime type)"
free -h || true
python3 --version

# Install only if any required package is missing (all are pre-installed on standard Colab runtimes)
python3 -c "import torch, numpy, scipy, pandas, sklearn" 2>/dev/null || {
  echo "Installing missing Python packages..."
  python3 -m pip install -q -r "$PCL_REPO/pod/requirements-pod.txt"
}

mkdir -p "$PCL_LEGACY2_CACHE_DIR" "$PCL_LEGACY2_PRETRAIN_DIR" "$PCL_LEGACY2_CKPT_ROOT" "$PCL_REPO/results/revision/logs"

# Restore any previously saved results from persistent storage (e.g. Google Drive after a runtime reconnect)
if [ -d "$PCL_COLAB_PERSIST/results_phase2" ]; then
  echo "=== Restoring previous results/phase2 from $PCL_COLAB_PERSIST/results_phase2 ==="
  mkdir -p "$PCL_REPO/results/phase2"
  cp -rn "$PCL_COLAB_PERSIST/results_phase2/"* "$PCL_REPO/results/phase2/" 2>/dev/null || true
fi

echo "=== 2. PhysioNet 2019 cache setup ==="
CACHE_PKL="$PCL_LEGACY2_CACHE_DIR/physionet_frac1.0.pkl"
CACHE_S42="$PCL_LEGACY2_CACHE_DIR/physionet_frac1.0_s42.pkl"

if [ -s "$CACHE_PKL" ]; then
  echo "[skip] PhysioNet cache already present at $CACHE_PKL"
else
  # Check if a gzipped cache exists in Drive, /content, or repo
  GZ_FOUND=""
  for candidate in \
    "$PCL_COLAB_PERSIST/physionet_frac1.0.pkl.gz" \
    "/content/physionet_frac1.0.pkl.gz" \
    "$HOME/pcl_cache_gz/physionet_frac1.0.pkl.gz" \
    "$PCL_REPO/physionet_frac1.0.pkl.gz"; do
    if [ -s "$candidate" ]; then
      GZ_FOUND="$candidate"
      break
    fi
  done

  if [ -n "$GZ_FOUND" ]; then
    echo "Restoring PhysioNet cache from $GZ_FOUND ..."
    gzip -dc "$GZ_FOUND" > "$CACHE_S42"
    ln -sf "$CACHE_S42" "$CACHE_PKL"
    if [ "$GZ_FOUND" != "$PCL_COLAB_PERSIST/physionet_frac1.0.pkl.gz" ]; then
      cp -f "$GZ_FOUND" "$PCL_COLAB_PERSIST/physionet_frac1.0.pkl.gz" 2>/dev/null || true
    fi
  else
    # Fast single-zip download of PhysioNet 2019 (~42 MB zip instead of 40,336 individual wget requests)
    if [ ! -d "$PHYSIONET_DIR/training_setA" ] || [ ! -d "$PHYSIONET_DIR/training_setB" ]; then
      echo "Downloading PhysioNet 2019 Challenge zip archive..."
      mkdir -p "$PHYSIONET_DIR"
      ZIP_URL="https://physionet.org/static/published-projects/challenge-2019/sepsis-early-prediction-from-clinical-data-the-physionet-computing-in-cardiology-challenge-2019-1.0.0.zip"
      ZIP_PATH="/tmp/physionet2019_all.zip"
      if wget -c -q --show-progress -O "$ZIP_PATH" "$ZIP_URL"; then
        echo "Extracting training_setA and training_setB..."
        python3 - "$ZIP_PATH" "$PHYSIONET_DIR" <<'PY'
import os, sys, zipfile, shutil
zip_path, out_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as zf:
    for member in zf.namelist():
        if member.endswith("training_setA.zip") or member.endswith("training_setB.zip"):
            zf.extract(member, "/tmp/pn_extract")
            inner_zip = os.path.join("/tmp/pn_extract", member)
            with zipfile.ZipFile(inner_zip) as izf:
                izf.extractall(out_dir)
            os.remove(inner_zip)
# Normalize directory structure if training_setA extracted as training/training_setA
for sub in ("training_setA", "training_setB"):
    nested = os.path.join(out_dir, "training", sub)
    target = os.path.join(out_dir, sub)
    if os.path.isdir(nested) and not os.path.isdir(target):
        shutil.move(nested, target)
shutil.rmtree("/tmp/pn_extract", ignore_errors=True)
PY
        rm -f "$ZIP_PATH"
      else
        echo "Single-zip download unavailable; falling back to pod/fetch_data.sh..."
        bash "$PCL_REPO/pod/fetch_data.sh" physionet
      fi
    fi

    echo "Building PhysioNet cache..."
    (cd "$PCL_REPO/results/revision/scripts" && python3 build_cache.py physionet 1.0)
    [ -s "$CACHE_S42" ] && ln -sf "$CACHE_S42" "$CACHE_PKL"
    echo "Saving compressed cache to $PCL_COLAB_PERSIST/physionet_frac1.0.pkl.gz for future sessions..."
    gzip -c "$CACHE_S42" > "$PCL_COLAB_PERSIST/physionet_frac1.0.pkl.gz" || true
  fi
fi

echo "=== 3. Preflight check ==="
python3 "$PCL_REPO/pod/preflight.py"
