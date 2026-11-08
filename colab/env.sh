#!/usr/bin/env bash
# Google Colab environment configuration (PhysioNet 2019 only; uses Colab's pre-installed CUDA PyTorch).
_here="${BASH_SOURCE[0]:-$0}"
export PCL_REPO="$(cd "$(dirname "$_here")/.." && pwd)"

# If Google Drive is mounted at /content/drive/MyDrive, persist caches/checkpoints/results there automatically
if [ -z "${PCL_COLAB_PERSIST:-}" ]; then
  if [ -d "/content/drive/MyDrive" ]; then
    export PCL_COLAB_PERSIST="/content/drive/MyDrive/pcl_colab"
  else
    export PCL_COLAB_PERSIST="/content/pcl_persist"
  fi
fi
mkdir -p "$PCL_COLAB_PERSIST"

export PCL_DATA_ROOT="${PCL_DATA_ROOT:-/tmp/pcl_data}"
export PHYSIONET_DIR="${PHYSIONET_DIR:-$PCL_DATA_ROOT/physionet2019}"
export MIMIC_DIR="${MIMIC_DIR:-$PCL_DATA_ROOT/mimiciv/3.1}"
export EICU_DIR="${EICU_DIR:-$PCL_DATA_ROOT/eicu-crd/2.0}"

# Enforce PhysioNet-only mode (credentialed MIMIC-IV / eICU-CRD must never be on Colab)
export PCL_PHYSIONET_ONLY=1
export PCL_LEGACY2_PRETRAIN_DIR="${PCL_LEGACY2_PRETRAIN_DIR:-$PCL_COLAB_PERSIST/pretrained}"
export PCL_LEGACY2_CACHE_DIR="${PCL_LEGACY2_CACHE_DIR:-/tmp/pcl_cache}"
export PCL_CACHE_DIR="$PCL_LEGACY2_CACHE_DIR"
export PCL_LEGACY2_CKPT_ROOT="${PCL_LEGACY2_CKPT_ROOT:-$PCL_COLAB_PERSIST/ckpt}"
export PCL_PREDS_DIR="${PCL_PREDS_DIR:-/tmp/pcl_preds}"

# ICUDataset keeps all tensors in RAM; NUM_WORKERS=0 avoids multiprocessing fork RAM bloat on 12 GB Colab VMs
export NUM_WORKERS="${NUM_WORKERS:-0}"
export PCL_TEST_MODE=0

echo "[colab env] repo=$PCL_REPO persist=$PCL_COLAB_PERSIST cache=$PCL_LEGACY2_CACHE_DIR pretrain=$PCL_LEGACY2_PRETRAIN_DIR ckpt=$PCL_LEGACY2_CKPT_ROOT"
