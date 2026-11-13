#!/usr/bin/env bash
# Head Pilot Driver for GPU Pod / JupyterHub / Colab
# Tests LayerNorm on ClassificationHead, LP-FT, and baseline on Sepsis (PhysioNet) and Mortality (if MIMIC present).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
CACHE_DIR="${CACHE_DIR:-/tmp/pcl_cache}"
RESULTS_DIR="${REPO_DIR}/results/head_pilot"

if [ -d "/content/drive/MyDrive" ]; then
    PERSIST_DIR="${PERSIST_DIR:-/content/drive/MyDrive/pcl_colab}"
else
    PERSIST_DIR="${PERSIST_DIR:-${REPO_DIR}/results/persist}"
fi

if [ -d "${REPO_DIR}/results/revision/encoders_slim/sweep" ]; then
    PRETRAIN_DIR="${PRETRAIN_DIR:-${REPO_DIR}/results/revision/encoders_slim/sweep}"
elif [ -d "/content/drive/MyDrive/pcl_colab/pretrained" ]; then
    PRETRAIN_DIR="${PRETRAIN_DIR:-/content/drive/MyDrive/pcl_colab/pretrained}"
else
    PRETRAIN_DIR="${PRETRAIN_DIR:-${REPO_DIR}/results/pretrained_pilot}"
fi

cd "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}" "${CACHE_DIR}" "${PRETRAIN_DIR}"

export PCL_TEST_MODE=0
export NUM_WORKERS="${NUM_WORKERS:-0}"
export PCL_LEGACY2_CACHE_DIR="${CACHE_DIR}"

echo "=== Running Head Pilot ==="
echo "[env] repo=${REPO_DIR} pretrain=${PRETRAIN_DIR} cache=${CACHE_DIR} results=${RESULTS_DIR}"
python -c "import torch; print(f'[torch] version={torch.__version__}, cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}')"
if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[WARNING] CUDA is NOT available in this Python environment (running on CPU)."
    echo "[WARNING] Full 30-epoch pretraining + fine-tuning on CPU will be slow; consider activating a CUDA env if a GPU is attached."
fi

# Unpack bundled PhysioNet cache if not already unpacked
if [ ! -s "${CACHE_DIR}/physionet_frac1.0.pkl" ]; then
    echo "=== Unpacking bundled PhysioNet 2019 dataset cache ==="
    if [ -f "${REPO_DIR}/colab/physionet_frac1.0.pkl.gz" ]; then
        gzip -dc "${REPO_DIR}/colab/physionet_frac1.0.pkl.gz" > "${CACHE_DIR}/physionet_frac1.0.pkl"
        echo "Unpacked ${CACHE_DIR}/physionet_frac1.0.pkl ($(( $(stat -c%s "${CACHE_DIR}/physionet_frac1.0.pkl") / 1048576 )) MB)"
    else
        echo "ERROR: ${REPO_DIR}/colab/physionet_frac1.0.pkl.gz not found."
        exit 1
    fi
fi

# Seeds 42, 43, 44 match the sweep encoders (seed 43 collapsed under mortality PCL lam0.5)
SEEDS=(42 43 44)
MODES=("baseline" "layernorm" "lpft")

get_or_pretrain_ckpt() {
    local lam=$1
    local seed=$2
    local ckpt1="${PRETRAIN_DIR}/lam${lam}_p${seed}.pt"
    local ckpt2="${PRETRAIN_DIR}/lam${lam}_p${seed}_fp32.pt"
    if [ -f "$ckpt1" ]; then
        echo "$ckpt1"
        return 0
    fi
    if [ -f "$ckpt2" ]; then
        echo "$ckpt2"
        return 0
    fi
    echo "=== [Pretrain] Training checkpoint: lam=${lam}, seed=${seed} on Site A ===" >&2
    python scripts/pretrain.py \
        --lam "${lam}" \
        --seed "${seed}" \
        --precision fp32 \
        --out-dir "${PRETRAIN_DIR}" \
        --cache-dir "${CACHE_DIR}" >&2
    if [ -f "$ckpt2" ]; then
        echo "$ckpt2"
        return 0
    elif [ -f "$ckpt1" ]; then
        echo "$ckpt1"
        return 0
    else
        echo "ERROR: Expected checkpoint not produced at $ckpt2 or $ckpt1" >&2
        return 1
    fi
}

# 1. Sepsis Head Pilot (PhysioNet Site A -> Site B, tests PCL lam0.5 and ERM lam0 across baseline, layernorm, lpft)
echo "=== Running Sepsis Head Pilot on PhysioNet 2019 ==="
for seed in "${SEEDS[@]}"; do
    CKPT_PCL=$(get_or_pretrain_ckpt 0.5 "${seed}")
    CKPT_ERM=$(get_or_pretrain_ckpt 0 "${seed}")

    for mode in "${MODES[@]}"; do
        echo "=== [Sepsis] Mode: ${mode} | Seed: ${seed} | PCL lam0.5 (${CKPT_PCL}) ==="
        python scripts/run_head_pilot.py \
            --task sepsis \
            --source physionet_a \
            --seed "${seed}" \
            --tag "pcl_lam0.5" \
            --head-mode "${mode}" \
            --pretrain-ckpt "${CKPT_PCL}" \
            --cache-dir "${CACHE_DIR}" \
            --fixed-T 24 \
            --epochs 30

        echo "=== [Sepsis] Mode: ${mode} | Seed: ${seed} | ERM lam0 (${CKPT_ERM}) ==="
        python scripts/run_head_pilot.py \
            --task sepsis \
            --source physionet_a \
            --seed "${seed}" \
            --tag "erm_lam0" \
            --head-mode "${mode}" \
            --pretrain-ckpt "${CKPT_ERM}" \
            --cache-dir "${CACHE_DIR}" \
            --fixed-T 24 \
            --epochs 30
    done
done

# 2. Mortality Head Pilot (only if MIMIC data / cache is present)
if [ -f "${REPO_DIR}/results/revision/cache/mimic_frac1.0_s42.pkl" ] || [ -f "${REPO_DIR}/results/revision/cache_gz/mimic_frac1.0.pkl.gz" ] || [ -d "${REPO_DIR}/data/mimic4-demo" ]; then
    echo "=== Running Mortality Head Pilot (MIMIC-IV data detected) ==="
    for seed in "${SEEDS[@]}"; do
        CKPT_PCL=$(get_or_pretrain_ckpt 0.5 "${seed}")
        CKPT_ERM=$(get_or_pretrain_ckpt 0 "${seed}")

        for mode in "${MODES[@]}"; do
            echo "=== [Mortality] Mode: ${mode} | Seed: ${seed} | PCL lam0.5 ==="
            python scripts/run_head_pilot.py \
                --task mortality_hospital \
                --source mimic \
                --seed "${seed}" \
                --tag "pcl_lam0.5" \
                --head-mode "${mode}" \
                --pretrain-ckpt "${CKPT_PCL}" \
                --cache-dir "${CACHE_DIR}" \
                --fixed-T 48 \
                --epochs 30

            echo "=== [Mortality] Mode: ${mode} | Seed: ${seed} | ERM lam0 ==="
            python scripts/run_head_pilot.py \
                --task mortality_hospital \
                --source mimic \
                --seed "${seed}" \
                --tag "erm_lam0" \
                --head-mode "${mode}" \
                --pretrain-ckpt "${CKPT_ERM}" \
                --cache-dir "${CACHE_DIR}" \
                --fixed-T 48 \
                --epochs 30
        done
    done
else
    echo "[info] MIMIC-IV data not present on pod (remote pods enforce PCL_PHYSIONET_ONLY=1)."
    echo "[info] Sepsis head pilot completed across all modes and seeds."
fi

# Sync results to Colab Google Drive only if mounted
if [ -d "/content/drive/MyDrive" ]; then
    mkdir -p "${PERSIST_DIR}/head_pilot"
    cp -ru "${RESULTS_DIR}"/* "${PERSIST_DIR}/head_pilot/" || true
    echo "Synced results to ${PERSIST_DIR}/head_pilot"
fi

echo "=== Head Pilot Completed ==="
