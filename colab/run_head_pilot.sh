#!/usr/bin/env bash
# Head Pilot Driver for Colab / GPU Pod
# Tests LayerNorm on ClassificationHead, LP-FT, and lower encoder LR on Sepsis & Mortality.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
PERSIST_DIR="${PERSIST_DIR:-/content/drive/MyDrive/pcl_colab}"
CACHE_DIR="${CACHE_DIR:-/tmp/pcl_cache}"
RESULTS_DIR="${REPO_DIR}/results/head_pilot"

if [ -d "${REPO_DIR}/results/revision/encoders_slim/sweep" ]; then
    PRETRAIN_DIR="${PRETRAIN_DIR:-${REPO_DIR}/results/revision/encoders_slim/sweep}"
else
    PRETRAIN_DIR="${PRETRAIN_DIR:-${PERSIST_DIR}/pretrained}"
fi

cd "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}" "${CACHE_DIR}"

echo "=== Running Head Pilot on GPU Pod ==="
echo "[env] repo=${REPO_DIR} persist=${PERSIST_DIR} pretrain=${PRETRAIN_DIR} cache=${CACHE_DIR}"

# Unpack bundled PhysioNet cache if not already unpacked
if [ ! -s "${CACHE_DIR}/physionet_frac1.0.pkl" ]; then
    echo "=== Unpacking bundled PhysioNet 2019 dataset cache ==="
    if [ -f "${REPO_DIR}/colab/physionet_frac1.0.pkl.gz" ]; then
        gzip -dc "${REPO_DIR}/colab/physionet_frac1.0.pkl.gz" > "${CACHE_DIR}/physionet_frac1.0.pkl"
        echo "Unpacked ${CACHE_DIR}/physionet_frac1.0.pkl ($(( $(stat -c%s "${CACHE_DIR}/physionet_frac1.0.pkl") / 1048576 )) MB)"
    fi
fi
export PCL_LEGACY2_CACHE_DIR="${CACHE_DIR}"

# Seeds 42, 43, 44 match the available sweep encoders (seed 43 is the collapsed run)
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
    mkdir -p "${PRETRAIN_DIR}"
    python scripts/pretrain.py \
        --lam "${lam}" \
        --seed "${seed}" \
        --precision fp32 \
        --out-dir "${PRETRAIN_DIR}" \
        --cache-dir "${CACHE_DIR}" >&2
    if [ -f "$ckpt2" ]; then
        echo "$ckpt2"
        return 0
    fi
    echo "$ckpt1"
}

# 1. Sepsis Head Pilot (PhysioNet Site A -> Site B, tests PCL lam0.5 and ERM lam0.0 across baseline, layernorm, lpft)
echo "=== Running Sepsis Head Pilot on PhysioNet 2019 ==="
for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CKPT_PCL=$(get_or_pretrain_ckpt 0.5 "${seed}")
        CKPT_ERM=$(get_or_pretrain_ckpt 0 "${seed}")

        echo "=== [Sepsis] Mode: ${mode} | Seed: ${seed} | PCL lam0.5 (${CKPT_PCL}) ==="
        python scripts/run_head_pilot.py \
            --task sepsis \
            --source physionet_a \
            --seed "${seed}" \
            --head-mode "${mode}" \
            --pretrain-ckpt "${CKPT_PCL}" \
            --fixed-T 24 \
            --epochs 30 || true

        echo "=== [Sepsis] Mode: ${mode} | Seed: ${seed} | ERM lam0.0 (${CKPT_ERM}) ==="
        python scripts/run_head_pilot.py \
            --task sepsis \
            --source physionet_a \
            --seed "${seed}" \
            --head-mode "${mode}" \
            --pretrain-ckpt "${CKPT_ERM}" \
            --fixed-T 24 \
            --epochs 30 || true
    done
done

# 2. Mortality Head Pilot (if MIMIC data / cache is present)
if [ -f "${REPO_DIR}/results/revision/cache/mimic_frac1.0_s42.pkl" ] || [ -f "${REPO_DIR}/results/revision/cache_gz/mimic_frac1.0.pkl.gz" ] || [ -d "${REPO_DIR}/data/mimic4-demo" ]; then
    echo "=== Running Mortality Head Pilot (MIMIC-IV data detected) ==="
    for mode in "${MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "=== [Mortality] Mode: ${mode} | Seed: ${seed} | PCL lam0.5 ==="
            python scripts/run_head_pilot.py \
                --task mortality_hospital \
                --source mimic \
                --seed "${seed}" \
                --head-mode "${mode}" \
                --pretrain-ckpt "${PRETRAIN_DIR}/lam0.5_p${seed}.pt" \
                --fixed-T 48 \
                --epochs 30 || true

            echo "=== [Mortality] Mode: ${mode} | Seed: ${seed} | ERM lam0.0 ==="
            python scripts/run_head_pilot.py \
                --task mortality_hospital \
                --source mimic \
                --seed "${seed}" \
                --head-mode "${mode}" \
                --pretrain-ckpt "${PRETRAIN_DIR}/lam0.0_p${seed}.pt" \
                --fixed-T 48 \
                --epochs 30 || true
        done
    done
else
    echo "[info] MIMIC-IV data not present on pod (remote pods enforce PCL_PHYSIONET_ONLY=1)."
    echo "[info] Sepsis head pilot completed across all modes and seeds."
fi

# Sync results to Google Drive if mounted
if [ -d "${PERSIST_DIR}" ]; then
    mkdir -p "${PERSIST_DIR}/head_pilot"
    cp -ru "${RESULTS_DIR}"/* "${PERSIST_DIR}/head_pilot/" || true
    echo "Synced results to ${PERSIST_DIR}/head_pilot"
fi

echo "=== Head Pilot Completed ==="
