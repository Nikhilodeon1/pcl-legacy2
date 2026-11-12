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
echo "[env] repo=${REPO_DIR} persist=${PERSIST_DIR} pretrain=${PRETRAIN_DIR}"

# Seeds 42, 43, 44 match the available sweep encoders (seed 43 is the collapsed run)
SEEDS=(42 43 44)
MODES=("baseline" "layernorm" "lpft")

# 1. Mortality Head Pilot (Testing on lam0.5 and lam0.0)
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

# Sync results to Google Drive if mounted
if [ -d "${PERSIST_DIR}" ]; then
    mkdir -p "${PERSIST_DIR}/head_pilot"
    cp -ru "${RESULTS_DIR}"/* "${PERSIST_DIR}/head_pilot/" || true
    echo "Synced results to ${PERSIST_DIR}/head_pilot"
fi

echo "=== Head Pilot Completed ==="
