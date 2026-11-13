#!/usr/bin/env bash
# Sepsis Lambda Sweep Driver: Addressing Mock Review Step 2
# Tests lambda in {0.01, 0.05, 0.1, 0.5, 1.0} plus ERM (0.0) on PhysioNet Site A -> Site B
# Logs pretraining gradient norms, loss shares, and downstream transfer performance.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
if [ -d "/content/drive/MyDrive" ]; then
    PERSIST_DIR="${PERSIST_DIR:-/content/drive/MyDrive/pcl_colab}"
else
    PERSIST_DIR="${PERSIST_DIR:-${REPO_DIR}/results/persist}"
fi
CACHE_DIR="${CACHE_DIR:-/tmp/pcl_cache}"
RESULTS_DIR="${REPO_DIR}/results/sepsis_lambda_sweep"
PRETRAIN_DIR="${REPO_DIR}/results/pretrain_lambda_sweep"

cd "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}" "${PRETRAIN_DIR}" "${CACHE_DIR}"

export PCL_TEST_MODE=0
export NUM_WORKERS="${NUM_WORKERS:-0}"
export PCL_LEGACY2_CACHE_DIR="${CACHE_DIR}"

echo "=== Running Sepsis Lambda Sweep ==="
echo "[env] repo=${REPO_DIR} cache=${CACHE_DIR} results=${RESULTS_DIR}"
python -c "import torch; print(f'[torch] version={torch.__version__}, cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}')"

# Unpack bundled PhysioNet cache if not present
if [ ! -s "${CACHE_DIR}/physionet_frac1.0.pkl" ]; then
    echo "=== Unpacking bundled PhysioNet 2019 dataset cache ==="
    if [ -f "${REPO_DIR}/colab/physionet_frac1.0.pkl.gz" ]; then
        gzip -dc "${REPO_DIR}/colab/physionet_frac1.0.pkl.gz" > "${CACHE_DIR}/physionet_frac1.0.pkl"
        echo "Unpacked ${CACHE_DIR}/physionet_frac1.0.pkl ($(( $(stat -c%s "${CACHE_DIR}/physionet_frac1.0.pkl") / 1048576 )) MB)"
    fi
fi

LAMBDAS=(0 0.01 0.05 0.1 0.5 1)
SEEDS=(42 43 44)

for lam in "${LAMBDAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        tag="lam${lam}_p${seed}_fp32"
        ckpt="${PRETRAIN_DIR}/${tag}.pt"
        pretrain_json="${PRETRAIN_DIR}/${tag}.json"
        ft_json="${RESULTS_DIR}/${tag}_s${seed}_T24.json"

        # 1. Pretrain on Site A
        if [ ! -f "${ckpt}" ]; then
            echo "=== [Pretrain] Lambda: ${lam} | Seed: ${seed} ==="
            python scripts/pretrain.py \
                --lam "${lam}" \
                --seed "${seed}" \
                --precision fp32 \
                --epochs 30 \
                --out-dir "${PRETRAIN_DIR}" \
                --cache-dir "${CACHE_DIR}"
        else
            echo "[skip] Pretrain checkpoint exists: ${ckpt}"
        fi

        # 2. Fine-tune on Sepsis (Site A -> Site B)
        if [ ! -f "${ft_json}" ]; then
            echo "=== [Fine-tune Sepsis] Lambda: ${lam} | Seed: ${seed} ==="
            python scripts/finetune_sepsis.py \
                --pretrain-ckpt "${ckpt}" \
                --encoder-id "${tag}" \
                --seed "${seed}" \
                --fixed-T 24 \
                --epochs 30 \
                --out-dir "${RESULTS_DIR}" \
                --cache-dir "${CACHE_DIR}"
        else
            echo "[skip] Sepsis fine-tune exists: ${ft_json}"
        fi
    done
done

# Sync results to Google Drive only if mounted
if [ -d "/content/drive/MyDrive" ]; then
    mkdir -p "${PERSIST_DIR}/sepsis_lambda_sweep"
    cp -ru "${RESULTS_DIR}"/* "${PERSIST_DIR}/sepsis_lambda_sweep/" || true
    echo "Synced results to ${PERSIST_DIR}/sepsis_lambda_sweep"
fi

echo "=== Sepsis Lambda Sweep Completed Successfully ==="
