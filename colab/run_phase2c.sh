#!/usr/bin/env bash
# Phase 2c Reviewer Experiments:
# 1. Steps-Matched PCL Control (pcl_matched_vol_p{seed}): PCL lambda=0.5 on 25k stays (10,800 steps)
# 2. Supervised Transformer From Scratch (scratch_p{seed}): No pretraining baseline
# 3. VICReg Anti-Collapse Intervention (pcl_varreg1_p{seed}): PCL lambda=0.5 + VICReg variance/covariance reg
# 4. MAP-Only Constraint Ablation (pcl_maponly_p{seed}): MAP constraint only (HH=0, SpO2=0)
# 5. No-HH Constraint Ablation (pcl_nohh_p{seed}): MAP + SpO2 only (HH=0)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
python3 pod/preflight.py >/dev/null || { python3 pod/preflight.py; exit 1; }

JOBS="${JOBS:-2}"
SEEDS="${SEEDS:-42 43 44 45 46}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
P2C_DIR="$PCL_LEGACY2_PRETRAIN_DIR/phase2c"
mkdir -p "$P2C_DIR" results/phase2c/pretrain results/phase2c/sepsis results/phase2c/los_T24 results/revision/logs "$PCL_COLAB_PERSIST/results_phase2c"

echo "=== Running Phase 2c Reviewer Experiments (JOBS=$JOBS) ==="
python3 scripts/run_phase2c_reviewer.py --jobs "$JOBS" --seeds $SEEDS --out-dir "$P2C_DIR"

cp -rf results/phase2c/* "$PCL_COLAB_PERSIST/results_phase2c/" 2>/dev/null || true
zip -q -r /content/phase2c_results.zip results/phase2c 2>/dev/null || true
echo "=== Phase 2c Reviewer Experiments ALL DONE ==="
