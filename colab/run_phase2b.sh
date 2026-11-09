#!/usr/bin/env bash
# Phase 2b targeted experiments for TMLR submission:
# 1. Matched-Volume Pretraining Control:
#    ERM trained on 10k Site A vs 25k matched Site A (volume control) vs 10k Site A + 14k Site B (target exposure).
# 2. Loss-Balancer & Constraint Ablation:
#    PCL with adaptive EMA balancer vs fixed weight (no balancer) vs detached balancer.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
python3 pod/preflight.py >/dev/null || { python3 pod/preflight.py; exit 1; }

JOBS="${JOBS:-2}"
SEEDS="${SEEDS:-42 43 44 45 46}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
P2B_DIR="$PCL_LEGACY2_PRETRAIN_DIR/phase2b"
mkdir -p "$P2B_DIR" results/phase2b/pretrain results/phase2b/sepsis results/phase2b/los_T24 results/revision/logs "$PCL_COLAB_PERSIST/results_phase2b"

echo "=== Running Phase 2b Targeted Ablations (JOBS=$JOBS) ==="
python3 scripts/run_phase2b_ablations.py --jobs "$JOBS" --seeds $SEEDS --out-dir "$P2B_DIR"

cp -rf results/phase2b/* "$PCL_COLAB_PERSIST/results_phase2b/" 2>/dev/null || true
echo "=== Phase 2b Ablations ALL DONE ==="
