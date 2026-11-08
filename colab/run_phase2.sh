#!/usr/bin/env bash
# Phase-2 pretraining + sepsis/LOS fine-tuning runner for Google Colab (PhysioNet 2019 only).
#
#   EXP=amp    lambda {0, 0.5, 1} x seeds x precision {amp, fp32}
#   EXP=leak   lambda {0, 0.5}    x seeds x fp32 x {no extra, Site B unlabeled in pretraining}
#   EXP=all    both (shared fp32 / no-extra jobs run once)
#
# Defaults to JOBS=2 and NUM_WORKERS=0 for standard 12.7 GB Colab T4/L4 runtimes.
# Automatically backs up results/phase2/ to $PCL_COLAB_PERSIST/results_phase2 after every job.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
[ "${PCL_PHYSIONET_ONLY:-1}" = "1" ] || { echo "this script is for the PhysioNet-only Colab setting"; exit 2; }
python3 pod/preflight.py >/dev/null || { python3 pod/preflight.py; exit 1; }

JOBS="${JOBS:-2}"
EXP="${EXP:-amp}"
SEEDS="${SEEDS:-42 43 44 45 46}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
PH2="$PCL_LEGACY2_PRETRAIN_DIR/phase2"
mkdir -p "$PH2" results/phase2/pretrain results/phase2/sepsis results/phase2/los_T24 results/revision/logs "$PCL_COLAB_PERSIST/results_phase2"

LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
add() { echo "$1 $2 $3 $4" >> "$LIST"; }
for S in $SEEDS; do
  if [ "$EXP" = "amp" ] || [ "$EXP" = "all" ]; then for L in 0 0.5 1; do for P in amp fp32; do add $L $S $P none; done; done; fi
  if [ "$EXP" = "leak" ] || [ "$EXP" = "all" ]; then for L in 0 0.5; do add $L $S fp32 none; add $L $S fp32 pb; done; fi
done
sort -u "$LIST" -o "$LIST"
echo "$(wc -l < "$LIST") jobs, $JOBS at a time (persist=$PCL_COLAB_PERSIST)"

run_job() {
  L=$1; S=$2; P=$3; X=$4; T0=$(date +%s)
  TAG="lam${L}_p${S}_${P}"; EXTRA=(); if [ "$X" = "pb" ]; then TAG="${TAG}_leakpb"; EXTRA=(--extra-unlabeled physionet_b); fi
  LOG="results/revision/logs/phase2_${TAG}.log"; : > "$LOG"
  env -u PCL_DISABLE_AMP python3 scripts/pretrain.py --lam "$L" --seed "$S" --precision "$P" ${EXTRA[@]+"${EXTRA[@]}"} --out-dir "$PH2" >> "$LOG" 2>&1 || { echo "[$(date +%H:%M:%S)] $TAG pretrain FAILED"; return 1; }
  cp -f "$PH2/$TAG.json" results/phase2/pretrain/
  PCL_DISABLE_AMP=1 python3 scripts/finetune_sepsis.py --pretrain-ckpt "$PH2/$TAG.pt" --encoder-id "$TAG" --seed "$S" --out-dir results/phase2/sepsis >> "$LOG" 2>&1 \
    || echo "[$(date +%H:%M:%S)] $TAG sepsis fine-tune FAILED"
  PCL_DISABLE_AMP=1 python3 scripts/finetune_los.py --method pcl --seed "$S" --fixed-T 24 --pretrain-ckpt "$PH2/$TAG.pt" --encoder-id "$TAG" \
    --results-subdir phase2/los_T24 >> "$LOG" 2>&1 || echo "[$(date +%H:%M:%S)] $TAG LOS fine-tune FAILED"
  cp -rf results/phase2/* "$PCL_COLAB_PERSIST/results_phase2/" 2>/dev/null || true
  echo "[$(date +%H:%M:%S)] $TAG done in $(( ($(date +%s)-T0)/60 )) min"
}
export -f run_job; export PH2
T0=$(date +%s)
xargs -P "$JOBS" -L1 bash -c 'run_job "$@"' _ < "$LIST"
cp -rf results/phase2/* "$PCL_COLAB_PERSIST/results_phase2/" 2>/dev/null || true
echo "ALL DONE in $(( ($(date +%s)-T0)/60 )) min"
