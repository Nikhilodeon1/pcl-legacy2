#!/usr/bin/env bash
# Phase-2 pretraining experiments (PhysioNet 2019 only; safe for the pod). Each job = pretrain -> sepsis fine-tune -> LOS fine-tune (T=24).
#
#   EXP=amp    lambda {0, 0.5, 1} x seeds x precision {amp, fp32}: does fp16 pretraining cause the near-constant PCL representation / unstable fine-tunes?
#   EXP=leak   lambda {0, 0.5}    x seeds x fp32 x {no extra, Site B unlabeled in pretraining}: the "unlabeled target data in pretraining" confound, controlled
#   EXP=all    both (the shared fp32 / no-extra jobs run once)
#
#   JOBS=4 SEEDS="42 43 44 45 46" EXP=amp bash pod/run_phase2.sh
#
# Resumable: finished stages are skipped (JSON exists), so after a crash/refresh rerun the same command.
# Outputs: results/phase2/{pretrain,sepsis,los_T24}/*.json (small, commit these); checkpoints in $PCL_LEGACY2_PRETRAIN_DIR/phase2 (pretrained, 13 MB each).
# Fine-tuning always runs in fp32 (PCL_DISABLE_AMP=1) so the only varied factor is the PRETRAINING precision.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
[ "${PCL_PHYSIONET_ONLY:-1}" = "1" ] || { echo "this script is for the PhysioNet-only pod setting"; exit 2; }
python pod/preflight.py >/dev/null || { python pod/preflight.py; exit 1; }
JOBS="${JOBS:-4}"; EXP="${EXP:-amp}"; SEEDS="${SEEDS:-42 43 44 45 46}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
PH2="$PCL_LEGACY2_PRETRAIN_DIR/phase2"; mkdir -p "$PH2" results/phase2/pretrain results/phase2/sepsis results/phase2/los_T24 results/revision/logs

LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
add() { echo "$1 $2 $3 $4" >> "$LIST"; }          # lambda seed precision extra(none|pb)
for S in $SEEDS; do
  if [ "$EXP" = "amp" ] || [ "$EXP" = "all" ]; then for L in 0 0.5 1; do for P in amp fp32; do add $L $S $P none; done; done; fi
  if [ "$EXP" = "leak" ] || [ "$EXP" = "all" ]; then for L in 0 0.5; do add $L $S fp32 none; add $L $S fp32 pb; done; fi
done
sort -u "$LIST" -o "$LIST"
echo "$(wc -l < "$LIST") jobs, $JOBS at a time"

run_job() {
  L=$1; S=$2; P=$3; X=$4; T0=$(date +%s)
  TAG="lam${L}_p${S}_${P}"; EXTRA=(); if [ "$X" = "pb" ]; then TAG="${TAG}_leakpb"; EXTRA=(--extra-unlabeled physionet_b); fi
  LOG="results/revision/logs/phase2_${TAG}.log"; : > "$LOG"
  env -u PCL_DISABLE_AMP python scripts/pretrain.py --lam "$L" --seed "$S" --precision "$P" ${EXTRA[@]+"${EXTRA[@]}"} --out-dir "$PH2" >> "$LOG" 2>&1 || { echo "[$(date +%H:%M:%S)] $TAG pretrain FAILED"; return 1; }
  cp -f "$PH2/$TAG.json" results/phase2/pretrain/
  PCL_DISABLE_AMP=1 python scripts/finetune_sepsis.py --pretrain-ckpt "$PH2/$TAG.pt" --encoder-id "$TAG" --seed "$S" --out-dir results/phase2/sepsis >> "$LOG" 2>&1 \
    || echo "[$(date +%H:%M:%S)] $TAG sepsis fine-tune FAILED"
  PCL_DISABLE_AMP=1 python scripts/finetune_los.py --method pcl --seed "$S" --fixed-T 24 --pretrain-ckpt "$PH2/$TAG.pt" --encoder-id "$TAG" \
    --results-subdir phase2/los_T24 >> "$LOG" 2>&1 || echo "[$(date +%H:%M:%S)] $TAG LOS fine-tune FAILED"
  echo "[$(date +%H:%M:%S)] $TAG done in $(( ($(date +%s)-T0)/60 )) min"
}
export -f run_job; export PH2
T0=$(date +%s)
xargs -P "$JOBS" -L1 bash -c 'run_job "$@"' _ < "$LIST"
echo "ALL DONE in $(( ($(date +%s)-T0)/60 )) min"
