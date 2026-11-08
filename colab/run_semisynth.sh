#!/usr/bin/env bash
# Semi-synthetic shift + length-shortcut positive-control experiments on Google Colab (PhysioNet 2019 only).
# Reads fine-tuned checkpoints from $PCL_LEGACY2_CKPT_ROOT and pretrained encoders from $PCL_LEGACY2_PRETRAIN_DIR/phase2.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
[ "${PCL_PHYSIONET_ONLY:-1}" = "1" ] || { echo "PhysioNet-only Colab script"; exit 2; }
LAMS="${LAMS:-0 0.5 1}"; SEEDS="${SEEDS:-42 43 44 45 46}"; PREC="${PREC:-fp32}"
PH2="$PCL_LEGACY2_PRETRAIN_DIR/phase2"; CK="$PCL_LEGACY2_CKPT_ROOT"
OUTS=results/phase2/semisynth_shift.csv; OUTL=results/phase2/semisynth_length.csv
mkdir -p results/phase2 results/revision/logs "$PCL_COLAB_PERSIST/results_phase2"
export PCL_DISABLE_AMP=1 NUM_WORKERS=0
have() { [ -f "$1" ] && grep -q "^$2," "$1"; }
for L in $LAMS; do for S in $SEEDS; do
  TAG="lam${L}_p${S}_${PREC}"; ENC="$PH2/$TAG.pt"
  [ -f "$ENC" ] || { echo "missing $ENC (run colab/run_phase2.sh first)"; continue; }
  LOG="results/revision/logs/semisynth_$TAG.log"
  if [ -f "$CK/phase2_sepsis/${TAG}_s${S}.pt" ] && ! have $OUTS "$TAG,sepsis"; then
    python3 scripts/semisynth_eval.py --mode shift --task sepsis --ft-ckpt "$CK/phase2_sepsis/${TAG}_s${S}.pt" --seed "$S" --tag "$TAG" --out $OUTS >> "$LOG" 2>&1 || echo "$TAG shift/sepsis FAILED"
  fi
  if [ -f "$CK/los_T24/ckpt/${TAG}_s${S}.pt" ] && ! have $OUTS "$TAG,los"; then
    python3 scripts/semisynth_eval.py --mode shift --task los --fixed-T 24 --ft-ckpt "$CK/los_T24/ckpt/${TAG}_s${S}.pt" --seed "$S" --tag "$TAG" --out $OUTS >> "$LOG" 2>&1 || echo "$TAG shift/los FAILED"
  fi
  for T in "" 24; do
    FT=(); [ -n "$T" ] && FT=(--fixed-T "$T")
    if ! grep -q "^$TAG,$S,$T," $OUTL 2>/dev/null; then
      python3 scripts/semisynth_eval.py --mode length --pretrain-ckpt "$ENC" --seed "$S" --tag "$TAG" ${FT[@]+"${FT[@]}"} --out $OUTL >> "$LOG" 2>&1 || echo "$TAG length T=$T FAILED"
    fi
  done
  cp -rf results/phase2/* "$PCL_COLAB_PERSIST/results_phase2/" 2>/dev/null || true
  echo "[$(date +%H:%M:%S)] $TAG done"
done; done
