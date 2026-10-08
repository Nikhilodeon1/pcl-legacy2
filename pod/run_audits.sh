#!/usr/bin/env bash
# Inference-only audits on the ORIGINAL fine-tuned checkpoints (a few minutes of GPU). Each step skips itself if its inputs are missing.
#   bash pod/run_audits.sh
# Needs: full caches (pod/build_caches.sh), results/los/ckpt (9) and/or results/mortality/ckpt (18) -- copy them into the repo first.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
S=results/revision/scripts
mkdir -p results/revision/logs "$PCL_PREDS_DIR"
PD="--out-dir $PCL_PREDS_DIR"
PR="--pred-dir $PCL_PREDS_DIR"
have() { [ -e "$1" ]; }
run() { echo; echo "=== $*"; "$@" || echo "!! step failed (continuing): $*"; }

have "$PCL_CACHE_DIR/mimic_frac1.0.pkl" || { echo "no full caches in $PCL_CACHE_DIR -> run: bash pod/build_caches.sh"; exit 1; }
run python $S/g1_probe.py --mimic-frac 1.0 --eicu-frac 1.0
run python $S/g2_cohort.py --mimic-frac 1.0 --eicu-frac 1.0
if [ "$(ls results/los/ckpt/*.pt 2>/dev/null | wc -l)" -ge 9 ]; then
  run python $S/g1_shuffle.py --task los --seed 42 --mimic-frac 1.0 --eicu-frac 1.0 --n-eval 20000 --device cuda
  run python $S/predict_los.py --mimic-frac 1.0 --eicu-frac 1.0 --device cuda $PD
  run python $S/a1_los_diagnostics.py $PR
  run python $S/a3_bootstrap_los.py $PR
else echo "[skip] LOS audits: need 9 ckpts in results/los/ckpt"; fi
if [ "$(ls results/mortality/ckpt/*.pt 2>/dev/null | wc -l)" -ge 18 ]; then
  run python $S/predict_mortality.py --ckpt-dir results/mortality/ckpt --cache-dir "$PCL_CACHE_DIR" --frac 1.0 --device cuda $PD
  run python $S/g1_shuffle.py --task mortality --mort-ckpt-dir results/mortality/ckpt --seed 42 --mimic-frac 1.0 --eicu-frac 1.0 --n-eval 20000 --device cuda
  run python $S/a3_bootstrap_mortality.py $PR
else echo "[skip] mortality audits: need 18 ckpts in results/mortality/ckpt (else rerun original-protocol fine-tunes: scripts/finetune_mortality.py without --fixed-T)"; fi
have results/mortality/selection_criteria.json && run python $S/a4_selection.py --task mortality
echo; echo "outputs: results/revision/{g1_*,g2_*,a1_*,a3_*,preds/}. Commit the small CSV/JSON; preds/ and cache/ are gitignored."
