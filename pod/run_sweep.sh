#!/usr/bin/env bash
# Lambda x pretraining-seed sweep under the fixed-observation protocol (revision B2 + B3).
# 3 independently pretrained encoders per lambda in {0.0 (=ERM), 0.1, 0.5, 1.0, 2.0, 5.0}; each fine-tuned once (fine-tune seed = pretraining seed)
# on LOS (T=24) and both mortality directions (T=48): 18 encoders x 3 runs = 54 runs, roughly 4-6 h. Resumable (finished JSONs are skipped).
#   bash pod/run_sweep.sh                 # everything
#   ONLY=los bash pod/run_sweep.sh        # LOS first (18 runs, about 1 h) -- or ONLY=mortality
#   LAMS="0.0 0.5" PSEEDS="43 44" bash pod/run_sweep.sh     # subsets
# Needs the slimmed encoders in $PCL_LEGACY2_PRETRAIN_DIR/sweep/lam{L}_p{S}.pt (see results/revision/scripts/slim_encoders.py).
# Outputs: results/los_fixed_T24_sweep/*.json, results/mortality_fixed_T48_sweep/*.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
python pod/preflight.py >/dev/null || { python pod/preflight.py; exit 1; }
LAMS="${LAMS:-0.0 0.5 1.0 0.1 2.0 5.0}"      # 0.0/0.5/1.0 first: the most informative if time runs short
PSEEDS="${PSEEDS:-42 43 44}"
ONLY="${ONLY:-all}"
mkdir -p results/revision/logs
LOG=results/revision/logs/run_sweep_$(date +%Y%m%d_%H%M).log
T0=$(date +%s)
el() { awk -v t0="$T0" -v now="$(date +%s)" 'BEGIN{printf "  [elapsed %.2f h]\n",(now-t0)/3600}'; }
{
for L in $LAMS; do for P in $PSEEDS; do
  CK="$PCL_LEGACY2_PRETRAIN_DIR/sweep/lam${L}_p${P}.pt"
  [ -s "$CK" ] || { echo "MISSING $CK -- skipping"; continue; }
  M=pcl; [ "$L" = "0.0" ] && M=erm
  ID="lam${L}_p${P}"
  if [ "$ONLY" != "mortality" ]; then
    echo "=== LOS | $ID"; python scripts/finetune_los.py --method $M --seed $P --fixed-T 24 --pretrain-ckpt "$CK" --encoder-id "$ID"; el
  fi
  if [ "$ONLY" != "los" ]; then
    for SRC in mimic eicu; do
      echo "=== mortality | $ID $SRC"; python scripts/finetune_mortality.py --method $M --source $SRC --seed $P --fixed-T 48 --pretrain-ckpt "$CK" --encoder-id "$ID"; el
    done
  fi
done; done
echo "SWEEP DONE"; el
} 2>&1 | tee -a "$LOG"
