set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
python pod/preflight.py >/dev/null || { python pod/preflight.py; exit 1; }
LAMS="${LAMS:-0.0 0.5 1.0 0.1 2.0 5.0}" 
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
