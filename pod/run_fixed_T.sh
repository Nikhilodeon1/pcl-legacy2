set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
python pod/preflight.py >/dev/null || { python pod/preflight.py; exit 1; }
mkdir -p results/revision/logs
LOG=results/revision/logs/run_fixed_T_$(date +%Y%m%d_%H%M).log
T0=$(date +%s)
cost() { if [ -n "${POD_RATE_USD_PER_H:-}" ]; then awk -v t0="$T0" -v now="$(date +%s)" -v r="$POD_RATE_USD_PER_H" 'BEGIN{h=(now-t0)/3600; printf "  elapsed %.2f h, ~$%.2f\n", h, h*r}'; fi; }
ONLY="${ONLY:-all}"
{
for S in 42 43 44; do for M in erm pcl dro; do
  if [ "$ONLY" != "mortality" ]; then
    echo "=== LOS fixed T=24 | $M seed $S"; python scripts/finetune_los.py --method $M --seed $S --fixed-T 24; cost
  fi
  if [ "$ONLY" != "los" ]; then
    for SRC in mimic eicu; do
      echo "=== mortality fixed T=48 | $M $SRC seed $S"; python scripts/finetune_mortality.py --method $M --source $SRC --seed $S --fixed-T 48; cost
    done
  fi
done; done
echo "ALL DONE"; cost
} 2>&1 | tee -a "$LOG"
echo "results: results/los_fixed_T24/*.json results/mortality_fixed_T48/*.json  (commit these JSONs; checkpoints are gitignored)"
