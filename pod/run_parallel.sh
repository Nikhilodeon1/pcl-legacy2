set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
python pod/preflight.py >/dev/null || { python pod/preflight.py; exit 1; }
JOBS="${JOBS:-3}"; ONLY="${ONLY:-all}"
export NUM_WORKERS="${NUM_WORKERS:-6}"
mkdir -p results/revision/logs
LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
for S in 42 43 44; do for M in erm pcl dro; do
  [ "$ONLY" != "mortality" ] && echo "los $M $S" >> "$LIST"
  if [ "$ONLY" != "los" ]; then for SRC in mimic eicu; do echo "mort $M $S $SRC" >> "$LIST"; done; fi
done; done
echo "$(wc -l < "$LIST") jobs, $JOBS at a time, NUM_WORKERS=$NUM_WORKERS each"
T0=$(date +%s)
xargs -P "$JOBS" -L1 bash pod/_one_job.sh < "$LIST"
echo "ALL DONE in $(( ($(date +%s)-T0)/60 )) min"
