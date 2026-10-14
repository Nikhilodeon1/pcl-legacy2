set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" >/dev/null
cd "$PCL_REPO"
KIND=$1; M=$2; S=$3; SRC=${4:-}
TAG="${KIND}_${M}_s${S}${SRC:+_$SRC}"
LOG="results/revision/logs/par_${TAG}.log"
T0=$(date +%s)
if [ "$KIND" = "los" ]; then
  python scripts/finetune_los.py --method "$M" --seed "$S" --fixed-T "${FIXEDT_LOS:-24}" > "$LOG" 2>&1
else
  python scripts/finetune_mortality.py --method "$M" --source "$SRC" --seed "$S" --fixed-T "${FIXEDT_MORT:-48}" > "$LOG" 2>&1
fi
rc=$?
echo "[$(date +%H:%M:%S)] $TAG exit=$rc in $(( ($(date +%s)-T0)/60 )) min"
