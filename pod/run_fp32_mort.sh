set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
python pod/preflight.py >/dev/null || { python pod/preflight.py; exit 1; }
JOBS="${JOBS:-3}"; METHODS="${METHODS:-pcl erm dro}"
export NUM_WORKERS="${NUM_WORKERS:-6}" PCL_DISABLE_AMP=1
mkdir -p results/revision/logs
LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
for M in $METHODS; do for S in 42 43 44; do for SRC in mimic eicu; do echo "$M $S $SRC" >> "$LIST"; done; done; done
echo "$(wc -l < "$LIST") jobs, $JOBS at a time, fp32 (PCL_DISABLE_AMP=1)"
run_one() {
  M=$1; S=$2; SRC=$3; T0=$(date +%s)
  python scripts/finetune_mortality.py --method "$M" --source "$SRC" --seed "$S" --fixed-T 48 \
    --pretrain-ckpt "$PCL_LEGACY2_PRETRAIN_DIR/${M}_pretrained.pt" --encoder-id "${M}_fp32" \
    > "results/revision/logs/fp32_${M}_s${S}_${SRC}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] ${M}_fp32 ${SRC} s${S} exit=$rc in $(( ($(date +%s)-T0)/60 )) min"
}
export -f run_one
T0=$(date +%s)
xargs -P "$JOBS" -L1 bash -c 'run_one "$@"' _ < "$LIST"
echo "ALL DONE in $(( ($(date +%s)-T0)/60 )) min"
