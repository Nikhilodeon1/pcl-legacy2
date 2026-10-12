#!/usr/bin/env bash
# Independent-pretraining sweep (revision B1/B2/B3), N fine-tunes at a time on ONE GPU, fp32 (AMP off; PCL fine-tunes are unstable in fp16 and results must match the fp32 main batch).
# Encoders: $PCL_LEGACY2_PRETRAIN_DIR/sweep/lam{L}_p{P}.pt  (lambda L, pretraining seed P; lambda 0.0 = ERM). Fine-tune seed = P. Fixed-observation protocol: LOS T=24, mortality T=48.
# Order: whole pretraining seeds first (42, then 43, then 44), lambda 0.5 / 1.0 / 0.0 within each, so partial results are still usable.
#   JOBS=4 bash pod/run_sweep_parallel.sh                      # lambda 0.0 0.5 1.0, seeds 42 43 44: 9 encoders x (LOS + 2 mortality) = 27 runs, about 2-2.5 h
#   LAMS="0.1 2.0 5.0" JOBS=4 bash pod/run_sweep_parallel.sh   # the other lambdas afterwards
#   ONLY=los | ONLY=mortality   # one task only
# Resumable: finished JSONs are skipped, so after a crash/refresh just rerun the same command.
# Outputs: results/los_fixed_T24_sweep/lam*_p*.json, results/mortality_fixed_T48_sweep/lam*_p*.json ; logs in results/revision/logs/sweep_*.log
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
python pod/preflight.py >/dev/null || { python pod/preflight.py; exit 1; }
JOBS="${JOBS:-4}"; LAMS="${LAMS:-0.5 1.0 0.0}"; PSEEDS="${PSEEDS:-42 43 44}"; ONLY="${ONLY:-all}"
export NUM_WORKERS="${NUM_WORKERS:-5}" PCL_DISABLE_AMP=1
mkdir -p results/revision/logs
LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
for P in $PSEEDS; do for L in $LAMS; do
  CK="$PCL_LEGACY2_PRETRAIN_DIR/sweep/lam${L}_p${P}.pt"
  [ -s "$CK" ] || { echo "MISSING $CK -- skipping" >&2; continue; }
  [ "$ONLY" != "mortality" ] && echo "los $L $P mimic" >> "$LIST"
  if [ "$ONLY" != "los" ]; then for SRC in mimic eicu; do echo "mort $L $P $SRC" >> "$LIST"; done; fi
done; done
# LOS has no source argument; keep one LOS line per encoder
awk '!($1=="los" && seen[$2" "$3]++)' "$LIST" > "$LIST.u" && mv "$LIST.u" "$LIST"
echo "$(wc -l < "$LIST") jobs, $JOBS at a time, fp32"
run_one() {
  KIND=$1; L=$2; P=$3; SRC=$4; T0=$(date +%s)
  M=pcl; [ "$L" = "0.0" ] && M=erm
  ID="lam${L}_p${P}"; CK="$PCL_LEGACY2_PRETRAIN_DIR/sweep/${ID}.pt"
  LOG="results/revision/logs/sweep_${KIND}_${ID}_${SRC}.log"
  if [ "$KIND" = "los" ]; then
    python scripts/finetune_los.py --method "$M" --seed "$P" --fixed-T 24 --pretrain-ckpt "$CK" --encoder-id "$ID" > "$LOG" 2>&1
  else
    python scripts/finetune_mortality.py --method "$M" --source "$SRC" --seed "$P" --fixed-T 48 --pretrain-ckpt "$CK" --encoder-id "$ID" > "$LOG" 2>&1
  fi
  rc=$?
  echo "[$(date +%H:%M:%S)] ${KIND} ${ID} ${SRC} exit=$rc in $(( ($(date +%s)-T0)/60 )) min"
}
export -f run_one
T0=$(date +%s)
xargs -P "$JOBS" -L1 bash -c 'run_one "$@"' _ < "$LIST"
echo "ALL DONE in $(( ($(date +%s)-T0)/60 )) min"
