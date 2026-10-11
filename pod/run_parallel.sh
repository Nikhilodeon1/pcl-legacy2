#!/usr/bin/env bash
# Run the fixed-observation fine-tunes N at a time on ONE GPU (same protocol per run as run_fixed_T.sh; only wall-clock changes).
#   JOBS=3 ONLY=mortality bash pod/run_parallel.sh
#   JOBS=4 bash pod/run_parallel.sh                 # LOS + mortality
# Per-job logs: results/revision/logs/par_<job>.log ; summary lines go to the terminal / your nohup file.
# Resumable: a job whose result JSON exists exits immediately. DO NOT run this at the same time as run_fixed_T.sh (same jobs would collide).
# Memory: every job loads the caches itself. Watch `free -g` / the cgroup line; if used memory nears the 64 GB limit, lower JOBS.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PCL_REPO"
python pod/preflight.py >/dev/null || { python pod/preflight.py; exit 1; }
JOBS="${JOBS:-3}"; ONLY="${ONLY:-all}"
export NUM_WORKERS="${NUM_WORKERS:-6}"          # dataloader workers PER job (32 cores / 3-4 jobs)
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
