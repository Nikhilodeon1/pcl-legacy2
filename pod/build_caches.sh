#!/usr/bin/env bash
# Build the FULL preprocessed caches (physionet, mimic, eicu) with the original loaders, in parallel (CPU-bound, ~45-60 min wall on 16 cores).
# Output: $PCL_LEGACY2_CACHE_DIR/{physionet,mimic,eicu}_frac1.0.pkl -- the exact names finetune_los.py / finetune_mortality.py / revision scripts read.
# This is CPU work on a GPU-priced pod; if you can, do it on a CPU-only pod and persist the result (see bottom).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$PCL_LEGACY2_CACHE_DIR" "$PCL_REPO/results/revision/logs"
cd "$PCL_REPO/results/revision/scripts"
for d in physionet mimic eicu; do
  if [ -s "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl" ]; then echo "[skip] $d already cached"; continue; fi
  python build_cache.py "$d" 1.0 > "$PCL_REPO/results/revision/logs/build_$d.log" 2>&1 &
done
wait
for d in physionet mimic eicu; do
  src="$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0_s42.pkl"
  [ -s "$src" ] && ln -sf "$src" "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl"
  ls -la "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl" || { echo "FAILED: $d (see results/revision/logs/build_$d.log)"; exit 1; }
done
echo "caches OK in $PCL_LEGACY2_CACHE_DIR"; du -sh "$PCL_LEGACY2_CACHE_DIR"
echo
echo "To survive a pod replacement (needs ~2-3 GB free in \$HOME; check 'df -h ~' first):"
echo "  mkdir -p ~/pcl_cache && cp -L $PCL_LEGACY2_CACHE_DIR/{physionet,mimic,eicu}_frac1.0.pkl ~/pcl_cache/"
