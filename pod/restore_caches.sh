#!/usr/bin/env bash
# Restore caches from the persistent gzip copies into $PCL_LEGACY2_CACHE_DIR (/tmp/pcl_cache). Run on every new pod instead of rebuilding.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SRC="${PCL_PERSIST_DIR:-$HOME/pcl_cache_gz}"
mkdir -p "$PCL_LEGACY2_CACHE_DIR"
for d in physionet mimic eicu; do
  [ -s "$SRC/${d}_frac1.0.pkl.gz" ] || { echo "missing $SRC/${d}_frac1.0.pkl.gz"; exit 1; }
  [ -s "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl" ] && { echo "[skip] $d already present"; continue; }
  echo "restoring $d ..."
  # write to a temp name, then rename: a crash must never leave a truncated .pkl that later looks like a finished cache
  rm -f "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl.part"
  gunzip -c "$SRC/${d}_frac1.0.pkl.gz" > "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl.part" && mv "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl.part" "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl"
done
ls -la "$PCL_LEGACY2_CACHE_DIR"
