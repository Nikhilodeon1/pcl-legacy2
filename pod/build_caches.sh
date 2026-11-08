set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$PCL_LEGACY2_CACHE_DIR" "$PCL_REPO/results/revision/logs"
cd "$PCL_REPO/results/revision/scripts"
DS=(physionet mimic eicu); [ "${PCL_PHYSIONET_ONLY:-1}" = "1" ] && DS=(physionet)
for d in "${DS[@]}"; do
  if [ -s "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl" ]; then echo "[skip] $d already cached"; continue; fi
  if [ "${SEQUENTIAL:-0}" = "1" ]; then python build_cache.py "$d" 1.0 > "$PCL_REPO/results/revision/logs/build_$d.log" 2>&1
  else python build_cache.py "$d" 1.0 > "$PCL_REPO/results/revision/logs/build_$d.log" 2>&1 &
  fi
done
wait
for d in "${DS[@]}"; do
  src="$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0_s42.pkl"
  [ -s "$src" ] && ln -sf "$src" "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl"
  ls -la "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl" || { echo "FAILED: $d (see results/revision/logs/build_$d.log)"; exit 1; }
done
echo "caches OK in $PCL_LEGACY2_CACHE_DIR"; du -sh "$PCL_LEGACY2_CACHE_DIR"
echo
echo "To survive a pod replacement (needs ~2-3 GB free in \$HOME; check 'df -h ~' first):"
echo "  mkdir -p ~/pcl_cache && cp -L $PCL_LEGACY2_CACHE_DIR/physionet_frac1.0.pkl ~/pcl_cache/"
