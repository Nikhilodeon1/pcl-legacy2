set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DEST="${PCL_PERSIST_DIR:-$HOME/pcl_cache_gz}"
mkdir -p "$DEST"
for d in physionet mimic eicu; do
  src="$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl"
  [ -e "$src" ] || { echo "missing $src (run pod/build_caches.sh first)"; exit 1; }
  free_mb=$(df -Pm "$DEST" | awk 'NR==2{print $4}')
  need_mb=$(( $(stat -L -c %s "$src") / 1024 / 1024 / 5 + 20 ))
  [ "$free_mb" -ge "$need_mb" ] || { echo "not enough space in $DEST for $d (free ${free_mb} MB, need ~${need_mb} MB). Free some space in ~ first."; exit 1; }
  echo "compressing $d ..."; gzip -3 -c -L "$src" > "$DEST/${d}_frac1.0.pkl.gz.tmp" && mv "$DEST/${d}_frac1.0.pkl.gz.tmp" "$DEST/${d}_frac1.0.pkl.gz"
done
ls -la "$DEST"; du -sh "$DEST"; df -h "$HOME" | tail -1
