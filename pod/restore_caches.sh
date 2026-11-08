set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SRC="${PCL_PERSIST_DIR:-$HOME/pcl_cache_gz}"
mkdir -p "$PCL_LEGACY2_CACHE_DIR"
DS=(physionet mimic eicu); [ "${PCL_PHYSIONET_ONLY:-1}" = "1" ] && DS=(physionet)
for d in "${DS[@]}"; do
  [ -s "$SRC/${d}_frac1.0.pkl.gz" ] || { echo "missing $SRC/${d}_frac1.0.pkl.gz"; exit 1; }
  [ -s "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl" ] && { echo "[skip] $d already present"; continue; }
  echo "restoring $d ..."
  rm -f "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl.part"
  gunzip -c "$SRC/${d}_frac1.0.pkl.gz" > "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl.part" && mv "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl.part" "$PCL_LEGACY2_CACHE_DIR/${d}_frac1.0.pkl"
done
ls -la "$PCL_LEGACY2_CACHE_DIR"
