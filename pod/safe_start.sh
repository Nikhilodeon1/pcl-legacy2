set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOGF="$HOME/pcl_setup.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGF"; }
snap() {
  say "  mem: $(free -g 2>/dev/null | awk '/Mem:/{printf "used %sG avail %sG", $3, $7}')  cgroup_limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo n/a)"
  say "  disk: home $(df -h ~ | awk 'NR==2{print $4" free"}'), /tmp $(df -h /tmp | awk 'NR==2{print $4" free"}')"
}
step() { say "=== START $1"; snap; shift; "$@"; rc=$?; say "=== END (exit $rc)"; snap; return $rc; }

say "------ safe_start $(date) host=$(hostname)"
say "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
step "setup_env"      bash pod/setup_env.sh                          || { say "setup_env failed"; exit 1; }
source pod/env.sh
if [ -s "$HOME/pcl_cache_gz/physionet_frac1.0.pkl.gz" ]; then
  step "restore_caches" bash pod/restore_caches.sh                   || { say "restore_caches failed"; exit 1; }
else
  # PhysioNet 2019 is open access: download it here and build the cache (no credentialed data on the pod)
  [ -d "$PHYSIONET_DIR/training_setA" ] || step "fetch_physionet" bash pod/fetch_data.sh physionet || { say "fetch failed"; exit 1; }
  step "build_cache"    bash pod/build_caches.sh                       || { say "build_caches failed"; exit 1; }
fi
step "preflight"      python pod/preflight.py
say "DONE. Full log: $LOGF"
