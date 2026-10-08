#!/usr/bin/env bash
# Source this (do not execute):  source pod/env.sh
# Layout rule: the repo lives in persistent $HOME; everything big or rebuildable lives outside it.
#   persistent : repo (code, results/*.json), optional ~/pcl_cache, ~/pcl_pretrained
#   ephemeral  : /tmp/venv, /tmp/pcl_data (raw PhysioNet data), /tmp/pcl_cache (unless persisted), pip/uv caches
# Every path below can be overridden by exporting the variable before sourcing.

_here="${BASH_SOURCE[0]:-$0}"
export PCL_REPO="$(cd "$(dirname "$_here")/.." && pwd)"

export PCL_VENV="${PCL_VENV:-/tmp/venv}"
export TMPDIR="${TMPDIR:-/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pipcache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/tmp/uvpy}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache}"   # keep __pycache__ out of the repo/home

export PCL_DATA_ROOT="${PCL_DATA_ROOT:-/tmp/pcl_data}"
export PHYSIONET_DIR="${PHYSIONET_DIR:-$PCL_DATA_ROOT/physionet2019}"        # contains training_setA/ training_setB/
export MIMIC_DIR="${MIMIC_DIR:-$PCL_DATA_ROOT/mimiciv/3.1}"                  # contains hosp/ icu/
export EICU_DIR="${EICU_DIR:-$PCL_DATA_ROOT/eicu-crd/2.0}"                   # contains patient.csv.gz ...

# pretrained encoders: {erm,pcl,dro}_pretrained.pt (NOT in git; ~13 MB each, keep in persistent home)
export PCL_LEGACY2_PRETRAIN_DIR="${PCL_LEGACY2_PRETRAIN_DIR:-$HOME/pcl_pretrained}"

# preprocessed-sample caches: use the persistent copy if it exists, else /tmp
if [ -z "${PCL_LEGACY2_CACHE_DIR:-}" ]; then
  if ls "$HOME/pcl_cache"/*.pkl >/dev/null 2>&1; then export PCL_LEGACY2_CACHE_DIR="$HOME/pcl_cache"
  else export PCL_LEGACY2_CACHE_DIR="/tmp/pcl_cache"; fi
fi
export PCL_CACHE_DIR="$PCL_LEGACY2_CACHE_DIR"

# big/regenerable outputs stay out of the (small) persistent home: fixed-T fine-tune checkpoints and per-stay prediction dumps
export PCL_LEGACY2_CKPT_ROOT="${PCL_LEGACY2_CKPT_ROOT:-/tmp/pcl_ckpt}"
export PCL_PREDS_DIR="${PCL_PREDS_DIR:-/tmp/pcl_preds}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export PCL_TEST_MODE=0
export POD_RATE_USD_PER_H="${POD_RATE_USD_PER_H:-}"    # set to your hourly rate to get cost lines in the run scripts

if [ -f "$PCL_VENV/bin/activate" ]; then . "$PCL_VENV/bin/activate"; fi
echo "[pcl env] repo=$PCL_REPO venv=$PCL_VENV cache=$PCL_LEGACY2_CACHE_DIR data=$PCL_DATA_ROOT pretrain=$PCL_LEGACY2_PRETRAIN_DIR"
