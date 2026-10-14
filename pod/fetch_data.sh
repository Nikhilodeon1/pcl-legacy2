set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
WHAT=("$@"); [ ${#WHAT[@]} -eq 0 ] && WHAT=(physionet mimic eicu)
AUTH=()
if [ -n "${PHYSIONET_USER:-}" ] && [ -n "${PHYSIONET_PASSWORD:-}" ]; then AUTH=(--user "$PHYSIONET_USER" --password "$PHYSIONET_PASSWORD"); fi
get() {
  mkdir -p "$2"; wget -c -N -q --show-progress "${AUTH[@]}" -P "$2" "$1"; }

for w in "${WHAT[@]}"; do case "$w" in
  physionet)
    echo "--- PhysioNet/CinC 2019 training sets A and B -> $PHYSIONET_DIR"
    mkdir -p "$PHYSIONET_DIR"
    wget -r -N -c -np -nH --cut-dirs=4 -R "index.html*" -q "${AUTH[@]}" -P "$PHYSIONET_DIR" \
      https://physionet.org/files/challenge-2019/1.0.0/training/ ;;
  mimic)
    echo "--- MIMIC-IV 3.1 (6 files) -> $MIMIC_DIR"
    B=https://physionet.org/files/mimiciv/3.1
    get $B/icu/icustays.csv.gz "$MIMIC_DIR/icu"; get $B/icu/chartevents.csv.gz "$MIMIC_DIR/icu"
    get $B/hosp/patients.csv.gz "$MIMIC_DIR/hosp"; get $B/hosp/admissions.csv.gz "$MIMIC_DIR/hosp"; get $B/hosp/labevents.csv.gz "$MIMIC_DIR/hosp" ;;
  eicu)
    echo "--- eICU-CRD 2.0 (5 files) -> $EICU_DIR"
    B=https://physionet.org/files/eicu-crd/2.0
    for f in patient vitalPeriodic vitalAperiodic lab nurseCharting; do get $B/$f.csv.gz "$EICU_DIR"; done ;;
  *) echo "unknown: $w"; exit 2 ;;
esac; done
echo "--- done. sizes:"; du -sh "$PHYSIONET_DIR" "$MIMIC_DIR" "$EICU_DIR" 2>/dev/null || true
