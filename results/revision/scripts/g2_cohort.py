"""
G2 — cohort table per dataset x task, from the same preprocessed caches the models used
plus raw ID tables for patient-level counts.

Reports: stays, unique patients, stays/patient, prevalence, LOS stats, inclusion filters,
source-split patient overlap (reproducing finetune_mortality.py's StratifiedShuffleSplit on
STAYS), normalization and imputation facts (read from code, asserted here).

Usage: python g2_cohort.py --pn-frac 1.0 --mimic-frac 0.25 --eicu-frac 0.25
Outputs: g2_cohort.csv, g2_cohort.json
Note: MIMIC/eICU are subsampled at the stay level when frac<1 (build_cache.py); counts scale ~1/frac.
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import revlib as R

ap = argparse.ArgumentParser()
ap.add_argument("--pn-frac", type=float, default=1.0)
ap.add_argument("--mimic-frac", type=float, default=0.25)
ap.add_argument("--eicu-frac", type=float, default=0.25)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--eicu-dir", default=None, help="defaults to $EICU_DIR")
args = ap.parse_args()

pn = R.load_cache("physionet", args.pn_frac)
mimic = R.load_cache("mimic", args.mimic_frac)
eicu = R.load_cache("eicu", args.eicu_frac)
site_a = [s for s in pn if s["site_id"] == 0]
site_b = [s for s in pn if s["site_id"] == 1]

# eICU: map patientunitstayid -> uniquepid (patient) for patient-level counts
stay2pid = R.eicu_stay2pid(args.eicu_dir) or {}

def stats(name, ss, patient_key):
    los = np.array([float(s["los_h"]) for s in ss])
    d = dict(dataset=name, stays=len(ss))
    if patient_key == "subject_id":
        pids = [s["subject_id"] for s in ss]
    elif patient_key == "eicu_uniquepid":
        pids = [stay2pid.get(str(s["patient_id"]), f"unk{s['patient_id']}") for s in ss]
    else:
        pids = [s["patient_id"] for s in ss]   # PhysioNet: one file = one patient/stay
    d["patients"] = len(set(pids))
    d["stays_per_patient_max"] = int(pd.Series(pids).value_counts().max())
    d["frac_patients_multi_stay"] = float((pd.Series(pids).value_counts() > 1).mean())
    d["los_mean_h"] = float(los.mean()); d["los_median_h"] = float(np.median(los))
    d["los_p99_h"] = float(np.percentile(los, 99)); d["los_max_h"] = float(los.max()); d["los_min_h"] = float(los.min())
    d["frac_los_lt_48h"] = float((los < 48).mean())
    if "mortality_hospital" in ss[0] and name in ("mimic", "eicu"):
        m = np.array([float(s["mortality_hospital"]) for s in ss])
        d["mortality_hospital_prev"] = float(m.mean()); d["n_deaths"] = int(m.sum())
    return d, pids

rows, pid_lists = [], {}
for name, ss, key in [("physionet_a", site_a, "psv"), ("physionet_b", site_b, "psv"),
                      ("mimic", mimic, "subject_id"), ("eicu", eicu, "eicu_uniquepid")]:
    d, pids = stats(name, ss, key)
    rows.append(d)
    pid_lists[name] = pids
    print(d)

# Source-split patient overlap: finetune_mortality.py -> make_patient_split_loaders -> StratifiedShuffleSplit on STAYS
overlap = {}
for name, ss in [("mimic", mimic), ("eicu", eicu)]:
    y = np.array([int(round(float(s["mortality_hospital"]))) for s in ss])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr, va = next(sss.split(np.arange(len(ss)), y))
    ptr = set(pid_lists[name][i] for i in tr)
    pva = [pid_lists[name][i] for i in va]
    n_overlap_stays = int(sum(p in ptr for p in pva))
    overlap[name] = dict(val_stays=len(va), val_stays_whose_patient_in_train=n_overlap_stays,
                         frac=n_overlap_stays / len(va),
                         note="split is by STAY (sample index), not patient; fraction is at the cache subsample "
                              f"(frac<1 understates overlap because sibling stays are subsampled away)")
    print(name, overlap[name])

# LOS source split (PhysioNet-A): one file = one patient, so stay-level == patient-level there
facts = {
    "window": "first 48 hourly bins from ICU admission (intime / offset 0); stays <48h are padded after discharge",
    "inclusion": "adults (age>=18), ICU LOS >= 24h (MIN_LOS_H=24), >=1 hour in which ALL hemodynamic variables observed "
                 "(has_hemodynamic_window); PhysioNet: len(df)>=24 and non-empty and hemodynamic window",
    "los_definition": {"physionet": "len(df) hourly rows (hours from ICU admission to last record)",
                       "mimic": "icustays.los*24 (ICU stay, hours)", "eicu": "unitdischargeoffset/60 (ICU unit stay, hours)"},
    "mortality_definition": {"mimic": "admissions.hospital_expire_flag (hospital)",
                             "eicu": "patient.hospitaldischargestatus=='Expired' (hospital)",
                             "physionet": "not available"},
    "prediction_time": "end of hour 48 (or end of stay if shorter) -- outcome (LOS / in-hospital death) is NOT yet known, but for stays <48h the window already spans the whole stay",
    "variables": "17 (9 core + 8 expanded) -- see src/data/variables.py",
    "normalization": "MinMaxNormalizer: FIXED affine map from clinical plausibility bounds (PLAUS); fit() is a no-op; no statistics derived from any split",
    "imputation": "hourly median bin; forward-fill gap limit 6h; remaining missing -> 0 with observation mask as separate input",
    "split_unit_mortality": "stratified random split of STAYS (make_patient_split_loaders despite name), 80/20, seed = fine-tune seed",
    "split_unit_los": "random permutation of PhysioNet-A stays (= patients), 80/20, seed = fine-tune seed",
    "repeated_admissions": "no de-duplication to one stay per patient in MIMIC/eICU",
}
with open(os.path.join(R.REV, "g2_cohort.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader(); w.writerows(rows)
json.dump(dict(cohorts=rows, split_overlap=overlap, facts=facts, caches=vars(args)),
          open(os.path.join(R.REV, "g2_cohort.json"), "w"), indent=1, default=float)
