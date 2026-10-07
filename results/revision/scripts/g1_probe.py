"""
G1 (i) — mask/length-only probe. Predicts LOS (R2, raw hours) and in-hospital mortality
(AUROC) from features of the OBSERVATION MASK ONLY (observed hours, last/first observed
hour, trailing padding, per-variable observed counts). No physiological values are used.

Protocol mirrors the paper: LOS = train on PhysioNet-A (same 80/20 split as finetune_los.py,
seed 42), evaluate on PN-A val and zero-shot on PN-B / MIMIC-IV / eICU. Mortality = train on
one of MIMIC-IV / eICU (80/20 stratified), evaluate on its val split and on the other site.

Flag thresholds (PREREG.md): LOS R2 > 0.05 or mortality AUROC > 0.60 on source val or any target.

Usage: python g1_probe.py --pn-frac 1.0 --mimic-frac 0.25 --eicu-frac 0.25
Outputs: g1_probe.csv, g1_probe.json, g1_cohort_leak_stats.json
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import revlib as R

ap = argparse.ArgumentParser()
ap.add_argument("--pn-frac", type=float, default=1.0)
ap.add_argument("--mimic-frac", type=float, default=0.25)
ap.add_argument("--eicu-frac", type=float, default=0.25)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fixed-T-los", type=int, default=None, help="re-run the probe under the fixed-observation protocol (LOS T)")
ap.add_argument("--fixed-T-mort", type=int, default=None, help="... (mortality T)")
args = ap.parse_args()
TAG = "" if not (args.fixed_T_los or args.fixed_T_mort) else f"_fixedT{args.fixed_T_los}_{args.fixed_T_mort}"

pn = R.load_cache("physionet", args.pn_frac)
mimic = R.load_cache("mimic", args.mimic_frac)
eicu = R.load_cache("eicu", args.eicu_frac)
site_a = [s for s in pn if s["site_id"] == 0]
site_b = [s for s in pn if s["site_id"] == 1]
if args.fixed_T_los:
    from src.data.fixed_window import fix_window
    site_a = fix_window(site_a, args.fixed_T_los, args.fixed_T_los); site_b = fix_window(site_b, args.fixed_T_los, args.fixed_T_los)
    mimic_los = fix_window(mimic, args.fixed_T_los, args.fixed_T_los); eicu_los = fix_window(eicu, args.fixed_T_los, args.fixed_T_los)
else:
    mimic_los, eicu_los = mimic, eicu
if args.fixed_T_mort:
    from src.data.fixed_window import fix_window
    mimic_m = fix_window(mimic, args.fixed_T_mort, args.fixed_T_mort); eicu_m = fix_window(eicu, args.fixed_T_mort, args.fixed_T_mort)
else:
    mimic_m, eicu_m = mimic, eicu
print(f"PN-A {len(site_a)}  PN-B {len(site_b)}  MIMIC {len(mimic)} (los {len(mimic_los)} / mort {len(mimic_m)})  eICU {len(eicu)} (los {len(eicu_los)} / mort {len(eicu_m)})")

rows = []
out = {"counts": dict(pn_a=len(site_a), pn_b=len(site_b), mimic=len(mimic), eicu=len(eicu))}


def los_arr(ss):
    return np.array([float(s["los_h"]) for s in ss])


# ---------- how deterministic is the mask -> LOS link? ----------
leak = {}
for name, ss in [("pn_a", site_a), ("pn_b", site_b), ("mimic", mimic), ("eicu", eicu)]:
    X, names = R.mask_features(ss)
    los = los_arr(ss)
    last, n_obs = X[:, 1], X[:, 0]
    short = los < R.HOURS
    d = dict(n=len(ss), frac_los_lt_48h=float(short.mean()), frac_window_full=float(X[:, 4].mean()),
             spearman_lastobs_vs_min_los48=float(spearmanr(last, np.minimum(los, R.HOURS)).statistic),
             exact_match_lastobs_plus1_eq_min_los48=float(np.mean((last + 1) == np.minimum(np.floor(los), R.HOURS))),
             within_6h_match=float(np.mean(np.abs((last + 1) - np.minimum(np.floor(los), R.HOURS)) <= 6)))
    if "mortality_hospital" in ss[0]:
        mort = np.array([float(s["mortality_hospital"]) for s in ss])
        d["mortality_prev"] = float(mort.mean())
        d["mortality_prev_short_lt48h"] = float(mort[short].mean()) if short.any() else None
        d["mortality_prev_long_ge48h"] = float(mort[~short].mean()) if (~short).any() else None
    leak[name] = d
out["mask_los_link"] = leak

# ---------- LOS probe ----------
tr, va = R.split_pn_a(len(site_a), args.seed)
sa_tr = [site_a[i] for i in tr]
sa_va = [site_a[i] for i in va]
for fs_name, featfn in [("mask+length", R.mask_features), ("length-only", R.length_only_features)]:
    Xtr, _ = featfn(sa_tr)
    ytr = np.log1p(los_arr(sa_tr))
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=args.seed).fit(Xtr, ytr)
    for tname, ss in [("pn_a_val", sa_va), ("pn_b", site_b), ("mimic", mimic_los), ("eicu", eicu_los)]:
        Xt, _ = featfn(ss)
        y = los_arr(ss)
        p = np.expm1(reg.predict(Xt))
        r2 = R.r2_score_raw(y, p)
        mae = float(np.mean(np.abs(y - p)))
        rows.append(dict(task="los", features=fs_name, train="pn_a_train", eval=tname, metric="R2", value=r2,
                         mae_hours=mae, n=len(ss)))
        print(f"LOS  {fs_name:12s} -> {tname:9s} R2={r2:+.3f} MAE={mae:.1f}h n={len(ss)}")

# ---------- Mortality probe ----------
def mort_arr(ss):
    return np.array([int(round(float(s["mortality_hospital"]))) for s in ss])

for src_name, src, tgt_name, tgt in [("mimic", mimic_m, "eicu", eicu_m), ("eicu", eicu_m, "mimic", mimic_m)]:
    idx = np.arange(len(src))
    ytr_all = mort_arr(src)
    i_tr, i_va = train_test_split(idx, test_size=0.2, stratify=ytr_all, random_state=args.seed)
    s_tr = [src[i] for i in i_tr]
    s_va = [src[i] for i in i_va]
    for fs_name, featfn in [("mask+length", R.mask_features), ("length-only", R.length_only_features)]:
        Xtr, _ = featfn(s_tr)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=args.seed).fit(Xtr, mort_arr(s_tr))
        for ename, ss in [(f"{src_name}_val", s_va), (f"{tgt_name}_zeroshot", tgt)]:
            Xe, _ = featfn(ss)
            auc = roc_auc_score(mort_arr(ss), clf.predict_proba(Xe)[:, 1])
            rows.append(dict(task="mortality", features=fs_name, train=f"{src_name}_train", eval=ename, metric="AUROC",
                             value=float(auc), mae_hours=None, n=len(ss)))
            print(f"MORT {fs_name:12s} {src_name}->{ename:16s} AUROC={auc:.3f} n={len(ss)}")

# ---------- flags ----------
flag_los = [r for r in rows if r["task"] == "los" and r["value"] > 0.05]
flag_mort = [r for r in rows if r["task"] == "mortality" and r["value"] > 0.60]
out["flag_los"] = bool(flag_los)
out["flag_mortality"] = bool(flag_mort)
out["flagged_rows"] = flag_los + flag_mort
out["thresholds"] = dict(los_r2=0.05, mortality_auroc=0.60)
out["probe_rows"] = rows

here = R.REV
with open(os.path.join(here, f"g1_probe{TAG}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
out["fixed_T_los"] = args.fixed_T_los; out["fixed_T_mort"] = args.fixed_T_mort
json.dump(out, open(os.path.join(here, f"g1_probe{TAG}.json"), "w"), indent=1, default=float)
print("\nFLAG LOS:", out["flag_los"], "| FLAG MORTALITY:", out["flag_mortality"])
print(json.dumps(leak, indent=1))
