"""
A3 (mortality) — patient-level paired cluster bootstrap of AUROC. 10,000 resamples of TEST patients,
same resample for every method and seed within a direction. Per direction: AUROC per method, and
PCL-ERM, PCL-DRO, ERM-DRO differences, averaged across seeds per resample, 95% percentile intervals.
Also per-seed intervals and in-domain (val) per-seed intervals.

SCOPE: test-sampling variance only. Excludes fine-tune-seed variance (averaged, not resampled) and
pre-training variance (one pretrained encoder per method).

AUROC under a weighted (multinomial patient-weight) resample: with scores sorted once, AUROC =
sum_i w_i y_i * (cumulative negative weight strictly below i + 0.5 * tied negative weight) / (W_pos W_neg).
Ties are negligible for continuous logits (checked: ties handled by sorting stable + midrank not needed).

Usage: python a3_bootstrap_mortality.py [--pred-dir ...] [--B 10000]
Outputs: a3_bootstrap_mortality.csv/json
"""
import argparse
import csv
import json
import os

import numpy as np

import revlib as R

ap = argparse.ArgumentParser()
ap.add_argument("--pred-dir", default=os.path.join(R.REV, "preds"))
ap.add_argument("--B", type=int, default=10000)
args = ap.parse_args()
rng = np.random.default_rng(777)
NOTE = "test-sampling variance only (patient-cluster bootstrap); excludes fine-tune-seed and pre-training variance"


def prep(logit, y):
    order = np.argsort(logit, kind="stable")
    return order, y[order]


def auroc_weighted(ys, w):
    """ys: labels in ascending-score order; w: per-stay weights in same order."""
    wneg = w * (1 - ys)
    cumneg = np.cumsum(wneg) - wneg          # negative weight strictly below
    num = np.sum(w * ys * cumneg)
    return num / (np.sum(w * ys) * np.sum(wneg))


def boot_auroc(logits_by_key, y, pid, B):
    """logits_by_key: dict key -> logits. Returns dict key -> (B,) AUROCs, same patient resample for all keys."""
    up, inv = np.unique(pid, return_inverse=True)
    k = len(up)
    prepped = {key: prep(l, y) for key, l in logits_by_key.items()}
    res = {key: np.empty(B) for key in logits_by_key}
    for b in range(B):
        wp = rng.multinomial(k, np.full(k, 1.0 / k)).astype(np.float64)
        w = wp[inv]
        for key, (order, ys) in prepped.items():
            res[key][b] = auroc_weighted(ys, w[order])
    return res


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


rows = []
pairs = [("pcl", "erm"), ("pcl", "dro"), ("erm", "dro")]
for direction in ("mimic_to_eicu", "eicu_to_mimic"):
    src, tgt = direction.split("_to_")
    for space in ("tgt", "val"):
        F = {(m, s): os.path.join(args.pred_dir, f"mort_{m}_{src}to{tgt}_s{s}.npz") for m in R.METHODS for s in R.SEEDS}
        if not all(os.path.exists(f) for f in F.values()):
            print("missing predictions for", direction)
            continue
        D = {k: np.load(f, allow_pickle=True) for k, f in F.items()}
        if space == "tgt":
            y = D[("erm", 42)]["tgt_true"]; pid = D[("erm", 42)]["tgt_pid"]
            res = boot_auroc({k: d["tgt_logit"] for k, d in D.items()}, y, pid, args.B)
            per_seed = res
            mode = "paired: same patient resample for all methods and seeds"
        else:
            per_seed = {}
            for s in R.SEEDS:
                y = D[("erm", s)]["val_true"]; pid = D[("erm", s)]["val_pid"]
                r = boot_auroc({(m, s): D[(m, s)]["val_logit"] for m in R.METHODS}, y, pid, args.B)
                per_seed.update(r)
            mode = "per-seed (validation splits differ by seed)"
        for m in R.METHODS:
            ms = np.mean([per_seed[(m, s)] for s in R.SEEDS], axis=0)
            lo, hi = ci(ms) if space == "tgt" else (float("nan"), float("nan"))
            rows.append(dict(direction=direction, space=space, quantity=m, scope="mean_over_seeds", point=float(ms.mean()),
                             ci_lo=lo, ci_hi=hi, B=args.B, mode=mode, note=NOTE))
        for a, b in pairs:
            d = np.mean([per_seed[(a, s)] - per_seed[(b, s)] for s in R.SEEDS], axis=0)
            lo, hi = ci(d) if space == "tgt" else (float("nan"), float("nan"))
            rows.append(dict(direction=direction, space=space, quantity=f"{a}-{b}", scope="mean_over_seeds", point=float(d.mean()),
                             ci_lo=lo, ci_hi=hi, B=args.B, mode=mode, note=NOTE))
            for s in R.SEEDS:
                ds = per_seed[(a, s)] - per_seed[(b, s)]
                lo, hi = ci(ds)
                rows.append(dict(direction=direction, space=space, quantity=f"{a}-{b}", scope=f"seed{s}", point=float(ds.mean()),
                                 ci_lo=lo, ci_hi=hi, B=args.B, mode=mode, note=NOTE))
        print(direction, space, "done")
with open(os.path.join(R.REV, "a3_bootstrap_mortality.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(rows, open(os.path.join(R.REV, "a3_bootstrap_mortality.json"), "w"), indent=1)
