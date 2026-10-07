"""
A3 (LOS) — patient-level paired cluster bootstrap. 10,000 resamples of TEST PATIENTS; the same
resample is used for every method and every seed, so differences are paired. Per seed and target:
R2 and MAE per method plus PCL-ERM, PCL-DRO, ERM-DRO differences; then averaged across seeds
(per resample) with 95% percentile intervals.

SCOPE: this interval covers TEST-SAMPLING variance only. It does not include fine-tune-seed variance
(the 3 seeds are averaged, not resampled) and, critically, it does not include PRE-TRAINING variance
(one pretrained encoder per method). It must not be read as a confidence interval for 'PCL is worse'.

Resampling is by patient (PhysioNet: one stay per patient; MIMIC: subject_id; eICU: uniquepid),
implemented with multinomial patient weights and per-patient sufficient statistics, which is
equivalent to the cluster bootstrap and exact for R2/MAE.

Usage: python a3_bootstrap_los.py [--pred-dir ...] [--B 10000]
Outputs: a3_bootstrap_los.csv, a3_bootstrap_los.json
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
ap.add_argument("--chunk", type=int, default=250)
ap.add_argument("--targets", nargs="+", default=["pn_b", "mimic", "eicu", "pn_a_val"])
args = ap.parse_args()
rng = np.random.default_rng(555)
NOTE = "test-sampling variance only (patient-cluster bootstrap); excludes fine-tune-seed and pre-training variance"

P = {(m, s): np.load(os.path.join(args.pred_dir, f"los_{m}_s{s}.npz"), allow_pickle=True)
     for m in R.METHODS for s in R.SEEDS if os.path.exists(os.path.join(args.pred_dir, f"los_{m}_s{s}.npz"))}
seeds = sorted({s for _, s in P})


def patient_stats(y, errs, pid):
    """Per-patient sufficient stats. errs: dict method->pred error array. Returns (stats matrix, names)."""
    up, inv = np.unique(pid, return_inverse=True)
    k = len(up)
    cols = {"n": np.bincount(inv, minlength=k).astype(float), "sy": np.bincount(inv, weights=y, minlength=k),
            "syy": np.bincount(inv, weights=y * y, minlength=k)}
    for m, e in errs.items():
        cols[f"sse_{m}"] = np.bincount(inv, weights=e * e, minlength=k)
        cols[f"sae_{m}"] = np.bincount(inv, weights=np.abs(e), minlength=k)
    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names, k


def boot(stats, names, k):
    """Returns dict name -> (B,) bootstrap totals."""
    tot = {n: [] for n in names}
    done = 0
    while done < args.B:
        b = min(args.chunk, args.B - done)
        W = rng.multinomial(k, np.full(k, 1.0 / k), size=b).astype(np.float32)   # (b, k)
        T = W @ stats.astype(np.float32)                                          # (b, ncols)
        for j, n in enumerate(names):
            tot[n].append(T[:, j].astype(np.float64))
        done += b
    return {n: np.concatenate(v) for n, v in tot.items()}


def ci(a, indep=False):
    # indep=True: draws were made independently per seed, so a cross-seed mean of draws has an artificially
    # narrow spread; report no interval for that aggregate (per-seed rows carry the intervals).
    if indep:
        return float("nan"), float("nan")
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


rows, out = [], {}
pairs = [("pcl", "erm"), ("pcl", "dro"), ("erm", "dro")]
for tgt in args.targets:
    if f"{tgt}_pred" not in P[(R.METHODS[0], seeds[0])].files:
        continue
    per_seed_metric = {}   # (m, s, metric) -> (B,)
    if tgt == "pn_a_val":
        # validation sets differ per seed -> bootstrap each seed separately (not paired across seeds)
        for s in seeds:
            y = P[(R.METHODS[0], s)][f"{tgt}_true"]; pid = P[(R.METHODS[0], s)][f"{tgt}_pid"]
            errs = {m: P[(m, s)][f"{tgt}_pred"] - y for m in R.METHODS}
            st, names, k = patient_stats(y, errs, pid)
            tot = boot(st, names, k)
            sst = tot["syy"] - tot["sy"] ** 2 / tot["n"]
            for m in R.METHODS:
                per_seed_metric[(m, s, "r2")] = 1 - tot[f"sse_{m}"] / sst
                per_seed_metric[(m, s, "mae")] = tot[f"sae_{m}"] / tot["n"]
        mode = "per-seed (val splits differ by seed; averaged point estimates, intervals are per-seed)"
    else:
        y = P[(R.METHODS[0], seeds[0])][f"{tgt}_true"]; pid = P[(R.METHODS[0], seeds[0])][f"{tgt}_pid"]
        errs = {(m, s): P[(m, s)][f"{tgt}_pred"] - y for m in R.METHODS for s in seeds}
        st, names, k = patient_stats(y, {f"{m}{s}": e for (m, s), e in errs.items()}, pid)
        tot = boot(st, names, k)
        sst = tot["syy"] - tot["sy"] ** 2 / tot["n"]
        for m in R.METHODS:
            for s in seeds:
                per_seed_metric[(m, s, "r2")] = 1 - tot[f"sse_{m}{s}"] / sst
                per_seed_metric[(m, s, "mae")] = tot[f"sae_{m}{s}"] / tot["n"]
        mode = "paired: same patient resample for all methods and seeds"
    for metric in ("r2", "mae"):
        mean_seed = {m: np.mean([per_seed_metric[(m, s, metric)] for s in seeds], axis=0) for m in R.METHODS}
        for m in R.METHODS:
            lo, hi = ci(mean_seed[m], tgt == 'pn_a_val')
            rows.append(dict(target=tgt, metric=metric, quantity=m, scope="mean_over_seeds", point=float(mean_seed[m].mean()),
                             ci_lo=lo, ci_hi=hi, B=args.B, mode=mode, note=NOTE))
        for a, b in pairs:
            d = mean_seed[a] - mean_seed[b]
            lo, hi = ci(d, tgt == 'pn_a_val')
            rows.append(dict(target=tgt, metric=metric, quantity=f"{a}-{b}", scope="mean_over_seeds", point=float(d.mean()),
                             ci_lo=lo, ci_hi=hi, B=args.B, mode=mode, note=NOTE))
            for s in seeds:
                ds = per_seed_metric[(a, s, metric)] - per_seed_metric[(b, s, metric)]
                lo, hi = ci(ds)
                rows.append(dict(target=tgt, metric=metric, quantity=f"{a}-{b}", scope=f"seed{s}", point=float(ds.mean()),
                                 ci_lo=lo, ci_hi=hi, B=args.B, mode=mode, note=NOTE))
    print(f"{tgt}: done ({mode})")
    for r in rows:
        if r["target"] == tgt and r["metric"] == "r2" and r["scope"] == "mean_over_seeds" and "-" in r["quantity"]:
            print(f"   {r['quantity']:9s} dR2 = {r['point']:+.4f}  95% [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")

with open(os.path.join(R.REV, "a3_bootstrap_los.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(rows, open(os.path.join(R.REV, "a3_bootstrap_los.json"), "w"), indent=1)
