"""
A1 — LOS label-support diagnostics (inference outputs only; needs preds/ from predict_los.py).

Per target (physionet_b, mimic, eicu), method, seed: R2, MAE, RMSE, Spearman(pred,true) on
  (i)   full target
  (ii)  stays <= 336 h
  (iii) label-matched resample: target stays resampled so the true-LOS histogram matches
        PhysioNet-A's decile histogram (equal mass per PN-A decile; target stays above PN-A's
        max LOS are outside the matched support and excluded). 200 resamples x 500 per decile.
        The same resample indices are used for every method/seed of a target (paired).
Plus constant baselines (source-train mean, target median), affine-recalibrated R2 (fit on a
random half of PATIENTS, evaluate on the other half; ORACLE DIAGNOSTIC, uses target labels),
mean signed error / MAE by true-LOS decile, and scatter data.

PREREG verdicts:
  H2  : on (ii) and (iii), far-target (mimic, eicu) R2 >= 0 and within 0.15 of PN-B R2 on the
        same subset, for every method (mean over seeds) -> met only if all hold.
  H2b : Spearman(full target) >= 0.30 while full-target R2 < 0, counted over method x far target.

Outputs (results/revision): a1_metrics.csv, a1_resample_summary.csv, a1_baselines.csv,
a1_decile_error.csv, a1_scatter.csv, a1_verdicts.json
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr

import revlib as R

ap = argparse.ArgumentParser()
ap.add_argument("--pn-frac", type=float, default=1.0)
ap.add_argument("--n-resamples", type=int, default=200)
ap.add_argument("--per-decile", type=int, default=500)
ap.add_argument("--pred-dir", default=os.path.join(R.REV, "preds"))
ap.add_argument("--targets", nargs="+", default=["physionet_b", "mimic", "eicu"])
args = ap.parse_args()
FAR = [t for t in ("mimic", "eicu") if t in args.targets]

pn = R.load_cache("physionet", args.pn_frac)
los_a_all = np.array([float(s["los_h"]) for s in pn if s["site_id"] == 0])
q = np.quantile(los_a_all, np.linspace(0.1, 0.9, 9))
edges = np.concatenate([[-np.inf], q, [los_a_all.max()]])       # 10 bins; upper edge = PN-A max
MAXA = float(los_a_all.max())
print("PN-A decile edges:", np.round(q, 1), "max", MAXA)


KEY = {"physionet_b": "pn_b", "mimic": "mimic", "eicu": "eicu"}


def metrics(y, p):
    sp = spearmanr(p, y).statistic if len(y) > 2 else float("nan")
    return dict(r2=R.r2_score_raw(y, p), mae=float(np.mean(np.abs(y - p))),
                rmse=float(np.sqrt(np.mean((y - p) ** 2))), spearman=float(sp))


# ---------------- load predictions ----------------
P = {}
for m in R.METHODS:
    for s in R.SEEDS:
        f = os.path.join(args.pred_dir, f"los_{m}_s{s}.npz")
        if os.path.exists(f):
            P[(m, s)] = np.load(f, allow_pickle=True)
keys = sorted(P)
print("loaded", len(keys), "prediction files")

# ---------------- fixed paired resample indices per target ----------------
rng = np.random.default_rng(2026)
resample_idx = {}
for t in args.targets:
    y = P[keys[0]][f"{KEY[t]}_true"]
    bins = np.digitize(y, edges[1:-1], right=False)               # 0..9
    in_support = y <= MAXA
    idxs = []
    for b in range(10):
        pool = np.where((bins == b) & in_support)[0]
        idxs.append(pool)
    if any(len(p) == 0 for p in idxs):
        print(f"WARN {t}: empty decile bin(s) {[b for b,p in enumerate(idxs) if len(p)==0]}")
    draws = np.stack([np.concatenate([rng.choice(p, args.per_decile, replace=True) for p in idxs if len(p)])
                      for _ in range(args.n_resamples)])           # (n_resamples, 10*per_decile)
    resample_idx[t] = draws

rows, resum, base_rows, dec_rows, scatter_rows = [], [], [], [], []
agg = {}   # (target, method, subset) -> list over seeds of metrics

for (m, s) in keys:
    d = P[(m, s)]
    _, va = R.split_pn_a(int(round(len(los_a_all))), s)
    src_train_mean = float(np.mean(np.delete(los_a_all, va)))     # PN-A train-split mean for this seed
    for t in args.targets:
        y, p = d[f"{KEY[t]}_true"], d[f"{KEY[t]}_pred"]
        pid = d[f"{KEY[t]}_pid"]
        subsets = {"full": np.ones(len(y), bool), "le336": y <= 336.0}
        for sub, msk in subsets.items():
            mt = metrics(y[msk], p[msk])
            rows.append(dict(target=t, method=m, seed=s, subset=sub, n=int(msk.sum()), **mt))
            agg.setdefault((t, m, sub), []).append(mt)
        # label-matched resample
        ms = [metrics(y[ix], p[ix]) for ix in resample_idx[t]]
        summ = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        for k in list(summ):
            summ[k + "_lo"] = float(np.percentile([x[k] for x in ms], 2.5))
            summ[k + "_hi"] = float(np.percentile([x[k] for x in ms], 97.5))
        resum.append(dict(target=t, method=m, seed=s, subset="label_matched", n_per_resample=resample_idx[t].shape[1], **summ))
        agg.setdefault((t, m, "label_matched"), []).append({k: summ[k] for k in ("r2", "mae", "rmse", "spearman")})

        # constant baselines (on each subset)
        for sub, msk in list(subsets.items()):
            yy = y[msk]
            base_rows.append(dict(target=t, method=m, seed=s, subset=sub, baseline="source_train_mean",
                                  r2=R.r2_score_raw(yy, np.full_like(yy, src_train_mean))))
            base_rows.append(dict(target=t, method=m, seed=s, subset=sub, baseline="target_median_oracle",
                                  r2=R.r2_score_raw(yy, np.full_like(yy, np.median(yy)))))
        # affine recalibration on random patient halves (oracle)
        up = np.unique(pid)
        r2s = []
        rr = np.random.default_rng(7)
        for _ in range(20):
            half = set(rr.choice(up, len(up) // 2, replace=False))
            a_m = np.array([x in half for x in pid])
            if a_m.sum() < 10 or (~a_m).sum() < 10:
                continue
            b, a = np.polyfit(p[a_m], y[a_m], 1)
            r2s.append(R.r2_score_raw(y[~a_m], a + b * p[~a_m]))
        base_rows.append(dict(target=t, method=m, seed=s, subset="full", baseline="affine_recalibrated_oracle_halfsplit",
                              r2=float(np.mean(r2s))))
        # error by true-LOS decile (PN-A decile edges, plus >max bin)
        bins = np.digitize(y, edges[1:-1])
        bins = np.where(y > MAXA, 10, bins)
        for b in range(11):
            mk = bins == b
            if mk.sum() == 0:
                continue
            dec_rows.append(dict(target=t, method=m, seed=s, decile=b if b < 10 else ">PN-A max", n=int(mk.sum()),
                                 mean_signed_error_h=float(np.mean(p[mk] - y[mk])), mae_h=float(np.mean(np.abs(p[mk] - y[mk]))),
                                 true_mean_h=float(y[mk].mean()), pred_mean_h=float(p[mk].mean())))
        if s == 42:
            ii = np.random.default_rng(3).choice(len(y), min(2000, len(y)), replace=False)
            for j in ii:
                scatter_rows.append(dict(target=t, method=m, seed=s, true_h=float(y[j]), pred_h=float(p[j])))


def write(name, rs):
    with open(os.path.join(R.REV, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
        w.writeheader(); w.writerows(rs)


write("a1_metrics.csv", rows); write("a1_resample_summary.csv", resum); write("a1_baselines.csv", base_rows)
write("a1_decile_error.csv", dec_rows); write("a1_scatter.csv", scatter_rows)

# ---------------- verdicts ----------------
def mean_over_seeds(t, m, sub, key):
    return float(np.mean([x[key] for x in agg[(t, m, sub)]]))

verd = {"H2": {}, "H2b": {}}
h2_all = True
for sub in ("le336", "label_matched"):
    for m in R.METHODS:
        pb = mean_over_seeds("physionet_b", m, sub, "r2") if "physionet_b" in args.targets else float("nan")
        for t in FAR:
            r2 = mean_over_seeds(t, m, sub, "r2")
            ok = (r2 >= 0.0) and (abs(r2 - pb) <= 0.15)
            verd["H2"][f"{sub}.{m}.{t}"] = dict(far_r2=r2, pn_b_r2=pb, ge0=bool(r2 >= 0), within_0p15=bool(abs(r2 - pb) <= 0.15), ok=bool(ok))
            h2_all &= ok
verd["H2_met"] = bool(h2_all)
n_h2b = 0
for m in R.METHODS:
    for t in FAR:
        sp = mean_over_seeds(t, m, "full", "spearman"); r2 = mean_over_seeds(t, m, "full", "r2")
        ok = (sp >= 0.30) and (r2 < 0)
        verd["H2b"][f"{m}.{t}"] = dict(spearman=sp, r2=r2, satisfied=bool(ok))
        n_h2b += int(ok)
verd["H2b_cells_satisfied"] = n_h2b
verd["H2b_cells_total"] = len(R.METHODS) * len(FAR)
json.dump(verd, open(os.path.join(R.REV, "a1_verdicts.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in verd.items() if k in ("H2_met", "H2b_cells_satisfied", "H2b_cells_total")}, indent=1))
