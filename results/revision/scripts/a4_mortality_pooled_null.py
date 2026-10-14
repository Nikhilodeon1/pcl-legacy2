"""
Mortality pooled-correlation null that RESPECTS direction (original protocol, one encoder per method; 18 checkpoints = 9 per direction).
For each criterion: observed pooled Spearman(criterion, true AUROC) over all 18, versus a null made by shuffling the true AUROC *within each
direction* (10,000 shuffles; keeps the between-direction structure, breaks any criterion-performance link inside a direction). Same logic as the LOS
A6 null. NOTE: the 18 points come from 3 encoders each fine-tuned in both directions, so a plain permutation p over all 18 is inflated; do not use it.
Output: a4_mortality_pooled_null.csv
"""
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
cells = json.load(open(os.path.join(ROOT, "results", "mortality", "selection_criteria.json")))["cells"]
dirs = np.array([x["source"] for x in cells]); y = np.array([x["true_auroc"] for x in cells])
rng = np.random.default_rng(0); B = 10000; rows = []
for s in ("violation", "recon_mse", "entropy", "repr_dist", "mmd", "atc"):
    v = np.array([x[s] for x in cells]); obs = float(spearmanr(v, y)[0]); null = np.empty(B)
    for b in range(B):
        yp = y.copy()
        for d in ("mimic", "eicu"):
            i = np.where(dirs == d)[0]; yp[i] = y[rng.permutation(i)]
        null[b] = spearmanr(v, yp)[0]
    lo, hi = np.percentile(null, [2.5, 97.5])
    rows.append(dict(signal=s, observed_pooled_rho=obs, null_lo=float(lo), null_hi=float(hi), inside_band=int(lo <= obs <= hi)))
    print(f"{s:10s} observed {obs:+.3f}  within-direction null 95% band [{lo:+.3f},{hi:+.3f}]  inside={lo <= obs <= hi}")
with open(os.path.join(REV, "a4_mortality_pooled_null.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
