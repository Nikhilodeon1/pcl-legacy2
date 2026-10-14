"""
A4 (mortality, ORIGINAL protocol): per-direction Spearman of each label-free criterion vs true AUROC, from the original run's
results/mortality/selection_criteria.json (18 checkpoints = 3 methods x 3 fine-tune seeds x 2 directions; ONE pretrained encoder per method).
Unit = one checkpoint, n = 9 per direction. Two-sided permutation p (20,000), seeded. Also the pooled (both directions, n = 18) values.
Lower-is-better signals (violation, recon_mse, entropy, repr_dist, mmd): negative rho = correct; ATC is higher-is-better: positive = correct.
Verifies the paper's Table 6 numbers (violation/recon rho in [-0.73, -0.40]; ATC +0.283 / +0.150).
Output: a4_mortality_by_direction.csv / .json
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
SIGS = ["violation", "recon_mse", "entropy", "repr_dist", "mmd", "atc"]
rng = np.random.default_rng(0)
rows = []
for name, sel in (("mimic->eicu", lambda x: x["source"] == "mimic"), ("eicu->mimic", lambda x: x["source"] == "eicu"), ("pooled", lambda x: True)):
    c = [x for x in cells if sel(x)]
    y = np.array([x["true_auroc"] for x in c])
    for s in SIGS:
        v = np.array([x[s] for x in c])
        r = float(spearmanr(v, y)[0])
        perm = np.array([spearmanr(v, rng.permutation(y))[0] for _ in range(20000)])
        rows.append(dict(domain=name, n=len(c), signal=s, rho=r, perm_p=float((np.abs(perm) >= abs(r) - 1e-12).mean()),
                         correct_sign=int((r > 0) if s == "atc" else (r < 0))))
        print(f"{name:12s} n={len(c):2d} {s:10s} rho={r:+.3f} p={rows[-1]['perm_p']:.3f} correct_sign={rows[-1]['correct_sign']}")
with open(os.path.join(REV, "a4_mortality_by_direction.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(rows, open(os.path.join(REV, "a4_mortality_by_direction.json"), "w"), indent=1)
