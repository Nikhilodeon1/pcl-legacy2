"""
Source-validation baseline for the mortality selection analysis (ORIGINAL protocol, one pretrained encoder per method).
Unit = one checkpoint (method x fine-tune seed), n = 9 per direction. Source-validation AUROC comes from the fine-tune result JSONs
(results/mortality/*.json, key in_domain.auroc, the same 18 checkpoints as results/mortality/selection_criteria.json); the target is the
zero-shot AUROC (ood.auroc). Reports Spearman rho (higher val -> higher OOD = correct), permutation p, and selection regret
(OOD AUROC of the best checkpoint minus OOD AUROC of the checkpoint chosen by highest source-val AUROC), random-choice expected regret,
and the within-seed version (choose 1 of 3 methods per seed). Output: a4_mortality_source_val.csv
"""
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
cells = json.load(open(os.path.join(ROOT, "results", "mortality", "selection_criteria.json")))["cells"]
val = {}
for f in glob.glob(os.path.join(ROOT, "results", "mortality", "*.json")):
    d = json.load(open(f))
    if "method" in d and "in_domain" in d:
        val[(d["method"], d["source"], d["seed"])] = (d["in_domain"]["auroc"], d["ood"]["auroc"])
rng = np.random.default_rng(0)
rows = []
for src in ("mimic", "eicu"):
    c = [x for x in cells if x["source"] == src]
    v = np.array([val[(x["method"], src, x["seed"])][0] for x in c])
    y = np.array([x["true_auroc"] for x in c])
    chk = max(abs(val[(x["method"], src, x["seed"])][1] - x["true_auroc"]) for x in c)
    r = float(spearmanr(v, y)[0])
    perm = np.array([spearmanr(v, rng.permutation(y))[0] for _ in range(20000)])
    p = float((np.abs(perm) >= abs(r) - 1e-12).mean())
    regret = float(y.max() - y[int(np.argmax(v))]); rand = float(y.max() - y.mean())
    sr = []
    for s in sorted(set(x["seed"] for x in c)):
        idx = [i for i, x in enumerate(c) if x["seed"] == s]
        sr.append(float(y[idx].max() - y[idx][int(np.argmax(v[idx]))]))
    rows.append(dict(direction=f"{src}->{'eicu' if src == 'mimic' else 'mimic'}", n=len(c), rho=r, perm_p=p, regret_ckpt=regret, random_regret=rand,
                     regret_within_seed=float(np.mean(sr)), json_vs_cells_max_abs_diff=chk))
    print(rows[-1])
# top-1 selection regret of every label-free criterion on the same 9 checkpoints, for comparison with the source-validation baseline
crit = []
for src in ("mimic", "eicu"):
    c = [x for x in cells if x["source"] == src]
    y = np.array([x["true_auroc"] for x in c])
    for sname in ("violation", "recon_mse", "entropy", "repr_dist", "mmd", "atc"):
        v = np.array([x[sname] for x in c])
        pick = int(np.argmax(v)) if sname == "atc" else int(np.argmin(v))
        crit.append(dict(direction=f"{src}->{'eicu' if src == 'mimic' else 'mimic'}", criterion=sname, regret_ckpt=float(y.max() - y[pick])))
with open(os.path.join(REV, "a4_mortality_criteria_regret.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(crit[0].keys())); w.writeheader(); w.writerows(crit)
with open(os.path.join(REV, "a4_mortality_source_val.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
