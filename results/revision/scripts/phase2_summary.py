"""
Summary of the phase-2 pretraining experiments (pod/run_phase2.sh): precision (amp vs fp32 pretraining) and the controlled pretraining-leak test.

Inputs : results/phase2/pretrain/*.json  (config, loss history, representation diagnostics of the pretrained encoder, Site A val stays)
         results/phase2/sepsis/*.json    (sepsis fine-tune: Site A val AUROC, zero-shot Site B AUROC)
         results/phase2/los_T24/*.json   (LOS fine-tune, fixed T=24: Site A val R2, zero-shot Site B R2)
Tag format: lam<L>_p<seed>_<amp|fp32>[_leakpb]; the pretraining seed also sets the fine-tune seed and the Site A split.
Outputs: results/revision/phase2_summary.csv / .json, printed tables.
Reports, per (lambda, precision, leak): mean (SD) over seeds of
  representation diagnostics at hour 47 (mean L2 norm, spread, const_frac), best pretraining val masked loss, sepsis AUROC (Site A val, Site B), LOS R2 (Site A val, Site B);
then paired-by-seed differences (mean, SD, t with df = n-1, two-sided p): fp32 - amp for every lambda, PCL - ERM within each precision,
and (leak - no leak) for the controlled leak test. With few seeds the p-values are rough; read the intervals/differences, not the stars.
"""
import csv
import glob
import json
import os
import re

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
P2 = os.path.join(ROOT, "results", "phase2")
TAG = re.compile(r"^lam(?P<lam>[0-9.]+)_p(?P<seed>\d+)_(?P<prec>amp|fp32)(?P<leak>_leakpb)?$")


def tag_of(path, strip_seed=False):
    b = os.path.basename(path)[:-5]
    if strip_seed:
        b = re.sub(r"_s\d+(_T\d+)?$", "", b)
    return b


def load_dir(sub, strip_seed):
    out = {}
    for f in glob.glob(os.path.join(P2, sub, "*.json")):
        m = TAG.match(tag_of(f, strip_seed))
        if not m:
            continue
        out[(float(m["lam"]), int(m["seed"]), m["prec"], bool(m["leak"]))] = json.load(open(f))
    return out


pre, sep, los = load_dir("pretrain", False), load_dir("sepsis", True), load_dir("los_T24", True)
print(f"pretrain {len(pre)}, sepsis {len(sep)}, LOS {len(los)} result files")
METRICS = {
    "l2_h47": lambda k: pre[k]["repr_diag_site_a"]["47"]["mean_l2"] if k in pre else None,
    "spread_h47": lambda k: pre[k]["repr_diag_site_a"]["47"]["spread"] if k in pre else None,
    "const_frac_h47": lambda k: pre[k]["repr_diag_site_a"]["47"]["const_frac"] if k in pre else None,
    "pretrain_val_masked": lambda k: min(pre[k]["history"]["val_masked"]) if k in pre else None,
    "sepsis_auroc_A": lambda k: sep[k]["in_domain"]["auroc"] if k in sep else None,
    "sepsis_auroc_B": lambda k: sep[k]["ood"]["physionet_b"]["auroc"] if k in sep else None,
    "los_r2_A": lambda k: los[k]["in_domain"]["r2"] if k in los else None,
    "los_r2_B": lambda k: los[k]["ood"]["physionet_b"]["r2"] if k in los else None,
}
keys = sorted(set(pre) | set(sep) | set(los))
cells = sorted({(k[0], k[2], k[3]) for k in keys})
seeds_of = {c: sorted(k[1] for k in keys if (k[0], k[2], k[3]) == c) for c in cells}


def vec(cell, metric):
    lam, prec, leak = cell
    vals = [(s, METRICS[metric]((lam, s, prec, leak))) for s in seeds_of[cell]]
    return {s: v for s, v in vals if v is not None}


rows = []
print(f"\n{'lam':>4s} {'prec':>5s} {'leak':>5s} {'n':>2s} " + " ".join(f"{m:>20s}" for m in METRICS))
for c in cells:
    row = dict(lam=c[0], precision=c[1], leak=int(c[2]), n_seeds=len(seeds_of[c]))
    line = f"{c[0]:4g} {c[1]:>5s} {int(c[2]):5d} {len(seeds_of[c]):2d} "
    for m in METRICS:
        v = list(vec(c, m).values())
        row[m + "_mean"] = float(np.mean(v)) if v else None
        row[m + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        line += f"{(f'{np.mean(v):.3f} ({np.std(v, ddof=1):.3f})' if len(v) > 1 else (f'{v[0]:.3f}' if v else '-')):>21s}"
    rows.append(row)
    print(line)


def paired(cell_a, cell_b, metric):
    a, b = vec(cell_a, metric), vec(cell_b, metric)
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    d = np.array([a[s] - b[s] for s in common])
    sd = d.std(ddof=1)
    t = float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 1e-9 else float("nan")
    return dict(n=len(d), mean=float(d.mean()), sd=float(sd), t=t, p=float(2 * stats.t.sf(abs(t), len(d) - 1)) if sd > 1e-9 else float("nan"))


contrasts = []


def report(title, pairs):
    print(f"\n== {title} (paired by seed: mean diff, t, p) ==")
    for name, ca, cb in pairs:
        if ca not in seeds_of or cb not in seeds_of:
            continue
        line = f"{name:22s}"
        for m in METRICS:
            r = paired(ca, cb, m)
            if r:
                contrasts.append(dict(contrast=name, metric=m, **r))
                line += f"  {m} {r['mean']:+.3f} (t={r['t']:+.2f}, p={r['p']:.2f}, n={r['n']})"
        print(line)


lams = sorted({c[0] for c in cells})
report("fp32 - amp pretraining", [(f"lam={l:g}", (l, "fp32", False), (l, "amp", False)) for l in lams])
report("PCL - ERM", [(f"lam={l:g} vs 0 [{p}]", (l, p, False), (0.0, p, False)) for l in lams if l > 0 for p in ("amp", "fp32")])
report("leak - no leak (fp32)", [(f"lam={l:g}", (l, "fp32", True), (l, "fp32", False)) for l in lams])

with open(os.path.join(REV, "phase2_summary.csv"), "w", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump({"cells": rows, "contrasts": contrasts}, open(os.path.join(REV, "phase2_summary.json"), "w"), indent=1)
