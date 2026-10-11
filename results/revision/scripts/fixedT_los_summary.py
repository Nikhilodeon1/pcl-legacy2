"""
Fixed-observation protocol (T=24h) LOS results vs the original protocol, from the saved result JSONs (full precision).

Inputs : results/los_fixed_T24/*.json (new), results/los/*_s4*.json (original)
Outputs: results/revision/fixedT24_los_summary.csv / .json, printed tables
Reports per method (mean, SD over 3 fine-tune seeds), paired-by-seed method differences with t (df=2, critical 4.303) and Cohen's d,
original-vs-fixed paired drop per method, and the constant-prediction reference (always predict the PhysioNet-A mean LOS).
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
M, S = ["erm", "pcl", "dro"], [42, 43, 44]
KEYS = [("in_domain", None, "PN-A val"), ("ood", "physionet_b", "PN-B"), ("ood", "mimic", "MIMIC"), ("ood", "eicu", "eICU")]


def load(dirname):
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results", dirname, "*.json")):
        d = json.load(open(f))
        if "method" not in d or d.get("encoder_id"):
            continue
        out[(d["method"], d["seed"])] = d
    return out


def r2(d, kind, tgt):
    return d[kind]["r2"] if tgt is None else d[kind][tgt]["r2"]


fixed, orig = load("los_fixed_T24"), load("los")
assert len(fixed) == 9, f"expected 9 fixed-T runs, found {len(fixed)}"
rows, summary = [], {"cohort_n": {}, "methods": {}, "paired": {}, "orig_vs_fixed": {}}
for m in M:
    for s in S:
        d = fixed[(m, s)]
        rows.append(dict(protocol="fixed_T24", method=m, seed=s, **{lab: r2(d, k, t) for k, t, lab in KEYS},
                         n_source=d["n_source"], **{f"n_{k}": v for k, v in d["n_targets"].items()}))
summary["cohort_n"] = {k: v for k, v in rows[0].items() if k.startswith("n_")}

def paired(a, b):
    d = np.asarray(a) - np.asarray(b); sd = d.std(ddof=1)
    return float(d.mean()), float(sd), (float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("inf")), (float(d.mean() / sd) if sd > 0 else float("inf"))

print(f"{'':6s}" + "".join(f"{lab:>20s}" for _, _, lab in KEYS))
for m in M:
    v = {lab: [r2(fixed[(m, s)], k, t) for s in S] for k, t, lab in KEYS}
    summary["methods"][m] = {lab: dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)), per_seed=x) for lab, x in v.items()}
    print(f"{m:6s}" + "".join(f"{np.mean(x):+12.4f} ({np.std(x, ddof=1):.4f})" for x in v.values()))
print("\npaired by seed (mean diff, t, d); |t| > 4.303 needed for p<0.05 at df=2")
for a, b in (("pcl", "erm"), ("pcl", "dro"), ("erm", "dro")):
    line = f"{a}-{b}:"
    for k, t, lab in KEYS:
        mean, sd, tt, dd = paired([r2(fixed[(a, s)], k, t) for s in S], [r2(fixed[(b, s)], k, t) for s in S])
        summary["paired"][f"{a}-{b}|{lab}"] = dict(mean=mean, sd=sd, t=tt, d=dd)
        line += f"  {lab} {mean:+.4f} (t={tt:+.2f})"
    print(line)
if all((m, s) in orig for m in M for s in S):
    print("\noriginal -> fixed (mean over seeds), paired drop")
    for m in M:
        line = f"{m}:"
        for k, t, lab in KEYS:
            o = [r2(orig[(m, s)], k, t) for s in S]; f = [r2(fixed[(m, s)], k, t) for s in S]
            mean, sd, tt, dd = paired(f, o)
            summary["orig_vs_fixed"][f"{m}|{lab}"] = dict(orig=float(np.mean(o)), fixed=float(np.mean(f)), drop=mean, t=tt)
            line += f"  {lab} {np.mean(o):+.3f}->{np.mean(f):+.3f}"
        print(line)
with open(os.path.join(REV, "fixedT24_los_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(summary, open(os.path.join(REV, "fixedT24_los_summary.json"), "w"), indent=1)
print("\ncohorts:", summary["cohort_n"])
