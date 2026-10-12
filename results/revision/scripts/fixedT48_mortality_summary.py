"""
Fixed-observation protocol (T=48h) mortality results, fp16-AMP vs fp32, from the saved result JSONs (full precision).

Inputs : results/mortality_fixed_T48/*.json            (fp16 AMP, first batch)
         results/mortality_fixed_T48_sweep/{m}_fp32_*.json  (fp32 rerun of all three arms)
Outputs: results/revision/fixedT48_mortality_summary.csv / .json, printed tables
Per precision, per method and direction (MIMIC->eICU, eICU->MIMIC): mean (SD) OOD AUROC over 3 fine-tune seeds, number of collapsed runs
(in-domain AUROC <= 0.51), paired-by-seed method differences with t (df=2, critical 4.303) and Cohen's d, and the fp16->fp32 paired change.
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
M, S, SRC = ["erm", "pcl", "dro"], [42, 43, 44], ["mimic", "eicu"]
DIR = {"mimic": "MIMIC->eICU", "eicu": "eICU->MIMIC"}


def load(dirname, suffix):
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results", dirname, "*.json")):
        d = json.load(open(f))
        if "method" not in d or "source" not in d:
            continue
        eid = d.get("encoder_id")
        if suffix is None and eid:
            continue
        if suffix is not None and eid != f"{d['method']}_{suffix}":
            continue
        out[(d["method"], d["source"], d["seed"])] = d
    return out


def paired(a, b):
    d = np.asarray(a) - np.asarray(b); sd = d.std(ddof=1)
    return float(d.mean()), float(sd), (float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("inf")), (float(d.mean() / sd) if sd > 0 else float("inf"))


runs = {"fp16": load("mortality_fixed_T48", None), "fp32": load("mortality_fixed_T48_sweep", "fp32")}
if not runs["fp16"]:   # the fp16 JSONs were lost with a pod refresh; fall back to the 3-decimal transcription of the terminal output
    tr = os.path.join(REV, "fixedT48_mortality_table_from_pod_paste.csv")
    for r in csv.DictReader(open(tr)):
        src = "mimic" if r["direction"] == "mimictoeicu" else "eicu"
        runs["fp16"][(r["method"], src, int(r["seed"]))] = {"in_domain": {"auroc": float(r["val_auroc"])}, "ood": {"auroc": float(r["ood_auroc"])}}
    print("[fp16 from the 3-decimal transcription, not JSONs]")
for k, r in runs.items():
    print(f"{k}: {len(r)} runs found (expected 18)")
rows, summary = [], {"n_runs": {k: len(v) for k, v in runs.items()}, "methods": {}, "paired": {}, "fp16_to_fp32": {}}
for prec, r in runs.items():
    for (m, src, s), d in sorted(r.items()):
        rows.append(dict(precision=prec, method=m, direction=DIR[src], seed=s, in_domain_auroc=d["in_domain"]["auroc"], ood_auroc=d["ood"]["auroc"],
                         collapsed=int(d["in_domain"]["auroc"] <= 0.51), n_source=d.get("n_source"), n_target=d.get("n_target")))


def vec(prec, m, src, key):
    return [runs[prec][(m, src, s)][key]["auroc"] for s in S]


for prec in runs:
    if not all((m, src, s) in runs[prec] for m in M for src in SRC for s in S):
        print(f"[skip {prec}: incomplete]"); continue
    print(f"\n== {prec}: OOD AUROC mean (SD) over 3 seeds; collapsed runs (in-domain AUROC<=0.51) ==")
    for m in M:
        line = f"{m:4s}"
        for src in SRC:
            o, i = vec(prec, m, src, "ood"), vec(prec, m, src, "in_domain")
            nc = int(sum(x <= 0.51 for x in i))
            summary["methods"][f"{prec}|{m}|{DIR[src]}"] = dict(mean=float(np.mean(o)), sd=float(np.std(o, ddof=1)), per_seed=o, in_domain_mean=float(np.mean(i)), collapsed=nc)
            line += f"   {DIR[src]} {np.mean(o):.3f} ({np.std(o, ddof=1):.3f})  in-dom {np.mean(i):.3f}  collapsed {nc}/3"
        print(line)
    print("paired by seed (mean diff, t, d); |t| > 4.303 needed for p<0.05 at df=2")
    for a, b in (("pcl", "erm"), ("pcl", "dro"), ("erm", "dro")):
        line = f"{a}-{b}:"
        for src in SRC:
            mean, sd, tt, dd = paired(vec(prec, a, src, "ood"), vec(prec, b, src, "ood"))
            summary["paired"][f"{prec}|{a}-{b}|{DIR[src]}"] = dict(mean=mean, sd=sd, t=tt, d=dd)
            line += f"   {DIR[src]} {mean:+.4f} (t={tt:+.2f}, d={dd:+.2f})"
        print(line)
if all(len(v) == 18 for v in runs.values()):
    print("\nfp16 -> fp32 change in OOD AUROC (paired by seed)")
    for m in M:
        line = f"{m:4s}"
        for src in SRC:
            mean, sd, tt, dd = paired(vec("fp32", m, src, "ood"), vec("fp16", m, src, "ood"))
            summary["fp16_to_fp32"][f"{m}|{DIR[src]}"] = dict(mean=mean, sd=sd, t=tt)
            line += f"   {DIR[src]} {mean:+.4f} (t={tt:+.2f})"
        print(line)
with open(os.path.join(REV, "fixedT48_mortality_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(summary, open(os.path.join(REV, "fixedT48_mortality_summary.json"), "w"), indent=1)
