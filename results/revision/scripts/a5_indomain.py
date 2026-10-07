"""
A5 — in-domain baselines per method and seed (from the saved result JSONs).
Mortality: AUROC MIMIC->MIMIC (source val of the mimic->eicu runs) and eICU->eICU (source val of eicu->mimic runs).
LOS: PhysioNet-A source-val R2 / MAE.
Also reports in-domain vs zero-shot gap. Outputs a5_indomain.csv, a5_indomain.json
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "results", "mortality", "*_s4*.json"))):
    d = json.load(open(f))
    rows.append(dict(task="mortality", split=f"{d['source']}->{d['source']} (val)", method=d["method"], seed=d["seed"],
                     metric="AUROC", in_domain=d["in_domain"]["auroc"], n_in_domain=None,
                     zero_shot=d["ood"]["auroc"], zero_shot_target=d["target"], gap=d["in_domain"]["auroc"] - d["ood"]["auroc"],
                     zero_shot_auroc_ci_lo=d["ood"].get("auroc_ci", [None, None])[0], zero_shot_auroc_ci_hi=d["ood"].get("auroc_ci", [None, None])[1]))
for f in sorted(glob.glob(os.path.join(ROOT, "results", "los", "*_s4*.json"))):
    d = json.load(open(f))
    for t, o in d["ood"].items():
        rows.append(dict(task="los", split="physionet_a (val)", method=d["method"], seed=d["seed"], metric="R2",
                         in_domain=d["in_domain"]["r2"], n_in_domain=d["in_domain"]["n"], zero_shot=o["r2"],
                         zero_shot_target=t, gap=d["in_domain"]["r2"] - o["r2"], zero_shot_auroc_ci_lo=None, zero_shot_auroc_ci_hi=None))
with open(os.path.join(REV, "a5_indomain.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
summ = {}
for task, key in (("mortality", "split"), ("los", "zero_shot_target")):
    for r in rows:
        if r["task"] != task:
            continue
        summ.setdefault(f"{task}|{r[key]}|{r['method']}", []).append((r["in_domain"], r["zero_shot"]))
out = {k: dict(in_domain_mean=float(np.mean([a for a, _ in v])), zero_shot_mean=float(np.mean([b for _, b in v])), n_runs=len(v)) for k, v in summ.items()}
json.dump(out, open(os.path.join(REV, "a5_indomain.json"), "w"), indent=1)
for k, v in sorted(out.items()):
    print(k, {a: round(b, 4) for a, b in v.items()})
