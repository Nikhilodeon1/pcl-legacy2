"""
Independent-pretraining sweep (revision B1/B2): lambda in {0.0 (=ERM), 0.5, 1.0} x 3 independently pretrained encoders (seeds 42/43/44), each fine-tuned ONCE
(fine-tune seed = pretraining seed), fp32, fixed-observation protocol (LOS T=24, mortality T=48).

Inputs : results/mortality_fixed_T48_sweep/lam*.json, results/los_fixed_T24_sweep/lam*.json
Outputs: results/revision/sweep_summary.csv / .json, printed tables
Per lambda: mean (SD) over the 3 pretraining seeds, per-seed values, number of failed fine-tunes (mortality: in-domain AUROC < 0.70, descriptive threshold chosen
because every ERM run is >= 0.76; exact collapse = in-domain AUROC <= 0.51), and paired-by-pretraining-seed differences vs lambda 0 (t with df=2; |t|>4.303 needed).
The pretraining seed and fine-tune seed are confounded by design (one run per encoder); this measures variability of the whole pipeline.
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
LAMS, PS = ["0.0", "0.5", "1.0"], [42, 43, 44]
FAIL_BELOW = 0.70


def load(dirname):
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results", dirname, "lam*.json")):
        d = json.load(open(f))
        eid = d.get("encoder_id", "")
        if not eid.startswith("lam"):
            continue
        lam, p = eid[3:].split("_p")
        out[(lam, int(p), d.get("source"))] = d
    return out


def paired(a, b):
    d = np.asarray(a) - np.asarray(b); sd = d.std(ddof=1)
    return float(d.mean()), float(sd), (float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("nan"))


rows, summary = [], {"mortality": {}, "los": {}, "paired": {}}

mort = load("mortality_fixed_T48_sweep")
print(f"mortality runs: {len(mort)} (expected 18)")
if mort:
    print("\n== mortality, OOD AUROC mean (SD) over 3 pretraining seeds; failed = in-domain AUROC < 0.70 ==")
    for src, name in (("mimic", "MIMIC->eICU"), ("eicu", "eICU->MIMIC")):
        vals = {}
        for lam in LAMS:
            runs = [mort.get((lam, p, src)) for p in PS]
            if any(r is None for r in runs):
                continue
            o, i = [r["ood"]["auroc"] for r in runs], [r["in_domain"]["auroc"] for r in runs]
            vals[lam] = o
            nf, nc = sum(x < FAIL_BELOW for x in i), sum(x <= 0.51 for x in i)
            ok = [x for x, y in zip(o, i) if y >= FAIL_BELOW]
            summary["mortality"][f"{name}|lam{lam}"] = dict(mean=float(np.mean(o)), sd=float(np.std(o, ddof=1)), per_seed=o, in_domain=i, failed=nf, collapsed=nc,
                                                           mean_of_non_failed=float(np.mean(ok)) if ok else None)
            for p, a, b in zip(PS, o, i):
                rows.append(dict(task="mortality", direction=name, lam=lam, pretrain_seed=p, ood=a, in_domain=b, failed=int(b < FAIL_BELOW), collapsed=int(b <= 0.51)))
            print(f"{name} lam={lam}: {np.mean(o):.3f} ({np.std(o, ddof=1):.3f})  per-seed {[round(x, 3) for x in o]}  failed {nf}/3 collapsed {nc}/3"
                  + (f"  non-failed mean {np.mean(ok):.3f}" if ok else ""))
        for lam in ("0.5", "1.0"):
            if lam in vals and "0.0" in vals:
                m, sd, t = paired(vals[lam], vals["0.0"])
                summary["paired"][f"mortality|{name}|lam{lam}-lam0.0"] = dict(mean=m, sd=sd, t=t)
                print(f"  paired lam{lam} - ERM: {m:+.4f} (t={t:+.2f})")
    allp = [summary["mortality"][k] for k in summary["mortality"] if "lam0.5" in k or "lam1.0" in k]
    allm = [summary["mortality"][k] for k in summary["mortality"] if "lam0.0" in k]
    print(f"\nPCL (lambda 0.5/1.0) failed fine-tunes: {sum(x['failed'] for x in allp)}/{3 * len(allp)}; ERM: {sum(x['failed'] for x in allm)}/{3 * len(allm)}")

los = load("los_fixed_T24_sweep")
print(f"\nLOS runs: {len(los)} (expected 9)")
KEYS = [("in_domain", None, "PN-A val"), ("ood", "physionet_b", "PN-B"), ("ood", "mimic", "MIMIC"), ("ood", "eicu", "eICU")]
if los:
    print("== LOS fixed T=24, R2 mean (SD) over 3 pretraining seeds ==")
    vals = {}
    for lam in LAMS:
        runs = [next((v for (l, p, _), v in los.items() if l == lam and p == pp), None) for pp in PS]
        if any(r is None for r in runs):
            continue
        line = f"lam={lam}:"
        for k, t, lab in KEYS:
            x = [(r[k]["r2"] if t is None else r[k][t]["r2"]) for r in runs]
            vals[(lam, lab)] = x
            summary["los"][f"{lab}|lam{lam}"] = dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)), per_seed=x)
            for p, v in zip(PS, x):
                rows.append(dict(task="los", direction=lab, lam=lam, pretrain_seed=p, ood=v, in_domain="", failed="", collapsed=""))
            line += f"  {lab} {np.mean(x):+.3f} ({np.std(x, ddof=1):.3f})"
        print(line)
    for lam in ("0.5", "1.0"):
        line = f"paired lam{lam} - ERM:"
        for _, _, lab in KEYS:
            if (lam, lab) in vals and ("0.0", lab) in vals:
                m, sd, t = paired(vals[(lam, lab)], vals[("0.0", lab)])
                summary["paired"][f"los|{lab}|lam{lam}-lam0.0"] = dict(mean=m, sd=sd, t=t)
                line += f"  {lab} {m:+.4f} (t={t:+.2f})"
        print(line)

if rows:
    with open(os.path.join(REV, "sweep_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(summary, open(os.path.join(REV, "sweep_summary.json"), "w"), indent=1)
