"""
Summary of the semi-synthetic experiments (scripts/semisynth_eval.py, pod/run_semisynth.sh).

Inputs : results/phase2/semisynth_shift.csv   (one row per tag x shift family x magnitude; model performance and four label-free criteria vs the unshifted source)
         results/phase2/semisynth_length.csv  (synthetic length-only outcome: fine-tuned model AUROC and mask/length probe AUROC, original vs fixed window)
Outputs: results/revision/semisynth_response_matrix.csv and printed tables.
Response matrix: for each task x shift family, the change from magnitude 0 to the largest magnitude of performance and of each criterion (mean over tags, relative to the
magnitude-0 value), the mean within-tag Spearman rho between magnitude and the quantity, and a verdict:
  responds (up/down) if mean |rho| >= 0.9 and |relative change| >= 5%;  flat otherwise.
A criterion is "fooled" by a family when it responds although performance does not, and "blind" when performance responds but the criterion stays flat.
"""
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
P2 = os.path.join(ROOT, "results", "phase2")
QTY = ["performance", "violation", "recon_mse", "repr_dist", "mmd"]


def verdict(rho, rel):
    if np.isnan(rho) or np.isnan(rel):
        return "n/a"
    return ("up" if rel > 0 else "down") if (abs(rho) >= 0.9 and abs(rel) >= 0.05) else "flat"


def shift_summary():
    f = os.path.join(P2, "semisynth_shift.csv")
    if not os.path.exists(f):
        print("no semisynth_shift.csv yet"); return []
    rows = list(csv.DictReader(open(f)))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["family"], r["tag"], r["seed"])].append(r)
    agg = defaultdict(lambda: defaultdict(list))
    for (task, fam, tag, seed), rs in groups.items():
        rs = sorted(rs, key=lambda r: float(r["magnitude"]))
        mags = np.array([float(r["magnitude"]) for r in rs])
        for q in QTY:
            v = np.array([float(r[q]) for r in rs])
            base = v[0]
            rel = (v[-1] - base) / (max(abs(base), abs(v[-1])) + 1e-9)   # bounded in [-1, 1]; magnitude-0 value can be exactly 0 (distances)
            rho = spearmanr(mags, v)[0] if np.ptp(v) > 1e-12 else 0.0
            agg[(task, fam)][q].append((rel, rho, base, v[-1]))
    out = []
    print(f"\n{'task':6s} {'family':28s} " + " ".join(f"{q:>20s}" for q in QTY))
    for (task, fam), d in sorted(agg.items()):
        line = f"{task:6s} {fam:28s} "
        row = dict(task=task, family=fam, n_tags=len(d["performance"]))
        for q in QTY:
            rel = float(np.mean([x[0] for x in d[q]])); rho = float(np.mean([x[1] for x in d[q]]))
            row[q + "_rel_change"], row[q + "_rho"], row[q + "_verdict"] = rel, rho, verdict(rho, rel)
            line += f"{verdict(rho, rel):>6s} ({rel:+7.1%})".rjust(21)
        out.append(row)
        print(line)
    return out


def length_summary():
    f = os.path.join(P2, "semisynth_length.csv")
    if not os.path.exists(f):
        print("no semisynth_length.csv yet"); return
    rows = list(csv.DictReader(open(f)))
    print("\nLength-only synthetic outcome (positive control for the stay-length shortcut):")
    for T in sorted({r["fixed_T"] for r in rows}):
        rs = [r for r in rows if r["fixed_T"] == T]
        m = np.array([float(r["model_auroc"]) for r in rs]); p = np.array([float(r["probe_auroc"]) for r in rs])
        print(f"  window {'original' if T == '' else 'fixed T=' + T:12s} n={len(rs):2d}  model AUROC {m.mean():.3f} ({m.std(ddof=1) if len(m) > 1 else 0:.3f})  "
              f"mask/length probe AUROC {p.mean():.3f}")


out = shift_summary()
length_summary()
if out:
    with open(os.path.join(REV, "semisynth_response_matrix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
