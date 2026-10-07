"""
G3 — numbers audit. Regenerates every quantitative claim of the submitted paper
from the raw result JSONs and prints a PASS/MISMATCH line per claim.

Inputs : pcl-legacy2/results/{mortality,los}/*.json
Outputs: results/revision/g3_numbers_audit.md, g3_numbers.csv, g3_numbers.json
Run    : python g3_numbers_audit.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))  # pcl-legacy2
MORT = os.path.join(ROOT, "results", "mortality")
LOS = os.path.join(ROOT, "results", "los")

rows = []      # (id, claim, paper_value, computed_value, verdict)
numbers = {}   # everything the paper may quote


def rec(cid, claim, paper, computed, ok=None, note=""):
    if ok is None:
        ok = (paper == computed)
    rows.append((cid, claim, paper, computed, "PASS" if ok else "MISMATCH", note))


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(n)) if sd > 0 else float("inf")
    return d.mean(), sd, t, d.mean() / sd if sd > 0 else float("inf")


# ---------------- load ----------------
mort = {}  # (direction, method, seed) -> json
for f in glob.glob(os.path.join(MORT, "*.json")):
    if "selection" in os.path.basename(f):
        continue
    d = json.load(open(f))
    mort[(f"{d['source']}->{d['target']}", d["method"], d["seed"])] = d
los = {}   # (method, seed) -> json
for f in glob.glob(os.path.join(LOS, "*.json")):
    if "selection" in os.path.basename(f):
        continue
    d = json.load(open(f))
    los[(d["method"], d["seed"])] = d

methods = ["erm", "pcl", "dro"]
seeds = [42, 43, 44]
dirs = ["mimic->eicu", "eicu->mimic"]
targets = ["physionet_b", "mimic", "eicu"]

rec("data.mort_n", "mortality runs found", 18, len(mort))
rec("data.los_n", "LOS runs found", 9, len(los))

# ---------------- Table 1: mortality mean AUROC ----------------
paper_t1 = {("mimic->eicu", "erm"): 0.760, ("mimic->eicu", "pcl"): 0.744, ("mimic->eicu", "dro"): 0.764,
            ("eicu->mimic", "erm"): 0.819, ("eicu->mimic", "pcl"): 0.808, ("eicu->mimic", "dro"): 0.819}
for (dr, m), pv in paper_t1.items():
    v = np.mean([mort[(dr, m, s)]["ood"]["auroc"] for s in seeds])
    numbers[f"mort.auroc.{dr}.{m}"] = v
    rec(f"T1.{dr}.{m}", f"Table 1 mean AUROC {dr} {m}", pv, round(v, 3), ok=abs(round(v, 3) - pv) < 1e-9)

# PCL lowest in every direction x seed cell
cnt = 0
for dr in dirs:
    for s in seeds:
        a = {m: mort[(dr, m, s)]["ood"]["auroc"] for m in methods}
        cnt += int(a["pcl"] == min(a.values()))
numbers["mort.pcl_lowest_cells"] = cnt
rec("mort.pcl_lowest", "PCL lowest AUROC in 6/6 direction x seed cells", 6, cnt)

# mortality paired tests
mt = {}
for dr in dirs:
    for a, b in [("pcl", "erm"), ("pcl", "dro"), ("erm", "dro")]:
        x = [mort[(dr, a, s)]["ood"]["auroc"] for s in seeds]
        y = [mort[(dr, b, s)]["ood"]["auroc"] for s in seeds]
        mean, sd, t, dd = paired(x, y)
        mt[(dr, a, b)] = (mean, sd, t, dd)
        numbers[f"mort.paired.{dr}.{a}-{b}"] = dict(mean=mean, sd=sd, t=t, d=dd)
m = mt[("eicu->mimic", "pcl", "dro")]
rec("mort.t_main", "eICU->MIMIC PCL-DRO t", -8.24, round(m[2], 2), ok=abs(round(m[2], 2) + 8.24) < 0.006)
rec("mort.d_main", "eICU->MIMIC PCL-DRO Cohen d", -4.8, round(m[3], 1), ok=abs(round(m[3], 1) + 4.8) < 0.06)
rec("mort.mean_main", "eICU->MIMIC PCL-DRO mean diff", -0.01085, round(m[0], 5), ok=abs(m[0] + 0.01085) < 1e-5)
rec("mort.sd_main", "eICU->MIMIC PCL-DRO SD of diffs", 0.00228, round(m[1], 5), ok=abs(m[1] - 0.00228) < 1e-5)
others = [abs(mt[(dr, a, b)][2]) for dr in dirs for (a, b) in [("pcl", "erm"), ("pcl", "dro")]
          if not (dr == "eicu->mimic" and (a, b) == ("pcl", "dro"))]
rec("mort.t_other_range", "other three PCL-vs-X |t| in 1.9-3.5", "1.9-3.5",
    f"{min(others):.2f}-{max(others):.2f}", ok=(round(min(others), 1) >= 1.9 - 0.05 and round(max(others), 1) <= 3.5 + 0.05))
rec("mort.all_pcl_neg", "all four PCL-vs-ERM/DRO mean diffs negative", True,
    all(mt[(dr, a, b)][0] < 0 for dr in dirs for (a, b) in [("pcl", "erm"), ("pcl", "dro")]))
numbers["mort.erm_dro_t"] = {dr: mt[(dr, "erm", "dro")][2] for dr in dirs}

# SD of AUROC across seeds per (direction, method) ("std range 0.002 to 0.014")
sds = {}
for dr in dirs:
    for me in methods:
        v = [mort[(dr, me, s)]["ood"]["auroc"] for s in seeds]
        sds[(dr, me)] = float(np.std(v, ddof=1))
numbers["mort.auroc_seed_sd"] = {f"{k[0]}.{k[1]}": v for k, v in sds.items()}
rec("mort.sd_range", "seed SD of OOD AUROC ranges 0.002-0.014 (claim from review; not in current main.tex)",
    "0.002-0.014", f"{min(sds.values()):.4f}-{max(sds.values()):.4f}",
    ok=(abs(min(sds.values()) - 0.002) < 0.0006 and abs(max(sds.values()) - 0.014) < 0.0006))

# in-domain AUROC range ("ATC source-AUROC 0.84-0.86" claim)
ind = {(dr, me, s): mort[(dr, me, s)]["in_domain"]["auroc"] for dr in dirs for me in methods for s in seeds}
numbers["mort.in_domain_auroc_range"] = [min(ind.values()), max(ind.values())]
rec("mort.atc_src_auroc", "source-domain AUROC range 0.84-0.86 (ATC explanation; not in current main.tex)",
    "0.84-0.86", f"{min(ind.values()):.3f}-{max(ind.values()):.3f}",
    ok=(min(ind.values()) >= 0.835 and max(ind.values()) <= 0.865))

# ---------------- LOS ----------------
def r2(m, s, t): return los[(m, s)]["ood"][t]["r2"]
def mae(m, s, t): return los[(m, s)]["ood"][t]["mae_hours"]

pm = {}  # per-method mean
for t in targets:
    for me in methods:
        pm[(t, me, "r2")] = float(np.mean([r2(me, s, t) for s in seeds]))
        pm[(t, me, "mae")] = float(np.mean([mae(me, s, t) for s in seeds]))
for me in methods:
    pm[("indomain", me, "r2")] = float(np.mean([los[(me, s)]["in_domain"]["r2"] for s in seeds]))
    pm[("indomain", me, "mae")] = float(np.mean([los[(me, s)]["in_domain"]["mae_hours"] for s in seeds]))
numbers["los.per_method_mean"] = {f"{k[0]}.{k[1]}.{k[2]}": v for k, v in pm.items()}

# Table 2 ranges, two readings: (a) range of per-method means, (b) range over all 9 runs
paper_t2 = {"indomain": ("6.4-6.7", "0.26-0.27", (0.26, 0.27)), "physionet_b": ("5.9-6.0", "0.21-0.23", (0.21, 0.23)),
            "mimic": ("62.6-63.1", "-0.13 to -0.12", (-0.13, -0.12)), "eicu": ("49.7-50.2", "-0.11 to -0.10", (-0.11, -0.10))}
for key, (pmae, pr2, (lo, hi)) in paper_t2.items():
    mae_means = [pm[(key, me, "mae")] for me in methods]
    r2_means = [pm[(key, me, "r2")] for me in methods]
    if key == "indomain":
        all9_mae = [los[(me, s)]["in_domain"]["mae_hours"] for me in methods for s in seeds]
        all9_r2 = [los[(me, s)]["in_domain"]["r2"] for me in methods for s in seeds]
    else:
        all9_mae = [mae(me, s, key) for me in methods for s in seeds]
        all9_r2 = [r2(me, s, key) for me in methods for s in seeds]
    numbers[f"T2.{key}"] = dict(mae_means=[min(mae_means), max(mae_means)], r2_means=[min(r2_means), max(r2_means)],
                                mae_all9=[min(all9_mae), max(all9_mae)], r2_all9=[min(all9_r2), max(all9_r2)])
    rec(f"T2.{key}.mae", f"Table 2 MAE range {key} (range of per-method means / over all 9 runs)", pmae,
        f"means {min(mae_means):.2f}-{max(mae_means):.2f}; all9 {min(all9_mae):.2f}-{max(all9_mae):.2f}",
        ok=(round(min(mae_means), 1) == float(pmae.split('-')[0]) and round(max(mae_means), 1) == float(pmae.split('-')[1])),
        note="caption says 'mean over 3 seeds x 3 methods' - ranges are over per-method means")
    rec(f"T2.{key}.r2", f"Table 2 R2 range {key}", pr2,
        f"means {min(r2_means):.3f}-{max(r2_means):.3f}; all9 {min(all9_r2):.3f}-{max(all9_r2):.3f}",
        ok=(round(min(r2_means), 2) == min(lo, hi) and round(max(r2_means), 2) == max(lo, hi)))

# pairwise method-difference means (LOS)
maxdiff = 0.0
maxdiff_where = None
lt = {}
for t in targets:
    for a, b in [("pcl", "erm"), ("pcl", "dro"), ("erm", "dro")]:
        x = [r2(a, s, t) for s in seeds]
        y = [r2(b, s, t) for s in seeds]
        mean, sd, tt, dd = paired(x, y)
        lt[(t, a, b)] = (mean, sd, tt, dd)
        numbers[f"los.paired.{t}.{a}-{b}"] = dict(mean=mean, sd=sd, t=tt, d=dd)
        if abs(mean) > maxdiff:
            maxdiff, maxdiff_where = abs(mean), (t, a, b)
numbers["los.max_pairwise_diff"] = dict(value=maxdiff, where=maxdiff_where)
rec("los.maxdiff", "all pairwise method-difference means <= 0.021 R2", "<=0.021", f"{maxdiff:.4f} at {maxdiff_where}",
    ok=maxdiff <= 0.0215)
m1 = lt[("eicu", "pcl", "erm")]
rec("los.t_eicu_pcl_erm", "LOS eICU PCL-ERM t", -5.61, round(m1[2], 2), ok=abs(round(m1[2], 2) + 5.61) < 0.006)
rec("los.d_eicu_pcl_erm", "LOS eICU PCL-ERM d", -3.2, round(m1[3], 1), ok=abs(round(m1[3], 1) + 3.2) < 0.06)
rec("los.mean_eicu_pcl_erm", "LOS eICU PCL-ERM mean", -0.0113, round(m1[0], 4), ok=abs(m1[0] + 0.0113) < 5e-5)
m2 = lt[("mimic", "pcl", "dro")]
rec("los.t_mimic_pcl_dro", "LOS MIMIC PCL-DRO t", -29.7, round(m2[2], 1), ok=abs(round(m2[2], 1) + 29.7) < 0.06)
rec("los.d_mimic_pcl_dro", "LOS MIMIC PCL-DRO d", -17.1, round(m2[3], 1), ok=abs(round(m2[3], 1) + 17.1) < 0.06)
rec("los.mean_mimic_pcl_dro", "LOS MIMIC PCL-DRO mean", -0.0056, round(m2[0], 4), ok=abs(m2[0] + 0.0056) < 5e-5)
rec("los.sd_mimic_pcl_dro", "LOS MIMIC PCL-DRO SD", 0.0003, round(m2[1], 4), ok=abs(m2[1] - 0.0003) < 5e-5)
sig = [(k, v[2]) for k, v in lt.items() if abs(v[2]) > 4.303]
numbers["los.significant_pairs"] = [(list(k), t) for k, t in sig]
rec("los.n_sig", "number of LOS paired comparisons clearing |t|>4.303 (paper: 'PCL below ERM at eICU, below DRO at MIMIC')",
    2, len(sig), note=f"cleared: {[(k, round(t, 2)) for k, t in sig]}")
rec("los.pb_none", "No comparison clears at PhysioNet-B", True, all(abs(lt[("physionet_b", a, b)][2]) < 4.303
                                                                   for a, b in [("pcl", "erm"), ("pcl", "dro"), ("erm", "dro")]))

# the ~0.38 swing and the 20-30x claim
indom = float(np.mean([pm[("indomain", me, "r2")] for me in methods]))
far = float(np.mean([pm[(t, me, "r2")] for t in ["mimic", "eicu"] for me in methods]))
pb = float(np.mean([pm[("physionet_b", me, "r2")] for me in methods]))
numbers["los.swing"] = dict(indomain=indom, far=far, physionet_b=pb, swing=indom - far)
rec("los.swing", "in-domain -> far zero-shot R2 swing ~0.38", 0.38, round(indom - far, 3), ok=abs((indom - far) - 0.38) < 0.01)
between_target = pb - min(pm[("mimic", me, "r2")] for me in methods)
within = {t: max(pm[(t, me, "r2")] for me in methods) - min(pm[(t, me, "r2")] for me in methods) for t in targets}
ratios = {t: between_target / within[t] for t in targets}
numbers["los.pooling_ratio"] = dict(between_target_range=between_target, within_target_method_range=within, ratio=ratios)
lo_r, hi_r = min(ratios.values()), max(ratios.values())
rec("los.ratio_20_30", "between-target R2 range is 20-30x the within-target between-method range", "20-30x",
    f"{lo_r:.1f}x-{hi_r:.1f}x (by target: " + ", ".join(f'{t} {v:.1f}x' for t, v in ratios.items()) + ")",
    ok=(lo_r >= 19.5 and hi_r <= 30.5),
    note="uses per-method-mean ranges; vs the <=0.02 bound the paper quotes the ratio is >= 0.35/0.0205 = 17x")

# LOS label tail ratios (Table 7 values as printed in the paper)
paper_max = {"A": 335.0, "B": 336.0, "mimic": 492.7, "eicu": 1108.3}
paper_p99 = {"A": 174.5, "B": 137.4, "mimic": 387.9, "eicu": 469.9}
numbers["los.tail_ratios_vs_physionetA"] = dict(
    max_mimic=paper_max["mimic"] / paper_max["A"], max_eicu=paper_max["eicu"] / paper_max["A"],
    p99_mimic=paper_p99["mimic"] / paper_p99["A"], p99_eicu=paper_p99["eicu"] / paper_p99["A"])
rec("los.tail_1p5_3", "real tails 1.5-3x longer (max)", "1.5-3x",
    f"MIMIC {paper_max['mimic']/paper_max['A']:.2f}x, eICU {paper_max['eicu']/paper_max['A']:.2f}x (max); "
    f"P99 {paper_p99['mimic']/paper_p99['A']:.2f}x/{paper_p99['eicu']/paper_p99['A']:.2f}x",
    ok=False, note="eICU max is 3.31x, not <=3x; text should say 1.5-3.3x (max) and cite P99 ratios 2.2-2.7x. "
                   "Table 7 values themselves need verification against the raw data (needs dataset; see G2)")

# ---------------- LOS selection criteria (recomputed from cells) ----------------
sc_path = os.path.join(LOS, "selection_criteria.json")
sc = json.load(open(sc_path))
cells = sc["cells"]
sig_names = ["violation", "recon_mse", "repr_dist", "mmd"]
print("LOS cell keys:", list(cells[0].keys()))
within_rho = {}
for t in targets:
    sub = [c for c in cells if c["target"] == t]
    for sn in sig_names:
        within_rho[(t, sn)] = stats.spearmanr([c[sn] for c in sub], [c["true_r2"] for c in sub]).statistic
pooled = {sn: stats.spearmanr([c[sn] for c in cells], [c["true_r2"] for c in cells]).statistic for sn in sig_names}
numbers["los.within_rho"] = {f"{k[0]}.{k[1]}": float(v) for k, v in within_rho.items()}
numbers["los.pooled_rho"] = {k: float(v) for k, v in pooled.items()}
paper_t3 = {("physionet_b", "violation"): -0.72, ("mimic", "violation"): -0.32, ("eicu", "violation"): -0.45,
            ("physionet_b", "recon_mse"): -0.73, ("mimic", "recon_mse"): -0.55, ("eicu", "recon_mse"): -0.60,
            ("physionet_b", "repr_dist"): 0.33, ("mimic", "repr_dist"): 0.63, ("eicu", "repr_dist"): 0.33,
            ("physionet_b", "mmd"): -0.30, ("mimic", "mmd"): 0.35, ("eicu", "mmd"): 0.02}
for k, pv in paper_t3.items():
    rec(f"T3.{k[0]}.{k[1]}", f"Table 3 within-target rho {k}", pv, round(float(within_rho[k]), 2),
        ok=abs(round(float(within_rho[k]), 2) - pv) < 1e-9, note="n=9 checkpoints per target")
paper_pool = {"violation": -0.198, "recon_mse": -0.374, "repr_dist": 0.896, "mmd": 0.888}
for k, pv in paper_pool.items():
    rec(f"T8.pooled.{k}", f"pooled rho {k}", pv, round(float(pooled[k]), 3),
        ok=abs(round(float(pooled[k]), 3) - pv) < 1e-9, note="n=27 pooled across 3 targets")

# mortality selection criteria JSON presence
mp = os.path.join(MORT, "selection_criteria.json")
rec("mort.selection_json", "results/mortality/selection_criteria.json present (needed to regenerate Table 6 + ATC rho)",
    True, os.path.exists(mp), note="absent locally: Table 6 and the ATC rho values cannot be regenerated here until copied from the pod")

# ---------------- write ----------------
with open(os.path.join(REV, "g3_numbers.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "claim", "paper_value", "computed_value", "verdict", "note"])
    w.writerows(rows)
json.dump(numbers, open(os.path.join(REV, "g3_numbers.json"), "w"), indent=1, default=float)
with open(os.path.join(REV, "g3_numbers_audit.md"), "w") as f:
    f.write("# G3 numbers audit\n\nGenerated by scripts/g3_numbers_audit.py from raw JSON.\n\n")
    f.write("| id | claim | paper | computed | verdict | note |\n|---|---|---|---|---|---|\n")
    for r in rows:
        f.write("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |\n")
bad = [r for r in rows if r[4] != "PASS"]
print(f"{len(rows)} checks, {len(bad)} not PASS")
for r in bad:
    print("  MISMATCH:", r[0], "| paper:", r[2], "| computed:", r[3], "|", r[5])
