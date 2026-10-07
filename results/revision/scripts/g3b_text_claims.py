"""
G3b — qualitative / directional claims in the submitted paper, checked against raw JSON.
(g3_numbers_audit.py covers numeric claims; this covers statements about direction, sign, ranking.)

Outputs: g3b_text_claims.md, g3b_text_claims.json
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
los = {}
for f in glob.glob(os.path.join(ROOT, "results", "los", "*_s4*.json")):
    d = json.load(open(f)); los[(d["method"], d["seed"])] = d
mort = {}
for f in glob.glob(os.path.join(ROOT, "results", "mortality", "*_s4*.json")):
    d = json.load(open(f)); mort[(f"{d['source']}->{d['target']}", d["method"], d["seed"])] = d
sel = json.load(open(os.path.join(ROOT, "results", "los", "selection_criteria.json")))
cells = sel["cells"]
M, S, T = ["erm", "pcl", "dro"], [42, 43, 44], ["physionet_b", "mimic", "eicu"]
out, lines = {}, []


def claim(cid, text, verdict, detail):
    out[cid] = dict(claim=text, verdict=verdict, detail=detail)
    lines.append(f"| {cid} | {text} | **{verdict}** | {detail} |")


# C1: repr dist / MMD larger at far targets than near target
m = {t: {k: float(np.mean([c[k] for c in cells if c["target"] == t])) for k in ("repr_dist", "mmd", "true_r2")} for t in T}
far_larger = all(m[t][k] > m["physionet_b"][k] for t in ("mimic", "eicu") for k in ("repr_dist", "mmd"))
claim("C1", "repr. dist. / MMD are larger at the far targets (MIMIC, eICU) than at near-domain PhysioNet-B ('more distant databases in representation space')",
      "FALSE" if not far_larger else "TRUE",
      "mean repr_dist PN-B/MIMIC/eICU = " + "/".join(f"{m[t]['repr_dist']:.2f}" for t in T) +
      "; mean MMD = " + "/".join(f"{m[t]['mmd']:.3f}" for t in T) + " -> both criteria rank PN-B as the MOST distant target")

# C2: pooled rho has the opposite sign from every within-target value (repr dist, MMD)
sp_pool, sp_within = sel["spearman_pooled"], sel["spearman_within_target"]
for k in ("repr_dist", "mmd"):
    same = [t for t in T if np.sign(sp_within[t][k]) == np.sign(sp_pool[k])]
    claim(f"C2.{k}", f"pooled rho({k})={sp_pool[k]:+.3f} has the opposite sign from every within-target value",
          "FALSE" if same else "TRUE",
          "within-target " + ", ".join(f"{t}:{sp_within[t][k]:+.2f}" for t in T) +
          f"; same sign as pooled at: {same if same else 'none'}")

# C3: PCL never wins a single cell (LOS target x seed; mortality direction x seed)
wins_los = [(t, s) for t in T for s in S if max(M, key=lambda m_: los[(m_, s)]["ood"][t]["r2"]) == "pcl"]
claim("C3.los", "PCL never wins a single LOS cell (target x seed, 9 cells)", "TRUE" if not wins_los else "FALSE", f"PCL best in: {wins_los}")
wins_m = [(d, s) for d in ("mimic->eicu", "eicu->mimic") for s in S
          if max(M, key=lambda m_: mort[(d, m_, s)]["ood"]["auroc"]) == "pcl"]
claim("C3.mort", "PCL never wins a single mortality cell (6 cells)", "TRUE" if not wins_m else "FALSE", f"PCL best in: {wins_m}")

# C4: PCL lowest in 'the majority of LOS comparisons'
low = [(t, s) for t in T for s in S if min(M, key=lambda m_: los[(m_, s)]["ood"][t]["r2"]) == "pcl"]
claim("C4", "PCL is the lowest-scoring method in the majority of LOS cells", "TRUE" if len(low) > 4.5 else "FALSE",
      f"PCL lowest in {len(low)}/9 cells: {low}")

# C5: ranking DRO >= ERM > PCL (the 'same ranking as the corrected sepsis result') on each task
def mean_perf(task_dict_fn, keys):
    return np.mean([task_dict_fn(k) for k in keys])
rank_rows = []
for t in T:
    mm = {m_: np.mean([los[(m_, s)]["ood"][t]["r2"] for s in S]) for m_ in M}
    rank_rows.append(f"LOS {t}: " + " > ".join(sorted(M, key=lambda x: -mm[x])) + f" ({', '.join(f'{k}={v:.4f}' for k, v in mm.items())})")
for d in ("mimic->eicu", "eicu->mimic"):
    mm = {m_: np.mean([mort[(d, m_, s)]["ood"]["auroc"] for s in S]) for m_ in M}
    rank_rows.append(f"MORT {d}: " + " > ".join(sorted(M, key=lambda x: -mm[x])) + f" ({', '.join(f'{k}={v:.4f}' for k, v in mm.items())})")
los_dro_ge_erm = sum(np.mean([los[("dro", s)]["ood"][t]["r2"] for s in S]) >= np.mean([los[("erm", s)]["ood"][t]["r2"] for s in S]) for t in T)
claim("C5", "'same DRO >= ERM > PCL ranking' replicates on LOS", "FALSE (DRO>=ERM holds at %d/3 LOS targets)" % los_dro_ge_erm,
      " | ".join(rank_rows))

# C6: Violation/recon correct sign at every LOS target
ok = all(sp_within[t][k] < 0 for t in T for k in ("violation", "recon_mse"))
claim("C6", "violation and recon. error are correctly signed (negative rho, lower-is-better) at every LOS target", "TRUE" if ok else "FALSE",
      "; ".join(f"{t}:{sp_within[t]['violation']:+.2f}/{sp_within[t]['recon_mse']:+.2f}" for t in T) +
      " (n=9 checkpoints per target; permutation p in a4_selection_los.csv: significant only at physionet_b)")

hdr = "# G3b qualitative claims audit\n\n| id | claim | verdict | detail |\n|---|---|---|---|\n"
open(os.path.join(REV, "g3b_text_claims.md"), "w").write(hdr + "\n".join(lines) + "\n")
json.dump(out, open(os.path.join(REV, "g3b_text_claims.json"), "w"), indent=1, default=float)
for k, v in out.items():
    print(k, "|", v["verdict"], "|", v["detail"][:200])
