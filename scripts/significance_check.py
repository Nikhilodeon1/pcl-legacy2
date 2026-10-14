import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEGACY2_ROOT = os.path.dirname(_HERE)

MORT_DIR = os.path.join(_LEGACY2_ROOT, "results", "mortality")
LOS_DIR = os.path.join(_LEGACY2_ROOT, "results", "los")

T_CRIT_DF2 = 4.303 


def paired_ttest(a, b):

    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    df = n - 1
    mean_d = d.mean()
    sd_d = d.std(ddof=1) if n > 1 else 0.0
    if sd_d == 0:
        t = float("inf") if mean_d != 0 else 0.0
        cohen_d = float("inf") if mean_d != 0 else 0.0
    else:
        t = mean_d / (sd_d / np.sqrt(n))
        cohen_d = mean_d / sd_d
    return mean_d, sd_d, t, df, cohen_d, d.tolist()


def verdict(t, df):
    if df != 2:
        return f"(df={df}, use judgement — critical value below is for df=2 only)"
    return "SIGNIFICANT at alpha=0.05" if abs(t) > T_CRIT_DF2 else "not significant at alpha=0.05"


def check_mortality():
    print("=" * 78)
    print("MORTALITY — paired by seed, within each direction")
    print("=" * 78)
    files = glob.glob(os.path.join(MORT_DIR, "*.json"))
    files = [f for f in files if os.path.basename(f) != "selection_criteria.json"]
    rows = {}
    for f in files:
        d = json.load(open(f))
        direction = f"{d['source']}->{d['target']}"
        key = (direction, d["method"])
        rows.setdefault(key, {})[d["seed"]] = d["ood"]["auroc"]

    directions = sorted(set(k[0] for k in rows))
    for direction in directions:
        seeds = sorted(set().union(*[rows[k].keys() for k in rows if k[0] == direction]))
        print(f"\n-- {direction} (seeds={seeds}) --")
        methods_present = sorted(set(k[1] for k in rows if k[0] == direction))
        per_method = {m: [rows[(direction, m)][s] for s in seeds] for m in methods_present}
        for m in methods_present:
            print(f"  {m:4s} per-seed AUROC: {[round(v, 6) for v in per_method[m]]}")
        for a, b in [("pcl", "erm"), ("pcl", "dro"), ("erm", "dro")]:
            if a not in per_method or b not in per_method:
                continue
            mean_d, sd_d, t, df, cohen_d, diffs = paired_ttest(per_method[a], per_method[b])
            print(f"  {a.upper()} - {b.upper()}: per-seed diffs={[round(x,6) for x in diffs]} "
                  f"mean={mean_d:+.6f}  sd={sd_d:.6f}  t={t:+.3f}  d={cohen_d:+.2f}  {verdict(t, df)}")


def check_los():
    print("\n" + "=" * 78)
    print("LOS — paired by seed, within each target (metric: R^2)")
    print("=" * 78)
    files = glob.glob(os.path.join(LOS_DIR, "*.json"))
    files = [f for f in files if os.path.basename(f) != "selection_criteria.json"]
    rows = {}
    all_targets = set()
    for f in files:
        d = json.load(open(f))
        for target, ood in d["ood"].items():
            if ood.get("r2") is None:
                continue
            all_targets.add(target)
            rows.setdefault((target, d["method"]), {})[d["seed"]] = ood["r2"]

    for target in sorted(all_targets):
        methods_present = sorted(set(k[1] for k in rows if k[0] == target))
        seeds = sorted(set().union(*[rows[(target, m)].keys() for m in methods_present]))
        print(f"\n-- {target} (seeds={seeds}) --")
        per_method = {m: [rows[(target, m)][s] for s in seeds] for m in methods_present}
        for m in methods_present:
            print(f"  {m:4s} per-seed R^2: {[round(v, 6) for v in per_method[m]]}")
        pairs = [("pcl", "erm"), ("pcl", "dro"), ("erm", "dro")]
        for a, b in pairs:
            if a not in per_method or b not in per_method:
                continue
            mean_d, sd_d, t, df, cohen_d, diffs = paired_ttest(per_method[a], per_method[b])
            print(f"  {a.upper()} - {b.upper()}: per-seed diffs={[round(x,6) for x in diffs]} "
                  f"mean={mean_d:+.6f}  sd={sd_d:.6f}  t={t:+.3f}  d={cohen_d:+.2f}  {verdict(t, df)}")


if __name__ == "__main__":
    check_mortality()
    check_los()
    print("\nNote: n=3 seeds -> df=2, very low power. A 'not significant' result here means")
    print("exactly that — it does not by itself prove the effect is zero, only that 3 seeds")
    print("cannot distinguish it from zero at the conventional threshold. Report honestly either way.")
