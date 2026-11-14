"""
Rigorous Statistical Inference for Phase 2c (AAAI Reviewer Standards):
1. Separates Confirmatory vs Exploratory hypothesis families.
2. Replaces t-test-only small-n inference with exact paired sign-flip permutation
   tests (for n <= 12; Monte Carlo for n > 12) + paired bootstrap 95% CIs.
3. Applies Holm-Bonferroni multiple-testing correction per family.
4. Adds TOST (Two One-Sided Tests) equivalence testing at pre-specified margins
   so non-significant differences are never conflated with statistical equivalence.
5. Provides exact tie-safe patient-level paired cluster bootstrap for Site B Sepsis
   when per-stay prediction .npz files are available.
"""
import argparse
import glob
import itertools
import json
import os
import sys
import numpy as np
import scipy.stats as stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from phase2c_summary import load_all_conditions

CANDIDATE_SEEDS = list(range(42, 52))  # Supports n=5 (42-46) and n=10 (42-51)


def paired_sign_flip_p(diffs, n_perm=10000, seed=42):
    """Two-sided paired sign-flip permutation p-value on per-seed differences."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0:
        return 1.0, "empty"
    obs = abs(float(np.mean(d)))
    if n <= 12:
        signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=n)), dtype=float)
        null_means = np.abs(signs @ d) / n
        p = float(np.mean(null_means >= obs - 1e-12))
        return p, f"exact_{1 << n}"
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null_means = np.abs(flips @ d) / n
    p = float((1.0 + np.sum(null_means >= obs - 1e-12)) / (n_perm + 1.0))
    return p, f"mc_{n_perm}"


def tost_paired(mean_d, se, n, margin):
    """Two One-Sided Tests (TOST) for paired differences within (-margin, +margin)."""
    if n < 2:
        eq = abs(mean_d) < margin
        return {"margin": float(margin), "p1": 1.0, "p2": 1.0, "p_tost": 0.0 if eq else 1.0, "equivalent": bool(eq)}
    df = n - 1
    if se <= 1e-12:
        if abs(mean_d) < margin:
            return {"margin": float(margin), "p1": 0.0, "p2": 0.0, "p_tost": 0.0, "equivalent": True}
        return {"margin": float(margin), "p1": 1.0, "p2": 1.0, "p_tost": 1.0, "equivalent": False}
    t1 = (mean_d + margin) / se
    t2 = (mean_d - margin) / se
    p1 = float(1.0 - stats.t.cdf(t1, df=df))
    p2 = float(stats.t.cdf(t2, df=df))
    p_tost = max(p1, p2)
    return {
        "margin": float(margin),
        "p1": p1,
        "p2": p2,
        "p_tost": float(p_tost),
        "equivalent": bool(p_tost < 0.05),
    }


def paired_contrast_stats(vals_a, vals_b, tost_margins=(0.01, 0.02), n_perm=10000, n_boot=10000, seed=42):
    """Compute paired t-test, exact/MC sign-flip permutation test, bootstrap CI, and TOST."""
    a = np.asarray(vals_a, dtype=float)
    b = np.asarray(vals_b, dtype=float)
    d = a - b
    n = len(d)
    mean_d = float(np.mean(d)) if n else float("nan")
    if n < 2:
        return {
            "n": n, "mean_diff": mean_d, "sd": 0.0, "se": 0.0,
            "t_stat": 0.0, "t_p_value": 1.0,
            "t_ci_95": [mean_d, mean_d], "t_ci_90": [mean_d, mean_d],
            "boot_ci_95": [mean_d, mean_d],
            "perm_p_value": 1.0, "perm_mode": "n<2",
            "tost": [tost_paired(mean_d, 0.0, n, m) for m in tost_margins],
        }
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)
    t_stat = mean_d / se if se > 1e-12 else (0.0 if abs(mean_d) <= 1e-12 else float(np.sign(mean_d) * 1e9))
    t_p = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1))) if se > 1e-12 else (1.0 if abs(mean_d) <= 1e-12 else 0.0)
    tcrit_95 = float(stats.t.ppf(0.975, df=n - 1))
    tcrit_90 = float(stats.t.ppf(0.95, df=n - 1))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = d[idx].mean(axis=1)
    boot_ci_95 = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]

    perm_p, perm_mode = paired_sign_flip_p(d, n_perm=n_perm, seed=seed)
    return {
        "n": n,
        "mean_diff": mean_d,
        "sd": sd,
        "se": float(se),
        "t_stat": float(t_stat),
        "t_p_value": t_p,
        "t_ci_95": [mean_d - tcrit_95 * se, mean_d + tcrit_95 * se],
        "t_ci_90": [mean_d - tcrit_90 * se, mean_d + tcrit_90 * se],
        "boot_ci_95": boot_ci_95,
        "perm_p_value": perm_p,
        "perm_mode": perm_mode,
        "tost": [tost_paired(mean_d, se, n, m) for m in tost_margins],
    }


def holm_bonferroni(p_values):
    """Return Holm-Bonferroni step-down adjusted p-values preserving input order."""
    m = len(p_values)
    if m == 0:
        return []
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * p[idx]
        running_max = max(running_max, scaled)
        adj_sorted[rank] = min(1.0, running_max)
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return [float(x) for x in adj]


def _auroc_weighted_tie_safe(y, score, w):
    """Exact weighted AUROC using mid-rank tie handling (collapsed constant pred -> 0.5)."""
    order = np.argsort(score, kind="mergesort")
    s_ord = score[order]
    y_ord = y[order]
    w_ord = w[order]
    _, inv = np.unique(s_ord, return_inverse=True)
    w_neg_group = np.bincount(inv, weights=w_ord * (1.0 - y_ord))
    w_pos_group = np.bincount(inv, weights=w_ord * y_ord)
    tot_pos = float(w_pos_group.sum())
    tot_neg = float(w_neg_group.sum())
    if tot_pos <= 0.0 or tot_neg <= 0.0:
        return float("nan")
    cum_neg_below = np.cumsum(w_neg_group) - w_neg_group
    num = np.sum(w_pos_group * (cum_neg_below + 0.5 * w_neg_group))
    return float(num / (tot_pos * tot_neg))


def siteb_patient_bootstrap(pred_records, B=10000, seed=777, contrast_pairs=None):
    """
    Paired patient-level cluster bootstrap across Site B prediction .npz files.
    pred_records: dict[(condition_name, seed)] -> dict(probs=..., labels=..., pids=...)
    """
    if not pred_records:
        return {"available": False, "reason": "No Site B .npz prediction files found."}

    ref_key = next(iter(pred_records))
    y = np.asarray(pred_records[ref_key]["labels"], dtype=float)
    pids = np.asarray(pred_records[ref_key]["pids"])
    for k, rec in pred_records.items():
        if len(rec["labels"]) != len(y) or not np.array_equal(rec["labels"], y):
            raise ValueError(f"Label mismatch in Site B prediction file for {k}")

    up, inv = np.unique(pids, return_inverse=True)
    n_patients = len(up)
    rng = np.random.default_rng(seed)

    # Pre-sort once per (condition, seed) for fast exact cluster bootstrap
    prepped = {}
    for k, rec in pred_records.items():
        score = np.asarray(rec["probs"], dtype=float)
        order = np.argsort(score, kind="mergesort")
        s_ord = score[order]
        y_ord = y[order]
        _, inv_score = np.unique(s_ord, return_inverse=True)
        prepped[k] = (order, y_ord, inv_score)

    boot_scores = {k: np.empty(B, dtype=float) for k in pred_records}
    for b in range(B):
        wp = rng.multinomial(n_patients, np.full(n_patients, 1.0 / n_patients)).astype(np.float64)
        w = wp[inv]
        for k, (order, y_ord, inv_score) in prepped.items():
            w_ord = w[order]
            w_neg_g = np.bincount(inv_score, weights=w_ord * (1.0 - y_ord))
            w_pos_g = np.bincount(inv_score, weights=w_ord * y_ord)
            cum_neg = np.cumsum(w_neg_g) - w_neg_g
            boot_scores[k][b] = np.sum(w_pos_g * (cum_neg + 0.5 * w_neg_g)) / (w_pos_g.sum() * w_neg_g.sum())

    by_condition = {}
    conditions_present = sorted({k[0] for k in pred_records})
    for c in conditions_present:
        c_keys = sorted([k for k in pred_records if k[0] == c], key=lambda x: x[1])
        stacked = np.mean([boot_scores[k] for k in c_keys], axis=0)
        by_condition[c] = {
            "seeds": [k[1] for k in c_keys],
            "mean": float(np.mean(stacked)),
            "boot_ci_95": [float(np.percentile(stacked, 2.5)), float(np.percentile(stacked, 97.5))],
        }

    contrasts = {}
    for title, ca, cb in (contrast_pairs or []):
        sa = {k[1]: k for k in pred_records if k[0] == ca}
        sb = {k[1]: k for k in pred_records if k[0] == cb}
        shared = sorted(set(sa) & set(sb))
        if not shared:
            continue
        diffs_b = np.mean([boot_scores[sa[s]] - boot_scores[sb[s]] for s in shared], axis=0)
        contrasts[title] = {
            "seeds": shared,
            "mean_diff": float(np.mean(diffs_b)),
            "boot_ci_95": [float(np.percentile(diffs_b, 2.5)), float(np.percentile(diffs_b, 97.5))],
            "p_boot_two_sided": float(2.0 * min(np.mean(diffs_b <= 0.0), np.mean(diffs_b >= 0.0))),
        }

    return {
        "available": True,
        "B": B,
        "n_stays": int(len(y)),
        "n_patients": int(n_patients),
        "n_positives": int(np.sum(y)),
        "by_condition": by_condition,
        "contrasts": contrasts,
    }


def _shared_seed_values(data, cond_a, cond_b, metric):
    shared = []
    for s in CANDIDATE_SEEDS:
        va = data.get(cond_a, {}).get(s, {}).get(metric)
        vb = data.get(cond_b, {}).get(s, {}).get(metric)
        if va is not None and vb is not None:
            shared.append((s, float(va), float(vb)))
    return shared


def _run_family(data, specs, tost_margins=(0.01, 0.02), seed=42):
    out = []
    for idx, spec in enumerate(specs):
        title, ca, cb, metric = spec[:4]
        if metric == "__interaction__":
            # Spec format: (title, (a1, a0), (b1, b0), "__interaction__", inner_metric)
            a1, a0 = ca
            b1, b0 = cb
            inner_m = spec[4]
            shared = []
            for s in CANDIDATE_SEEDS:
                vals = [data.get(c, {}).get(s, {}).get(inner_m) for c in (a1, a0, b1, b0)]
                if all(v is not None for v in vals):
                    shared.append((s, float(vals[0]) - float(vals[1]), float(vals[2]) - float(vals[3])))
            metric_label = inner_m
        else:
            shared = _shared_seed_values(data, ca, cb, metric)
            metric_label = metric

        if not shared:
            continue
        seeds_used = [x[0] for x in shared]
        vals_a = [x[1] for x in shared]
        vals_b = [x[2] for x in shared]
        res = paired_contrast_stats(vals_a, vals_b, tost_margins=tost_margins, seed=seed + idx)
        res["title"] = title
        res["metric"] = metric_label
        res["seeds"] = seeds_used
        out.append(res)

    if out:
        holm_perm = holm_bonferroni([r["perm_p_value"] for r in out])
        holm_t = holm_bonferroni([r["t_p_value"] for r in out])
        for r, hp, ht in zip(out, holm_perm, holm_t):
            r["p_holm_perm"] = hp
            r["p_holm_t"] = ht
    return out


CONFIRMATORY_SPECS = [
    ("C1. Sepsis: PCL Clean vs ERM Clean (10k)", "PCL Clean (10k)", "ERM Clean (10k)", "site_b_auroc"),
    ("C2. Sepsis: Volume Effect on ERM (25k - 10k)", "ERM Matched-Vol (25k)", "ERM Clean (10k)", "site_b_auroc"),
    ("C3. Sepsis: Volume Effect on PCL (25k - 10k)", "PCL Matched-Vol (25k)", "PCL Clean (10k)", "site_b_auroc"),
    ("C4. Sepsis: PCL vs ERM at Matched 25k", "PCL Matched-Vol (25k)", "ERM Matched-Vol (25k)", "site_b_auroc"),
    ("C5. Sepsis: 2x2 Volume x Method Interaction", ("ERM Matched-Vol (25k)", "ERM Clean (10k)"), ("PCL Matched-Vol (25k)", "PCL Clean (10k)"), "__interaction__", "site_b_auroc"),
    ("C6. LOS T=24: PCL Clean vs ERM Clean (10k)", "PCL Clean (10k)", "ERM Clean (10k)", "los_site_b_r2"),
    ("C7. LOS T=24: PCL vs ERM at Matched 25k", "PCL Matched-Vol (25k)", "ERM Matched-Vol (25k)", "los_site_b_r2"),
]

EXPLORATORY_SPECS = [
    ("E1. Sepsis: PCL High-Budget 80k vs 10k", "PCL Matched-Vol (80k)", "PCL Clean (10k)", "site_b_auroc"),
    ("E2. Sepsis: PCL 80k vs ERM 25k", "PCL Matched-Vol (80k)", "ERM Matched-Vol (25k)", "site_b_auroc"),
    ("E3. Sepsis: ERM 10k vs Supervised Scratch", "ERM Clean (10k)", "Supervised Scratch", "site_b_auroc"),
    ("E4. Sepsis: PCL 10k vs Supervised Scratch", "PCL Clean (10k)", "Supervised Scratch", "site_b_auroc"),
    ("E5. Sepsis: PCL+VICReg vs PCL Clean (10k)", "PCL + VICReg (10k)", "PCL Clean (10k)", "site_b_auroc"),
    ("E6. Sepsis: MAP-Only vs Full PCL (10k)", "PCL MAP-Only (10k)", "PCL Clean (10k)", "site_b_auroc"),
    ("E7. Sepsis: No-HH vs Full PCL (10k)", "PCL No-HH (10k)", "PCL Clean (10k)", "site_b_auroc"),
    ("E8. Sepsis: Placebo vs Full PCL (10k)", "PCL Placebo (10k)", "PCL Clean (10k)", "site_b_auroc"),
    ("E9. Sepsis: Placebo vs ERM Clean (10k)", "PCL Placebo (10k)", "ERM Clean (10k)", "site_b_auroc"),
]


def discover_siteb_preds(pred_dirs):
    """Locate saved Site B .npz prediction files keyed to canonical condition names."""
    prefix_map = {
        "lam0_p": "ERM Clean (10k)",
        "erm_matched_vol_p": "ERM Matched-Vol (25k)",
        "lam0.5_p": "PCL Clean (10k)",
        "pcl_matched_vol_p": "PCL Matched-Vol (25k)",
        "pcl_matched_vol_n80000_p": "PCL Matched-Vol (80k)",
        "pcl_maponly_p": "PCL MAP-Only (10k)",
        "pcl_nohh_p": "PCL No-HH (10k)",
        "pcl_placebo_p": "PCL Placebo (10k)",
        "pcl_varreg1_p": "PCL + VICReg (10k)",
        "scratch_p": "Supervised Scratch",
    }
    records = {}
    for d in pred_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*_site_b.npz"))):
            base = os.path.basename(f)
            if "_leakpb_" in base:
                continue
            matched_cond = None
            for pref, cname in sorted(prefix_map.items(), key=lambda x: -len(x[0])):
                if base.startswith(pref):
                    matched_cond = cname
                    break
            if not matched_cond:
                continue
            # Extract trailing fine-tune seed _s<seed>_site_b.npz
            try:
                seed_part = base.rsplit("_s", 1)[1].split("_")[0]
                seed_val = int(seed_part)
            except (IndexError, ValueError):
                continue
            npz = np.load(f, allow_pickle=True)
            records[(matched_cond, seed_val)] = {
                "probs": npz["probs"],
                "labels": npz["labels"],
                "pids": npz["pids"] if "pids" in npz else np.arange(len(npz["labels"])),
            }
    return records


def print_family_table(header, rows):
    print(f"\n=== {header} ===")
    print(f"{'Contrast':<48} {'n':>2} {'Diff':>8} {'95% t-CI':>19} {'p_t':>7} {'p_perm':>7} {'p_Holm':>7} {'TOST<=0.01':>11} {'TOST<=0.02':>11}")
    print("-" * 130)
    for r in rows:
        ci = f"[{r['t_ci_95'][0]:+.4f},{r['t_ci_95'][1]:+.4f}]"
        tost1 = f"{r['tost'][0]['p_tost']:.3f}({'Y' if r['tost'][0]['equivalent'] else 'N'})"
        tost2 = f"{r['tost'][1]['p_tost']:.3f}({'Y' if r['tost'][1]['equivalent'] else 'N'})"
        print(f"{r['title']:<48} {r['n']:>2} {r['mean_diff']:>+8.4f} {ci:>19} "
              f"{r['t_p_value']:>7.4f} {r['perm_p_value']:>7.4f} {r['p_holm_perm']:>7.4f} {tost1:>11} {tost2:>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tost-margins", type=float, nargs="+", default=[0.01, 0.02])
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "revision", "phase2c_rigor_summary.json"))
    args = ap.parse_args()

    data = load_all_conditions(seeds=CANDIDATE_SEEDS)
    confirmatory = _run_family(data, CONFIRMATORY_SPECS, tost_margins=tuple(args.tost_margins), seed=42)
    exploratory = _run_family(data, EXPLORATORY_SPECS, tost_margins=tuple(args.tost_margins), seed=142)

    print_family_table("CONFIRMATORY FAMILY (Holm-corrected within family)", confirmatory)
    print_family_table("EXPLORATORY / ABLATION FAMILY (Holm-corrected within family)", exploratory)

    pred_dirs = [
        os.path.join(_ROOT, "results", "phase2", "sepsis", "preds"),
        os.path.join(_ROOT, "results", "phase2b", "sepsis", "preds"),
        os.path.join(_ROOT, "phase2b_results", "results", "phase2b", "sepsis", "preds"),
        os.path.join(_ROOT, "results", "phase2c", "sepsis", "preds"),
    ]
    pred_records = discover_siteb_preds(pred_dirs)
    contrast_pairs = [(s[0], s[1], s[2]) for s in (CONFIRMATORY_SPECS + EXPLORATORY_SPECS) if s[3] == "site_b_auroc"]
    patient_boot = siteb_patient_bootstrap(pred_records, B=args.B, contrast_pairs=contrast_pairs)
    print(f"\n=== SITE B PATIENT-LEVEL PAIRED CLUSTER BOOTSTRAP ===")
    if patient_boot["available"]:
        print(f"n_stays={patient_boot['n_stays']}, n_patients={patient_boot['n_patients']}, positives={patient_boot['n_positives']}")
        for cname, info in patient_boot["by_condition"].items():
            print(f"  {cname:<28} mean={info['mean']:.4f}  95% CI=[{info['boot_ci_95'][0]:.4f}, {info['boot_ci_95'][1]:.4f}]  (seeds={info['seeds']})")
    else:
        print(f"  Skipped: {patient_boot['reason']} (re-run finetune_sepsis.py on GPU to dump *_site_b.npz files)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "tost_margins": args.tost_margins,
            "confirmatory_family": confirmatory,
            "exploratory_family": exploratory,
            "site_b_patient_bootstrap": patient_boot,
        }, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
