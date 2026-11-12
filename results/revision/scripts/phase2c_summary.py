"""
Unified Statistical Synthesis for Phase 2c Reviewer Experiments.
Aggregates Phase 2, Phase 2b, and Phase 2c results across seeds 42-46.
"""
import glob
import json
import os
import sys
import numpy as np
import scipy.stats as stats
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
SEEDS = [42, 43, 44, 45, 46]


def paired_stats(diffs):
    diffs = np.array(diffs, dtype=float)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    if n < 2:
        return {"n": n, "mean_diff": mean_d, "sd": 0.0, "se": 0.0, "ci_low": mean_d, "ci_high": mean_d, "t_stat": 0.0, "p_value": 1.0}
    sd = float(np.std(diffs, ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    t_stat = mean_d / se if se > 1e-12 else 0.0
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))
    return {
        "n": n,
        "mean_diff": mean_d,
        "sd": sd,
        "se": se,
        "ci_low": mean_d - tcrit * se,
        "ci_high": mean_d + tcrit * se,
        "t_stat": t_stat,
        "p_value": p_val,
    }


def find_json(patterns):
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return matches[0]
    return None


def load_all_conditions():
    conditions = {
        "ERM Clean (10k)": {
            "sepsis": lambda s: [f"results/phase2/sepsis/lam0_p{s}_fp32_s{s}.json"],
            "los": lambda s: [f"results/phase2/los_fixed_T24/lam0_p{s}_fp32_s{s}.json"],
            "pretrain": lambda s: [f"results/phase2/pretrain/lam0_p{s}_fp32.json"],
        },
        "ERM Matched-Vol (25k)": {
            "sepsis": lambda s: [
                f"results/phase2b/sepsis/erm_matched_vol_p{s}_s{s}.json",
                f"phase2b_results/results/phase2b/sepsis/erm_matched_vol_p{s}_s{s}.json",
            ],
            "los": lambda s: [
                f"results/phase2b/los_T24/erm_matched_vol_p{s}_s{s}.json",
                f"phase2b_results/results/phase2b/los_T24/erm_matched_vol_p{s}_s{s}.json",
            ],
            "pretrain": lambda s: [
                f"results/phase2b/pretrain/erm_matched_vol_p{s}.json",
                f"phase2b_results/results/phase2b/pretrain/erm_matched_vol_p{s}.json",
            ],
        },
        "ERM Target-Exposed (25k)": {
            "sepsis": lambda s: [f"results/phase2/sepsis/lam0_p{s}_fp32_leakpb_s{s}.json"],
            "los": lambda s: [f"results/phase2/los_fixed_T24/lam0_p{s}_fp32_leakpb_s{s}.json"],
            "pretrain": lambda s: [f"results/phase2/pretrain/lam0_p{s}_fp32_leakpb.json"],
        },
        "PCL Clean (10k)": {
            "sepsis": lambda s: [f"results/phase2/sepsis/lam0.5_p{s}_fp32_s{s}.json"],
            "los": lambda s: [f"results/phase2/los_fixed_T24/lam0.5_p{s}_fp32_s{s}.json"],
            "pretrain": lambda s: [f"results/phase2/pretrain/lam0.5_p{s}_fp32.json"],
        },
        "PCL No-Balancer (10k)": {
            "sepsis": lambda s: [
                f"results/phase2b/sepsis/pcl_fixed_nobalance_p{s}_s{s}.json",
                f"phase2b_results/results/phase2b/sepsis/pcl_fixed_nobalance_p{s}_s{s}.json",
            ],
            "los": lambda s: [
                f"results/phase2b/los_T24/pcl_fixed_nobalance_p{s}_s{s}.json",
                f"phase2b_results/results/phase2b/los_T24/pcl_fixed_nobalance_p{s}_s{s}.json",
            ],
            "pretrain": lambda s: [
                f"results/phase2b/pretrain/pcl_fixed_nobalance_p{s}.json",
                f"phase2b_results/results/phase2b/pretrain/pcl_fixed_nobalance_p{s}.json",
            ],
        },
        "PCL Target-Exposed (25k)": {
            "sepsis": lambda s: [f"results/phase2/sepsis/lam0.5_p{s}_fp32_leakpb_s{s}.json"],
            "los": lambda s: [f"results/phase2/los_fixed_T24/lam0.5_p{s}_fp32_leakpb_s{s}.json"],
            "pretrain": lambda s: [f"results/phase2/pretrain/lam0.5_p{s}_fp32_leakpb.json"],
        },
        "PCL Matched-Vol (25k)": {
            "sepsis": lambda s: [f"results/phase2c/sepsis/pcl_matched_vol_p{s}_s{s}.json"],
            "los": lambda s: [f"results/phase2c/los_T24/pcl_matched_vol_p{s}_s{s}.json"],
            "pretrain": lambda s: [f"results/phase2c/pretrain/pcl_matched_vol_p{s}.json"],
        },
        "PCL Matched-Vol (80k)": {
            "sepsis": lambda s: [f"results/phase2c/sepsis/pcl_matched_vol_n80000_p{s}_s{s}.json"],
            "los": lambda s: [f"results/phase2c/los_T24/pcl_matched_vol_n80000_p{s}_s{s}.json"],
            "pretrain": lambda s: [f"results/phase2c/pretrain/pcl_matched_vol_n80000_p{s}.json"],
        },
        "PCL MAP-Only (10k)": {
            "sepsis": lambda s: [f"results/phase2c/sepsis/pcl_maponly_p{s}_s{s}.json"],
            "los": lambda s: [f"results/phase2c/los_T24/pcl_maponly_p{s}_s{s}.json"],
            "pretrain": lambda s: [f"results/phase2c/pretrain/pcl_maponly_p{s}.json"],
        },
        "PCL No-HH (10k)": {
            "sepsis": lambda s: [f"results/phase2c/sepsis/pcl_nohh_p{s}_s{s}.json"],
            "los": lambda s: [f"results/phase2c/los_T24/pcl_nohh_p{s}_s{s}.json"],
            "pretrain": lambda s: [f"results/phase2c/pretrain/pcl_nohh_p{s}.json"],
        },
        "PCL Placebo (10k)": {
            "sepsis": lambda s: [f"results/phase2c/sepsis/pcl_placebo_p{s}_s{s}.json"],
            "los": lambda s: [f"results/phase2c/los_T24/pcl_placebo_p{s}_s{s}.json"],
            "pretrain": lambda s: [f"results/phase2c/pretrain/pcl_placebo_p{s}.json"],
        },
        "PCL + VICReg (10k)": {
            "sepsis": lambda s: [f"results/phase2c/sepsis/pcl_varreg1_p{s}_s{s}.json"],
            "los": lambda s: [f"results/phase2c/los_T24/pcl_varreg1_p{s}_s{s}.json"],
            "pretrain": lambda s: [f"results/phase2c/pretrain/pcl_varreg1_p{s}.json"],
        },
        "Supervised Scratch": {
            "sepsis": lambda s: [f"results/phase2c/sepsis/scratch_p{s}_s{s}.json"],
            "los": lambda s: [f"results/phase2c/los_T24/scratch_p{s}_s{s}.json"],
            "pretrain": lambda s: [],
        },
    }

    data = {}
    for cname, loaders in conditions.items():
        data[cname] = {}
        for s in SEEDS:
            # Sepsis
            sep_f = find_json(loaders["sepsis"](s))
            sep_d = json.load(open(sep_f)) if sep_f and os.path.exists(sep_f) else {}
            # LOS
            los_f = find_json(loaders["los"](s))
            los_d = json.load(open(los_f)) if los_f and os.path.exists(los_f) else {}
            # Pretrain
            pt_f = find_json(loaders["pretrain"](s)) if loaders["pretrain"](s) else None
            pt_d = json.load(open(pt_f)) if pt_f and os.path.exists(pt_f) else {}
            r47 = pt_d.get("repr_diag_site_a", {}).get("47", {})

            data[cname][s] = {
                "site_a_auroc": sep_d.get("in_domain", {}).get("auroc"),
                "site_b_auroc": sep_d.get("ood", {}).get("physionet_b", {}).get("auroc"),
                "site_b_auprc": sep_d.get("ood", {}).get("physionet_b", {}).get("auprc"),
                "site_b_ece": sep_d.get("ood", {}).get("physionet_b", {}).get("ece"),
                "site_b_brier": sep_d.get("ood", {}).get("physionet_b", {}).get("brier"),
                "los_site_a_r2": los_d.get("in_domain", {}).get("r2"),
                "los_site_a_mae": los_d.get("in_domain", {}).get("mae_hours"),
                "los_site_b_r2": (los_d.get("ood", {}).get("physionet_b", {}).get("r2")
                                 if "physionet_b" in los_d.get("ood", {})
                                 else los_d.get("ood", {}).get("physionet", {}).get("r2")),
                "los_site_b_mae": (los_d.get("ood", {}).get("physionet_b", {}).get("mae_hours")
                                  if "physionet_b" in los_d.get("ood", {})
                                  else los_d.get("ood", {}).get("physionet", {}).get("mae_hours")),
                "l2_norm": r47.get("mean_l2"),
                "spread": r47.get("spread"),
                "const_frac": r47.get("const_frac"),
            }
    return data


def format_table(data):
    print("=" * 115)
    print("EMPIRICAL SYNTHESIS ACROSS ALL EXPERIMENTAL CONDITIONS (5 SEEDS: 42-46)")
    print("=" * 115)
    header = f"{'Condition':<26} {'L2 Norm (h47)':<15} {'Spread (h47)':<14} {'Const Frac':<13} {'Site B AUROC':<18} {'Site B AUPRC':<16} {'LOS Site B R2':<12}"
    print(header)
    print("-" * 115)

    summary_rows = {}
    for cname, sdict in data.items():
        metrics = {k: [sdict[s][k] for s in SEEDS if sdict[s][k] is not None] for k in [
            "l2_norm", "spread", "const_frac", "site_b_auroc", "site_b_auprc", "los_site_b_r2"
        ]}
        row_str = f"{cname:<26} "
        row_summary = {}
        for m in ["l2_norm", "spread", "const_frac", "site_b_auroc", "site_b_auprc", "los_site_b_r2"]:
            vals = metrics[m]
            if len(vals) == len(SEEDS):
                mean = np.mean(vals)
                sd = np.std(vals, ddof=1)
                row_summary[m] = {"mean": float(mean), "sd": float(sd), "vals": [float(x) for x in vals]}
                if m in ["l2_norm"]:
                    row_str += f"{mean:6.2f} ± {sd:5.2f}    "
                elif m in ["spread", "const_frac"]:
                    row_str += f"{mean:6.4f} ± {sd:6.4f}  "
                elif m in ["site_b_auroc", "site_b_auprc"]:
                    row_str += f"{mean:6.4f} ± {sd:6.4f}   "
                elif m in ["los_site_b_r2"]:
                    row_str += f"{mean:7.4f} ± {sd:6.4f}"
            else:
                row_str += f"{'—':<15}"
                row_summary[m] = None
        print(row_str)
        summary_rows[cname] = row_summary
    print("-" * 115)
    return summary_rows


def run_hypothesis_tests(data):
    print("\n" + "=" * 80)
    print("TARGETED HYPOTHESIS TESTS (PAIRED SEEDS 42-46)")
    print("=" * 80)

    tests = [
        ("1. 2x2 Factorial: Volume Effect on ERM (25k Matched-Vol - 10k Clean)",
         "ERM Matched-Vol (25k)", "ERM Clean (10k)", "site_b_auroc"),
        ("2. 2x2 Factorial: Volume Effect on PCL (25k Matched-Vol - 10k Clean)",
         "PCL Matched-Vol (25k)", "PCL Clean (10k)", "site_b_auroc"),
        ("3. 2x2 Factorial: PCL vs ERM at Matched 25k Volume (PCL 25k - ERM 25k)",
         "PCL Matched-Vol (25k)", "ERM Matched-Vol (25k)", "site_b_auroc"),
        ("4. Budget Curve: PCL High-Budget (80k Matched-Vol - 10k Clean)",
         "PCL Matched-Vol (80k)", "PCL Clean (10k)", "site_b_auroc"),
        ("5. Budget Curve: PCL High-Budget vs ERM Mid-Budget (PCL 80k - ERM 25k)",
         "PCL Matched-Vol (80k)", "ERM Matched-Vol (25k)", "site_b_auroc"),
        ("6. Pretraining Utility: ERM Pretraining vs Scratch (ERM 10k - Scratch)",
         "ERM Clean (10k)", "Supervised Scratch", "site_b_auroc"),
        ("7. Pretraining Utility: PCL Pretraining vs Scratch (PCL 10k - Scratch)",
         "PCL Clean (10k)", "Supervised Scratch", "site_b_auroc"),
        ("8. Pretraining Utility: ERM 25k vs Scratch (ERM 25k - Scratch)",
         "ERM Matched-Vol (25k)", "Supervised Scratch", "site_b_auroc"),
        ("9. Anti-Collapse: VICReg Regularizer Effect on Transfer (PCL+VICReg - PCL Clean)",
         "PCL + VICReg (10k)", "PCL Clean (10k)", "site_b_auroc"),
        ("10. Constraint Ablation: MAP-Only vs Full PCL (PCL MAP-Only - PCL Clean)",
         "PCL MAP-Only (10k)", "PCL Clean (10k)", "site_b_auroc"),
        ("11. Constraint Ablation: No-HH vs Full PCL (PCL No-HH - PCL Clean)",
         "PCL No-HH (10k)", "PCL Clean (10k)", "site_b_auroc"),
        ("12. Placebo Control: Placebo Constraint vs Full PCL (PCL Placebo - PCL Clean)",
         "PCL Placebo (10k)", "PCL Clean (10k)", "site_b_auroc"),
        ("13. Placebo Control: Placebo Constraint vs ERM Clean (PCL Placebo - ERM Clean)",
         "PCL Placebo (10k)", "ERM Clean (10k)", "site_b_auroc"),
    ]

    results = {}
    for title, c_a, c_b, m in tests:
        vals_a = [data[c_a][s][m] for s in SEEDS]
        vals_b = [data[c_b][s][m] for s in SEEDS]
        diffs = [a - b for a, b in zip(vals_a, vals_b)]
        res = paired_stats(diffs)
        results[title] = res
        print(f"\n[{title}]")
        print(f"  Δ {m} = {res['mean_diff']:+.4f} (95% CI: [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}], t = {res['t_stat']:+.2f}, p = {res['p_value']:.4f})")

    # Interaction: (Δ_ERM - Δ_PCL)
    diff_erm = [data["ERM Matched-Vol (25k)"][s]["site_b_auroc"] - data["ERM Clean (10k)"][s]["site_b_auroc"] for s in SEEDS]
    diff_pcl = [data["PCL Matched-Vol (25k)"][s]["site_b_auroc"] - data["PCL Clean (10k)"][s]["site_b_auroc"] for s in SEEDS]
    interaction_diffs = [e - p for e, p in zip(diff_erm, diff_pcl)]
    int_res = paired_stats(interaction_diffs)
    results["2x2 Factorial Interaction: (Δ_vol_ERM - Δ_vol_PCL)"] = int_res
    print(f"\n[2x2 Factorial Interaction: (Δ_vol_ERM - Δ_vol_PCL)]")
    print(f"  Δ_interaction = {int_res['mean_diff']:+.4f} (95% CI: [{int_res['ci_low']:+.4f}, {int_res['ci_high']:+.4f}], t = {int_res['t_stat']:+.2f}, p = {int_res['p_value']:.4f})")

    return results


def main():
    data = load_all_conditions()
    summary = format_table(data)
    test_results = run_hypothesis_tests(data)

    out_file = os.path.join(_ROOT, "results", "revision", "phase2c_full_summary.json")
    with open(out_file, "w") as f:
        json.dump({
            "summary_rows": summary,
            "hypothesis_tests": test_results,
            "raw_data": data,
        }, f, indent=2)
    print(f"\nSaved complete synthesis to {out_file}")


if __name__ == "__main__":
    main()
