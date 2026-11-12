"""
Master Table Regeneration Script.
Loads all empirical JSON/CSV files and outputs exact, verified markdown tables
for PRELIMINARY_REVIEWER_REPORT.md and COMPREHENSIVE_AUDIT_REPORT.md.
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
SEEDS_5 = [42, 43, 44, 45, 46]
SEEDS_10 = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def stats_str(vals, fmt="6.4f"):
    vals = [float(v) for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "—"
    mean = np.mean(vals)
    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return f"{mean:{fmt}} ± {sd:{fmt}}"


def main():
    print("=== Generating Master Verified Tables ===")

    # 1. Load phase2c_full_summary.json
    p2c_sum_path = os.path.join(_ROOT, "results", "revision", "phase2c_full_summary.json")
    if not os.path.exists(p2c_sum_path):
        import phase2c_summary
        phase2c_summary.main()
    p2c_sum = json.load(open(p2c_sum_path))
    s_rows = p2c_sum["summary_rows"]

    # Table 2.2: Sepsis Multi-Metric Evaluation across all 13 conditions
    print("\n### Table 2.2 Sepsis Performance & Calibration Table:")
    print("| Condition | Site A Val AUROC | Site B OOD AUROC | Site B OOD AUPRC | Site B ECE | Site B Brier |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    # Load detailed raw sepsis data to get site A and calibration stats
    raw = p2c_sum["raw_data"]
    for cname, sdict in raw.items():
        sa = [sdict[str(s)]["site_a_auroc"] for s in SEEDS_5]
        sb = [sdict[str(s)]["site_b_auroc"] for s in SEEDS_5]
        pr = [sdict[str(s)]["site_b_auprc"] for s in SEEDS_5]
        ece = [sdict[str(s)]["site_b_ece"] for s in SEEDS_5]
        br = [sdict[str(s)]["site_b_brier"] for s in SEEDS_5]
        print(f"| **{cname}** | {stats_str(sa)} | {stats_str(sb)} | {stats_str(pr)} | {stats_str(ece)} | {stats_str(br)} |")

    # Table 3: 2x2 Factorial Tests
    print("\n### 2x2 Factorial & Budget Hypothesis Tests:")
    for tname, res in p2c_sum["hypothesis_tests"].items():
        print(f"- **{tname}:** Δ = {res['mean_diff']:+.4f} (95% CI: [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}], t = {res['t_stat']:+.2f}, p = {res['p_value']:.4f})")

    # Table 5: LOS T=24
    print("\n### Table 3.1 LOS T=24 Continuous Prediction Table:")
    print("| Condition | Site A Val R² | Site A Val MAE (h) | Site B OOD R² | Site B OOD MAE (h) |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for cname, sdict in raw.items():
        r2_a = [sdict[str(s)]["los_site_a_r2"] for s in SEEDS_5]
        mae_a = [sdict[str(s)]["los_site_a_mae"] for s in SEEDS_5]
        r2_b = [sdict[str(s)]["los_site_b_r2"] for s in SEEDS_5]
        mae_b = [sdict[str(s)]["los_site_b_mae"] for s in SEEDS_5]
        if any(v is not None for v in r2_b):
            print(f"| **{cname}** | {stats_str(r2_a)} | {stats_str(mae_a, '5.2f')} | {stats_str(r2_b)} | {stats_str(mae_b, '5.2f')} |")


if __name__ == "__main__":
    main()
