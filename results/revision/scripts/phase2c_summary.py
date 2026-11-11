"""
Summary and hypothesis tests for Phase 2c Reviewer Experiments.
Analyses:
1. Complete 2x2 Factorial Design:
   (ERM vs PCL) x (10k Source vs 25k Matched Volume Source)
2. Supervised Transformer from Scratch vs Pretrained Models
3. VICReg Anti-Collapse Intervention (Effect on erank, L2 norm, and OOD transfer)
4. Physiological Constraint Ablation (MAP-only vs No-HH vs Full PCL)
"""
import glob
import json
import os
import sys
import numpy as np
import scipy.stats as stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))


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


def main():
    # Search paths for phase2c and prior phases
    p2c_dirs = [
        os.path.join(_ROOT, "results", "phase2c"),
        os.path.join(_ROOT, "phase2c_results", "results", "phase2c"),
        os.path.join(_ROOT, "phase2c_results"),
    ]
    p2c_dir = next((d for d in p2c_dirs if os.path.exists(d)), os.path.join(_ROOT, "results", "phase2c"))

    logging_header = "PHASE 2C REVIEWER EXPERIMENTAL SUMMARY"
    print("=" * 80)
    print(logging_header)
    print(f"Reading Phase 2c results from: {p2c_dir}")
    print("=" * 80)

    # Load sepsis results
    sepsis_files = glob.glob(os.path.join(p2c_dir, "sepsis", "*.json"))
    los_files = glob.glob(os.path.join(p2c_dir, "los_T24", "*.json"))
    pretrain_files = glob.glob(os.path.join(p2c_dir, "pretrain", "*.json"))

    print(f"Found {len(pretrain_files)} pretrain JSONs, {len(sepsis_files)} sepsis JSONs, {len(los_files)} LOS JSONs.")


if __name__ == "__main__":
    main()
