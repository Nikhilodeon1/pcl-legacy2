"""
Phase 2b Comprehensive Analysis Script:
Computes matched-volume control analysis, factorial leakage tests,
and loss-balancer mechanics ablation across 5 seeds (42-46).
"""
import glob
import json
import os
import re
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REV_DIR = os.path.dirname(HERE)
RESULTS_DIR = os.path.dirname(REV_DIR)
ROOT = os.path.dirname(RESULTS_DIR)
P2_DIR = os.path.join(RESULTS_DIR, "phase2")
P2B_DIR = os.path.join(RESULTS_DIR, "phase2b")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def run_analysis():
    seeds = [42, 43, 44, 45, 46]

    # Data structures: condition -> seed -> dict
    # Conditions:
    # 'erm_src': ERM fp32 source-only (10k)
    # 'erm_mv': ERM fp32 matched-volume (25k Site A)
    # 'erm_leak': ERM fp32 target-leak (10k Site A + 14k Site B)
    # 'pcl_bal': PCL lambda=0.5 fp32 with balancer (10k)
    # 'pcl_nobal': PCL lambda=0.5 fp32 fixed without balancer (10k)
    # 'pcl_leak': PCL lambda=0.5 fp32 target-leak with balancer (10k Site A + 14k Site B)

    data = {
        'erm_src': {},
        'erm_mv': {},
        'erm_leak': {},
        'pcl_bal': {},
        'pcl_nobal': {},
        'pcl_leak': {}
    }

    for s in seeds:
        # 1. ERM Source-Only
        pre_f = os.path.join(P2_DIR, "pretrain", f"lam0_p{s}_fp32.json")
        sep_f = os.path.join(P2_DIR, "sepsis", f"lam0_p{s}_fp32_s{s}.json")
        los_f = os.path.join(P2_DIR, "los_T24", f"lam0_p{s}_fp32_s{s}.json")
        if os.path.exists(pre_f) and os.path.exists(sep_f) and os.path.exists(los_f):
            pre, sep, los = load_json(pre_f), load_json(sep_f), load_json(los_f)
            data['erm_src'][s] = {
                'l2_h47': pre["repr_diag_site_a"]["47"]["mean_l2"],
                'spread_h47': pre["repr_diag_site_a"]["47"]["spread"],
                'const_frac_h47': pre["repr_diag_site_a"]["47"]["const_frac"],
                'sep_auroc_A': sep["in_domain"]["auroc"],
                'sep_auroc_B': sep["ood"]["physionet_b"]["auroc"],
                'los_r2_A': los["in_domain"]["r2"],
                'los_r2_B': los["ood"]["physionet_b"]["r2"]
            }

        # 2. ERM Matched-Volume
        pre_f = os.path.join(P2B_DIR, "pretrain", f"erm_matched_vol_p{s}.json")
        sep_f = os.path.join(P2B_DIR, "sepsis", f"erm_matched_vol_p{s}_s{s}.json")
        los_f = os.path.join(P2B_DIR, "los_T24", f"erm_matched_vol_p{s}_s{s}.json")
        if os.path.exists(pre_f) and os.path.exists(sep_f) and os.path.exists(los_f):
            pre, sep, los = load_json(pre_f), load_json(sep_f), load_json(los_f)
            data['erm_mv'][s] = {
                'l2_h47': pre["repr_diag_site_a"]["47"]["mean_l2"],
                'spread_h47': pre["repr_diag_site_a"]["47"]["spread"],
                'const_frac_h47': pre["repr_diag_site_a"]["47"]["const_frac"],
                'sep_auroc_A': sep["in_domain"]["auroc"],
                'sep_auroc_B': sep["ood"]["physionet_b"]["auroc"],
                'los_r2_A': los["in_domain"]["r2"],
                'los_r2_B': los["ood"]["physionet_b"]["r2"]
            }

        # 3. ERM Target-Leak
        pre_f = os.path.join(P2_DIR, "pretrain", f"lam0_p{s}_fp32_leakpb.json")
        sep_f = os.path.join(P2_DIR, "sepsis", f"lam0_p{s}_fp32_leakpb_s{s}.json")
        los_f = os.path.join(P2_DIR, "los_T24", f"lam0_p{s}_fp32_leakpb_s{s}.json")
        if os.path.exists(pre_f) and os.path.exists(sep_f) and os.path.exists(los_f):
            pre, sep, los = load_json(pre_f), load_json(sep_f), load_json(los_f)
            data['erm_leak'][s] = {
                'l2_h47': pre["repr_diag_site_a"]["47"]["mean_l2"],
                'spread_h47': pre["repr_diag_site_a"]["47"]["spread"],
                'const_frac_h47': pre["repr_diag_site_a"]["47"]["const_frac"],
                'sep_auroc_A': sep["in_domain"]["auroc"],
                'sep_auroc_B': sep["ood"]["physionet_b"]["auroc"],
                'los_r2_A': los["in_domain"]["r2"],
                'los_r2_B': los["ood"]["physionet_b"]["r2"]
            }

        # 4. PCL with Balancer (lambda=0.5, fp32)
        pre_f = os.path.join(P2_DIR, "pretrain", f"lam0.5_p{s}_fp32.json")
        sep_f = os.path.join(P2_DIR, "sepsis", f"lam0.5_p{s}_fp32_s{s}.json")
        los_f = os.path.join(P2_DIR, "los_T24", f"lam0.5_p{s}_fp32_s{s}.json")
        if os.path.exists(pre_f) and os.path.exists(sep_f) and os.path.exists(los_f):
            pre, sep, los = load_json(pre_f), load_json(sep_f), load_json(los_f)
            data['pcl_bal'][s] = {
                'l2_h47': pre["repr_diag_site_a"]["47"]["mean_l2"],
                'spread_h47': pre["repr_diag_site_a"]["47"]["spread"],
                'const_frac_h47': pre["repr_diag_site_a"]["47"]["const_frac"],
                'sep_auroc_A': sep["in_domain"]["auroc"],
                'sep_auroc_B': sep["ood"]["physionet_b"]["auroc"],
                'los_r2_A': los["in_domain"]["r2"],
                'los_r2_B': los["ood"]["physionet_b"]["r2"]
            }

        # 5. PCL Fixed without Balancer (lambda=0.5, fp32)
        pre_f = os.path.join(P2B_DIR, "pretrain", f"pcl_fixed_nobalance_p{s}.json")
        sep_f = os.path.join(P2B_DIR, "sepsis", f"pcl_fixed_nobalance_p{s}_s{s}.json")
        los_f = os.path.join(P2B_DIR, "los_T24", f"pcl_fixed_nobalance_p{s}_s{s}.json")
        if os.path.exists(pre_f) and os.path.exists(sep_f) and os.path.exists(los_f):
            pre, sep, los = load_json(pre_f), load_json(sep_f), load_json(los_f)
            data['pcl_nobal'][s] = {
                'l2_h47': pre["repr_diag_site_a"]["47"]["mean_l2"],
                'spread_h47': pre["repr_diag_site_a"]["47"]["spread"],
                'const_frac_h47': pre["repr_diag_site_a"]["47"]["const_frac"],
                'sep_auroc_A': sep["in_domain"]["auroc"],
                'sep_auroc_B': sep["ood"]["physionet_b"]["auroc"],
                'los_r2_A': los["in_domain"]["r2"],
                'los_r2_B': los["ood"]["physionet_b"]["r2"]
            }

        # 6. PCL Target-Leak with Balancer (lambda=0.5, fp32)
        pre_f = os.path.join(P2_DIR, "pretrain", f"lam0.5_p{s}_fp32_leakpb.json")
        sep_f = os.path.join(P2_DIR, "sepsis", f"lam0.5_p{s}_fp32_leakpb_s{s}.json")
        los_f = os.path.join(P2_DIR, "los_T24", f"lam0.5_p{s}_fp32_leakpb_s{s}.json")
        if os.path.exists(pre_f) and os.path.exists(sep_f) and os.path.exists(los_f):
            pre, sep, los = load_json(pre_f), load_json(sep_f), load_json(los_f)
            data['pcl_leak'][s] = {
                'l2_h47': pre["repr_diag_site_a"]["47"]["mean_l2"],
                'spread_h47': pre["repr_diag_site_a"]["47"]["spread"],
                'const_frac_h47': pre["repr_diag_site_a"]["47"]["const_frac"],
                'sep_auroc_A': sep["in_domain"]["auroc"],
                'sep_auroc_B': sep["ood"]["physionet_b"]["auroc"],
                'los_r2_A': los["in_domain"]["r2"],
                'los_r2_B': los["ood"]["physionet_b"]["r2"]
            }

    print("\n" + "="*80)
    print("PHASE 2 & 2B COMPLETE EMPIRICAL AUDIT RESULTS (5 SEEDS: 42-46)")
    print("="*80)

    # Condition Summaries
    print("\n--- 1. Condition Means & Standard Deviations ---")
    headers = ["Condition", "N", "L2 Norm (h47)", "Spread (h47)", "Const Frac", "Sepsis Site B AUROC", "LOS Site B R2"]
    print(f"{headers[0]:<15} {headers[1]:<3} {headers[2]:<16} {headers[3]:<14} {headers[4]:<12} {headers[5]:<22} {headers[6]:<16}")
    print("-" * 105)

    cond_order = [
        ('erm_src', 'ERM Src (10k)'),
        ('erm_mv', 'ERM Mv (25k)'),
        ('erm_leak', 'ERM Leak (25k)'),
        ('pcl_bal', 'PCL Bal (10k)'),
        ('pcl_nobal', 'PCL NoBal(10k)'),
        ('pcl_leak', 'PCL Leak (25k)'),
    ]

    summary_table = {}
    for c_key, c_name in cond_order:
        d = data[c_key]
        n = len(d)
        if n == 0:
            continue
        l2 = [d[s]['l2_h47'] for s in d]
        sp = [d[s]['spread_h47'] for s in d]
        cf = [d[s]['const_frac_h47'] for s in d]
        sb = [d[s]['sep_auroc_B'] for s in d]
        lb = [d[s]['los_r2_B'] for s in d]

        summary_table[c_key] = {
            'l2_mean': np.mean(l2), 'l2_sd': np.std(l2, ddof=1),
            'sp_mean': np.mean(sp), 'sp_sd': np.std(sp, ddof=1),
            'cf_mean': np.mean(cf), 'cf_sd': np.std(cf, ddof=1),
            'sb_mean': np.mean(sb), 'sb_sd': np.std(sb, ddof=1),
            'lb_mean': np.mean(lb), 'lb_sd': np.std(lb, ddof=1),
        }

        print(f"{c_name:<15} {n:<3} {np.mean(l2):6.2f} ± {np.std(l2, ddof=1):5.2f}   "
              f"{np.mean(sp):.4f} ± {np.std(sp, ddof=1):.4f} "
              f"{np.mean(cf):.4f} ± {np.std(cf, ddof=1):.4f} "
              f"{np.mean(sb):.4f} ± {np.std(sb, ddof=1):.4f}        "
              f"{np.mean(lb):.4f} ± {np.std(lb, ddof=1):.4f}")

    def paired_diff(c1, c2, metric):
        d1, d2 = data[c1], data[c2]
        common = sorted(set(d1) & set(d2))
        vals = [d1[s][metric] - d2[s][metric] for s in common]
        n = len(vals)
        mean_d = np.mean(vals)
        sd_d = np.std(vals, ddof=1) if n > 1 else 0.0
        se = sd_d / np.sqrt(n) if n > 0 else 0.0
        t_stat = mean_d / se if se > 1e-12 else np.nan
        p_val = 2 * stats.t.sf(abs(t_stat), df=n-1) if n > 1 and not np.isnan(t_stat) else np.nan
        ci95 = (mean_d - stats.t.ppf(0.975, df=n-1)*se, mean_d + stats.t.ppf(0.975, df=n-1)*se) if n > 1 else (np.nan, np.nan)
        return {'diff': mean_d, 'sd': sd_d, 'se': se, 't': t_stat, 'p': p_val, 'ci95': ci95, 'n': n}

    print("\n--- 2. Targeted Hypothesis Tests (Paired across identical seeds) ---")

    # A. Matched-Volume Control vs Leakage
    print("\n[A] Pretraining Leakage Decomposition for ERM:")
    # 1. Total Volume Effect (25k Site A vs 10k Site A)
    vol_res = paired_diff('erm_mv', 'erm_src', 'sep_auroc_B')
    print(f"  1. Pure Volume Effect (ERM 25k Src - ERM 10k Src):")
    print(f"     Δ AUROC = {vol_res['diff']:+.4f} (95% CI: [{vol_res['ci95'][0]:+.4f}, {vol_res['ci95'][1]:+.4f}], t={vol_res['t']:+.2f}, p={vol_res['p']:.4f})")

    # 2. Target Domain Specificity (25k Leak vs 25k Volume-Matched Src)
    tgt_res = paired_diff('erm_leak', 'erm_mv', 'sep_auroc_B')
    print(f"  2. Target Domain Specificity (ERM 25k Leak - ERM 25k Matched-Src):")
    print(f"     Δ AUROC = {tgt_res['diff']:+.4f} (95% CI: [{tgt_res['ci95'][0]:+.4f}, {tgt_res['ci95'][1]:+.4f}], t={tgt_res['t']:+.2f}, p={tgt_res['p']:.4f})")

    # 3. Overall Leakage Effect (ERM 25k Leak - ERM 10k Src)
    tot_res = paired_diff('erm_leak', 'erm_src', 'sep_auroc_B')
    print(f"  3. Overall Leakage Benefit (ERM 25k Leak - ERM 10k Src):")
    print(f"     Δ AUROC = {tot_res['diff']:+.4f} (95% CI: [{tot_res['ci95'][0]:+.4f}, {tot_res['ci95'][1]:+.4f}], t={tot_res['t']:+.2f}, p={tot_res['p']:.4f})")

    # B. PCL Target Leakage Benefit
    pcl_leak_res = paired_diff('pcl_leak', 'pcl_bal', 'sep_auroc_B')
    print(f"\n[B] Pretraining Leakage for PCL (PCL 25k Leak - PCL 10k Src):")
    print(f"     Δ AUROC = {pcl_leak_res['diff']:+.4f} (95% CI: [{pcl_leak_res['ci95'][0]:+.4f}, {pcl_leak_res['ci95'][1]:+.4f}], t={pcl_leak_res['t']:+.2f}, p={pcl_leak_res['p']:.4f})")

    # C. Factorial Interaction: ERM Leak Benefit vs PCL Leak Benefit
    interaction_vals = [
        (data['erm_leak'][s]['sep_auroc_B'] - data['erm_src'][s]['sep_auroc_B']) -
        (data['pcl_leak'][s]['sep_auroc_B'] - data['pcl_bal'][s]['sep_auroc_B'])
        for s in seeds
    ]
    mean_int = np.mean(interaction_vals)
    sd_int = np.std(interaction_vals, ddof=1)
    se_int = sd_int / np.sqrt(len(seeds))
    t_int = mean_int / se_int
    p_int = 2 * stats.t.sf(abs(t_int), df=len(seeds)-1)
    ci_int = (mean_int - stats.t.ppf(0.975, df=len(seeds)-1)*se_int, mean_int + stats.t.ppf(0.975, df=len(seeds)-1)*se_int)
    print(f"\n[C] Factorial Interaction: (Δ_ERM - Δ_PCL):")
    print(f"     Δ_interaction = {mean_int:+.4f} (95% CI: [{ci_int[0]:+.4f}, {ci_int[1]:+.4f}], t={t_int:+.2f}, p={p_int:.4f})")

    # D. Loss Balancer Mechanics Ablation: PCL Fixed (No Balancer) vs PCL Balancer
    print("\n[D] Loss Balancer Mechanics Ablation (Fixed Weight vs EMA Balancer at lambda=0.5):")
    nobal_l2 = paired_diff('pcl_nobal', 'pcl_bal', 'l2_h47')
    nobal_sp = paired_diff('pcl_nobal', 'pcl_bal', 'spread_h47')
    nobal_cf = paired_diff('pcl_nobal', 'pcl_bal', 'const_frac_h47')
    nobal_sep = paired_diff('pcl_nobal', 'pcl_bal', 'sep_auroc_B')

    print(f"  1. Representation L2 Norm:  Δ = {nobal_l2['diff']:+.2f} (t={nobal_l2['t']:+.2f}, p={nobal_l2['p']:.4f})")
    print(f"  2. Latent Vector Spread:    Δ = {nobal_sp['diff']:+.4f} (t={nobal_sp['t']:+.2f}, p={nobal_sp['p']:.4f})")
    print(f"  3. Constant Coordinate Frac:Δ = {nobal_cf['diff']:+.4f} (t={nobal_cf['t']:+.2f}, p={nobal_cf['p']:.4f})")
    print(f"  4. Sepsis Site B Transfer:  Δ = {nobal_sep['diff']:+.4f} (t={nobal_sep['t']:+.2f}, p={nobal_sep['p']:.4f})")

    # Save complete JSON summary
    out_obj = {
        'summary_table': summary_table,
        'paired_contrasts': {
            'erm_volume_effect': vol_res,
            'erm_target_domain_specificity': tgt_res,
            'erm_total_leakage': tot_res,
            'pcl_total_leakage': pcl_leak_res,
            'factorial_interaction': {
                'diff': mean_int, 'sd': sd_int, 'se': se_int, 't': t_int, 'p': p_int, 'ci95': ci_int
            },
            'nobalancer_ablation': {
                'l2': nobal_l2,
                'spread': nobal_sp,
                'const_frac': nobal_cf,
                'sepsis_auroc_B': nobal_sep
            }
        }
    }
    with open(os.path.join(REV_DIR, "phase2b_summary.json"), "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"\nSaved analysis to {os.path.join(REV_DIR, 'phase2b_summary.json')}")

if __name__ == "__main__":
    run_analysis()
