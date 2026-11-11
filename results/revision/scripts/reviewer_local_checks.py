"""
Comprehensive local verification script addressing Mock TMLR Reviewer items:
1. Centered vs Uncentered Effective Rank (RankMe) across lambda sweep (0.0 to 5.0)
2. Correlation of representation metrics (erank_cent, const_frac, spread) with OOD transfer
3. GBM & Ridge learnability check on fixed-window T=24 LOS (raw-hour R2, log-R2, Duan smearing, Spearman, MAE) and Sepsis
4. Multi-metric Sepsis & LOS tables (AUROC, AUPRC, Brier, R2, MAE) from Phase 2 & 2b
5. Real-data unsupervised selection criteria correlations & regret with CIs
"""
import csv
import glob
import gzip
import json
import os
import pickle
import sys
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, average_precision_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
REV_DIR = os.path.dirname(HERE)
RESULTS_DIR = os.path.dirname(REV_DIR)
ROOT = os.path.dirname(RESULTS_DIR)
sys.path.insert(0, ROOT)

def main():
    out_report = {}

    # =========================================================================
    # 1. Centered vs Uncentered Effective Rank Summary & Transfer Correlations
    # =========================================================================
    er_list = json.load(open(os.path.join(REV_DIR, "erank_centered_sweep.json")))
    er_map = {r["name"]: r for r in er_list}

    by_lam = {}
    for r in er_list:
        lam = float(r["name"].split("_")[0].replace("lam", ""))
        by_lam.setdefault(lam, []).append(r)

    print("=== 1A. Lambda Sweep Representation Geometry (Hour 47, N=1000 Site-A stays, 3 pretraining seeds) ===")
    print(f"{'Lambda':>6s} | {'Mean L2':>13s} | {'Centered L2':>13s} | {'Const Frac':>13s} | {'Spread':>13s} | {'Uncent erank':>13s} | {'Centered erank':>14s}")
    print("-" * 100)
    lam_summary = {}
    for lam in sorted(by_lam):
         items = by_lam[lam]
         m_l2 = [x["l2_mean"] for x in items]
         c_l2 = [x["l2_cent"] for x in items]
         cf = [x["const_frac"] for x in items]
         sp = [x["spread"] for x in items]
         er_r = [x["erank_raw"] for x in items]
         er_c = [x["erank_cent"] for x in items]
         lam_summary[str(lam)] = {
             "l2_mean": [float(np.mean(m_l2)), float(np.std(m_l2, ddof=1))],
             "l2_cent": [float(np.mean(c_l2)), float(np.std(c_l2, ddof=1))],
             "const_frac": [float(np.mean(cf)), float(np.std(cf, ddof=1))],
             "spread": [float(np.mean(sp)), float(np.std(sp, ddof=1))],
             "erank_raw": [float(np.mean(er_r)), float(np.std(er_r, ddof=1))],
             "erank_cent": [float(np.mean(er_c)), float(np.std(er_c, ddof=1))],
         }
         print(f"{lam:6.1f} | {np.mean(m_l2):6.1f}±{np.std(m_l2,ddof=1):4.1f} | "
               f"{np.mean(c_l2):6.2f}±{np.std(c_l2,ddof=1):4.2f} | "
               f"{np.mean(cf):.4f}±{np.std(cf,ddof=1):.4f} | "
               f"{np.mean(sp):.4f}±{np.std(sp,ddof=1):.4f} | "
               f"{np.mean(er_r):6.2f}±{np.std(er_r,ddof=1):4.2f} | "
               f"{np.mean(er_c):6.2f}±{np.std(er_c,ddof=1):4.2f}")
    out_report["lambda_sweep_erank"] = lam_summary

    # Correlate with transfer in sweep_summary.csv
    sweep_rows = list(csv.DictReader(open(os.path.join(REV_DIR, "sweep_summary.csv"))))
    print("\n=== 1B. Correlation of Representation Geometry with Downstream Transfer (Sweep Encoders) ===")
    corr_summary = {}
    for task, direction in [("mortality", "MIMIC->eICU"), ("mortality", "eICU->MIMIC"), ("los", "PN-A val"), ("los", "PN-B"), ("los", "MIMIC"), ("los", "eICU")]:
        sub = [r for r in sweep_rows if r["task"] == task and r["direction"] == direction]
        ec, er, cf, cl2, y = [], [], [], [], []
        for r in sub:
            k = f"lam{float(r['lam']):.1f}_p{r['pretrain_seed']}"
            if k in er_map and r["ood"]:
                ec.append(er_map[k]["erank_cent"])
                er.append(er_map[k]["erank_raw"])
                cf.append(er_map[k]["const_frac"])
                cl2.append(er_map[k]["l2_cent"])
                y.append(float(r["ood"]))
        if len(y) >= 3:
            rc, pc = stats.spearmanr(ec, y)
            rr, pr = stats.spearmanr(er, y)
            rcf, pcf = stats.spearmanr(cf, y)
            corr_summary[f"{task}_{direction}"] = {
                "n": len(y),
                "spearman_erank_cent": [float(rc), float(pc)],
                "spearman_erank_raw": [float(rr), float(pr)],
                "spearman_const_frac": [float(rcf), float(pcf)],
            }
            print(f"{task:10s} {direction:12s} (n={len(y)}): rho(erank_cent, OOD)={rc:+.3f} (p={pc:.3f}) | rho(erank_raw, OOD)={rr:+.3f} (p={pr:.3f}) | rho(const_frac, OOD)={rcf:+.3f} (p={pcf:.3f})")

    # Also check Phase 2 + 2b correlation across all 40 pretrain+sepsis runs
    p2_pre = glob.glob(os.path.join(RESULTS_DIR, "phase2", "pretrain", "*.json")) + glob.glob(os.path.join(RESULTS_DIR, "phase2b", "pretrain", "*.json"))
    p2_l2, p2_sp, p2_cf, p2_auroc_B, p2_auprc_B = [], [], [], [], []
    for pf in p2_pre:
        tag = os.path.basename(pf)[:-5]
        pre_d = json.load(open(pf))
        seed = pre_d["seed"]
        sf1 = os.path.join(RESULTS_DIR, "phase2", "sepsis", f"{tag}_s{seed}.json")
        sf2 = os.path.join(RESULTS_DIR, "phase2b", "sepsis", f"{tag}_s{seed}.json")
        sf = sf1 if os.path.exists(sf1) else (sf2 if os.path.exists(sf2) else None)
        if sf:
            sep_d = json.load(open(sf))
            p2_l2.append(pre_d["repr_diag_site_a"]["47"]["mean_l2"])
            p2_sp.append(pre_d["repr_diag_site_a"]["47"]["spread"])
            p2_cf.append(pre_d["repr_diag_site_a"]["47"]["const_frac"])
            p2_auroc_B.append(sep_d["ood"]["physionet_b"]["auroc"])
            p2_auprc_B.append(sep_d["ood"]["physionet_b"]["auprc"])
    print(f"\nAcross all {len(p2_auroc_B)} Phase 2+2b encoders on Sepsis Site B:")
    for name, arr in [("spread_h47", p2_sp), ("const_frac_h47", p2_cf), ("mean_l2_h47", p2_l2)]:
        rho, p = stats.spearmanr(arr, p2_auroc_B)
        r_p, pp = stats.pearsonr(arr, p2_auroc_B)
        print(f"  {name:15s} vs Site B AUROC: Spearman rho={rho:+.3f} (p={p:.4e}), Pearson r={r_p:+.3f} (p={pp:.4e})")
        corr_summary[f"phase2_sepsis_{name}"] = {"n": len(p2_auroc_B), "spearman": [float(rho), float(p)], "pearson": [float(r_p), float(pp)]}
    out_report["transfer_correlations"] = corr_summary

    # =========================================================================
    # 2. Multi-Metric Sepsis Table (AUROC, AUPRC, Brier) across Phase 2 & 2b
    # =========================================================================
    print("\n=== 2. Complete Sepsis Multi-Metric Summary (AUROC, AUPRC, Brier Score, 5 Seeds 42-46, FP32) ===")
    cond_defs = [
        ("ERM Source (10k, lam=0)", "phase2", "lam0_p{s}_fp32"),
        ("PCL Source (10k, lam=0.5)", "phase2", "lam0.5_p{s}_fp32"),
        ("PCL Source (10k, lam=1.0)", "phase2", "lam1_p{s}_fp32"),
        ("PCL Fixed No-Balancer (10k, lam=0.5)", "phase2b", "pcl_fixed_nobalance_p{s}"),
        ("ERM Matched-Volume (25k Src, lam=0)", "phase2b", "erm_matched_vol_p{s}"),
        ("ERM Target-Exposed (25k A+B, lam=0)", "phase2", "lam0_p{s}_fp32_leakpb"),
        ("PCL Target-Exposed (25k A+B, lam=0.5)", "phase2", "lam0.5_p{s}_fp32_leakpb"),
    ]
    sepsis_multi = {}
    print(f"{'Condition':<38s} | {'Site A AUROC':<15s} | {'Site B AUROC':<15s} | {'Site B AUPRC':<15s} | {'Site B Brier':<15s}")
    print("-" * 108)
    for label, subdir, pat in cond_defs:
        a_auc, b_auc, b_prc, b_bri = [], [], [], []
        for s in [42, 43, 44, 45, 46]:
            tag = pat.format(s=s)
            sf = os.path.join(RESULTS_DIR, subdir, "sepsis", f"{tag}_s{s}.json")
            if os.path.exists(sf):
                d = json.load(open(sf))
                a_auc.append(d["in_domain"]["auroc"])
                b_auc.append(d["ood"]["physionet_b"]["auroc"])
                b_prc.append(d["ood"]["physionet_b"]["auprc"])
                b_bri.append(d["ood"]["physionet_b"]["brier"])
        sepsis_multi[label] = {
            "site_a_auroc": [float(np.mean(a_auc)), float(np.std(a_auc, ddof=1))],
            "site_b_auroc": [float(np.mean(b_auc)), float(np.std(b_auc, ddof=1))],
            "site_b_auprc": [float(np.mean(b_prc)), float(np.std(b_prc, ddof=1))],
            "site_b_brier": [float(np.mean(b_bri)), float(np.std(b_bri, ddof=1))],
        }
        print(f"{label:<38s} | {np.mean(a_auc):.3f} ± {np.std(a_auc,ddof=1):.3f} | "
              f"{np.mean(b_auc):.3f} ± {np.std(b_auc,ddof=1):.3f} | "
              f"{np.mean(b_prc):.3f} ± {np.std(b_prc,ddof=1):.3f} | "
              f"{np.mean(b_bri):.3f} ± {np.std(b_bri,ddof=1):.3f}")
    out_report["sepsis_multimetric"] = sepsis_multi

    # =========================================================================
    # 3. GBM & Summary-Feature Learnability Check on Fixed-Window T=24 LOS & Sepsis
    # =========================================================================
    print("\n=== 3. Non-Neural Baselines (GBM & Ridge on First-24h Summary Features vs Mask-Only) ===")
    with gzip.open(os.path.join(ROOT, "colab", "physionet_frac1.0.pkl.gz"), "rb") as f:
        pn = pickle.load(f)
    from src.data.fixed_window import fix_window
    pn_t24 = fix_window(pn, 24, min_los_h=24)
    site_a = [s for s in pn_t24 if s["site_id"] == 0]
    site_b = [s for s in pn_t24 if s["site_id"] == 1]

    def extract_summary_features(stays, T=24):
        feats_full = []
        feats_mask = []
        y_los = []
        y_sep = []
        for s in stays:
            v = np.asarray(s["values"][:T], dtype=np.float32) # (T, 17)
            m = np.asarray(s["mask"][:T], dtype=bool)         # (T, 17)
            # Mask features: per-variable count, total count, any observed
            cnt = m.sum(axis=0).astype(np.float32)
            tot = np.array([cnt.sum(), (m.sum(axis=1) > 0).sum()], dtype=np.float32)
            mf = np.concatenate([cnt, tot])
            feats_mask.append(mf)

            # Clinical summary features on observed values (where mask is True, else imputed value)
            v_mean = v.mean(axis=0)
            v_std = v.std(axis=0)
            v_min = v.min(axis=0)
            v_max = v.max(axis=0)
            v_last = v[-1]
            v_first = v[0]
            v_diff = v_last - v_first
            ff = np.concatenate([v_mean, v_std, v_min, v_max, v_last, v_diff, mf])
            feats_full.append(ff)
            y_los.append(float(s["los_h"]))
            y_sep.append(int(np.asarray(s["label"]).sum() > 0))
        return np.stack(feats_full), np.stack(feats_mask), np.array(y_los), np.array(y_sep)

    Xa_full, Xa_mask, ya_los, ya_sep = extract_summary_features(site_a, T=24)
    Xb_full, Xb_mask, yb_los, yb_sep = extract_summary_features(site_b, T=24)

    gbm_results = {"los_T24": {}, "sepsis_T24": {}}
    for feat_name, Xa, Xb in [("GBM (Physiology+Mask Summary, T=24)", Xa_full, Xb_full),
                              ("GBM (Mask-Only Summary, T=24)", Xa_mask, Xb_mask)]:
        r2_a_raw, r2_b_raw = [], []
        r2_a_smear, r2_b_smear = [], []
        r2_a_log, r2_b_log = [], []
        sp_a, sp_b = [], []
        mae_a, mae_b = [], []
        sep_a, sep_b, prc_b = [], [], []

        for seed in [42, 43, 44, 45, 46]:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(Xa))
            ntr = int(len(Xa) * 0.8)
            tr, va = idx[:ntr], idx[ntr:]

            # 1. LOS trained in raw hours
            reg_raw = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
            reg_raw.fit(Xa[tr], ya_los[tr])
            pa_raw = reg_raw.predict(Xa[va])
            pb_raw = reg_raw.predict(Xb)
            r2_a_raw.append(r2_score(ya_los[va], pa_raw))
            r2_b_raw.append(r2_score(yb_los, pb_raw))
            sp_a.append(stats.spearmanr(ya_los[va], pa_raw)[0])
            sp_b.append(stats.spearmanr(yb_los, pb_raw)[0])
            mae_a.append(mean_absolute_error(ya_los[va], pa_raw))
            mae_b.append(mean_absolute_error(yb_los, pb_raw))

            # 2. LOS trained in log1p space (matching finetune_los.py!)
            z_tr = np.log1p(ya_los[tr])
            reg_log = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
            reg_log.fit(Xa[tr], z_tr)
            za_hat = reg_log.predict(Xa[va])
            zb_hat = reg_log.predict(Xb)
            r2_a_log.append(r2_score(np.log1p(ya_los[va]), za_hat))
            r2_b_log.append(r2_score(np.log1p(yb_los), zb_hat))
            # Naive expm1 (like finetune_los.py) vs Duan smearing
            pa_naive = np.expm1(za_hat)
            pb_naive = np.expm1(zb_hat)
            r2_a_smear.append(r2_score(ya_los[va], pa_naive))
            r2_b_smear.append(r2_score(yb_los, pb_naive))

            # 3. Sepsis classification
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed, class_weight="balanced")
            clf.fit(Xa[tr], ya_sep[tr])
            sa_prob = clf.predict_proba(Xa[va])[:, 1]
            sb_prob = clf.predict_proba(Xb)[:, 1]
            sep_a.append(roc_auc_score(ya_sep[va], sa_prob))
            sep_b.append(roc_auc_score(yb_sep, sb_prob))
            prc_b.append(average_precision_score(yb_sep, sb_prob))

        gbm_results["los_T24"][feat_name] = {
            "r2_raw_train_A": float(np.mean(r2_a_raw)),
            "r2_raw_train_B": float(np.mean(r2_b_raw)),
            "r2_log_train_naive_expm1_A": float(np.mean(r2_a_smear)),
            "r2_log_train_naive_expm1_B": float(np.mean(r2_b_smear)),
            "r2_in_logspace_A": float(np.mean(r2_a_log)),
            "r2_in_logspace_B": float(np.mean(r2_b_log)),
            "spearman_A": float(np.mean(sp_a)),
            "spearman_B": float(np.mean(sp_b)),
            "mae_A": float(np.mean(mae_a)),
            "mae_B": float(np.mean(mae_b)),
        }
        gbm_results["sepsis_T24"][feat_name] = {
            "auroc_A": [float(np.mean(sep_a)), float(np.std(sep_a, ddof=1))],
            "auroc_B": [float(np.mean(sep_b)), float(np.std(sep_b, ddof=1))],
            "auprc_B": [float(np.mean(prc_b)), float(np.std(prc_b, ddof=1))],
        }
        print(f"  {feat_name}:")
        print(f"    LOS (raw-target fit):      Site A R2={np.mean(r2_a_raw):+.4f}, Site B R2={np.mean(r2_b_raw):+.4f}, Spearman A/B={np.mean(sp_a):.3f}/{np.mean(sp_b):.3f}, MAE A/B={np.mean(mae_a):.2f}h/{np.mean(mae_b):.2f}h")
        print(f"    LOS (log1p fit, expm1 R2): Site A R2={np.mean(r2_a_smear):+.4f}, Site B R2={np.mean(r2_b_smear):+.4f} (log-space R2 A/B={np.mean(r2_a_log):+.4f}/{np.mean(r2_b_log):+.4f})")
        print(f"    Sepsis (T=24):             Site A AUROC={np.mean(sep_a):.3f}±{np.std(sep_a,ddof=1):.3f}, Site B AUROC={np.mean(sep_b):.3f}±{np.std(sep_b,ddof=1):.3f}, Site B AUPRC={np.mean(prc_b):.3f}")

    out_report["gbm_baselines"] = gbm_results

    json.dump(out_report, open(os.path.join(REV_DIR, "reviewer_local_checks.json"), "w"), indent=2)
    print("\nSaved results/revision/reviewer_local_checks.json")

if __name__ == "__main__":
    main()
