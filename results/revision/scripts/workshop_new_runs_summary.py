"""
AAAI Workshop Synthesis Script:
Summarizes all newly run experiments once synced from Google Drive/Colab:
  1. Head Normalization & LP-FT Pilot (results/head_pilot/*.json)
  2. Sepsis Small-Lambda Dose-Response + Pretrain Gradient-Norm / Loss-Share Logs
     (results/pretrain_lambda_sweep/*.json & results/sepsis_lambda_sweep/*.json)
  3. Reconciled Multi-Site Cohort Accounting (results/revision/g2_cohort.json)
Outputs results/revision/workshop_new_runs_summary.json.
"""
import argparse
import collections
import glob
import json
import os
import sys
import numpy as np
import scipy.stats as stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from phase2c_stats_rigor import holm_bonferroni, paired_contrast_stats, siteb_patient_bootstrap


def summarize_head_pilot(head_dir):
    files = sorted(glob.glob(os.path.join(head_dir, "*.json")))
    if not files:
        return {"available": False, "rows": []}

    grouped = collections.defaultdict(list)
    for f in files:
        d = json.load(open(f))
        grouped[(d["task"], d["tag"], d["head_mode"])].append({
            "seed": d["seed"],
            "in_auroc": d.get("in_domain", {}).get("auroc"),
            "ood_auroc": d.get("ood", {}).get("auroc"),
            "ood_auprc": d.get("ood", {}).get("auprc"),
            "ood_brier": d.get("ood", {}).get("brier"),
            "constant_collapse": bool(d.get("logs", {}).get("constant_collapse", False)),
        })

    rows = []
    print("\n=== 1. HEAD NORMALIZATION & LP-FT PILOT ===")
    print(f"{'Task':<10} {'Ckpt':<12} {'Mode':<11} {'n':>2} {'In-Dom AUROC':>17} {'OOD AUROC':>17} {'Collapse'}")
    print("-" * 82)
    for (task, tag, mode), recs in sorted(grouped.items()):
        in_vals = [r["in_auroc"] for r in recs if r["in_auroc"] is not None]
        ood_vals = [r["ood_auroc"] for r in recs if r["ood_auroc"] is not None]
        n_col = sum(1 for r in recs if r["constant_collapse"])
        row = {
            "task": task, "tag": tag, "head_mode": mode, "n": len(recs),
            "in_auroc_mean": float(np.mean(in_vals)) if in_vals else None,
            "in_auroc_sd": float(np.std(in_vals, ddof=1)) if len(in_vals) > 1 else 0.0,
            "ood_auroc_mean": float(np.mean(ood_vals)) if ood_vals else None,
            "ood_auroc_sd": float(np.std(ood_vals, ddof=1)) if len(ood_vals) > 1 else 0.0,
            "n_collapses": n_col,
        }
        rows.append(row)
        print(f"{task:<10} {tag:<12} {mode:<11} {len(recs):>2} "
              f"{row['in_auroc_mean']:>7.4f} +/- {row['in_auroc_sd']:<6.4f} "
              f"{row['ood_auroc_mean']:>7.4f} +/- {row['ood_auroc_sd']:<6.4f} {n_col}/{len(recs)}")
    return {"available": True, "rows": rows}


def summarize_lambda_sweep(pt_dir, ft_dir):
    pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.json")))
    ft_files = sorted(glob.glob(os.path.join(ft_dir, "*.json")))
    if not pt_files and not ft_files:
        return {"available": False}

    pt_by_ls = {}
    for f in pt_files:
        d = json.load(open(f))
        h = d.get("history", {})
        r47 = d.get("repr_diag_site_a", {}).get("47", {})
        pt_by_ls[(float(d["lambda"]), int(d["seed"]))] = {
            "mean_grad_norm": float(np.nanmean(h["train_grad_norm"])) if h.get("train_grad_norm") else None,
            "final_grad_norm": float(h["train_grad_norm"][-1]) if h.get("train_grad_norm") else None,
            "mean_pcl_weighted": float(np.nanmean(h["train_pcl_weighted"])) if h.get("train_pcl_weighted") else None,
            "final_pcl_share": float(h["train_pcl_loss_share"][-1]) if h.get("train_pcl_loss_share") else None,
            "mean_pcl_share": float(np.nanmean(h["train_pcl_loss_share"])) if h.get("train_pcl_loss_share") else None,
            "l2_h47": r47.get("mean_l2"),
            "spread_h47": r47.get("spread"),
            "const_frac_h47": r47.get("const_frac"),
        }

    ft_by_ls = {}
    for f in ft_files:
        d = json.load(open(f))
        lam_str = d["encoder_id"].split("_")[0].replace("lam", "")
        lam = float(lam_str)
        seed = int(d["seed"])
        ood_b = d.get("ood", {}).get("physionet_b", {})
        ft_by_ls[(lam, seed)] = {
            "site_a_auroc": d.get("in_domain", {}).get("auroc"),
            "site_b_auroc": ood_b.get("auroc"),
            "site_b_auprc": ood_b.get("auprc"),
            "site_b_brier": ood_b.get("brier"),
        }

    lambdas = sorted({k[0] for k in pt_by_ls} | {k[0] for k in ft_by_ls})
    summary = []
    print("\n=== 2. SEPSIS SMALL-LAMBDA DOSE-RESPONSE SWEEP ===")
    print(f"{'lam':>6} {'n':>2} {'Site B AUROC':>17} {'GradNorm(end)':>14} {'PCL Share(avg)':>14} {'L2(h47)':>10} {'Spread(h47)':>12}")
    print("-" * 86)
    for lam in lambdas:
        pt_recs = [v for (l, _), v in sorted(pt_by_ls.items()) if l == lam]
        ft_recs = [v for (l, _), v in sorted(ft_by_ls.items()) if l == lam]
        aurocs = [r["site_b_auroc"] for r in ft_recs if r["site_b_auroc"] is not None]
        gnorms = [r["final_grad_norm"] for r in pt_recs if r["final_grad_norm"] is not None]
        shares = [r["mean_pcl_share"] for r in pt_recs if r["mean_pcl_share"] is not None]
        l2s = [r["l2_h47"] for r in pt_recs if r["l2_h47"] is not None]
        spreads = [r["spread_h47"] for r in pt_recs if r["spread_h47"] is not None]
        row = {
            "lambda": lam,
            "n_pretrain": len(pt_recs),
            "n_finetune": len(ft_recs),
            "site_b_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
            "site_b_auroc_sd": float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0,
            "final_grad_norm_mean": float(np.mean(gnorms)) if gnorms else None,
            "mean_pcl_share_mean": float(np.mean(shares)) if shares else None,
            "l2_h47_mean": float(np.mean(l2s)) if l2s else None,
            "spread_h47_mean": float(np.mean(spreads)) if spreads else None,
        }
        summary.append(row)
        auroc_s = f"{row['site_b_auroc_mean']:.4f}+/-{row['site_b_auroc_sd']:.4f}" if aurocs else "—"
        gn_s = f"{row['final_grad_norm_mean']:.4f}" if gnorms else "—"
        sh_s = f"{row['mean_pcl_share_mean']:.4f}" if shares else "—"
        l2_s = f"{row['l2_h47_mean']:.1f}" if l2s else "—"
        sp_s = f"{row['spread_h47_mean']:.4f}" if spreads else "—"
        print(f"{lam:>6g} {len(ft_recs):>2} {auroc_s:>17} {gn_s:>14} {sh_s:>14} {l2_s:>10} {sp_s:>12}")

    # Monotone dose-response Spearman correlations across checkpoints
    lam_vec, auroc_vec = [], []
    for (lam, _), v in sorted(ft_by_ls.items()):
        if v["site_b_auroc"] is not None:
            lam_vec.append(lam)
            auroc_vec.append(v["site_b_auroc"])
    rho_auroc = stats.spearmanr(lam_vec, auroc_vec) if len(lam_vec) >= 3 else None

    lam_p, share_p = [], []
    for (lam, _), v in sorted(pt_by_ls.items()):
        if v["mean_pcl_share"] is not None:
            lam_p.append(lam)
            share_p.append(v["mean_pcl_share"])
    rho_share = stats.spearmanr(lam_p, share_p) if len(lam_p) >= 3 else None

    # Paired contrasts vs lambda=0
    contrasts = []
    for lam in lambdas:
        if lam == 0.0:
            continue
        seeds_shared = sorted({s for (l, s) in ft_by_ls if l == lam} & {s for (l, s) in ft_by_ls if l == 0.0})
        if not seeds_shared:
            continue
        va = [ft_by_ls[(lam, s)]["site_b_auroc"] for s in seeds_shared]
        vb = [ft_by_ls[(0.0, s)]["site_b_auroc"] for s in seeds_shared]
        res = paired_contrast_stats(va, vb, seed=42 + int(round(lam * 100)))
        res["lambda"] = lam
        res["seeds"] = seeds_shared
        contrasts.append(res)
    if contrasts:
        holm_p = holm_bonferroni([c["perm_p_value"] for c in contrasts])
        for c, hp in zip(contrasts, holm_p):
            c["p_holm_perm"] = hp

    # Patient-level bootstrap on Site B sweep predictions if present
    pred_dir = os.path.join(ft_dir, "preds")
    pred_records = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, "*_site_b.npz"))):
        base = os.path.basename(f)
        lam_part = base.split("_")[0].replace("lam", "")
        seed_part = base.rsplit("_s", 1)[1].split("_")[0]
        npz = np.load(f, allow_pickle=True)
        pred_records[(f"lam={lam_part}", int(seed_part))] = {
            "probs": npz["probs"], "labels": npz["labels"],
            "pids": npz["pids"] if "pids" in npz else np.arange(len(npz["labels"])),
        }
    boot = siteb_patient_bootstrap(
        pred_records, B=5000,
        contrast_pairs=[(f"lam={l} - lam=0", f"lam={l:g}", "lam=0") for l in lambdas if l != 0.0],
    )

    return {
        "available": True,
        "by_lambda": summary,
        "spearman_lambda_vs_auroc": ({"rho": float(rho_auroc.statistic), "p": float(rho_auroc.pvalue)} if rho_auroc else None),
        "spearman_lambda_vs_pcl_share": ({"rho": float(rho_share.statistic), "p": float(rho_share.pvalue)} if rho_share else None),
        "contrasts_vs_erm": contrasts,
        "site_b_patient_bootstrap": boot,
    }


def summarize_cohort_accounting(cohort_json):
    if not os.path.exists(cohort_json):
        return None
    g2 = json.load(open(cohort_json))
    print("\n=== 3. COHORT ACCOUNTING (RECONCILED) ===")
    total_stays = sum(c["stays"] for c in g2["cohorts"])
    total_pts = sum(c["patients"] for c in g2["cohorts"])
    print(f"Total across 4 cohorts: {total_stays:,} stays ({total_pts:,} cohort-unique patients)")
    for c in g2["cohorts"]:
        print(f"  {c['dataset']:<13} stays={c['stays']:>7,}  patients={c['patients']:>7,}  "
              f"<48h_frac={c['frac_los_lt_48h']:.3f}  mort_prev={c.get('mortality_hospital_prev', 'n/a')}")
    return {"total_stays": total_stays, "total_cohort_patients": total_pts, "cohorts": g2["cohorts"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dir", default=os.path.join(_ROOT, "results", "head_pilot"))
    ap.add_argument("--pretrain-sweep-dir", default=os.path.join(_ROOT, "results", "pretrain_lambda_sweep"))
    ap.add_argument("--sepsis-sweep-dir", default=os.path.join(_ROOT, "results", "sepsis_lambda_sweep"))
    ap.add_argument("--cohort-json", default=os.path.join(_ROOT, "results", "revision", "g2_cohort.json"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "revision", "workshop_new_runs_summary.json"))
    args = ap.parse_args()

    out = {
        "head_pilot": summarize_head_pilot(args.head_dir),
        "lambda_sweep": summarize_lambda_sweep(args.pretrain_sweep_dir, args.sepsis_sweep_dir),
        "cohort_accounting": summarize_cohort_accounting(args.cohort_json),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
