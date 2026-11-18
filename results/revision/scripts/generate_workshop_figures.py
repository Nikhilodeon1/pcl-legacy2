"""
Generate publication-grade AAAI Workshop figures (PDF + PNG) directly from verified JSONs:
  - paper/figures/fig1_budget_and_ablations.{pdf,png}
  - paper/figures/fig2_lambda_dose_response.{pdf,png}
  - paper/figures/fig3_head_pilot_and_tasks.{pdf,png}
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_REV = os.path.join(_ROOT, "results", "revision")
_OUT = os.path.join(_ROOT, "paper", "figures")


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def make_fig1(p2c_sum):
    srows = p2c_sum["summary_rows"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8), gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Panel A: Budget scaling curve (10k -> 25k -> 80k)
    budgets = ["10k\n(3.9k steps)", "25k\n(10.8k steps)", "80k\n(30k steps)"]
    x = np.arange(len(budgets))
    erm_keys = ["ERM Clean (10k)", "ERM Matched-Vol (25k)", "ERM Matched-Vol (80k)"]
    pcl_keys = ["PCL Clean (10k)", "PCL Matched-Vol (25k)", "PCL Matched-Vol (80k)"]

    erm_m = [srows[k]["site_b_auroc"]["mean"] for k in erm_keys]
    erm_sd = [srows[k]["site_b_auroc"]["sd"] for k in erm_keys]
    pcl_m = [srows[k]["site_b_auroc"]["mean"] for k in pcl_keys]
    pcl_sd = [srows[k]["site_b_auroc"]["sd"] for k in pcl_keys]

    ax1.errorbar(x - 0.04, erm_m, yerr=erm_sd, marker="o", lw=2.0, capsize=4, color="#1f77b4", label="ERM (λ = 0.0)")
    ax1.errorbar(x + 0.04, pcl_m, yerr=pcl_sd, marker="s", lw=2.0, capsize=4, color="#d62728", label="PCL (λ = 0.5)")

    for xi, ym, ys in zip(x, erm_m, erm_sd):
        ax1.annotate(f"{ym:.3f}", (xi - 0.04, ym + ys + 0.007), ha="center", fontsize=8, color="#1f77b4", fontweight="bold")
    for xi, ym, ys in zip(x, pcl_m, pcl_sd):
        ax1.annotate(f"{ym:.3f}", (xi + 0.04, ym - ys - 0.015), ha="center", fontsize=8, color="#d62728", fontweight="bold")

    scratch_m = srows["Supervised Scratch"]["site_b_auroc"]["mean"]
    ax1.axhline(scratch_m, ls="--", lw=1.2, color="#555555", label=f"Supervised Scratch ({scratch_m:.3f})")
    ax1.axhline(0.713, ls=":", lw=1.3, color="#2ca02c", label="HistGBM 119-feat T=24 (0.713)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(budgets)
    ax1.set_ylabel("Site B OOD Sepsis AUROC")
    ax1.set_title("A. Pretraining Budget Scaling (n = 5 paired seeds)")
    ax1.set_ylim(0.59, 0.845)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left", ncol=2, fontsize=7.2, frameon=True, framealpha=0.92)

    # Panel B: 10k Mechanistic & Constraint Ablations
    abl_spec = [
        ("ERM Clean", "ERM Clean (10k)", "#1f77b4"),
        ("PCL MAP-Only", "PCL MAP-Only (10k)", "#ff7f0e"),
        ("PCL No-HH", "PCL No-HH (10k)", "#ff7f0e"),
        ("PCL Full (EMA)", "PCL Clean (10k)", "#d62728"),
        ("PCL Placebo", "PCL Placebo (10k)", "#9467bd"),
        ("PCL No-Balancer", "PCL No-Balancer (10k)", "#8c564b"),
        ("Scratch (No PT)", "Supervised Scratch", "#555555"),
        ("PCL + VICReg", "PCL + VICReg (10k)", "#e377c2"),
    ]
    labels = [a[0] for a in abl_spec][::-1]
    means = [srows[a[1]]["site_b_auroc"]["mean"] for a in abl_spec][::-1]
    sds = [srows[a[1]]["site_b_auroc"]["sd"] for a in abl_spec][::-1]
    colors = [a[2] for a in abl_spec][::-1]

    y = np.arange(len(labels))
    ax2.barh(y, means, xerr=sds, height=0.62, color=colors, alpha=0.88, capsize=3, edgecolor="black", linewidth=0.5)
    for yi, m, s in zip(y, means, sds):
        ax2.text(m + s + 0.005, yi, f"{m:.3f}", va="center", fontsize=8)
    ax2.axvline(srows["ERM Clean (10k)"]["site_b_auroc"]["mean"], ls="--", lw=1.0, color="#1f77b4", alpha=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlim(0.54, 0.77)
    ax2.set_xlabel("Site B Sepsis AUROC (± SD across 5 seeds)")
    ax2.set_title("B. Constraint, Placebo & VICReg Ablations (10k)")
    ax2.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(_OUT, f"fig1_budget_and_ablations.{ext}"), bbox_inches="tight")
    plt.close(fig)


def make_fig2(wk_sum, p2c_sum, gcos_sum):
    sw = wk_sum["lambda_sweep"]
    rows = sw["by_lambda"]
    boot_by_c = sw["site_b_patient_bootstrap"]["by_condition"]
    g_rows = {r["lambda"]: r for r in gcos_sum["by_lambda"]}

    lams = [r["lambda"] for r in rows]
    x = np.arange(len(lams))
    x_labels = [f"{l:g}" for l in lams]

    # Non-zero lambdas for log gradient ratio
    lams_nz = [l for l in lams if l > 0]
    x_nz = np.arange(len(lams_nz))
    ratios = [g_rows[l]["eff_grad_ratio_mean"] for l in lams_nz]
    cos_vals = [g_rows[l]["cos_enc_mean"] for l in lams_nz]

    aurocs = [r["site_b_auroc_mean"] for r in rows]
    boot_lo = [aurocs[i] - boot_by_c[f"lam={l:g}"]["boot_ci_95"][0] for i, l in enumerate(lams)]
    boot_hi = [boot_by_c[f"lam={l:g}"]["boot_ci_95"][1] - aurocs[i] for i, l in enumerate(lams)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.6, 3.5))

    # Panel A: Pure Gradient-Scale Dominance & Near-Orthogonal Cosine
    c1, c2 = "#d62728", "#1f77b4"
    ax1.plot(x_nz, ratios, marker="o", lw=2.0, color=c1, label="||∇(λL_pcl)|| / ||∇L_mask||")
    ax1.axhline(1.0, ls=":", lw=1.1, color=c1, alpha=0.7, label="Equal Gradient Scale (1×)")
    ax1.set_yscale("log")
    ax1.set_xticks(x_nz)
    ax1.set_xticklabels([f"{l:g}" for l in lams_nz])
    ax1.set_xlabel("Constraint Weight λ > 0")
    ax1.set_ylabel("Gradient Norm Ratio (log)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(0.1, 450)
    ax1.set_title("A. Gradient-Scale Dominance (cos ≈ 0)")
    ax1.grid(axis="y", alpha=0.25)

    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)
    ax1_r.plot(x_nz, cos_vals, marker="s", ls="--", lw=1.7, color=c2, label="Encoder Grad Cosine")
    ax1_r.axhline(0.0, ls="-.", lw=0.9, color=c2, alpha=0.45)
    ax1_r.set_ylabel("cos(∇L_mask, ∇L_pcl)", color=c2)
    ax1_r.tick_params(axis="y", labelcolor=c2)
    ax1_r.set_ylim(-0.35, 0.35)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=7.0, frameon=True)

    # Panel B: Norm vs Transfer across ALL Phase 2c arms (disproving naive norm causality)
    srows = p2c_sum["summary_rows"]
    pts_x, pts_y, pts_lbl, pts_clr = [], [], [], []
    arm_meta = [
        ("ERM Clean (10k)", "ERM 10k", "#1f77b4", (6, 4)),
        ("ERM Matched-Vol (25k)", "ERM 25k", "#1f77b4", (6, 3)),
        ("ERM Matched-Vol (80k)", "ERM 80k", "#1f77b4", (6, -2)),
        ("PCL Clean (10k)", "PCL 10k", "#d62728", (7, -3)),
        ("PCL Matched-Vol (25k)", "PCL 25k", "#d62728", (5, -9)),
        ("PCL Matched-Vol (80k)", "PCL 80k", "#d62728", (-38, 6)),
        ("PCL + VICReg (10k)", "VICReg", "#e377c2", (6, 3)),
        ("PCL Placebo (10k)", "Placebo", "#9467bd", (-36, -10)),
        ("PCL MAP-Only (10k)", "MAP", "#ff7f0e", (-32, 4)),
        ("PCL No-HH (10k)", "No-HH", "#ff7f0e", (7, 4)),
    ]
    for key, short, clr, offset in arm_meta:
        l2 = srows[key]["l2_norm"]["mean"]
        auroc = srows[key]["site_b_auroc"]["mean"]
        ax2.scatter([l2], [auroc], color=clr, s=48, edgecolor="black", linewidth=0.6, zorder=4)
        ax2.annotate(short, (l2, auroc), xytext=offset, textcoords="offset points", fontsize=7.2, color=clr, fontweight="bold")
        pts_x.append(l2); pts_y.append(auroc)
    ax2.set_xlabel("Hour-47 Representation L2 Norm")
    ax2.set_ylabel("Site B Sepsis AUROC (n = 5)")
    ax2.set_xlim(40, 315)
    ax2.set_ylim(0.595, 0.775)
    ax2.set_title("B. Representation Norm Does Not Predict Transfer")
    ax2.grid(alpha=0.25)

    # Panel C: Downstream Sepsis Site B AUROC vs Lambda + Source-Val Tuned PCL
    ax3.errorbar(x, aurocs, yerr=[boot_lo, boot_hi], marker="o", lw=2.0, capsize=4, color="#1f77b4", label="Mean ± 95% Patient CI")
    ax3.axhline(aurocs[0], ls="--", lw=1.1, color="#555555", label=f"ERM Baseline ({aurocs[0]:.3f})")
    ax3.axhline(0.6792, ls=":", lw=1.3, color="#ff7f0e", label="Val-Tuned PCL (0.679)")
    ax3.axvspan(3.5, 5.4, color="#d62728", alpha=0.08, label="High-λ Regime (≥0.5)")
    for xi, ym, hi in zip(x, aurocs, boot_hi):
        ax3.annotate(f"{ym:.3f}", (xi, ym + hi + 0.004), ha="center", fontsize=7.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels)
    ax3.set_xlabel("Constraint Weight λ")
    ax3.set_ylabel("Site B Sepsis AUROC (n = 3)")
    ax3.set_ylim(0.555, 0.775)
    ax3.set_title("C. Site B Sepsis Transfer vs λ")
    ax3.grid(axis="y", alpha=0.25)
    ax3.legend(loc="lower left", fontsize=7.0, frameon=True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(_OUT, f"fig2_lambda_dose_response.{ext}"), bbox_inches="tight")
    plt.close(fig)


def make_fig3(wk_sum, mort_sum, gbm_sum):
    hp_rows = {(r["tag"], r["head_mode"]): r for r in wk_sum["head_pilot"]["rows"]}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1.0, 1.1]})

    modes = ["baseline", "layernorm", "lpft"]
    mode_labels = ["Baseline\n(Unnorm.)", "LayerNorm\nHead", "LP-FT\n(LN + 1e-5)"]
    x = np.arange(len(modes))
    w = 0.34

    erm_m = [hp_rows[("erm_lam0", m)]["ood_auroc_mean"] for m in modes]
    erm_sd = [hp_rows[("erm_lam0", m)]["ood_auroc_sd"] for m in modes]
    pcl_m = [hp_rows[("pcl_lam0.5", m)]["ood_auroc_mean"] for m in modes]
    pcl_sd = [hp_rows[("pcl_lam0.5", m)]["ood_auroc_sd"] for m in modes]
    pcl_col = [hp_rows[("pcl_lam0.5", m)]["n_collapses"] for m in modes]

    ax1.bar(x - w / 2, erm_m, yerr=erm_sd, width=w, capsize=3, color="#1f77b4", alpha=0.88, edgecolor="black", lw=0.5, label="ERM (λ = 0.0)")
    ax1.bar(x + w / 2, pcl_m, yerr=pcl_sd, width=w, capsize=3, color="#d62728", alpha=0.88, edgecolor="black", lw=0.5, label="PCL (λ = 0.5)")

    for xi, ym, ys in zip(x, erm_m, erm_sd):
        ax1.text(xi - w / 2, ym + ys + 0.006, f"{ym:.3f}\n(0/3)", ha="center", va="bottom", fontsize=7.5)
    for xi, ym, ys, nc in zip(x, pcl_m, pcl_sd, pcl_col):
        tag_col = f"{ym:.3f}\n({nc}/3 col.)" if nc > 0 else f"{ym:.3f}\n(0/3)"
        col_clr = "#b00020" if nc > 0 else "black"
        ax1.text(xi + w / 2, ym + ys + 0.006, tag_col, ha="center", va="bottom", fontsize=7.5, color=col_clr, fontweight="bold" if nc > 0 else "normal")

    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_labels)
    ax1.set_ylim(0.50, 0.70)
    ax1.set_ylabel("Site B AUROC (T=24 Protocol)")
    ax1.set_title("A. Exploratory Head Conditioning Pilot (T=24, n=3)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="lower right", frameon=True)

    mort_m = mort_sum["methods"]
    dirs = ["MIMIC->eICU", "eICU->MIMIC"]
    x2 = np.arange(len(dirs))
    w2 = 0.25

    erm_fp32 = [mort_m[f"fp32|erm|{d}"]["mean"] for d in dirs]
    erm_fp32_sd = [mort_m[f"fp32|erm|{d}"]["sd"] for d in dirs]
    pcl_fp32 = [mort_m[f"fp32|pcl|{d}"]["mean"] for d in dirs]
    pcl_fp32_sd = [mort_m[f"fp32|pcl|{d}"]["sd"] for d in dirs]
    gbm_m = [gbm_sum["mortality_T48"]["mimic_to_eicu_ood"]["mean"], gbm_sum["mortality_T48"]["eicu_to_mimic_ood"]["mean"]]
    gbm_sd = [gbm_sum["mortality_T48"]["mimic_to_eicu_ood"]["sd"], gbm_sum["mortality_T48"]["eicu_to_mimic_ood"]["sd"]]

    ax2.bar(x2 - w2, erm_fp32, yerr=erm_fp32_sd, width=w2, capsize=3, color="#1f77b4", alpha=0.88, edgecolor="black", lw=0.5, label="ERM Transformer (fp32)")
    ax2.bar(x2, pcl_fp32, yerr=pcl_fp32_sd, width=w2, capsize=3, color="#d62728", alpha=0.88, edgecolor="black", lw=0.5, label="PCL Transformer (fp32)")
    ax2.bar(x2 + w2, gbm_m, yerr=gbm_sd, width=w2, capsize=3, color="#2ca02c", alpha=0.88, edgecolor="black", lw=0.5, label="Tabular HistGBM (T=48)")

    for bars, vals, sds in [(x2 - w2, erm_fp32, erm_fp32_sd), (x2, pcl_fp32, pcl_fp32_sd), (x2 + w2, gbm_m, gbm_sd)]:
        for xi, ym, ys in zip(bars, vals, sds):
            ax2.text(xi, ym + ys + 0.008, f"{ym:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(["MIMIC-IV → eICU-CRD", "eICU-CRD → MIMIC-IV"])
    ax2.set_ylim(0.45, 0.92)
    ax2.set_ylabel("Fixed T=48 Mortality AUROC")
    ax2.set_title("B. Fixed T=48 Mortality: Neural Ceiling & PCL Collapses")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(_OUT, f"fig3_head_pilot_and_tasks.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    set_style()
    os.makedirs(_OUT, exist_ok=True)
    p2c_sum = json.load(open(os.path.join(_REV, "phase2c_full_summary.json")))
    wk_sum = json.load(open(os.path.join(_REV, "workshop_new_runs_summary.json")))
    mort_sum = json.load(open(os.path.join(_REV, "fixedT48_mortality_summary.json")))
    gbm_sum = json.load(open(os.path.join(_REV, "verified_gbm_baselines.json")))
    gcos_sum = json.load(open(os.path.join(_REV, "grad_cosine_sweep.json")))

    make_fig1(p2c_sum)
    make_fig2(wk_sum, p2c_sum, gcos_sum)
    make_fig3(wk_sum, mort_sum, gbm_sum)
    print(f"Generated 3 figures (PDF + PNG) in {_OUT}")


if __name__ == "__main__":
    main()
