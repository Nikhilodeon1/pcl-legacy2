"""
Computes exact 5-seed means, SDs, and 95% CIs from results/phase2/semisynth_shift.csv
and semisynth_length.csv, plus complete real-data selection tables from
results/los/selection_criteria.json and results/mortality/selection_criteria.json.
"""
import csv
import json
import os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REV_DIR = os.path.dirname(HERE)
RESULTS_DIR = os.path.dirname(REV_DIR)

def ci95(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    m = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tval = stats.t.ppf(0.975, df=n-1) if n > 1 else 0.0
    return {"mean": m, "sd": sd, "ci_lo": m - tval * se, "ci_hi": m + tval * se, "n": n}

def main():
    out = {}

    # 1. Semisynthetic Length Control (5 seeds x 3 lambdas = 15 runs)
    len_rows = list(csv.DictReader(open(os.path.join(RESULTS_DIR, "phase2", "semisynth_length.csv"))))
    print("=== 1. Semi-Synthetic Length Control (5 Seeds x 3 Lambdas) ===")
    by_win = {}
    for r in len_rows:
        lam = r["tag"].split("_")[0]
        win = f"T={r['fixed_T']}" if r["fixed_T"] else "variable"
        by_win.setdefault((win, lam), []).append(r)
        by_win.setdefault((win, "all"), []).append(r)

    len_summary = {}
    for (win, lam), rows in sorted(by_win.items()):
        m_auc = ci95([float(r["model_auroc"]) for r in rows])
        p_auc = ci95([float(r["probe_auroc"]) for r in rows])
        len_summary[f"{win}_{lam}"] = {"model": m_auc, "probe_mask_len": p_auc}
        print(f"  Window={win:12s} | {lam:6s} (n={m_auc['n']:2d}): Model AUROC={m_auc['mean']:.3f}±{m_auc['sd']:.3f} [{m_auc['ci_lo']:.3f}, {m_auc['ci_hi']:.3f}] | "
              f"Mask+Len Probe={p_auc['mean']:.3f}±{p_auc['sd']:.3f} [{p_auc['ci_lo']:.3f}, {p_auc['ci_hi']:.3f}]")
    out["semisynth_length"] = len_summary

    # 2. Semisynthetic Shift (endpoint change from mag_min to mag_max across 15 runs, plus per-lambda)
    shift_rows = list(csv.DictReader(open(os.path.join(RESULTS_DIR, "phase2", "semisynth_shift.csv"))))
    print("\n=== 2. Semi-Synthetic Shift Endpoint Changes (Max vs Baseline Magnitude, n=15 & per-lambda n=5) ===")
    # Group by (task, family, tag) -> list sorted by magnitude
    by_tft = {}
    for r in shift_rows:
        by_tft.setdefault((r["task"], r["family"], r["tag"]), []).append(r)

    shift_summary = {}
    for (task, fam) in sorted({(r["task"], r["family"]) for r in shift_rows}):
        d_perf, d_viol, d_recon, d_repr, d_mmd = [], [], [], [], []
        base_perf, max_perf = [], []
        for (t, f, tag), rlist in by_tft.items():
            if t != task or f != fam:
                continue
            rlist = sorted(rlist, key=lambda x: float(x["magnitude"]))
            r0, r1 = rlist[0], rlist[-1]
            base_perf.append(float(r0["performance"]))
            max_perf.append(float(r1["performance"]))
            d_perf.append(float(r1["performance"]) - float(r0["performance"]))
            d_viol.append(float(r1["violation"]) - float(r0["violation"]))
            d_recon.append(float(r1["recon_mse"]) - float(r0["recon_mse"]))
            d_repr.append(float(r1["repr_dist"]) - float(r0["repr_dist"]))
            d_mmd.append(float(r1["mmd"]) - float(r0["mmd"]))

        res = {
            "base_perf": ci95(base_perf),
            "max_perf": ci95(max_perf),
            "delta_perf": ci95(d_perf),
            "delta_viol": ci95(d_viol),
            "delta_recon": ci95(d_recon),
            "delta_repr": ci95(d_repr),
            "delta_mmd": ci95(d_mmd),
        }
        shift_summary[f"{task}_{fam}"] = res
        print(f"  {task:6s} | {fam:26s} (n={res['delta_perf']['n']}): "
              f"ΔPerf={res['delta_perf']['mean']:+.4f}±{res['delta_perf']['sd']:.4f} | "
              f"ΔViol={res['delta_viol']['mean']:+.4f}±{res['delta_viol']['sd']:.4f} | "
              f"ΔRecon={res['delta_recon']['mean']:+.4f}±{res['delta_recon']['sd']:.4f} | "
              f"ΔRepr={res['delta_repr']['mean']:+.3f}±{res['delta_repr']['sd']:.3f} | "
              f"ΔMMD={res['delta_mmd']['mean']:+.4f}±{res['delta_mmd']['sd']:.4f}")
    out["semisynth_shift"] = shift_summary

    # 3. Real-data selection criteria summary (Mortality & LOS)
    mort_sel = json.load(open(os.path.join(RESULTS_DIR, "mortality", "selection_criteria.json")))
    los_sel = json.load(open(os.path.join(RESULTS_DIR, "los", "selection_criteria.json")))

    print("\n=== 3. Real Cross-Hospital Criterion Magnitudes by Target Domain ===")
    # Check mean criterion values per target in LOS and Mortality
    for name, sel_obj, tkey in [("LOS", los_sel, "target"), ("Mortality", mort_sel, "target")]:
        cells = sel_obj["cells"]
        targets = sorted({c[tkey] for c in cells})
        for tgt in targets:
            sub = [c for c in cells if c[tkey] == tgt]
            v_m = np.mean([c["violation"] for c in sub])
            r_m = np.mean([c["recon_mse"] for c in sub])
            d_m = np.mean([c["repr_dist"] for c in sub])
            m_m = np.mean([c["mmd"] for c in sub])
            print(f"  {name:10s} Target={tgt:12s} (n={len(sub)}): Viol={v_m:.4f}, Recon={r_m:.4f}, ReprDist={d_m:.3f}, MMD={m_m:.4f}")

    json.dump(out, open(os.path.join(REV_DIR, "semisynth_and_selection_ci.json"), "w"), indent=2)
    print("\nSaved results/revision/semisynth_and_selection_ci.json")

if __name__ == "__main__":
    main()
