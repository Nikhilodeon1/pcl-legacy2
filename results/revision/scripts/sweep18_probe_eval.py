"""
Evaluates Linear Probe and Whitened (ZCA / Variance-Standardized) Probe across all 18
independently pretrained sweep encoders (lambda in {0.0, 0.1, 0.5, 1.0, 2.0, 5.0} x seeds 42, 43, 44)
on PhysioNet Site A (in-domain val) and Site B (zero-shot OOD) for Sepsis and LOS.
Also correlates centered effective rank (erank_cent) with Sepsis OOD transfer across the 18 encoders.
"""
import glob
import gzip
import json
import os
import pickle
import sys
import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
REV_DIR = os.path.dirname(HERE)
RESULTS_DIR = os.path.dirname(REV_DIR)
ROOT = os.path.dirname(RESULTS_DIR)
sys.path.insert(0, ROOT)
os.environ["PCL_TEST_MODE"] = "0"

from src.baselines import fresh_model
from src.training.train_utils import load_state_dict_flexible

def extract_reps(model, X, M, h=47, batch=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            r = model.encode(X[i:i+batch], M[i:i+batch])[:, h].float().numpy()
            outs.append(r)
    return np.concatenate(outs, axis=0)

def main():
    with gzip.open(os.path.join(ROOT, "colab", "physionet_frac1.0.pkl.gz"), "rb") as f:
        pn = pickle.load(f)
    site_a = [s for s in pn if s["site_id"] == 0]
    site_b = [s for s in pn if s["site_id"] == 1]

    # Subsample 3000 Site A and 3000 Site B stays (stratified/seeded) for fast CPU evaluation
    rng = np.random.default_rng(42)
    ia = rng.choice(len(site_a), 3000, replace=False)
    ib = rng.choice(len(site_b), 3000, replace=False)

    Xa = torch.as_tensor(np.stack([np.asarray(site_a[i]["values"]) for i in ia]), dtype=torch.float32)
    Ma = torch.as_tensor(np.stack([np.asarray(site_a[i]["mask"]) for i in ia]), dtype=torch.bool)
    ya_sep = np.array([int(np.asarray(site_a[i]["label"]).sum() > 0) for i in ia])

    Xb = torch.as_tensor(np.stack([np.asarray(site_b[i]["values"]) for i in ib]), dtype=torch.float32)
    Mb = torch.as_tensor(np.stack([np.asarray(site_b[i]["mask"]) for i in ib]), dtype=torch.bool)
    yb_sep = np.array([int(np.asarray(site_b[i]["label"]).sum() > 0) for i in ib])

    er_map = {r["name"]: r for r in json.load(open(os.path.join(REV_DIR, "erank_centered_sweep.json")))}
    ckpts = sorted(glob.glob(os.path.join(REV_DIR, "encoders_slim", "sweep", "*.pt")))

    rows = []
    print(f"{'Encoder':<14s} | {'erank_c':>7s} | {'Probe A':>7s} | {'Probe B':>7s} | {'Whiten B':>8s} | {'AUPRC B':>7s}")
    print("-" * 68)
    for p in ckpts:
        name = os.path.basename(p)[:-3]
        lam = float(name.split("_")[0].replace("lam", ""))
        seed = int(name.split("_p")[1])
        model = fresh_model(seed=0)
        model.load_state_dict(load_state_dict_flexible(p, "cpu"))

        Za = extract_reps(model, Xa, Ma, h=47)
        Zb = extract_reps(model, Xb, Mb, h=47)

        # 80/20 split on Za
        r_sp = np.random.default_rng(seed)
        perm = r_sp.permutation(len(Za))
        ntr = int(len(Za) * 0.8)
        tr, va = perm[:ntr], perm[ntr:]

        # 1. Raw linear probe
        clf = LogisticRegression(max_iter=500, C=0.1, class_weight="balanced")
        clf.fit(Za[tr], ya_sep[tr])
        auc_a = roc_auc_score(ya_sep[va], clf.predict_proba(Za[va])[:, 1])
        prob_b = clf.predict_proba(Zb)[:, 1]
        auc_b = roc_auc_score(yb_sep, prob_b)
        prc_b = average_precision_score(yb_sep, prob_b)

        # 2. Centered + variance-standardized (whitened diagonal) probe (fitted on Source Train only)
        scaler = StandardScaler()
        Za_tr_w = scaler.fit_transform(Za[tr])
        Za_va_w = scaler.transform(Za[va])
        Zb_w = scaler.transform(Zb)
        clf_w = LogisticRegression(max_iter=500, C=0.1, class_weight="balanced")
        clf_w.fit(Za_tr_w, ya_sep[tr])
        auc_b_w = roc_auc_score(yb_sep, clf_w.predict_proba(Zb_w)[:, 1])

        ec = er_map[name]["erank_cent"]
        print(f"{name:<14s} | {ec:7.2f} | {auc_a:7.3f} | {auc_b:7.3f} | {auc_b_w:8.3f} | {prc_b:7.3f}")
        rows.append({
            "name": name, "lam": lam, "seed": seed, "erank_cent": ec,
            "probe_auroc_A": float(auc_a), "probe_auroc_B": float(auc_b),
            "whiten_auroc_B": float(auc_b_w), "probe_auprc_B": float(prc_b)
        })

    # Correlate erank_cent with probe_auroc_B across all 18 encoders
    ec_arr = [r["erank_cent"] for r in rows]
    ab_arr = [r["probe_auroc_B"] for r in rows]
    aw_arr = [r["whiten_auroc_B"] for r in rows]
    rho, pval = stats.spearmanr(ec_arr, ab_arr)
    pr, ppval = stats.pearsonr(ec_arr, ab_arr)
    print(f"\nAcross 18 sweep encoders: Spearman(erank_cent, Sepsis Probe B) = {rho:+.3f} (p={pval:.4f}), Pearson r = {pr:+.3f} (p={ppval:.4f})")

    # Group by lambda
    by_lam = {}
    for r in rows:
        by_lam.setdefault(r["lam"], []).append(r)
    print("\n=== Mean ± SD by Lambda ===")
    for lam in sorted(by_lam):
        sub = by_lam[lam]
        print(f"lam={lam:4.1f}: erank_cent={np.mean([x['erank_cent'] for x in sub]):5.2f} | "
              f"Site A={np.mean([x['probe_auroc_A'] for x in sub]):.3f}±{np.std([x['probe_auroc_A'] for x in sub],ddof=1):.3f} | "
              f"Site B={np.mean([x['probe_auroc_B'] for x in sub]):.3f}±{np.std([x['probe_auroc_B'] for x in sub],ddof=1):.3f} | "
              f"Whiten B={np.mean([x['whiten_auroc_B'] for x in sub]):.3f}±{np.std([x['whiten_auroc_B'] for x in sub],ddof=1):.3f}")

    json.dump({"rows": rows, "spearman_erank_probeB": [float(rho), float(pval)], "pearson_erank_probeB": [float(pr), float(ppval)]},
              open(os.path.join(REV_DIR, "sweep18_sepsis_probe.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
