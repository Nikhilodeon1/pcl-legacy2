"""
G1 (ii) — values-shuffled test. Takes the FINE-TUNED checkpoints, keeps the real observation
masks, and replaces observed values by (a) values permuted across stays within each variable
among observed cells, (b) uniform random noise in [0,1]. If performance stays near the
real-values performance, the model is reading the mask/length pattern, not physiology.

LOS (local ckpts exist for seed 42..44):
    python g1_shuffle.py --task los --seed 42 --n-eval 5000
Mortality (needs fine-tuned mortality ckpts, on the pod):
    python g1_shuffle.py --task mortality --mort-ckpt-dir <dir> --seed 42

Flag rule (PREREG.md): shuffled-value performance clearly above chance =
LOS R2 within 0.05 of real-values R2, or mortality AUROC > 0.60.
Outputs: g1_shuffle_{task}.csv / .json
"""
import argparse
import csv
import json
import os

import numpy as np

import revlib as R

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["los", "mortality"], required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n-eval", type=int, default=5000, help="max stays per eval set (CPU budget)")
ap.add_argument("--pn-frac", type=float, default=1.0)
ap.add_argument("--mimic-frac", type=float, default=0.25)
ap.add_argument("--eicu-frac", type=float, default=0.25)
ap.add_argument("--mort-ckpt-dir", default=None)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()
rng = np.random.default_rng(1234)


def subsample(ss, n):
    if len(ss) <= n:
        return ss
    idx = rng.choice(len(ss), n, replace=False)
    return [ss[i] for i in sorted(idx)]


def permute_values(samples):
    """Permute observed values across stays, independently per variable (mask fixed)."""
    X = R.stack(samples, "values").copy()
    M = R.stack(samples, "mask").astype(bool)
    N, T, V = X.shape
    for v in range(V):
        cells = M[:, :, v]
        vals = X[:, :, v][cells]
        X[:, :, v][cells] = rng.permutation(vals)
    return X


def uniform_values(samples):
    X = R.stack(samples, "values").copy()
    M = R.stack(samples, "mask").astype(bool)
    X[M] = rng.random(M.sum())
    return X


rows = []
if args.task == "los":
    from sklearn.metrics import roc_auc_score  # noqa: F401
    pn = R.load_cache("physionet", args.pn_frac)
    site_a = [s for s in pn if s["site_id"] == 0]
    site_b = [s for s in pn if s["site_id"] == 1]
    mimic = R.load_cache("mimic", args.mimic_frac)
    eicu = R.load_cache("eicu", args.eicu_frac)
    _, va = R.split_pn_a(len(site_a), args.seed)
    sets = {"pn_a_val": subsample([site_a[i] for i in va], args.n_eval), "pn_b": subsample(site_b, args.n_eval),
            "mimic": subsample(mimic, args.n_eval), "eicu": subsample(eicu, args.n_eval)}
    for method in R.METHODS:
        ck = os.path.join(R.LOS_CKPT_DIR, f"{method}_s{args.seed}.pt")
        model = R.build_model("los_h", ck, args.device, seed=args.seed)
        for sname, ss in sets.items():
            y = np.array([float(s["los_h"]) for s in ss])
            variants = {"real": None, "permuted": permute_values(ss), "uniform": uniform_values(ss)}
            for vname, vals in variants.items():
                p = np.expm1(R.predict(model, ss, "los_h", device=args.device, values=vals))
                r2 = R.r2_score_raw(y, p)
                rows.append(dict(task="los", method=method, seed=args.seed, eval=sname, variant=vname, metric="R2",
                                 value=r2, n=len(ss)))
                print(f"{method} {sname:9s} {vname:9s} R2={r2:+.3f} n={len(ss)}", flush=True)
else:
    from sklearn.metrics import roc_auc_score
    mimic = R.load_cache("mimic", args.mimic_frac)
    eicu = R.load_cache("eicu", args.eicu_frac)
    data = {"mimic": mimic, "eicu": eicu}
    for method in R.METHODS:
        for src, tgt in [("mimic", "eicu"), ("eicu", "mimic")]:
            ck = os.path.join(args.mort_ckpt_dir, f"{method}_{src}to{tgt}_s{args.seed}.pt")
            model = R.build_model("mortality_hospital", ck, args.device, seed=args.seed)
            ss = subsample(data[tgt], args.n_eval)
            y = np.array([int(round(float(s["mortality_hospital"]))) for s in ss])
            variants = {"real": None, "permuted": permute_values(ss), "uniform": uniform_values(ss)}
            for vname, vals in variants.items():
                lg = R.predict(model, ss, "mortality_hospital", device=args.device, values=vals)
                auc = float(roc_auc_score(y, lg))
                rows.append(dict(task="mortality", method=method, seed=args.seed, eval=f"{src}->{tgt}", variant=vname,
                                 metric="AUROC", value=auc, n=len(ss)))
                print(f"{method} {src}->{tgt} {vname:9s} AUROC={auc:.3f} n={len(ss)}", flush=True)

with open(os.path.join(R.REV, f"g1_shuffle_{args.task}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
json.dump(rows, open(os.path.join(R.REV, f"g1_shuffle_{args.task}.json"), "w"), indent=1, default=float)
