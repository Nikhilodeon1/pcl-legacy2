"""
A4 — selection analysis. Observation unit = ONE CHECKPOINT (method x fine-tune seed).

For every domain (LOS: target in {physionet_b, mimic, eicu}; mortality: direction) and for the
pooled set, and every signal:
  - n (printed/stored with every correlation)
  - Spearman rho, permutation p (100k, two-sided), Kendall tau-b, leave-one-seed-out range of rho
  - selection regret and top-1 hit rate, two definitions:
      "ckpt"        : choose 1 of all checkpoints in the domain (n = 9 for LOS)
      "within_seed" : for each seed choose 1 of 3 methods (the original study's definition), averaged
  - baselines: source-validation metric (in-domain R2 / AUROC), random (expected value), oracle (0)

Signals are all lower-is-better except source_val (higher-is-better) and ATC (higher-is-better).

Usage: python a4_selection.py --task los
       python a4_selection.py --task mortality --sel-json <path to mortality selection_criteria.json>
Outputs: a4_selection_{task}.csv, a4_selection_{task}.json
"""
import argparse
import csv
import itertools
import json
import os

import numpy as np
from scipy.stats import kendalltau, spearmanr

import revlib as R

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["los", "mortality"], default="los")
ap.add_argument("--sel-json", default=None)
ap.add_argument("--n-perm", type=int, default=100000)
args = ap.parse_args()
rng = np.random.default_rng(99)
RESULTS = os.path.join(R.ROOT, "results", args.task)

sel = json.load(open(args.sel_json or os.path.join(RESULTS, "selection_criteria.json")))
cells = sel["cells"]
print("cell keys:", list(cells[0].keys()))

if args.task == "los":
    domain_key = "target"
    perf_key = "true_r2"
    signals = {"violation": -1, "recon_mse": -1, "repr_dist": -1, "mmd": -1}      # sign: +1 higher-better, -1 lower-better
    src_val = {}
    for m in R.METHODS:
        for s in R.SEEDS:
            d = json.load(open(os.path.join(RESULTS, f"{m}_s{s}.json")))
            src_val[(m, s)] = d["in_domain"]["r2"]
    for c in cells:
        c["source_val"] = src_val[(c["method"], c["seed"])]
else:
    perf_key = "true_auroc" if "true_auroc" in cells[0] else [k for k in cells[0] if "auroc" in k][0]
    domain_key = "direction" if "direction" in cells[0] else None
    cand = ["violation", "recon_mse", "repr_dist", "mmd", "entropy", "atc"]
    signals = {k: (+1 if k == "atc" else -1) for k in cand if k in cells[0]}
    ivk = [k for k in cells[0] if "in_domain" in k or "val_auroc" in k or "source_val" in k]
    if ivk:
        for c in cells:
            c["source_val"] = c[ivk[0]]
signals_all = dict(signals)
if "source_val" in cells[0]:
    signals_all["source_val"] = +1

domains = sorted({c[domain_key] for c in cells}) if domain_key else ["all"]


def sub(domain):
    return [c for c in cells if (domain_key is None or c[domain_key] == domain)] if domain != "pooled" else cells


def corr_block(xs, ys, sign):
    """Spearman with sign so that positive = criterion is 'correct' (score ordering matches performance)."""
    x = sign * np.asarray(xs, float)
    y = np.asarray(ys, float)
    rho = spearmanr(x, y).statistic
    tau = kendalltau(x, y, variant="b").statistic
    null = np.empty(args.n_perm)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx = np.argsort(np.argsort(x)).astype(float)
    rxc = rx - rx.mean()
    ryc = ry - ry.mean()
    den = np.sqrt((rxc ** 2).sum() * (ryc ** 2).sum())
    for i in range(args.n_perm):
        null[i] = (rxc * rng.permutation(ryc)).sum() / den
    p = float((np.sum(np.abs(null) >= abs(rho) - 1e-12) + 1) / (args.n_perm + 1))
    return rho, tau, p


def regret_hit(cs, sig, sign, perf):
    """Returns dict for both definitions."""
    out = {}
    # checkpoint level
    scores = np.array([sign * c[sig] for c in cs])
    perfs = np.array([c[perf] for c in cs])
    pick = int(np.argmax(scores))
    out["ckpt_regret"] = float(perfs.max() - perfs[pick])
    out["ckpt_hit"] = float(perfs[pick] == perfs.max())
    out["ckpt_random_regret"] = float(np.mean(perfs.max() - perfs))
    out["ckpt_random_hit"] = float(1.0 / len(cs))
    # within seed among methods
    regs, hits, rregs = [], [], []
    for s in sorted({c["seed"] for c in cs}):
        g = [c for c in cs if c["seed"] == s]
        sc = np.array([sign * c[sig] for c in g]); pf = np.array([c[perf] for c in g])
        k = int(np.argmax(sc))
        regs.append(pf.max() - pf[k]); hits.append(float(pf[k] == pf.max())); rregs.append(np.mean(pf.max() - pf))
    out["seed_regret"] = float(np.mean(regs)); out["seed_hit"] = float(np.mean(hits))
    out["seed_random_regret"] = float(np.mean(rregs)); out["seed_random_hit"] = float(1.0 / 3)
    return out


rows = []
for dom in domains + ["pooled"]:
    cs = sub(dom)
    if dom == "pooled" and len(domains) == 1:
        continue
    seeds = sorted({c["seed"] for c in cs})
    for sig, sign in signals_all.items():
        xs = [c[sig] for c in cs]; ys = [c[perf_key] for c in cs]
        rho, tau, p = corr_block(xs, ys, sign)
        loo = []
        for s in seeds:
            cc = [c for c in cs if c["seed"] != s]
            loo.append(spearmanr(sign * np.array([c[sig] for c in cc]), [c[perf_key] for c in cc]).statistic)
        rh = regret_hit(cs, sig, sign, perf_key) if dom != "pooled" else {}
        rows.append(dict(domain=dom, signal=sig, n=len(cs), higher_is_better=(sign > 0),
                         rho_correct_positive=float(rho), perm_p=p, kendall_tau_b=float(tau),
                         loso_min=float(min(loo)), loso_max=float(max(loo)), **rh))
        print(f"{dom:12s} {sig:10s} n={len(cs):2d} rho(+correct)={rho:+.3f} p={p:.4f} tau={tau:+.3f} "
              f"LOSO[{min(loo):+.2f},{max(loo):+.2f}]" +
              (f" regret(ckpt)={rh['ckpt_regret']:.4f} hit={rh['ckpt_hit']:.0f} | within-seed regret={rh['seed_regret']:.4f}" if rh else ""))

# oracle row per domain
for dom in domains:
    rows.append(dict(domain=dom, signal="oracle", n=len(sub(dom)), ckpt_regret=0.0, ckpt_hit=1.0, seed_regret=0.0, seed_hit=1.0))

fn = f"a4_selection_{args.task}"
with open(os.path.join(R.REV, fn + ".csv"), "w", newline="") as f:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("domain", "signal", "n"), k))
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader(); w.writerows(rows)
json.dump(rows, open(os.path.join(R.REV, fn + ".json"), "w"), indent=1, default=float)
