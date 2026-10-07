"""
A6 — pooling demonstration.

(a) Within-domain shuffle null. Within each target, permute true performance across the
    checkpoints of that target (breaks any method/criterion link, keeps between-domain structure),
    recompute the POOLED Spearman 10,000 times. Compare observed pooled rho to the null band.
    PREREG H5: pooled rho for repr_dist and MMD lies inside the 95% null band.
(b) Simulation. D domains x M methods x K seeds; criterion = domain offset + noise;
    performance = domain offset + small method effect + noise; sweep between/within variance
    ratio; report pooled vs mean within-domain rho (criterion with NO within-domain signal and
    criterion WITH within-domain signal).

Outputs: a6_pooling_null.csv/json, a6_pooling_sim.csv, a6_pooling_sim.png
"""
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr

import revlib as R

rng = np.random.default_rng(314)
sel = json.load(open(os.path.join(R.ROOT, "results", "los", "selection_criteria.json")))
cells = sel["cells"]
T = ["physionet_b", "mimic", "eicu"]
SIG = ["violation", "recon_mse", "repr_dist", "mmd"]
B = 10000

perf = np.array([c["true_r2"] for c in cells])
dom = np.array([c["target"] for c in cells])
idx_by = {t: np.where(dom == t)[0] for t in T}

rows = []
for sg in SIG:
    x = np.array([c[sg] for c in cells], float)
    obs = spearmanr(x, perf).statistic
    null = np.empty(B)
    for b in range(B):
        p = perf.copy()
        for t in T:
            p[idx_by[t]] = rng.permutation(p[idx_by[t]])
        null[b] = spearmanr(x, p).statistic
    lo, hi = np.percentile(null, [2.5, 97.5])
    within_obs = {t: float(spearmanr(x[idx_by[t]], perf[idx_by[t]]).statistic) for t in T}
    rows.append(dict(signal=sg, pooled_obs=float(obs), null_mean=float(null.mean()), null_lo=float(lo), null_hi=float(hi),
                     inside_band=bool(lo <= obs <= hi), frac_null_abs_ge_obs=float(np.mean(np.abs(null) >= abs(obs))),
                     within_physionet_b=within_obs["physionet_b"], within_mimic=within_obs["mimic"], within_eicu=within_obs["eicu"]))
    print(f"{sg:10s} pooled={obs:+.3f}  null95=[{lo:+.3f},{hi:+.3f}]  inside={lo <= obs <= hi}  P(|null|>=|obs|)={np.mean(np.abs(null) >= abs(obs)):.4f}")

with open(os.path.join(R.REV, "a6_pooling_null.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
h5 = {r["signal"]: r["inside_band"] for r in rows if r["signal"] in ("repr_dist", "mmd")}
json.dump(dict(rows=rows, H5_inside_band=h5, H5_met=bool(all(h5.values())), n_shuffles=B), open(os.path.join(R.REV, "a6_pooling_null.json"), "w"), indent=1)
print("H5 (repr_dist & MMD pooled rho inside shuffle-null band):", all(h5.values()))

# ---------------- (b) simulation ----------------
def simulate(ratio, D=3, M=3, K=3, signal=0.0, reps=400, method_sd=0.01, noise_sd=0.005, domain_sign=+1.0):
    """ratio = between-domain sd / within-domain sd of performance. signal in [0,1]: how much the criterion
    tracks the within-domain (method) performance component. domain_sign=+1: criterion is HIGHER in
    higher-performing domains (wrong-signed for a lower-is-better criterion, as repr_dist is in the LOS data)."""
    pooled, within = [], []
    for _ in range(reps):
        dom_off = rng.normal(0, ratio * method_sd, D)
        meth_eff = rng.normal(0, method_sd, M)
        perf, crit, dd = [], [], []
        for d in range(D):
            for m in range(M):
                for k in range(K):
                    p = dom_off[d] + meth_eff[m] + rng.normal(0, noise_sd)
                    c = domain_sign * dom_off[d] + signal * (-meth_eff[m]) + rng.normal(0, method_sd)
                    perf.append(p); crit.append(c); dd.append(d)
        perf, crit, dd = map(np.array, (perf, crit, dd))
        pooled.append(spearmanr(crit, perf).statistic)
        within.append(np.mean([spearmanr(crit[dd == d], perf[dd == d]).statistic for d in range(D)]))
    return float(np.mean(pooled)), float(np.mean(within))

sim = []
for ratio in [0.1, 0.3, 1, 3, 10, 30]:
    for sig in [0.0, 1.0]:
        pr, wi = simulate(ratio, signal=sig)
        sim.append(dict(between_to_within_ratio=ratio, criterion_has_within_signal=bool(sig), pooled_rho_mean=pr, within_rho_mean=wi))
        print(f"ratio={ratio:5.1f} within_signal={sig:.0f}  pooled={pr:+.3f} within={wi:+.3f}")
with open(os.path.join(R.REV, "a6_pooling_sim.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sim[0].keys())); w.writeheader(); w.writerows(sim)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for sig, ls in [(False, "--"), (True, "-")]:
        xs = [s["between_to_within_ratio"] for s in sim if s["criterion_has_within_signal"] == sig]
        ax.plot(xs, [s["pooled_rho_mean"] for s in sim if s["criterion_has_within_signal"] == sig], ls, marker="o", label=f"pooled, within-signal={sig}")
        ax.plot(xs, [s["within_rho_mean"] for s in sim if s["criterion_has_within_signal"] == sig], ls, marker="s", alpha=.5, label=f"within, within-signal={sig}")
    ax.set_xscale("log"); ax.set_xlabel("between-domain / within-domain sd"); ax.set_ylabel("mean Spearman rho"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(R.REV, "a6_pooling_sim.png"), dpi=200)
except Exception as e:
    print("plot skipped:", e)
