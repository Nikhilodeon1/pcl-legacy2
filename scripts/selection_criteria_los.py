"""
pcl-legacy2, Step 4 — LOS task, selection-criteria comparison.

Single-source setup (unlike mortality's both-directions): PhysioNet-A is the
only source, physionet_b/mimic/eicu are all targets. Cell = (method, seed,
target) = 3 methods x 3 seeds x 3 targets = 27 cells, using the 9 already
fine-tuned checkpoints from finetune_los.py.

Only 4 of the 6 mortality signals apply here: violation, recon_mse,
repr_dist, mmd. Entropy (defined on sigmoid(logits), a probability) and ATC
(defined via a binary correct/incorrect notion calibrated to accuracy) have
no standard non-hacky continuous-target analog — reported as N/A rather than
forced. MMD is still in: it's the actually-required new baseline that does
transfer.

True performance metric correlated against: R^2 (higher is better, same role
AUROC played for mortality — MAE/RMSE are also in each cell for reference).

Computes BOTH pooled and within-target Spearman correlation from the start
this time — mortality's first pass pooled across its 2 directions and got
misleadingly wrong signs on 3 of 6 signals; no reason to make that mistake
twice when the target dimension here could plausibly show the same pattern
(physionet_b is near-domain, mimic/eicu are far — very different score
scales likely).

Usage (needs the 9 finetune_los.py result JSONs + checkpoints, and the
cached physionet/mimic/eicu data — all already built):
    python selection_criteria_los.py
"""
import argparse
import glob
import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

os.environ["PCL_TEST_MODE"] = "0"

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEGACY2_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _LEGACY2_ROOT)  # config.py, src/, pod_monitor.py vendored here

RESULTS_DIR = os.path.join(_LEGACY2_ROOT, "results", "los")
FT_CKPT_DIR = os.path.join(RESULTS_DIR, "ckpt")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
OUT_PATH = os.path.join(RESULTS_DIR, "selection_criteria.json")

TASK = "los_h"
TARGETS = ("physionet_b", "mimic", "eicu")
SIGNALS = ["violation", "recon_mse", "repr_dist", "mmd"]  # all lower-is-better


# ── pure helpers, same as selection_criteria_mortality.py — copied, not
# imported (keeps this script standalone; both are small enough it's not
# worth a shared-module refactor under time pressure) ──────────────────────
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")

    def rank(v):
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        for u in np.unique(v):
            idx = np.where(v == u)[0]
            if len(idx) > 1:
                r[idx] = r[idx].mean()
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def regret(scores_by_key, perf_by_key, lower_is_better=True):
    keys = [k for k in scores_by_key if scores_by_key[k] is not None and perf_by_key.get(k) is not None]
    if not keys:
        return float("nan"), None
    pick = (min if lower_is_better else max)(keys, key=lambda k: scores_by_key[k])
    best = max(keys, key=lambda k: perf_by_key[k])
    return perf_by_key[pick] - perf_by_key[best], pick


def _rbf_mmd(x, y, n_sub=500, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) > n_sub:
        x = x[rng.choice(len(x), n_sub, replace=False)]
    if len(y) > n_sub:
        y = y[rng.choice(len(y), n_sub, replace=False)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")

    def _sqdist(a, b):
        return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)

    both = np.concatenate([x, y], axis=0)
    d2 = _sqdist(both, both)
    bandwidth = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    bandwidth = max(bandwidth, 1e-6)

    def _k(a, b):
        return np.exp(-_sqdist(a, b) / bandwidth)

    kxx = _k(x, x); kyy = _k(y, y); kxy = _k(x, y)
    m, n = len(x), len(y)
    txx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) if m > 1 else 0.0
    tyy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) if n > 1 else 0.0
    txy = kxy.mean()
    return float(max(txx + tyy - 2 * txy, 0.0))


# ── model-dependent scoring ─────────────────────────────────────────────────
def score_checkpoint(model, src_loader, tgt_loader, device, seed=0):
    import torch
    from src.losses.pcl_loss import PhysiologicalConstraintLoss
    from src.models.backbone import apply_random_mask
    from src.training.train_utils import masked_prediction_loss
    from config import MASK_PROB

    pcl_loss = PhysiologicalConstraintLoss().to(device)
    model.eval()

    @torch.no_grad()
    def _pass(loader):
        viol_sum, viol_n = 0.0, 0
        mse_sum, mse_n = 0.0, 0
        rep_sum, rep_n = None, 0
        for bi, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            c = batch["c_mask"].to(device, non_blocking=True)

            torch.manual_seed(seed * 100003 + bi)
            x_masked, pretrain_mask = apply_random_mask(x, m, MASK_PROB)
            reps_masked = model.encode(x_masked, m)
            preds = model.predict(reps_masked)
            out = pcl_loss(preds.float(), c, pretrain_mask)
            parts = [out["losses"][k].item() for k in ("MAP", "HH", "SpO2") if out["active"][k] > 0]
            if parts:
                viol_sum += float(np.mean(parts)) * x.shape[0]
                viol_n += x.shape[0]
            mse = masked_prediction_loss(preds, x, pretrain_mask)
            if torch.isfinite(mse):
                mse_sum += float(mse.item()) * x.shape[0]
                mse_n += x.shape[0]

            reps_full = model.encode(x, m)
            rp = reps_full.mean(dim=1).sum(dim=0).double()
            rep_sum = rp if rep_sum is None else rep_sum + rp
            rep_n += reps_full.shape[0]

        return {
            "violation": viol_sum / max(viol_n, 1),
            "recon_mse": mse_sum / max(mse_n, 1),
            "centroid": (rep_sum / rep_n) if rep_n else None,
        }

    @torch.no_grad()
    def _pooled_reps(loader, n_sub=500):
        out, n_total = [], 0
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            reps = model.encode(x, m).mean(dim=1).cpu().numpy()
            out.append(reps)
            n_total += len(reps)
            if n_total >= n_sub * 3:
                break
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1))

    src = _pass(src_loader)
    tgt = _pass(tgt_loader)

    repr_dist = float("nan")
    if src["centroid"] is not None and tgt["centroid"] is not None:
        repr_dist = float(torch.linalg.norm(tgt["centroid"] - src["centroid"]).item())

    src_reps = _pooled_reps(src_loader)
    tgt_reps = _pooled_reps(tgt_loader)
    mmd = _rbf_mmd(src_reps, tgt_reps, seed=seed)

    return {
        "violation": tgt["violation"],
        "recon_mse": tgt["recon_mse"],
        "repr_dist": repr_dist,
        "mmd": mmd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=1.0)
    args = ap.parse_args()

    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    result_files = [f for f in result_files if os.path.basename(f) != "selection_criteria.json"]
    if len(result_files) != 9:
        logging.warning(f"Expected 9 fine-tune result JSONs, found {len(result_files)} in {RESULTS_DIR}.")

    import torch
    from torch.utils.data import DataLoader
    from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TEST_MODE, D_MODEL, N_LAYERS
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset
    from src.training.train_utils import load_state_dict_flexible

    if TEST_MODE or D_MODEL != 256 or N_LAYERS != 6:
        raise RuntimeError(f"TEST_MODE={TEST_MODE} D_MODEL={D_MODEL} N_LAYERS={N_LAYERS} — see finetune_los.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _cached_site(name):
        cache_path = os.path.join(CACHE_DIR, f"{name}_frac{args.fraction}.pkl")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"{cache_path} missing — run finetune_los.py first so all three are cached.")
        import pickle
        with open(cache_path, "rb") as fp:
            return pickle.load(fp)

    pn_samples = _cached_site("physionet")
    mimic_samples = _cached_site("mimic")
    eicu_samples = _cached_site("eicu")
    site_a = [s for s in pn_samples if s["site_id"] == 0]
    site_b = [s for s in pn_samples if s["site_id"] == 1]
    logging.info(f"[CACHE] physionet-A: {len(site_a)}  physionet-B: {len(site_b)}  "
                 f"mimic: {len(mimic_samples)}  eicu: {len(eicu_samples)}")

    src_loader = DataLoader(ICUDataset(site_a), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    target_samples = {"physionet_b": site_b, "mimic": mimic_samples, "eicu": eicu_samples}
    target_loaders = {name: DataLoader(ICUDataset(s), batch_size=BATCH_SIZE, shuffle=False,
                                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                       for name, s in target_samples.items() if s}

    cells = []
    for f in result_files:
        with open(f) as fp:
            d = json.load(fp)
        method, seed = d["method"], d["seed"]
        ckpt_path = os.path.join(FT_CKPT_DIR, f"{method}_s{seed}.pt")
        model = fresh_model(seed=seed).to(device)
        model.add_classification_head(TASK)
        model.load_state_dict(load_state_dict_flexible(ckpt_path, device))

        for target, tgt_loader in target_loaders.items():
            if target not in d["ood"] or d["ood"][target].get("r2") is None:
                logging.warning(f"{f}: no valid r2 for target '{target}', skipping cell")
                continue
            true_r2 = d["ood"][target]["r2"]
            true_mae = d["ood"][target]["mae_hours"]

            scores = score_checkpoint(model, src_loader, tgt_loader, device, seed=seed)
            cell = {"method": method, "seed": seed, "target": target,
                    "true_r2": true_r2, "true_mae_hours": true_mae, **scores}
            cells.append(cell)
            logging.info(f"  {method:4s} s{seed} -> {target:12s}: "
                         f"viol={scores['violation']:.5f} recon={scores['recon_mse']:.5f} "
                         f"repr_dist={scores['repr_dist']:.3f} mmd={scores['mmd']:.5f} "
                         f"true_r2={true_r2:.4f}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── pooled + within-target Spearman (both, from the start — see docstring) ──
    def _rho(sig, subset):
        cs = [c for c in subset if c[sig] == c[sig]]
        return spearman([c[sig] for c in cs], [c["true_r2"] for c in cs])

    rhos_pooled = {s: _rho(s, cells) for s in SIGNALS}
    rhos_within_target = {t: {s: _rho(s, [c for c in cells if c["target"] == t]) for s in SIGNALS}
                           for t in TARGETS}

    # ── selection regret per target, averaged over seeds per method ─────────
    regret_by_target = {}
    for target in TARGETS:
        tcells = [c for c in cells if c["target"] == target]
        if not tcells:
            continue
        methods = sorted(set(c["method"] for c in tcells))
        perf_by_method = {m: float(np.mean([c["true_r2"] for c in tcells if c["method"] == m])) for m in methods}
        entry = {}
        for sig in SIGNALS:
            score_by_method = {m: float(np.mean([c[sig] for c in tcells if c["method"] == m and c[sig] == c[sig]]))
                                for m in methods}
            r, picked = regret(score_by_method, perf_by_method, lower_is_better=True)
            entry[sig] = {"regret": r, "picked": picked}
        entry["best_method"] = max(perf_by_method, key=lambda m: perf_by_method[m])
        entry["perf_spread"] = max(perf_by_method.values()) - min(perf_by_method.values())
        regret_by_target[target] = entry

    out = {
        "n_cells": len(cells), "signals": SIGNALS,
        "signals_not_applicable": ["entropy", "atc"],
        "spearman_pooled": rhos_pooled,
        "spearman_within_target": rhos_within_target,
        "regret_by_target": regret_by_target,
        "cells": cells,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"SELECTION CRITERIA — LOS  ({len(cells)} cells)")
    print("=" * 78)
    print("entropy/atc: N/A for continuous targets, not computed (see script docstring).\n")
    print("Spearman rho vs true R^2 (all 4 signals are lower-is-better; negative rho = good):")
    print(f"{'signal':<12}{'pooled':>10}" + "".join(f"{t:>16}" for t in TARGETS))
    for s in SIGNALS:
        row = f"{s:<12}{rhos_pooled[s]:>10.3f}"
        row += "".join(f"{rhos_within_target[t][s]:>16.3f}" for t in TARGETS)
        print(row)
    print("\nSelection regret by target (0 = perfect; perf_spread shows how much there even was to pick between):")
    for target, entry in regret_by_target.items():
        print(f"  {target}  (best={entry['best_method']}, true-R2 spread={entry['perf_spread']:.4f})")
        for sig in SIGNALS:
            r = entry[sig]
            print(f"    {sig:<12} regret={r['regret']:+.4f}  picked={r['picked']}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
