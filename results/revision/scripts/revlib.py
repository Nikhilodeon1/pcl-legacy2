"""
Shared helpers for the revision analyses. Imports the project's own model/data code
(vendored in pcl-legacy2/src) so nothing is re-implemented.

Cache layout: results/revision/cache/{dataset}_frac{fraction}_s{seed}.pkl, built by
build_cache.py with the original loaders.
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
os.environ["PCL_TEST_MODE"] = "0"
sys.path.insert(0, ROOT)

HOURS = 48
LOS_CKPT_DIR = os.path.join(ROOT, "results", "los", "ckpt")
METHODS = ["erm", "pcl", "dro"]
SEEDS = [42, 43, 44]


def cache_dir():
    """PCL_CACHE_DIR (preferred) or PCL_LEGACY2_CACHE_DIR (the finetune scripts' cache) or results/revision/cache."""
    return os.environ.get("PCL_CACHE_DIR") or os.environ.get("PCL_LEGACY2_CACHE_DIR") or os.path.join(REV, "cache")


def load_cache(name, frac, seed=42, cache_dir=None):
    d = cache_dir or globals()["cache_dir"]()
    cands = [os.path.join(d, f"{name}_frac{frac}_s{seed}.pkl"), os.path.join(d, f"{name}_frac{frac}.pkl")]
    for p in cands:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError("no cache for %s frac=%s in %s (tried %s)" % (name, frac, d, [os.path.basename(c) for c in cands]))


def eicu_stay2pid(eicu_dir=None):
    """patientunitstayid -> uniquepid from raw eICU patient.csv.gz. Returns None if the raw table is not on this machine
    (caller then falls back to stay-level clusters and must say so)."""
    import pandas as pd
    d = eicu_dir or os.environ.get("EICU_DIR")
    f = os.path.join(d, "patient.csv.gz") if d else None
    if not f or not os.path.exists(f):
        print("[WARN] eICU patient.csv.gz not found (set EICU_DIR): using stay-level clusters for eICU")
        return None
    pat = pd.read_csv(f, usecols=["patientunitstayid", "uniquepid"])
    return dict(zip(pat.patientunitstayid.astype(str), pat.uniquepid))


def split_pn_a(n, seed):
    """Reproduces finetune_los.py's source split exactly (80/20, default_rng(seed).permutation)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)
    return idx[:n_train], idx[n_train:]


def stack(samples, key):
    return np.stack([np.asarray(s[key]) for s in samples])


def mask_features(samples):
    """Features computable from the observation mask alone (no values).

    Returns (X, names). Includes observed-hours count, last/first observed hour,
    trailing padding length, window-full flag, total observed cells, and per-variable
    observed-hour counts.
    """
    m = stack(samples, "mask").astype(bool)          # (N, T, V)
    t_obs = m.any(axis=2)                            # (N, T)
    T = t_obs.shape[1]
    n_obs_h = t_obs.sum(1)
    last = np.where(t_obs.any(1), T - 1 - np.argmax(t_obs[:, ::-1], axis=1), 0)
    first = np.where(t_obs.any(1), np.argmax(t_obs, axis=1), 0)
    pad = T - 1 - last
    full = (last == T - 1).astype(int)
    cells = m.sum((1, 2))
    per_var = m.sum(1)                               # (N, V)
    X = np.column_stack([n_obs_h, last, first, pad, full, cells, per_var])
    names = ["n_obs_hours", "last_obs_hour", "first_obs_hour", "pad_len", "window_full", "n_obs_cells"] + \
            [f"obs_count_v{i}" for i in range(per_var.shape[1])]
    return X.astype(float), names


def length_only_features(samples):
    X, names = mask_features(samples)
    return X[:, :4], names[:4]


def build_model(task, ckpt_path, device="cpu", seed=42):
    from src.baselines import fresh_model
    from src.training.train_utils import load_state_dict_flexible
    model = fresh_model(seed=seed)
    model.add_classification_head(task)
    model.load_state_dict(load_state_dict_flexible(ckpt_path, device))
    model.to(device).eval()
    return model


def predict(model, samples, task, batch_size=256, device="cpu", values=None, with_reps=False):
    """Model outputs for each sample (log-space for LOS, logit for mortality).

    values: optional (N, T, V) array replacing sample['values'] (mask unchanged).
    with_reps: also return mean-pooled encoder representation over observed timesteps.
    """
    import torch
    N = len(samples)
    X = values if values is not None else stack(samples, "values")
    M = stack(samples, "mask")
    outs, reps = [], []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            m = torch.as_tensor(M[i:i + batch_size], dtype=torch.bool, device=device)
            r = model.encode(x, m)
            p = model.classify(r, task, obs_mask=m).squeeze(-1)
            outs.append(p.float().cpu().numpy())
            if with_reps:
                t_obs = m.any(-1).float().unsqueeze(-1)
                reps.append(((r * t_obs).sum(1) / t_obs.sum(1).clamp(min=1)).cpu().numpy())
    out = np.concatenate(outs)
    return (out, np.concatenate(reps)) if with_reps else out


def r2_score_raw(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
