"""
Dump per-stay LOS predictions (raw hours) of every fine-tuned checkpoint on every eval set.
Inference only. Foundation for A1 (label-support diagnostics), A3 (bootstrap), A4 (selection).

Eval sets: pn_a_val (that seed's 20% PhysioNet-A split), pn_b, mimic, eicu.
Output: results/revision/preds/los_{method}_s{seed}.npz with, per eval set e:
    {e}_pred (hours), {e}_true (hours), {e}_pid (patient id string; PhysioNet = file id)

Usage: python predict_los.py [--mimic-frac 0.25 --eicu-frac 0.25] [--methods erm pcl dro] [--seeds 42 43 44]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import revlib as R

ap = argparse.ArgumentParser()
ap.add_argument("--pn-frac", type=float, default=1.0)
ap.add_argument("--mimic-frac", type=float, default=0.25)
ap.add_argument("--eicu-frac", type=float, default=0.25)
ap.add_argument("--methods", nargs="+", default=R.METHODS)
ap.add_argument("--seeds", nargs="+", type=int, default=R.SEEDS)
ap.add_argument("--ckpt-dir", default=R.LOS_CKPT_DIR)
ap.add_argument("--out-dir", default=os.path.join(R.REV, "preds"))
ap.add_argument("--eicu-dir", default=None, help="defaults to $EICU_DIR")
ap.add_argument("--device", default="cpu")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

pn = R.load_cache("physionet", args.pn_frac)
site_a = [s for s in pn if s["site_id"] == 0]
site_b = [s for s in pn if s["site_id"] == 1]
def _try(name, frac):
    try:
        return R.load_cache(name, frac)
    except FileNotFoundError:
        print(f"[skip] no cache for {name}")
        return []


mimic = _try("mimic", args.mimic_frac)
eicu = _try("eicu", args.eicu_frac)

stay2pid = R.eicu_stay2pid(args.eicu_dir) or {}


def pids(name, ss):
    if name == "mimic":
        return np.array([str(s["subject_id"]) for s in ss])
    if name == "eicu":
        return np.array([str(stay2pid.get(str(s["patient_id"]), f"unk{s['patient_id']}")) for s in ss])
    return np.array([str(s["patient_id"]) for s in ss])


def true_h(ss):
    return np.array([float(s["los_h"]) for s in ss])


for method in args.methods:
    for seed in args.seeds:
        out = os.path.join(args.out_dir, f"los_{method}_s{seed}.npz")
        if os.path.exists(out):
            print("exists", out)
            continue
        t0 = time.time()
        model = R.build_model("los_h", os.path.join(args.ckpt_dir, f"{method}_s{seed}.pt"), args.device, seed=seed)
        _, va = R.split_pn_a(len(site_a), seed)
        sets = {"pn_a_val": [site_a[i] for i in va], "pn_b": site_b, "mimic": mimic, "eicu": eicu}
        sets = {k: v for k, v in sets.items() if len(v)}
        arrays = {}
        for name, ss in sets.items():
            arrays[f"{name}_pred"] = np.expm1(R.predict(model, ss, "los_h", device=args.device))
            arrays[f"{name}_true"] = true_h(ss)
            arrays[f"{name}_pid"] = pids(name, ss)
        np.savez_compressed(out, **arrays)
        r2 = {n: R.r2_score_raw(arrays[f"{n}_true"], arrays[f"{n}_pred"]) for n in sets}
        print(f"{method} s{seed}: " + " ".join(f"{n}={v:+.3f}" for n, v in r2.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
