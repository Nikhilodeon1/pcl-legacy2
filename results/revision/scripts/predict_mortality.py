"""
Dump per-stay mortality logits of every fine-tuned mortality checkpoint on its in-domain validation split
and on the zero-shot target. Inference only. Run where the mortality ckpts + caches live (the pod).

Source-val split reproduces finetune_mortality.py exactly: make_patient_split_loaders ->
StratifiedShuffleSplit(test_size=0.2, random_state=seed) over the cached source samples, stratified on
mortality_hospital.

Usage (pod):
  python predict_mortality.py --ckpt-dir /workspace/.../pcl-legacy2/results/mortality/ckpt \
        --cache-dir /workspace/.../pcl-legacy2/results/mortality/cache --frac 1.0
Output: results/revision/preds/mort_{method}_{src}to{tgt}_s{seed}.npz with
  val_logit, val_true, val_pid, tgt_logit, tgt_true, tgt_pid   (pid = patient: subject_id for MIMIC, uniquepid for eICU)
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import revlib as R

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", required=True)
ap.add_argument("--cache-dir", required=True)
ap.add_argument("--frac", type=float, default=1.0)
ap.add_argument("--out-dir", default=os.path.join(R.REV, "preds"))
ap.add_argument("--eicu-dir", default=None, help="defaults to $EICU_DIR")
ap.add_argument("--methods", nargs="+", default=R.METHODS)
ap.add_argument("--seeds", nargs="+", type=int, default=R.SEEDS)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

TASK = "mortality_hospital"
data = {}
for site in ("mimic", "eicu"):
    with open(os.path.join(args.cache_dir, f"{site}_frac{args.frac}.pkl"), "rb") as f:
        data[site] = pickle.load(f)
stay2pid = R.eicu_stay2pid(args.eicu_dir) or {}


def pids(site, ss):
    if site == "mimic":
        return np.array([str(s["subject_id"]) for s in ss])
    return np.array([str(stay2pid.get(str(s["patient_id"]), f"unk{s['patient_id']}")) for s in ss])


def labels(ss):
    return np.array([int(round(float(s[TASK]))) for s in ss])


for method in args.methods:
    for src, tgt in (("mimic", "eicu"), ("eicu", "mimic")):
        for seed in args.seeds:
            out = os.path.join(args.out_dir, f"mort_{method}_{src}to{tgt}_s{seed}.npz")
            if os.path.exists(out):
                continue
            t0 = time.time()
            ck = os.path.join(args.ckpt_dir, f"{method}_{src}to{tgt}_s{seed}.pt")
            model = R.build_model(TASK, ck, args.device, seed=seed)
            ss = data[src]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            _, va = next(sss.split(np.arange(len(ss)), labels(ss)))
            val = [ss[i] for i in va]
            tg = data[tgt]
            np.savez_compressed(
                out,
                val_logit=R.predict(model, val, TASK, device=args.device), val_true=labels(val), val_pid=pids(src, val),
                tgt_logit=R.predict(model, tg, TASK, device=args.device), tgt_true=labels(tg), tgt_pid=pids(tgt, tg))
            print(f"{method} {src}->{tgt} s{seed} done ({time.time() - t0:.0f}s)", flush=True)
