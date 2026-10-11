"""
Why do some PCL fine-tunes sit at AUROC exactly 0.500 from epoch 1?  Looks only at the pretrained ENCODERS (no fine-tuning).

For each encoder, on 4,000 MIMIC stays under the fixed T=48 protocol, take the pooled representation at hour 47 (what the head sees) and report
  - max |activation| and mean L2 norm (fp32)          -> huge values suggest fp16 overflow risk / saturated head inputs
  - fraction of stays with any non-finite value under fp16 autocast (what training uses on GPU)
  - across-stay spread of the representation (mean per-dimension std / mean per-dimension |mean|) -> ~0 means a collapsed, input-independent representation
  - a plain logistic-regression probe on standardized fp32 features (AUROC on 1,000 held-out stays) -> is mortality information present at all?
Compare encoders that fine-tuned fine (erm, dro) with the PCL encoder (and the lambda sweep encoders when present).

Usage (pod):  python results/revision/scripts/diag_pcl_collapse.py
"""
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import revlib as R
from src.baselines import fresh_model
from src.data.fixed_window import fix_window
from src.training.train_utils import load_state_dict_flexible

dev = "cuda" if torch.cuda.is_available() else "cpu"
pre = os.environ.get("PCL_LEGACY2_PRETRAIN_DIR", os.path.expanduser("~/pcl_pretrained"))
encs = {n: os.path.join(pre, f"{n}_pretrained.pt") for n in ("erm", "dro", "pcl")}
for n in ("lam0.0_p42", "lam0.5_p42", "lam1.0_p42", "lam0.5_p43", "lam1.0_p43", "lam0.5_p44", "lam1.0_p44"):
    p = os.path.join(pre, "sweep", f"{n}.pt")
    if os.path.exists(p):
        encs[n] = p
encs = {k: v for k, v in encs.items() if os.path.exists(v)}

samples = fix_window(R.load_cache("mimic", 1.0), 48, 48)
rng = np.random.default_rng(0)
idx = rng.choice(len(samples), 4000, replace=False)
S = [samples[i] for i in idx]
X = torch.as_tensor(np.stack([s["values"] for s in S]), dtype=torch.float32)
M = torch.as_tensor(np.stack([s["mask"] for s in S]), dtype=torch.bool)
y = np.array([int(round(float(s["mortality_hospital"]))) for s in S])
print(f"{len(S)} MIMIC stays (T=48 cohort), mortality prevalence {y.mean():.3f}, device {dev}\n")
print(f"{'encoder':12s} {'max|act|':>9s} {'mean L2':>8s} {'nonfinite fp16':>14s} {'spread':>8s} {'probe AUROC':>11s}")


def reps(model, autocast):
    out = []
    with torch.no_grad():
        for i in range(0, len(S), 256):
            x, m = X[i:i + 256].to(dev), M[i:i + 256].to(dev)
            with torch.amp.autocast("cuda", enabled=autocast and dev == "cuda"):
                r = model.encode(x, m)[:, 47]
            out.append(r.float().cpu())
    return torch.cat(out).numpy()


for name, path in encs.items():
    model = fresh_model(seed=0)
    model.load_state_dict(load_state_dict_flexible(path, "cpu"))
    model.to(dev).eval()
    r32, r16 = reps(model, False), reps(model, True)
    nonfinite = float((~np.isfinite(r16).all(axis=1)).mean())
    spread = float(np.mean(r32.std(axis=0)) / (np.mean(np.abs(r32.mean(axis=0))) + 1e-12))
    sc = StandardScaler().fit(r32[:3000])
    clf = LogisticRegression(max_iter=300, C=0.1).fit(sc.transform(r32[:3000]), y[:3000])
    auc = roc_auc_score(y[3000:], clf.predict_proba(sc.transform(r32[3000:]))[:, 1])
    print(f"{name:12s} {np.abs(r32).max():9.1f} {np.linalg.norm(r32, axis=1).mean():8.1f} {nonfinite:14.3f} {spread:8.3f} {auc:11.3f}")
