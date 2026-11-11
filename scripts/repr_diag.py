"""
Representation diagnostics for a pretrained encoder, on unlabeled stays (inputs only).

Why: PCL encoders were found to carry a large near-constant component in the pooled representation (mean L2 norm ~165-178 vs ~103 for ERM,
relative across-stay spread 0.07 vs 0.29), and PCL fine-tunes were unstable (some collapsed to AUROC 0.50). These numbers let a pretraining run
be checked for that pattern without any labels or fine-tuning.

For each requested hour index h (the head reads the encoder state at one hour; 23 for the T=24 protocol, 47 for T=48):
  mean_l2        mean over stays of ||rep||_2
  max_abs        max |activation|
  spread         mean over dimensions of the across-stay std, divided by mean over dimensions of |across-stay mean| (small = near-constant)
  const_frac     squared norm of the across-stay mean vector / mean squared norm (share of the representation that is the same for every stay)
  nonfinite_fp16 fraction of stays with any non-finite value when run under fp16 autocast (GPU only; 0.0 on CPU)
Everything is computed in fp32 except nonfinite_fp16.

Library use:  from repr_diag import diagnose
CLI:          python scripts/repr_diag.py --ckpt path/to/encoder.pt [--ckpt ...] [--n 4000] [--hours 23 47]
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def _reps(model, X, M, h, device, autocast, batch=256):
    import torch
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x, m = X[i:i + batch].to(device), M[i:i + batch].to(device)
            with torch.amp.autocast("cuda", enabled=bool(autocast and device == "cuda")):
                r = model.encode(x, m)[:, h]
            out.append(r.float().cpu())
    return torch.cat(out).numpy()


def diagnose(model, samples, device="cpu", hours=(23, 47), n=4000, seed=0):
    """samples: list of dicts with 'values' (48 x V) and 'mask'. Returns {str(h): {...}}."""
    import torch
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), min(n, len(samples)), replace=False)
    X = torch.as_tensor(np.stack([np.asarray(samples[i]["values"]) for i in idx]), dtype=torch.float32)
    M = torch.as_tensor(np.stack([np.asarray(samples[i]["mask"]) for i in idx]), dtype=torch.bool)
    model = model.to(device)
    res = {}
    def _erank(mat):
        s = np.linalg.svd(mat, compute_uv=False)
        s = s[s > 1e-12]
        p = s / s.sum()
        return float(np.exp(-(p * np.log(p)).sum()))

    for h in hours:
        r32 = _reps(model, X, M, h, device, autocast=False)
        r16 = _reps(model, X, M, h, device, autocast=True)
        mean_vec = r32.mean(axis=0)
        rc = r32 - mean_vec[None, :]
        res[str(h)] = dict(
            n=int(len(idx)),
            mean_l2=float(np.linalg.norm(r32, axis=1).mean()),
            centered_l2=float(np.linalg.norm(rc, axis=1).mean()),
            max_abs=float(np.abs(r32).max()),
            spread=float(r32.std(axis=0).mean() / (np.abs(mean_vec).mean() + 1e-12)),
            const_frac=float((mean_vec ** 2).sum() / ((r32 ** 2).sum(axis=1).mean() + 1e-12)),
            erank_raw=_erank(r32),
            erank_cent=_erank(rc),
            nonfinite_fp16=float((~np.isfinite(r16).all(axis=1)).mean()),
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--hours", type=int, nargs="+", default=[23, 47])
    ap.add_argument("--cache-dir", default=os.environ.get("PCL_LEGACY2_CACHE_DIR"))
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    args = ap.parse_args()
    import torch
    os.environ.setdefault("PCL_TEST_MODE", "0")
    from src.baselines import fresh_model
    from src.training.train_utils import load_state_dict_flexible
    import pickle
    pn = pickle.load(open(os.path.join(args.cache_dir, "physionet_frac1.0.pkl"), "rb"))
    samples = [s for s in pn if s["site_id"] == 0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    allres = {}
    for p in args.ckpt:
        model = fresh_model(seed=0)
        model.load_state_dict(load_state_dict_flexible(p, "cpu"))
        allres[p] = diagnose(model, samples, device, tuple(args.hours), args.n)
        for h, d in allres[p].items():
            print(f"{os.path.basename(p):40s} h={h:>2s} L2={d['mean_l2']:7.1f} max|a|={d['max_abs']:6.1f} spread={d['spread']:.3f} "
                  f"const_frac={d['const_frac']:.3f} nonfinite_fp16={d['nonfinite_fp16']:.3f}")
    if args.out:
        json.dump(allres, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
