"""
Extract bare encoder weights (raw state_dicts, ~13 MB each, no optimizer/history) from the original pretraining runs, ready to upload.

Output dir (gitignored): results/revision/encoders_slim/
  erm_pretrained.pt  pcl_pretrained.pt  dro_pretrained.pt          <- the three encoders the submitted paper used (seed 42, PCL lambda=1.0)
  sweep/lam{L}_p{S}.pt   L in {0.0,0.1,0.5,1.0,2.0,5.0}, S in {42,43,44}  <- independent pretraining seeds (lambda 0.0 == ERM)

Each slimmed file is verified tensor-for-tensor against the source. Usage: python slim_encoders.py [SRC_ROOT]
"""
import os
import sys

import torch

SRC = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/nikhi/Codes/IMPORTED/pclCodebase"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "encoders_slim")
os.makedirs(os.path.join(OUT, "sweep"), exist_ok=True)


def sd_of(p):
    o = torch.load(p, map_location="cpu", weights_only=False)
    return o["model_state"] if isinstance(o, dict) and "model_state" in o else o


def slim(src, dst):
    sd = {k: v.clone().contiguous() for k, v in sd_of(src).items()}
    torch.save(sd, dst)
    back = torch.load(dst, map_location="cpu", weights_only=False)
    ref = sd_of(src)
    assert back.keys() == ref.keys() and all(torch.equal(back[k], ref[k]) for k in ref), f"mismatch {src}"
    print(f"{os.path.getsize(dst) / 1e6:5.1f} MB  {os.path.relpath(dst, OUT)}")


base = {42: "results_lambda17", 43: "results_lambda17_s43", 44: "results_lambda17_s44"}
for m in ("erm", "pcl", "dro"):
    slim(os.path.join(SRC, base[42], "ckpt", f"{m}_pretrained.pt"), os.path.join(OUT, f"{m}_pretrained.pt"))
for s, d in base.items():
    for lam in ("0.0", "0.1", "0.5", "1.0", "2.0", "5.0"):
        slim(os.path.join(SRC, d, "ckpt", f"lambda_{lam}_pretrained.pt"), os.path.join(OUT, "sweep", f"lam{lam}_p{s}.pt"))
print("total", round(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs) / 1e6), "MB in", OUT)
