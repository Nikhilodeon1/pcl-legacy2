"""
Measure empirical gradient cosine similarity between L_masked and L_pcl
plus gradient-norm ratio on Site A minibatches for pretrained sweep checkpoints
(lambda in {0.01, 0.05, 0.1, 0.5, 1.0}, seeds 42-44).
Directly resolves Reviewer Critique #4 (distinguishing gradient conflict vs loss-scale dominance).
"""
import gzip
import json
import os
import pickle
import sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
os.environ["PCL_TEST_MODE"] = "0"

from config import BATCH_SIZE, MASK_PROB
from src.baselines import fresh_model
from src.data.dataset import ICUDataset, make_patient_split_loaders
from src.losses.pcl_loss import PhysiologicalConstraintLoss
from src.models.backbone import apply_random_mask
from src.training.train_utils import load_state_dict_flexible, masked_prediction_loss


def flat_grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    vecs = []
    for p, g in zip(params, grads):
        if g is None:
            vecs.append(torch.zeros_like(p).reshape(-1))
        else:
            vecs.append(g.detach().reshape(-1))
    return torch.cat(vecs)


def main():
    gz_path = os.path.join(_ROOT, "colab", "physionet_frac1.0.pkl.gz")
    with gzip.open(gz_path, "rb") as f:
        pn = pickle.load(f)
    site_a = [s for s in pn if s["site_id"] == 0]
    ds = ICUDataset(site_a)
    pcl_fn = PhysiologicalConstraintLoss()

    lambdas = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    seeds = [42, 43, 44]
    results = []

    print(f"{'lam':>6} {'cos_enc':>10} {'norm_masked':>12} {'norm_pcl_raw':>13} {'ratio(lam*pcl/mask)':>20}")
    print("-" * 68)
    for lam in lambdas:
        cos_list, nm_list, np_list, ratio_list = [], [], [], []
        for s in seeds:
            ckpt = os.path.join(_ROOT, "results", "pretrain_lambda_sweep", f"lam{lam:g}_p{s}_fp32.pt")
            if not os.path.exists(ckpt):
                continue
            pt_json = os.path.join(_ROOT, "results", "pretrain_lambda_sweep", f"lam{lam:g}_p{s}_fp32.json")
            bal_ratio = 1.0
            if os.path.exists(pt_json):
                br = json.load(open(pt_json)).get("history", {}).get("balance_ratio", [1.0])
                if br:
                    bal_ratio = float(br[-1])

            torch.manual_seed(s)
            train_loader, _ = make_patient_split_loaders(ds, train_frac=0.8, batch_size=BATCH_SIZE, seed=s, stratify_key="sepsis", num_workers=0)
            model = fresh_model(seed=s)
            model.load_state_dict(load_state_dict_flexible(ckpt, "cpu"))
            model.eval()  # Deterministic weights without dropout noise across backward passes
            enc_params = [p for p in model.encoder.parameters() if p.requires_grad]

            for b_idx, batch in enumerate(train_loader):
                if b_idx >= 5:
                    break
                x = batch["x"]
                mask = batch["mask"]
                c_mask = batch["c_mask"]
                x_m, pm = apply_random_mask(x, mask, mask_prob=MASK_PROB)
                reps = model.encode(x_m, mask)
                preds = model.predict(reps)
                L_m = masked_prediction_loss(preds, x, pm).float()
                L_p = pcl_fn(preds.float(), c_mask, pretrain_mask=pm)["L_PCL"]
                g_m = flat_grad(L_m, enc_params)
                g_p = flat_grad(L_p, enc_params)
                nm = float(torch.linalg.norm(g_m).item())
                np_raw = float(torch.linalg.norm(g_p).item())
                if nm > 1e-12 and np_raw > 1e-12:
                    cos_v = float(torch.dot(g_m, g_p).item() / (nm * np_raw))
                    eff_lam = lam * bal_ratio
                    cos_list.append(cos_v)
                    nm_list.append(nm)
                    np_list.append(np_raw)
                    ratio_list.append((eff_lam * np_raw) / nm)

        row = {
            "lambda": lam,
            "n_measurements": len(cos_list),
            "cos_enc_mean": float(np.mean(cos_list)) if cos_list else None,
            "cos_enc_sd": float(np.std(cos_list, ddof=1)) if len(cos_list) > 1 else 0.0,
            "grad_norm_masked_mean": float(np.mean(nm_list)) if nm_list else None,
            "grad_norm_pcl_raw_mean": float(np.mean(np_list)) if np_list else None,
            "eff_grad_ratio_mean": float(np.mean(ratio_list)) if ratio_list else None,
        }
        results.append(row)
        print(f"{lam:>6g} {row['cos_enc_mean']:>+10.4f} {row['grad_norm_masked_mean']:>12.4f} "
              f"{row['grad_norm_pcl_raw_mean']:>13.4f} {row['eff_grad_ratio_mean']:>20.2f}")

    out_path = os.path.join(_ROOT, "results", "revision", "grad_cosine_sweep.json")
    json.dump({"by_lambda": results}, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
