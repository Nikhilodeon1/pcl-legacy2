"""
Sepsis fine-tuning on PhysioNet 2019 Site A with zero-shot evaluation on Site B (the Dead PCL task, restricted to PhysioNet so it can run on the pod).

Label: the stay-level "sepsis ever in the stay" label broadcast over the 48 hours (ICUDataset 'sepsis' key). MIMIC-IV / eICU-CRD scoring is NOT done here
(credentialed data stay on the approved local machine; score the saved checkpoint there).

Same two-phase schedule as the other fine-tune scripts (30 epochs, head-only for the first third, AdamW), seed sets the 80/20 Site A split and the head init.

  python scripts/finetune_sepsis.py --pretrain-ckpt ~/pcl_pretrained/phase2/lam0.5_p42_fp32.pt --encoder-id lam0.5_p42_fp32 --seed 42 --out-dir results/phase2/sepsis
  --fixed-T 24   restricts every stay to its first 24 h (stays shorter than 24 h dropped), the fixed-observation protocol

Output: <out-dir>/<encoder-id>_s<seed>[_T<T>].json (+ the fine-tuned checkpoint under $PCL_LEGACY2_CKPT_ROOT, default /tmp/pcl_ckpt). Resumable.
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
os.environ["PCL_TEST_MODE"] = "0"
TASK = "sepsis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-ckpt", required=True)
    ap.add_argument("--encoder-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fixed-T", type=int, default=None)
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "results", "phase2", "sepsis"))
    ap.add_argument("--cache-dir", default=os.environ.get("PCL_LEGACY2_CACHE_DIR"))
    ap.add_argument("--max-samples", type=int, default=None, help="smoke test")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tag = f"{args.encoder_id}_s{args.seed}" + (f"_T{args.fixed_T}" if args.fixed_T else "")
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, tag + ".json")
    if os.path.exists(out_path) and not args.overwrite:
        logging.info(f"[RESUME] {out_path} exists, skipping")
        return

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import config
    from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, FINETUNE_EPOCHS, D_MODEL, N_LAYERS
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset, make_patient_split_loaders
    from src.eval.evaluate_utils import evaluate_model, run_finetuning
    from src.training.train_utils import load_state_dict_flexible

    if D_MODEL != 256 or N_LAYERS != 6:
        raise RuntimeError("config is in TEST_MODE; need 256 / 6")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = args.epochs or FINETUNE_EPOCHS
    pn = pickle.load(open(os.path.join(args.cache_dir, "physionet_frac1.0.pkl"), "rb"))
    site_a = [s for s in pn if s["site_id"] == 0]
    site_b = [s for s in pn if s["site_id"] == 1]
    if args.fixed_T:
        from src.data.fixed_window import fix_window
        site_a = fix_window(site_a, args.fixed_T, min_los_h=args.fixed_T)
        site_b = fix_window(site_b, args.fixed_T, min_los_h=args.fixed_T)
    if args.max_samples:
        site_a, site_b = site_a[:args.max_samples], site_b[:args.max_samples]
    logging.info(f"{tag}: Site A {len(site_a)} stays, Site B {len(site_b)} stays, device={device}, epochs={n_epochs}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ds_a = ICUDataset(site_a)
    train_loader, val_loader = make_patient_split_loaders(ds_a, train_frac=0.8, batch_size=BATCH_SIZE, seed=args.seed, stratify_key=TASK,
                                                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    b_loader = DataLoader(ICUDataset(site_b), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = fresh_model(seed=args.seed)
    model.load_state_dict(load_state_dict_flexible(args.pretrain_ckpt, "cpu"))
    model.add_classification_head(TASK)
    if args.fixed_T:
        model.cls_heads[TASK].fixed_index = args.fixed_T - 1
    ck_root = os.environ.get("PCL_LEGACY2_CKPT_ROOT", "/tmp/pcl_ckpt")
    ft_path = os.path.join(ck_root, "phase2_sepsis", tag + ".pt")
    os.makedirs(os.path.dirname(ft_path), exist_ok=True)

    t0 = time.time()
    run_finetuning(model, train_loader, val_loader, TASK, n_epochs=n_epochs, device=device, save_path=ft_path)
    model.load_state_dict(load_state_dict_flexible(ft_path, device))
    in_domain = evaluate_model(model, val_loader, TASK, device=device, split_name=f"{tag} in-domain (Site A val)")
    ood = {"physionet_b": evaluate_model(model, b_loader, TASK, device=device, split_name=f"{tag} zero-shot (Site B)")}
    json.dump({"encoder_id": args.encoder_id, "pretrain_ckpt": os.path.basename(args.pretrain_ckpt), "seed": args.seed, "task": TASK, "epochs": n_epochs,
               "fixed_T": args.fixed_T, "n_source": len(site_a), "n_targets": {"physionet_b": len(site_b)},
               "in_domain": in_domain, "ood": ood, "elapsed_sec": time.time() - t0, "device": device}, open(out_path, "w"), indent=1, default=str)
    logging.info(f"saved {out_path}")


if __name__ == "__main__":
    main()
