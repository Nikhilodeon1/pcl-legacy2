import argparse
import json
import logging
import os
import pickle
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

os.environ["PCL_TEST_MODE"] = "0" 

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEGACY2_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _LEGACY2_ROOT)  

PRETRAIN_CKPT_DIR = os.environ.get(
    "PCL_LEGACY2_PRETRAIN_DIR",
    os.path.join(_LEGACY2_ROOT, "results_lambda17", "ckpt"),
)

OUT_DIR = os.path.join(_LEGACY2_ROOT, "results", "los")
FT_CKPT_DIR = os.path.join(OUT_DIR, "ckpt")
CACHE_DIR = os.environ.get("PCL_LEGACY2_CACHE_DIR") or os.path.join(OUT_DIR, "cache") 

TASK = "los_h"
DATASETS = ("physionet", "mimic", "eicu")
# On the pod MIMIC-IV / eICU-CRD (credentialed) must not be present: only PhysioNet 2019 is used and the far targets are skipped.
PHYS_ONLY = os.environ.get("PCL_PHYSIONET_ONLY", "0") == "1"
if PHYS_ONLY:
    DATASETS = ("physionet",)


def _cached_load(name, cache_fraction, loader_fn, *args, **kwargs):

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{name}_frac{cache_fraction}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fp:
            cached = pickle.load(fp)
        if len(cached) == 0:

            logging.warning(f"[CACHE] {cache_path} is empty (0 samples) — treating as stale, deleting and re-reading.")
            os.remove(cache_path)
        else:
            logging.info(f"[CACHE] Loading {name} from {cache_path} ({len(cached)} samples)")
            return cached
    samples, _ = loader_fn(*args, **kwargs)
    if len(samples) == 0:
        raise RuntimeError(
            f"{name} loader returned 0 samples — not caching this. Check the source "
            f"data path/directory actually has data before rerunning."
        )
    try:
        with open(cache_path, "wb") as fp:
            pickle.dump(samples, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"[CACHE] Saved {name} ({len(samples)}) -> {cache_path}")
    except Exception as e:
        logging.warning(f"[CACHE] Could not write {name} cache: {e}")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["erm", "pcl", "dro"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fraction", type=float, default=1.0,
                     help="Subsample fraction per site, for a cheap dry run before full scale.")
    ap.add_argument("--epochs", type=int, default=None,
                     help="Override FINETUNE_EPOCHS. Small values are for plumbing checks only.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pretrain-ckpt", default=None,
                     help="Explicit pretrained-encoder file (default: $PCL_LEGACY2_PRETRAIN_DIR/{method}_pretrained.pt). "
                          "Used for the lambda / pretraining-seed sweep (B2/B3).")
    ap.add_argument("--encoder-id", default=None,
                     help="Unique name for a non-default encoder, e.g. lam0.5_p43 (lambda 0.5, pretraining seed 43). "
                          "Replaces --method in output file names and writes to a separate *_sweep results dir.")
    ap.add_argument("--fixed-T", type=int, default=None,
                     help="Revision G1 remedy: fixed-observation protocol. Every stay sees exactly the first T hours "
                          "(identical padding), stays with LOS < T are dropped, head pools at hour T-1. "
                          "Writes to results/{task}_fixed_T{T}/ and leaves the original results untouched.")
    ap.add_argument("--results-subdir", default=None,
                     help="write results/<subdir>/ instead of the default los_fixed_T{T}[_sweep] folder (used by the phase-2 pretraining experiments)")
    ap.add_argument("--cache-only", action="store_true",
                     help="Load + cache PhysioNet/MIMIC/eICU, then exit before touching the GPU. "
                          "Same pattern as finetune_mortality.py's --cache-only.")
    args = ap.parse_args()

    global OUT_DIR, FT_CKPT_DIR
    if args.fixed_T:
        OUT_DIR = os.path.join(_LEGACY2_ROOT, "results", f"los_fixed_T{args.fixed_T}")
    if args.encoder_id:
        OUT_DIR = OUT_DIR + "_sweep" 
    if args.results_subdir:
        OUT_DIR = os.path.join(_LEGACY2_ROOT, "results", args.results_subdir)
    FT_CKPT_DIR = os.path.join(OUT_DIR, "ckpt")
    if os.environ.get("PCL_LEGACY2_CKPT_ROOT"):
        FT_CKPT_DIR = os.path.join(os.environ["PCL_LEGACY2_CKPT_ROOT"], os.path.basename(OUT_DIR), "ckpt")

    tag = f"{args.encoder_id or args.method}_s{args.seed}"
    out_path = os.path.join(OUT_DIR, f"{tag}.json")
    if os.path.exists(out_path) and not args.overwrite:
        logging.info(f"[RESUME] {out_path} already exists — skipping. Pass --overwrite to force.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FT_CKPT_DIR, exist_ok=True)

    import torch
    _uncached = [n for n in DATASETS
                 if not os.path.exists(os.path.join(CACHE_DIR, f"{n}_frac{args.fraction}.pkl"))]
    if _uncached and torch.cuda.is_available():
        logging.warning(
            "\n" + "=" * 70 +
            f"\nSWITCH PODS NOW — about to read {', '.join(_uncached)} from raw files, uncached."
            "\nThis is CPU-only work on a GPU-priced pod."
            "\nCtrl-C, switch to a cheap pod, rerun this exact command with --cache-only,"
            "\nthen switch back and rerun without it — it'll hit the cache instantly."
            "\n" + "=" * 70
        )

    from pod_monitor import watch_pod
    watch_pod(verbose=True)

    from torch.utils.data import DataLoader, Subset
    import numpy as np
    from config import (BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TEST_MODE, D_MODEL, N_LAYERS,
                         PHYSIONET_DIR, MIMIC_DIR, EICU_DIR)
    from config import FINETUNE_EPOCHS as _CFG_FT_EPOCHS
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset
    from src.data.physionet2019 import load_physionet2019
    from src.data.mimic4 import load_mimic4
    from src.data.eicu import load_eicu
    from src.eval.evaluate_utils import evaluate_model_regression, run_finetuning_regression
    from src.training.train_utils import load_state_dict_flexible

    if TEST_MODE or D_MODEL != 256 or N_LAYERS != 6:
        raise RuntimeError(
            f"config resolved to TEST_MODE={TEST_MODE}, D_MODEL={D_MODEL}, N_LAYERS={N_LAYERS} — "
            f"incompatible with the production checkpoints in {PRETRAIN_CKPT_DIR}. See finetune_mortality.py."
        )

    n_epochs = args.epochs if args.epochs is not None else _CFG_FT_EPOCHS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Run: method={args.method} seed={args.seed} epochs={n_epochs} device={device} "
                 f"D_MODEL={D_MODEL} N_LAYERS={N_LAYERS}")

    t0 = time.time()

    pn_samples = _cached_load("physionet", args.fraction, load_physionet2019, PHYSIONET_DIR, fraction=args.fraction, seed=args.seed)
    if PHYS_ONLY:
        logging.info("[PHYSIONET-ONLY] skipping MIMIC-IV and eICU-CRD (credentialed data are not allowed on this machine)")
        mimic_samples, eicu_samples = [], []
    else:
        mimic_samples = _cached_load("mimic", args.fraction, load_mimic4, MIMIC_DIR, fraction=args.fraction, seed=args.seed)
        eicu_samples = _cached_load("eicu", args.fraction, load_eicu, EICU_DIR, fraction=args.fraction, seed=args.seed)

    if args.cache_only:
        logging.info(f"[CACHE-ONLY] All three datasets cached in {time.time() - t0:.0f}s. "
                      f"Switch to the GPU pod and rerun without --cache-only.")
        return

    if args.fixed_T:
        from src.data.fixed_window import fix_window
        pn_samples = fix_window(pn_samples, args.fixed_T, min_los_h=args.fixed_T)
        mimic_samples = fix_window(mimic_samples, args.fixed_T, min_los_h=args.fixed_T)
        eicu_samples = fix_window(eicu_samples, args.fixed_T, min_los_h=args.fixed_T)
        logging.info(f"[FIXED-T={args.fixed_T}] cohorts after LOS>={args.fixed_T}h: pn={len(pn_samples)} "
                     f"mimic={len(mimic_samples)} eicu={len(eicu_samples)}")
    site_a = [s for s in pn_samples if s["site_id"] == 0]
    site_b = [s for s in pn_samples if s["site_id"] == 1]
    if len(site_a) < 10:
        raise RuntimeError(f"Only {len(site_a)} PhysioNet Site-A samples — too few to fine-tune on. "
                            "Check PHYSIONET_DIR / training_setA.")

    ds_a = ICUDataset(site_a)
    
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(ds_a))
    n_train = int(len(ds_a) * 0.8)
    train_loader = DataLoader(Subset(ds_a, idx[:n_train].tolist()), batch_size=BATCH_SIZE,
                               shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(Subset(ds_a, idx[n_train:].tolist()), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    target_samples = {"physionet_b": site_b, "mimic": mimic_samples, "eicu": eicu_samples}
    target_loaders = {}
    for name, samples in target_samples.items():
        if not samples:
            logging.warning(f"No samples for target '{name}' — skipping.")
            continue
        target_loaders[name] = DataLoader(ICUDataset(samples), batch_size=BATCH_SIZE, shuffle=False,
                                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    pretrain_path = args.pretrain_ckpt or os.path.join(PRETRAIN_CKPT_DIR, f"{args.method}_pretrained.pt")
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(
            f"{pretrain_path} not found — expected the prior-study pretrained checkpoint here. "
            "Do not fall back to training a fresh encoder silently."
        )
    model = fresh_model(seed=args.seed)
    model.load_state_dict(load_state_dict_flexible(pretrain_path, device="cpu"))
    model.add_classification_head(TASK)
    if args.fixed_T:
        model.cls_heads[TASK].fixed_index = args.fixed_T - 1
    logging.info(f"Loaded pretrained encoder: {pretrain_path}")

    ft_ckpt_path = os.path.join(FT_CKPT_DIR, f"{tag}.pt")
    run_finetuning_regression(
        model, train_loader, val_loader, TASK,
        n_epochs=n_epochs, device=device, save_path=ft_ckpt_path,
    )
    model.load_state_dict(load_state_dict_flexible(ft_ckpt_path, device))

    in_domain = evaluate_model_regression(model, val_loader, TASK, device=device,
                                           split_name=f"{tag} in-domain (physionet-A)")
    ood = {}
    for name, loader in target_loaders.items():
        ood[name] = evaluate_model_regression(model, loader, TASK, device=device,
                                                split_name=f"{tag} zero-shot ({name})")

    elapsed = time.time() - t0
    result = {
        "fixed_T": args.fixed_T, "encoder_id": args.encoder_id, "pretrain_ckpt": args.pretrain_ckpt, "method": args.method, "seed": args.seed, "task": TASK, "epochs": n_epochs, "fraction": args.fraction,
        "n_source": len(ds_a), "n_targets": {k: len(v) for k, v in target_samples.items() if v},
        "in_domain": in_domain, "ood": ood, "elapsed_sec": elapsed,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logging.info(f"Saved {out_path} | elapsed={elapsed / 60:.1f} min")
    logging.info(
        f"Projected cost for remaining 8 runs at this pace: ~{8 * elapsed / 3600:.2f} GPU-hours "
        f"(excludes data-load caching speedup on reruns)"
    )


if __name__ == "__main__":
    main()
