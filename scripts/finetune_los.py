"""
pcl-legacy2, Step 3/4 — LOS task (continuous, log1p(hours)).

Single-source protocol — matches URTC exactly, unlike mortality which needed
a 2-site-both-directions workaround because PhysioNet lacks mortality labels.
PhysioNet DOES have valid continuous LOS (`los_h`, hours — see README and
src/data/physionet2019.py), so: train on PhysioNet Site A, zero-shot eval on
Site B + MIMIC-IV + eICU. 3 methods x 3 seeds = 9 runs total (half of
mortality's 18), each producing 3 zero-shot scores instead of 1.

Reuses the pretrained ERM/PCL/DRO encoders as-is, same as finetune_mortality.py
(pretraining is task-agnostic). Only difference in the fine-tune/eval machinery:
run_finetuning_regression / evaluate_model_regression (evaluate_utils.py) —
MSE loss on log1p(los_h), MAE/RMSE/R^2 eval — since there's no AUROC equivalent
for a continuous target. Those are new, additive functions; the existing
binary run_finetuning/evaluate_model (vendored from the reboot project's src/,
see VENDORED.md) are untouched.

IMPORTANT — separate cache dir from mortality's: mortality's cache
(results/mortality/cache/{mimic,eicu}_frac1.0.pkl) predates the `los_h` field
being added to the loaders. Reusing it here would silently zero-fill los_h via
ICUDataset's .get(..., 0) default — wrong data, not an error. This script uses
its own results/los/cache/ instead, forcing a fresh read that includes los_h.

Usage (same env as finetune_mortality.py; PHYSIONET_DIR/MIMIC_DIR/EICU_DIR):
    python finetune_los.py --method erm --seed 42
    python finetune_los.py --method erm --seed 42 --cache-only   # data-load only, run on a cheap pod first
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

os.environ["PCL_TEST_MODE"] = "0"  # see finetune_mortality.py — must precede any config import

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEGACY2_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _LEGACY2_ROOT)  # config.py, src/, pod_monitor.py vendored here

PRETRAIN_CKPT_DIR = os.environ.get(
    "PCL_LEGACY2_PRETRAIN_DIR",
    os.path.join(_LEGACY2_ROOT, "results_lambda17", "ckpt"),
)

OUT_DIR = os.path.join(_LEGACY2_ROOT, "results", "los")
FT_CKPT_DIR = os.path.join(OUT_DIR, "ckpt")
CACHE_DIR = os.path.join(OUT_DIR, "cache")  # NOT mortality's cache dir — see module docstring

TASK = "los_h"
DATASETS = ("physionet", "mimic", "eicu")


def _cached_load(name, cache_fraction, loader_fn, *args, **kwargs):
    # cache_fraction is named to NOT collide with a `fraction=` kwarg meant
    # for loader_fn (both mimic4/eicu/physionet2019 loaders take one) — a
    # plain `fraction` param here shadowed that and crashed with
    # "got multiple values for argument 'fraction'".
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{name}_frac{cache_fraction}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fp:
            cached = pickle.load(fp)
        if len(cached) == 0:
            # A previous run cached an empty result (e.g. the source data was
            # missing/misconfigured at the time) with no error — an empty
            # cache is never valid to trust silently. Delete it and re-read.
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
    ap.add_argument("--cache-only", action="store_true",
                     help="Load + cache PhysioNet/MIMIC/eICU, then exit before touching the GPU. "
                          "Same pattern as finetune_mortality.py's --cache-only.")
    args = ap.parse_args()

    tag = f"{args.method}_s{args.seed}"
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
    mimic_samples = _cached_load("mimic", args.fraction, load_mimic4, MIMIC_DIR, fraction=args.fraction, seed=args.seed)
    eicu_samples = _cached_load("eicu", args.fraction, load_eicu, EICU_DIR, fraction=args.fraction, seed=args.seed)

    if args.cache_only:
        logging.info(f"[CACHE-ONLY] All three datasets cached in {time.time() - t0:.0f}s. "
                      f"Switch to the GPU pod and rerun without --cache-only.")
        return

    site_a = [s for s in pn_samples if s["site_id"] == 0]
    site_b = [s for s in pn_samples if s["site_id"] == 1]
    if len(site_a) < 10:
        raise RuntimeError(f"Only {len(site_a)} PhysioNet Site-A samples — too few to fine-tune on. "
                            "Check PHYSIONET_DIR / training_setA.")

    ds_a = ICUDataset(site_a)

    # Plain random patient-level split — NOT make_patient_split_loaders, which
    # stratifies by a binary label; los_h is continuous, stratification doesn't
    # apply the same way.
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

    pretrain_path = os.path.join(PRETRAIN_CKPT_DIR, f"{args.method}_pretrained.pt")
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(
            f"{pretrain_path} not found — expected the URTC-era pretrained checkpoint here. "
            "Do not fall back to training a fresh encoder silently."
        )
    model = fresh_model(seed=args.seed)
    model.load_state_dict(load_state_dict_flexible(pretrain_path, device="cpu"))
    model.add_classification_head(TASK)
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
        "method": args.method, "seed": args.seed, "task": TASK, "epochs": n_epochs, "fraction": args.fraction,
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
