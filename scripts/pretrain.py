"""
Unified pretraining driver (PhysioNet 2019 Site A only, unless --extra-unlabeled or --matched-volume-total is given).
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
sys.path.insert(0, _HERE)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def make_tag(lam, seed, precision, extra, warmup, no_balancer, matched_vol, var_reg=0.0, constraints="all"):
    if constraints == "map_only":
        return f"pcl_maponly_p{seed}"
    if constraints == "nohh":
        return f"pcl_nohh_p{seed}"
    if matched_vol:
        prefix = "pcl" if lam > 0 else "erm"
        return f"{prefix}_matched_vol_p{seed}"
    if no_balancer:
        return f"pcl_fixed_nobalance_p{seed}"
    if var_reg and var_reg > 0.0:
        return f"pcl_varreg{var_reg:g}_p{seed}"
    t = f"lam{lam:g}_p{seed}_{precision}"
    if extra:
        t += "_leak" + "".join(w[0] + w[-1] for w in extra)
    if abs(warmup - 0.4) > 1e-9:
        t += f"_wu{warmup:g}"
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True, help="constraint weight lambda (0 = ERM, same code path)")
    ap.add_argument("--seed", type=int, required=True, help="sets the initialisation AND the Site A 80/20 split")
    ap.add_argument("--precision", choices=["amp", "fp32"], default="amp")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-fraction", type=float, default=0.4)
    ap.add_argument("--extra-unlabeled", nargs="*", default=[], choices=["physionet_b"])
    ap.add_argument("--matched-volume-total", type=int, default=None,
                    help="resample Site A to match this exact total stay count (volume control)")
    ap.add_argument("--no-balancer", action="store_true", help="disable adaptive EMA balancer (fixed weight PCL)")
    ap.add_argument("--var-reg", type=float, default=0.0,
                    help="VICReg-style variance+covariance anti-collapse regularization weight")
    ap.add_argument("--constraints", choices=["all", "map_only", "nohh"], default="all",
                    help="physiological constraints to apply (all, map_only, nohh)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=os.environ.get("PCL_LEGACY2_CACHE_DIR"))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.precision == "fp32":
        os.environ["PCL_DISABLE_AMP"] = "1"
    os.environ["PCL_TEST_MODE"] = "0"
    tag = make_tag(args.lam, args.seed, args.precision, args.extra_unlabeled, args.warmup_fraction,
                   args.no_balancer, args.matched_volume_total, args.var_reg, args.constraints)
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, tag + ".json")
    out_pt = os.path.join(args.out_dir, tag + ".pt")
    if os.path.exists(out_json) and not args.overwrite:
        logging.info(f"[RESUME] {out_json} exists, skipping")
        return

    import numpy as np
    import torch
    from torch.utils.data import ConcatDataset, DataLoader
    import config
    from config import BATCH_SIZE, MASK_PROB, NUM_WORKERS, PIN_MEMORY, USE_AMP, D_MODEL, N_LAYERS
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset, make_patient_split_loaders
    from src.losses.pcl_loss import PhysiologicalConstraintLoss
    from src.training.train_utils import load_state_dict_flexible, run_pretraining
    from repr_diag import diagnose

    if D_MODEL != 256 or N_LAYERS != 6:
        raise RuntimeError(f"config resolved D_MODEL={D_MODEL}, N_LAYERS={N_LAYERS}; need 256 / 6")
    if (args.precision == "fp32") == bool(USE_AMP):
        raise RuntimeError(f"precision={args.precision} but config.USE_AMP={USE_AMP}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"tag={tag} device={device} USE_AMP={USE_AMP} lambda={args.lam} seed={args.seed} epochs={args.epochs}")

    cache = os.path.join(args.cache_dir or "", "physionet_frac1.0.pkl")
    if os.path.exists(cache):
        pn = pickle.load(open(cache, "rb"))
    else:
        from src.data.physionet2019 import load_physionet2019
        pn, _ = load_physionet2019(config.PHYSIONET_DIR, fraction=1.0, seed=args.seed)
    site_a = [s for s in pn if s["site_id"] == 0]
    extra = [s for s in pn if s["site_id"] == 1] if "physionet_b" in args.extra_unlabeled else []
    if args.max_samples:
        site_a, extra = site_a[:args.max_samples], extra[:args.max_samples]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.matched_volume_total:
        # Resample Site A stays with replacement to match total volume (e.g. 25,160)
        rng = np.random.default_rng(args.seed)
        n_needed = max(0, args.matched_volume_total - len(site_a))
        if len(site_a) > 0 and n_needed > 0:
            resampled_idx = rng.choice(len(site_a), size=n_needed, replace=True)
            extra = [site_a[i] for i in resampled_idx]
            logging.info(f"[MATCHED-VOLUME] Resampled {len(extra)} Site A stays to match total volume {args.matched_volume_total}")

    logging.info(f"Site A stays: {len(site_a)}; extra unlabeled stays: {len(extra)}")

    ds = ICUDataset(site_a)
    train_loader, val_loader = make_patient_split_loaders(ds, train_frac=0.8, batch_size=BATCH_SIZE, seed=args.seed, stratify_key="sepsis",
                                                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if extra:
        train_ds = ConcatDataset([train_loader.dataset, ICUDataset(extra)])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = fresh_model(seed=args.seed)
    tmp_ckpt = out_pt + ".full.tmp"
    t0 = time.time()
    if args.constraints == "map_only":
        loss_fn = PhysiologicalConstraintLoss(map_weight=1.0, hh_weight=0.0, spo2_weight=0.0)
    elif args.constraints == "nohh":
        loss_fn = PhysiologicalConstraintLoss(map_weight=1.0, hh_weight=0.0, spo2_weight=1.0)
    else:
        loss_fn = PhysiologicalConstraintLoss(map_weight=1.0, hh_weight=1.0, spo2_weight=1.0)

    history = run_pretraining(model, loss_fn, train_loader, val_loader, n_epochs=args.epochs, lam=args.lam,
                              mask_prob=MASK_PROB, device=device, save_path=tmp_ckpt, warmup_fraction=args.warmup_fraction,
                              use_balancer=(not args.no_balancer), var_reg_weight=args.var_reg)
    secs = time.time() - t0

    model.load_state_dict(load_state_dict_flexible(tmp_ckpt, "cpu"))
    torch.save(model.state_dict(), out_pt)
    os.remove(tmp_ckpt)

    diag = diagnose(model, site_a, device=device, hours=(23, 47), n=min(4000, len(site_a)))
    clean = {k: [float(x) for x in v] for k, v in history.items() if isinstance(v, list)}
    json.dump({
        "tag": tag, "lambda": args.lam, "seed": args.seed, "precision": args.precision, "epochs": args.epochs,
        "warmup_fraction": args.warmup_fraction, "mask_prob": MASK_PROB, "batch_size": BATCH_SIZE,
        "extra_unlabeled": args.extra_unlabeled, "matched_volume_total": args.matched_volume_total,
        "no_balancer": args.no_balancer, "var_reg": args.var_reg, "constraints": args.constraints,
        "n_site_a": len(site_a), "n_extra": len(extra),
        "device": device, "torch": torch.__version__, "seconds": secs, "repr_diag_site_a": diag, "history": clean,
        "checkpoint": os.path.basename(out_pt),
    }, open(out_json, "w"), indent=1)
    d = diag["47"]
    logging.info(f"done in {secs / 60:.1f} min; best epoch weights -> {out_pt}; repr(h=47): L2={d['mean_l2']:.1f} spread={d['spread']:.3f} "
                 f"const_frac={d['const_frac']:.3f}")


if __name__ == "__main__":
    main()
