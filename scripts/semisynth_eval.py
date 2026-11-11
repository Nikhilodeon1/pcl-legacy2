"""
Semi-synthetic shift experiments on PhysioNet 2019 (PhysioNet only, safe for the pod).  Two modes.

--mode shift   Take a fine-tuned model and the Site A validation stays (the source), apply ONE known manipulation to a copy of those stays
               (src/synth/shifts.py), and record, for each manipulation and magnitude: model performance on the shifted copy and the four label-free
               criteria computed against the unshifted source (constraint violation, reconstruction MSE, representation distance, MMD).
               Ground truth is known by construction, e.g. a label-scale shift must leave every input-based criterion unchanged while performance drops;
               removing blood-gas labs should move the violation score without any change in physiology; a MAP bias breaks a constraint relation.

--mode length  Positive control for the stay-length shortcut: a synthetic outcome that depends only on observed stay length (values taken from other
               stays), fine-tuned from the same pretrained encoder under (a) the original window and (b) the fixed-observation window (--fixed-T). A
               last-observed-hour model must score well on (a) and fall to chance on (b); a gradient-boosted probe on mask/length features is reported too.

Usage:
  python scripts/semisynth_eval.py --mode shift  --task sepsis --pretrain-ckpt E.pt --ft-ckpt F.pt --seed 42 --tag lam0.5_p42_fp32 --out results/phase2/semisynth_shift.csv
  python scripts/semisynth_eval.py --mode length --pretrain-ckpt E.pt --seed 42 --tag lam0.5_p42_fp32 --fixed-T 24 --out results/phase2/semisynth_length.csv
The fine-tuned checkpoint for --mode shift is the one written by finetune_sepsis.py / finetune_los.py (under $PCL_LEGACY2_CKPT_ROOT).
"""
import argparse
import csv
import logging
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
os.environ["PCL_TEST_MODE"] = "0"
LABS = ["pH", "HCO3", "pCO2", "PaO2"]


def _load_site_a(args):
    pn = pickle.load(open(os.path.join(args.cache_dir, "physionet_frac1.0.pkl"), "rb"))
    a = [s for s in pn if s["site_id"] == 0]
    if args.fixed_T:
        from src.data.fixed_window import fix_window
        a = fix_window(a, args.fixed_T, min_los_h=args.fixed_T)
    return a[:args.max_samples] if args.max_samples else a


def _append(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def mode_shift(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from config import BATCH_SIZE
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset, make_patient_split_loaders
    from src.eval.evaluate_utils import evaluate_model, evaluate_model_regression
    from src.synth import shifts as SH
    from src.training.train_utils import load_state_dict_flexible
    from selection_criteria_los import score_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "sepsis" if args.task == "sepsis" else "los_h"
    site_a = _load_site_a(args)
    if args.task == "sepsis":                       # same split as finetune_sepsis.py
        ds = ICUDataset(site_a)
        _, val_loader = make_patient_split_loaders(ds, 0.8, BATCH_SIZE, args.seed, stratify_key="sepsis")
        val_idx = list(val_loader.dataset.indices)
    else:                                           # same split as finetune_los.py
        idx = np.random.default_rng(args.seed).permutation(len(site_a))
        val_idx = idx[int(len(site_a) * 0.8):].tolist()
    val = [site_a[i] for i in val_idx]
    model = fresh_model(seed=args.seed)
    model.add_classification_head(task)
    if args.fixed_T:
        model.cls_heads[task].fixed_index = args.fixed_T - 1
    model.load_state_dict(load_state_dict_flexible(args.ft_ckpt, "cpu"))
    model = model.to(device).eval()

    def loader(samples):
        return DataLoader(ICUDataset(samples), batch_size=BATCH_SIZE, shuffle=False)

    src_loader = loader(val)

    def performance(samples):
        if args.task == "sepsis":
            return "auroc", evaluate_model(model, loader(samples), task, device=device, split_name="semisynth")["auroc"]
        return "r2", evaluate_model_regression(model, loader(samples), task, device=device, split_name="semisynth")["r2"]

    families = [("composition_labs", m, lambda s, m=m: SH.composition_shift(s, LABS, m, seed=1)) for m in (0.0, 0.25, 0.5, 0.75, 1.0)]
    families += [("physiology_MAP_bias_mmHg", b, lambda s, b=b: SH.physiology_shift(s, "MAP", b)) for b in (0.0, 5.0, 10.0, 20.0)]
    families += [("physiology_pH_bias", b, lambda s, b=b: SH.physiology_shift(s, "pH", b)) for b in (0.0, 0.05, 0.1, 0.2)]
    families += [("physiology_PaO2_bias_mmHg", b, lambda s, b=b: SH.physiology_shift(s, "PaO2", b)) for b in (0.0, 25.0, 50.0, 100.0)]
    if args.task == "los":
        families += [("label_scale", f, lambda s, f=f: SH.label_scale_shift(s, f)) for f in (1.0, 1.5, 2.0, 3.0, 5.0)]
    rows = []
    for fam, mag, fn in families:
        tgt = fn(val)
        crit = score_checkpoint(model, src_loader, loader(tgt), device, seed=args.seed)
        mname, mval = performance(tgt)
        rows.append(dict(tag=args.tag, task=args.task, seed=args.seed, fixed_T=args.fixed_T or "", family=fam, magnitude=mag, metric=mname, performance=mval,
                         violation=crit["violation"], recon_mse=crit["recon_mse"], repr_dist=crit["repr_dist"], mmd=crit["mmd"], n=len(tgt)))
        print(f"{fam:28s} {mag:7.2f} {mname}={mval:+.4f} viol={crit['violation']:.4f} recon={crit['recon_mse']:.4f} "
              f"repr={crit['repr_dist']:.3f} mmd={crit['mmd']:.4f}")
    _append(args.out, rows)


def mode_length(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from config import BATCH_SIZE, FINETUNE_EPOCHS
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset, make_patient_split_loaders
    from src.eval.evaluate_utils import evaluate_model, run_finetuning
    from src.synth import shifts as SH
    from src.training.train_utils import load_state_dict_flexible

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pn = pickle.load(open(os.path.join(args.cache_dir, "physionet_frac1.0.pkl"), "rb"))
    a = [s for s in pn if s["site_id"] == 0]
    if args.max_samples:
        a = a[:args.max_samples]
    syn, thr = SH.length_outcome(a, flip=0.05, seed=args.seed)          # outcome defined by the ORIGINAL observed length
    if args.fixed_T:
        from src.data.fixed_window import fix_window
        syn = fix_window(syn, args.fixed_T, min_los_h=args.fixed_T)
    torch.manual_seed(args.seed)
    ds = ICUDataset(syn)
    train_loader, val_loader = make_patient_split_loaders(ds, 0.8, BATCH_SIZE, args.seed, stratify_key="sepsis")
    model = fresh_model(seed=args.seed)
    model.load_state_dict(load_state_dict_flexible(args.pretrain_ckpt, "cpu"))
    model.add_classification_head("sepsis")
    if args.fixed_T:
        model.cls_heads["sepsis"].fixed_index = args.fixed_T - 1
    import tempfile
    default_ck_root = os.path.join(tempfile.gettempdir(), "pcl_ckpt") if os.name == "nt" else "/tmp/pcl_ckpt"
    ck = os.path.join(os.environ.get("PCL_LEGACY2_CKPT_ROOT", default_ck_root), "phase2_length", f"{args.tag}_T{args.fixed_T or 0}_s{args.seed}.pt")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    run_finetuning(model, train_loader, val_loader, "sepsis", n_epochs=args.epochs or FINETUNE_EPOCHS, device=device, save_path=ck)
    model.load_state_dict(load_state_dict_flexible(ck, device))
    auroc = evaluate_model(model, val_loader, "sepsis", device=device, split_name="length-outcome val")["auroc"]
    # mask/length probe on the same split (features: per-variable observation counts and observed length)
    val_idx = set(val_loader.dataset.indices)
    def feats(s):
        m = np.asarray(s["mask"], bool)
        return np.concatenate([m.sum(0), [SH.observed_length(m)]])
    X = np.array([feats(s) for s in syn]); y = np.array([float(np.asarray(s["label"])[0]) for s in syn])
    tr = np.array([i not in val_idx for i in range(len(syn))])
    probe = HistGradientBoostingClassifier(max_iter=100).fit(X[tr], y[tr])
    probe_auc = roc_auc_score(y[~tr], probe.predict_proba(X[~tr])[:, 1])
    row = dict(tag=args.tag, seed=args.seed, fixed_T=args.fixed_T or "", threshold_h=thr, n=len(syn), prevalence=float(y.mean()), model_auroc=auroc, probe_auroc=probe_auc)
    print(row)
    _append(args.out, [row])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shift", "length"], required=True)
    ap.add_argument("--task", choices=["sepsis", "los"], default="sepsis")
    ap.add_argument("--pretrain-ckpt")
    ap.add_argument("--ft-ckpt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fixed-T", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--cache-dir", default=os.environ.get("PCL_LEGACY2_CACHE_DIR"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "shift":
        if not args.ft_ckpt:
            raise SystemExit("--mode shift needs --ft-ckpt")
        mode_shift(args)
    else:
        if not args.pretrain_ckpt:
            raise SystemExit("--mode length needs --pretrain-ckpt")
        mode_length(args)


if __name__ == "__main__":
    main()
