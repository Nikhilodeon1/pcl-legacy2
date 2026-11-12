"""
Head Pilot Experiment: Addressing Mock Review Step 1.
Tests whether head conditioning (LayerNorm before head, Linear Probing then Fine-Tuning,
or lower encoder LR) resolves fine-tuning collapse on mortality and sepsis.
Logs gradient norms, initial logit statistics, NaNs, and checks for constant outputs.
"""
import argparse
import copy
import json
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, D_MODEL, N_LAYERS
from src.baselines import fresh_model
from src.data.dataset import ICUDataset, make_patient_split_loaders
from src.training.train_utils import load_state_dict_flexible
from src.eval.evaluate_utils import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class NormalizedClassificationHead(nn.Module):
    """ClassificationHead with input LayerNorm to prevent saturation from large representation norms."""
    def __init__(self, d_model: int, dropout: float = 0.2, use_layernorm: bool = True):
        super().__init__()
        layers = []
        if use_layernorm:
            layers.append(nn.LayerNorm(d_model))
        layers.extend([
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        ])
        self.head = nn.Sequential(*layers)
        self.fixed_index = None

    def forward(self, reps, obs_mask=None):
        if self.fixed_index is not None:
            pooled = reps[:, self.fixed_index]
        elif obs_mask is not None:
            t_obs = obs_mask.any(dim=-1)
            last_idx = (t_obs.size(1) - 1 - t_obs.flip(dims=[1]).long().argmax(dim=1)).clamp(min=0)
            batch_idx = torch.arange(reps.size(0), device=reps.device)
            pooled = reps[batch_idx, last_idx]
        else:
            pooled = reps[:, -1, :]
        return self.head(pooled)


def run_pilot_finetuning(
    model,
    train_loader,
    val_loader,
    task_name,
    n_epochs=30,
    head_lr=1e-3,
    encoder_lr=1e-4,
    unfreeze_epoch=10,
    grad_clip=0.5,
    device="cuda",
    save_path=None,
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.cls_heads[task_name].parameters(), lr=head_lr, weight_decay=1e-4)

    # Calculate class weight
    all_train_labels = []
    if hasattr(train_loader.dataset, "indices"):
        ds = train_loader.dataset.dataset
        all_train_labels = [ds.samples[i][task_name].item() for i in train_loader.dataset.indices]
    else:
        all_train_labels = [s[task_name].item() for s in train_loader.dataset.samples]
    num_pos = float(np.sum(all_train_labels))
    num_neg = float(len(all_train_labels) - num_pos)
    pos_weight = torch.tensor([min(num_neg / (num_pos + 1e-6), 100.0)], device=device)

    encoder_unfrozen = False
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    logs = {"grad_norms": [], "val_losses": [], "train_losses": [], "nan_detected": False, "constant_collapse": False}

    for epoch in range(1, n_epochs + 1):
        if epoch >= unfreeze_epoch and not encoder_unfrozen:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {"params": model.cls_heads[task_name].parameters(), "lr": head_lr},
            ], weight_decay=1e-4)
            encoder_unfrozen = True
            logging.info(f"[Epoch {epoch}] Encoder unfrozen: encoder_lr={encoder_lr:.1e}, head_lr={head_lr:.1e}")

        model.train()
        epoch_losses, epoch_grad_norms = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch[task_name].to(device, non_blocking=True).float()

            reps = model.encode(x, mask)
            logits = model.classify(reps, task_name, obs_mask=mask).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"[Epoch {epoch}] NaN/Inf loss detected!")
                logs["nan_detected"] = True
                continue

            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            epoch_grad_norms.append(float(total_norm))
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses, val_logits = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                labels = batch[task_name].to(device, non_blocking=True).float()
                reps = model.encode(x, mask)
                logits = model.classify(reps, task_name, obs_mask=mask).squeeze(-1)
                v_loss = F.binary_cross_entropy_with_logits(logits, labels)
                val_losses.append(v_loss.item())
                val_logits.extend(logits.cpu().numpy().tolist())

        mean_vloss = float(np.mean(val_losses))
        mean_gnorm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0
        val_logit_std = float(np.std(val_logits)) if val_logits else 0.0
        logs["train_losses"].append(float(np.mean(epoch_losses)))
        logs["val_losses"].append(mean_vloss)
        logs["grad_norms"].append(mean_gnorm)

        if val_logit_std < 1e-4:
            logging.warning(f"[Epoch {epoch}] Constant logit collapse detected! std(logits)={val_logit_std:.6f}")
            logs["constant_collapse"] = True

        if mean_vloss < best_val_loss:
            best_val_loss = mean_vloss
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == n_epochs:
            logging.info(f"Epoch {epoch:2d}/{n_epochs:2d}: Train Loss={np.mean(epoch_losses):.4f}, Val Loss={mean_vloss:.4f}, GradNorm={mean_gnorm:.4f}, LogitStd={val_logit_std:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state, save_path)
    model.load_state_dict(best_state)
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mortality_hospital", "sepsis"], default="mortality_hospital")
    ap.add_argument("--source", choices=["mimic", "eicu", "physionet_a"], default="mimic")
    ap.add_argument("--seed", type=int, default=43, help="Seed 43 previously collapsed under PCL lam0.5")
    ap.add_argument("--pretrain-ckpt", required=True)
    ap.add_argument("--head-mode", choices=["baseline", "layernorm", "lpft", "low_lr"], default="layernorm")
    ap.add_argument("--fixed-T", type=int, default=48)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"=== Head Pilot: Task={args.task}, Source={args.source}, Seed={args.seed}, Mode={args.head_mode} ===")

    # Setup head parameters based on mode
    use_layernorm = args.head_mode in ["layernorm", "lpft"]
    unfreeze_epoch = 10 if args.head_mode == "lpft" else (10 if args.epochs == 30 else max(1, args.epochs // 3))
    encoder_lr = 1e-5 if args.head_mode in ["low_lr", "lpft"] else 1e-4

    # Load data
    target = "eicu" if args.source == "mimic" else ("mimic" if args.source == "eicu" else "physionet_b")
    from scripts.finetune_mortality import _load_site
    src_samples = _load_site(args.source, args.fraction, args.seed)
    tgt_samples = _load_site(target, args.fraction, args.seed)

    if args.fixed_T:
        from src.data.fixed_window import fix_window
        src_samples = fix_window(src_samples, args.fixed_T, min_los_h=args.fixed_T)
        tgt_samples = fix_window(tgt_samples, args.fixed_T, min_los_h=args.fixed_T)

    src_ds = ICUDataset(src_samples)
    tgt_ds = ICUDataset(tgt_samples)
    train_loader, val_loader = make_patient_split_loaders(src_ds, train_frac=0.8, batch_size=BATCH_SIZE, seed=args.seed, stratify_key=args.task)
    tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = fresh_model(seed=args.seed)
    model.load_state_dict(load_state_dict_flexible(args.pretrain_ckpt, device="cpu"))

    # Attach custom classification head
    head = NormalizedClassificationHead(D_MODEL, dropout=0.2, use_layernorm=use_layernorm)
    if args.fixed_T:
        head.fixed_index = args.fixed_T - 1
    model.cls_heads[args.task] = head

    # Train
    pilot_logs = run_pilot_finetuning(
        model, train_loader, val_loader, args.task,
        n_epochs=args.epochs,
        head_lr=1e-3,
        encoder_lr=encoder_lr,
        unfreeze_epoch=unfreeze_epoch,
        device=device,
    )

    # Evaluate
    in_domain = evaluate_model(model, val_loader, args.task, device=device, split_name=f"In-domain ({args.source})")
    ood = evaluate_model(model, tgt_loader, args.task, device=device, split_name=f"Zero-shot OOD ({target})")

    out_res = {
        "task": args.task,
        "source": args.source,
        "target": target,
        "seed": args.seed,
        "head_mode": args.head_mode,
        "use_layernorm": use_layernorm,
        "encoder_lr": encoder_lr,
        "unfreeze_epoch": unfreeze_epoch,
        "in_domain": in_domain,
        "ood": ood,
        "logs": pilot_logs,
    }

    out_path = os.path.join(_ROOT, "results", "head_pilot", f"{args.head_mode}_{args.source}to{target}_s{args.seed}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(out_res, fp, indent=2)
    logging.info(f"Pilot completed. Saved to {out_path}")
    print(f"RESULTS: In-domain AUROC = {in_domain.get('auroc')}, OOD AUROC = {ood.get('auroc')}")


if __name__ == "__main__":
    main()
