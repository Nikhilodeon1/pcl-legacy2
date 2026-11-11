import logging
logging.basicConfig(level=logging.INFO)
import torch
import numpy as np
from config import LAMBDA_PCL, MASK_PROB, USE_AMP

class PCLLossBalancer:

    def __init__(self, momentum=0.95, target_ratio=0.5):
        self.momentum = momentum
        self.target_ratio = target_ratio
        self.ratio_ema = None

    def balance(self, L_pcl, L_masked):
        pcl_val = L_pcl.item()
        masked_val = L_masked.item()
        if torch.isnan(L_pcl) or torch.isnan(L_masked):
            return L_pcl
        if pcl_val < 1e-06 or masked_val < 1e-06:
            return L_pcl
        ratio = self.target_ratio * masked_val / pcl_val
        ratio = min(ratio, 20.0)
        if self.ratio_ema is None:
            self.ratio_ema = ratio
        else:
            self.ratio_ema = self.momentum * self.ratio_ema + (1 - self.momentum) * ratio
        return L_pcl * self.ratio_ema

def masked_prediction_loss(preds, x_original, pretrain_mask):
    if pretrain_mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    diff = (preds - x_original)[pretrain_mask]
    return (diff ** 2).mean()

def vicreg_variance_covariance_loss(reps, gamma=1.0, eps=1e-4):
    """VICReg variance hinge + covariance decorrelation on pooled token representations."""
    z = reps[:, -1, :].float()  # (B, D) at terminal token
    if z.shape[0] < 4:
        return torch.tensor(0.0, device=reps.device)
    z_cent = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z_cent.var(dim=0) + eps)
    var_loss = torch.relu(gamma - std).mean()
    cov = (z_cent.T @ z_cent) / (z.shape[0] - 1)
    d = z.shape[1]
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / d
    return var_loss + 0.04 * cov_loss

def pretrain_step(batch, model, pcl_loss_fn, optimizer, lam=LAMBDA_PCL, mask_prob=MASK_PROB, device='cpu', scaler=None, site_adversary=None, lam_site=0.5, pcl_balancer=None, var_reg_weight=0.0):
    from src.models.backbone import apply_random_mask
    model.train()
    optimizer.zero_grad()
    x = batch['x'].to(device, non_blocking=True)
    mask = batch['mask'].to(device, non_blocking=True)
    c_mask = batch['c_mask'].to(device, non_blocking=True)
    x_masked, pretrain_mask = apply_random_mask(x, mask, mask_prob=mask_prob)
    with torch.amp.autocast('cuda', enabled=USE_AMP):
        reps = model.encode(x_masked, mask)
        preds = model.predict(reps)
        L_masked = masked_prediction_loss(preds, x, pretrain_mask)
    with torch.amp.autocast('cuda', enabled=False):
        L_masked = L_masked.float()
        pcl_out = pcl_loss_fn(preds.float(), c_mask, pretrain_mask=pretrain_mask)
        L_pcl = pcl_out['L_PCL']
        L_pcl_balanced = pcl_balancer.balance(L_pcl, L_masked) if pcl_balancer is not None else L_pcl
        L_total = L_masked + lam * L_pcl_balanced
        if var_reg_weight > 0.0:
            L_total = L_total + var_reg_weight * vicreg_variance_covariance_loss(reps)
        if torch.isnan(L_total):
            optimizer.zero_grad()
            return {k: 0.0 for k in ['L_total', 'L_masked', 'L_pcl', 'L_site', 'L_MAP', 'L_PP', 'L_SI', 'L_HH', 'L_SpO2']}
        L_site_val = 0.0
        if site_adversary is not None and 'site_id' in batch:
            site_labels = batch['site_id'].to(device, non_blocking=True).long()
            site_logits = site_adversary(reps.float(), mask, alpha=lam_site)
            L_site = torch.nn.functional.cross_entropy(site_logits, site_labels)
            L_total = L_total + lam_site * L_site
            L_site_val = L_site.item()
    if scaler is not None:
        scaler.scale(L_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return {'L_total': L_total.item(), 'L_masked': L_masked.item(), 'L_pcl': L_pcl.item(), 'L_site': L_site_val, 'L_MAP': pcl_out['losses']['MAP'].item(), 'L_PP': pcl_out['losses']['PP'].item(), 'L_SI': pcl_out['losses']['SI'].item(), 'L_HH': pcl_out['losses']['HH'].item(), 'L_SpO2': pcl_out['losses']['SpO2'].item()}

def pretrain_val_step(batch, model, pcl_loss_fn, lam=LAMBDA_PCL, mask_prob=MASK_PROB, device='cpu'):
    from src.models.backbone import apply_random_mask
    model.eval()
    with torch.no_grad():
        x = batch['x'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        c_mask = batch['c_mask'].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            x_masked, pretrain_mask = apply_random_mask(x, mask, mask_prob=mask_prob)
            reps = model.encode(x_masked, mask)
            preds = model.predict(reps)
            L_masked = masked_prediction_loss(preds, x, pretrain_mask)
        with torch.amp.autocast('cuda', enabled=False):
            pcl_out = pcl_loss_fn(preds.float(), c_mask, pretrain_mask=pretrain_mask)
            L_pcl = pcl_out['L_PCL']
            L_total = L_masked.float() + lam * L_pcl
    return {'L_total': L_total.item(), 'L_masked': L_masked.item(), 'L_pcl': L_pcl.item()}

def run_pretraining(model, pcl_loss_fn, train_loader, val_loader, n_epochs=20, lr=0.0003, lam=LAMBDA_PCL, mask_prob=MASK_PROB, device='cpu', save_path='pcl_pretrained.pt', log_every=1, warmup_fraction=0.3, use_site_adversary=False, lam_site=0.1, use_balancer=True, var_reg_weight=0.0):
    model = model.to(device)
    site_adversary = None
    if use_site_adversary and lam > 0:
        from src.models.backbone import SiteAdversary
        site_adversary = SiteAdversary(model.d_model, n_sites=4).to(device)
    all_params = list(model.parameters())
    if site_adversary is not None:
        all_params += list(site_adversary.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.0001)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.1)
    pcl_balancer = PCLLossBalancer(target_ratio=1.0) if use_balancer else None
    warmup_epochs = max(2, int(n_epochs * warmup_fraction))
    history = {'train_total': [], 'train_masked': [], 'train_pcl': [], 'val_total': [], 'val_masked': [], 'val_pcl': [], 'train_MAP': [], 'train_PP': [], 'train_SI': [], 'train_HH': [], 'train_SpO2': [], 'effective_lam': [], 'balance_ratio': []}
    best_val_masked = float('inf')
    logging.info(f'Starting pretraining — {n_epochs} epochs, λ={lam} (balancer={use_balancer}, var_reg={var_reg_weight}, warmup over {warmup_epochs} epochs), device={device}')
    logging.info(f"{'Epoch':<8} {'Train Total':<14} {'Train Masked':<14} {'Train PCL':<12} {'Val Total':<12} {'λ_eff':<10} {'LR'}")
    logging.info('─' * 85)
    for epoch in range(1, n_epochs + 1):
        if epoch <= warmup_epochs:
            lam_eff = lam * (epoch / warmup_epochs)
        else:
            lam_eff = lam
        train_metrics = {'L_total': [], 'L_masked': [], 'L_pcl': [], 'L_site': [], 'L_MAP': [], 'L_PP': [], 'L_SI': [], 'L_HH': [], 'L_SpO2': []}
        for batch in train_loader:
            step = pretrain_step(batch, model, pcl_loss_fn, optimizer, lam=lam_eff, mask_prob=mask_prob, device=device, scaler=scaler, site_adversary=site_adversary, lam_site=lam_site, pcl_balancer=pcl_balancer, var_reg_weight=var_reg_weight)
            for k in train_metrics:
                train_metrics[k].append(step[k])
        val_metrics = {'L_total': [], 'L_masked': [], 'L_pcl': []}
        for batch in val_loader:
            step = pretrain_val_step(batch, model, pcl_loss_fn, lam=lam, mask_prob=mask_prob, device=device)
            for k in val_metrics:
                val_metrics[k].append(step[k])
        scheduler.step()
        t_total = np.mean(train_metrics['L_total']) if train_metrics['L_total'] else np.nan
        t_masked = np.mean(train_metrics['L_masked']) if train_metrics['L_masked'] else np.nan
        t_pcl = np.mean(train_metrics['L_pcl']) if train_metrics['L_pcl'] else np.nan
        v_total = np.mean(val_metrics['L_total']) if val_metrics['L_total'] else np.nan
        v_masked = np.mean(val_metrics['L_masked']) if val_metrics['L_masked'] else np.nan
        v_pcl = np.mean(val_metrics['L_pcl']) if val_metrics['L_pcl'] else np.nan
        if np.isnan(t_total):
            logging.warning(f'Epoch {epoch}: Train loss is NaN. This usually means the DataLoader is empty (check BATCH_SIZE) or the model exploded.')
        cur_lr = optimizer.param_groups[0]['lr']
        history['train_total'].append(t_total)
        history['train_masked'].append(t_masked)
        history['train_pcl'].append(t_pcl)
        history['val_total'].append(v_total)
        history['val_masked'].append(v_masked)
        history['val_pcl'].append(v_pcl)
        history['train_MAP'].append(np.mean(train_metrics['L_MAP']))
        history['train_PP'].append(np.mean(train_metrics['L_PP']))
        history['train_SI'].append(np.mean(train_metrics['L_SI']))
        history['train_HH'].append(np.mean(train_metrics['L_HH']))
        history['train_SpO2'].append(np.mean(train_metrics['L_SpO2']))
        history['effective_lam'].append(lam_eff)
        history['balance_ratio'].append(float(pcl_balancer.ratio_ema) if (pcl_balancer and pcl_balancer.ratio_ema is not None) else 1.0)
        if epoch % log_every == 0 or epoch == n_epochs:
            logging.info(f'{epoch:<8} {t_total:<14.4f} {t_masked:<14.4f} {t_pcl:<12.4f} {v_total:<12.4f} {lam_eff:<10.3f} {cur_lr:.2e}')
        if v_masked < best_val_masked:
            best_val_masked = v_masked
            torch.save(model.state_dict(), save_path)
            logging.info(f'  -> Checkpoint saved (val_masked={best_val_masked:.4f})')
    logging.info(f'Pretraining complete. Best val_masked: {best_val_masked:.4f}')
    return history


def save_checkpoint(model, optimizer, epoch, history, path):
    torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'history': history}, path)


def load_checkpoint(model, path, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    logging.info(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    return ckpt['history']


def load_state_dict_flexible(path_or_dict, device='cpu'):
    if isinstance(path_or_dict, str):
        o = torch.load(path_or_dict, map_location=device, weights_only=False)
    else:
        o = path_or_dict
    if isinstance(o, dict) and 'model_state' in o:
        o = o['model_state']
    elif isinstance(o, dict) and 'model_state_dict' in o:
        o = o['model_state_dict']
    elif isinstance(o, dict) and 'state_dict' in o:
        o = o['state_dict']
    if isinstance(o, dict):
        cleaned = {}
        for k, v in o.items():
            k_clean = k[7:] if k.startswith('module.') else k
            cleaned[k_clean] = v
        return cleaned
    return o


def plot_loss_curves(history, save_path=None):
    import matplotlib.pyplot as plt
    epochs = range(1, len(history['train_total']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history['train_total'], label='Train')
    axes[0].plot(epochs, history['val_total'], label='Val')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')
    axes[1].plot(epochs, history['train_masked'], label='Train')
    axes[1].plot(epochs, history['val_masked'], label='Val')
    axes[1].set_title('Masked Prediction Loss')
    axes[1].legend()
    axes[1].set_xlabel('Epoch')
    axes[2].plot(epochs, history['train_pcl'], label='Train')
    axes[2].plot(epochs, history['val_pcl'], label='Val')
    axes[2].set_title('PCL Loss')
    axes[2].legend()
    axes[2].set_xlabel('Epoch')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_constraint_curves(history, save_path=None):
    import matplotlib.pyplot as plt
    epochs = range(1, len(history['train_total']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for cname in ['MAP', 'PP', 'SI', 'HH', 'SpO2']:
        key = f'train_{cname}'
        if key in history and history[key]:
            axes[0].plot(epochs, history[key], label=cname)
    axes[0].set_title('Per-Constraint Losses')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].set_yscale('log')
    if 'effective_lam' in history and history['effective_lam']:
        axes[1].plot(epochs, history['effective_lam'], 'k-', linewidth=2)
        axes[1].set_title('Effective λ (Warmup Schedule)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('λ')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

