import copy
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
NOISE_GRID = [0.05, 0.1, 0.2]
DROPOUT_GRID = [0.1, 0.2, 0.3]
WD_GRID = [0.0001, 0.001, 0.01]
MIXUP_ALPHA = 0.2

def _configs(grids=None):
    g = grids or {}
    noise = g.get('noise', NOISE_GRID)
    drops = g.get('dropout', DROPOUT_GRID)
    wds = g.get('wd', WD_GRID)
    mix = g.get('mixup_alpha', MIXUP_ALPHA)
    cfgs = []
    for v in noise:
        cfgs.append(('DenoisingAE', {'pre_noise': v, 'ft_noise': 0.0}))
    for v in noise:
        cfgs.append(('NoiseAug', {'pre_noise': v, 'ft_noise': v}))
    for d in drops:
        for w in wds:
            cfgs.append(('DropoutWD', {'dropout': d, 'wd': w}))
    cfgs.append(('Mixup', {'mixup_alpha': mix}))
    return cfgs

def _train_one(name, setting, train_loader, val_loader, *, task, device, pretrain_epochs, finetune_epochs, base_init_path, seed):
    from src.baselines import fresh_model, run_erm_pretraining
    from src.eval.evaluate_utils import run_finetuning, evaluate_model
    import torch, os
    dropout = setting.get('dropout')
    wd = setting.get('wd', 0.0001)
    pre_noise = setting.get('pre_noise', 0.0)
    ft_noise = setting.get('ft_noise', 0.0)
    mixup_alpha = setting.get('mixup_alpha', 0.0)
    model = fresh_model(seed=seed, dropout=dropout)
    if base_init_path and os.path.exists(base_init_path):
        model.load_state_dict(torch.load(base_init_path, map_location='cpu', weights_only=True))
    ckpt_dir = os.path.dirname(base_init_path) if base_init_path else '.'
    scratch = os.path.join(ckpt_dir, f'_reg_pre_{name}.pt')
    run_erm_pretraining(model, train_loader, val_loader, n_epochs=pretrain_epochs, device=device, save_path=scratch, input_noise=pre_noise, weight_decay=wd)
    model.add_classification_head(task)
    run_finetuning(model, train_loader, val_loader, task, n_epochs=finetune_epochs, device=device, save_path=None, input_noise=ft_noise, mixup_alpha=mixup_alpha, weight_decay=wd)
    val = evaluate_model(model, val_loader, task, device=device, split_name=f'{name} {setting} Val')
    return (model, val)

def run_regularization_baselines(train_loader, val_loader, ood_loaders, *, task, device, pretrain_epochs, finetune_epochs, base_init_path=None, seed=42, grids=None):
    from src.eval.evaluate_utils import evaluate_model
    results = {}
    best = {}
    sweeps = {}
    for name, setting in _configs(grids):
        logging.info(f'\n[A3] {name} sweep point: {setting}')
        try:
            model, val = _train_one(name, setting, train_loader, val_loader, task=task, device=device, pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs, base_init_path=base_init_path, seed=seed)
        except Exception as e:
            logging.warning(f'[A3] {name} {setting} failed: {e}')
            continue
        va = val.get('auroc')
        sweeps.setdefault(name, []).append({'setting': setting, 'val_auroc': va})
        if va is not None and (name not in best or va > best[name][0]):
            best[name] = (va, copy.deepcopy(model), setting)
    attempted = {n for n, _ in _configs(grids)}
    dropped = attempted - set(best)
    if dropped:
        logging.warning(f'[A3] No valid (2-class) sweep point for: {sorted(dropped)} — excluded from results (expected in tiny test-mode splits).')
    for name, (va, model, setting) in best.items():
        ood = {}
        for site, loader in ood_loaders.items():
            ood[site] = evaluate_model(model, loader, task, device=device, split_name=f'{name} {site}')
        val = evaluate_model(model, val_loader, task, device=device, split_name=f'{name} Val(best)')
        results[name] = {'val': val, 'ood': ood, 'chosen': setting, 'sweep': sweeps.get(name, [])}
        logging.info(f'[A3] {name} BEST setting={setting} val_auroc={va:.4f}')
    return results
