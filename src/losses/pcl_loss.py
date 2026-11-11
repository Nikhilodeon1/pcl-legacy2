import torch
import torch.nn as nn
try:
    from src.data.variables import CANONICAL_VARIABLES as VARIABLES
except ImportError:
    from src.data_utils import VARIABLES

def severinghaus_sao2_fraction(PaO2_mmhg: torch.Tensor, eps: float=1e-08) -> torch.Tensor:
    p = torch.clamp(PaO2_mmhg, min=1.0)
    inner = p ** 3 + 150.0 * p
    return inner / (inner + 23400.0 + eps)

class PhysiologicalConstraintLoss(nn.Module):

    def __init__(self, eps: float=1e-08, si_weight: float=0.0, pp_weight: float=0.0,
                 map_weight: float=1.0, hh_weight: float=1.0, spo2_weight: float=1.0,
                 placebo_seed=None):
        super().__init__()
        self.eps = eps
        self.si_weight = si_weight
        self.pp_weight = pp_weight
        self.map_weight = map_weight
        self.hh_weight = hh_weight
        self.spo2_weight = spo2_weight
        self.idx = {v: i for i, v in enumerate(VARIABLES)}
        if placebo_seed is not None:
            # Placebo: identical loss forms, but each constraint variable reads a different (fixed, seeded,
            # deranged) channel. Preserves loss structure/scale class while removing the physiological relationship.
            import random
            names = [v for v in ('SBP', 'DBP', 'MAP', 'SpO2', 'pH', 'HCO3', 'pCO2', 'PaO2') if v in self.idx]
            rng = random.Random(placebo_seed)
            while True:
                perm = names[:]
                rng.shuffle(perm)
                if all(a != b for a, b in zip(names, perm)):
                    break
            orig = dict(self.idx)
            self.idx = dict(orig)
            for a, b in zip(names, perm):
                self.idx[a] = orig[b]

    def _get(self, x: torch.Tensor, var: str) -> torch.Tensor:
        return x[:, :, self.idx[var]]

    def _mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return ((pred - target)[mask] ** 2).mean()

    def _l1(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return torch.abs((pred - target)[mask]).mean()

    def _constraint_mask(self, c_mask_k, pretrain_mask, var_indices):
        if pretrain_mask is None:
            return c_mask_k
        any_var_masked = torch.zeros_like(c_mask_k)
        for vi in var_indices:
            any_var_masked = any_var_masked | pretrain_mask[:, :, vi]
        return c_mask_k & any_var_masked

    def forward(self, x: torch.Tensor, c_mask: torch.Tensor, pretrain_mask: torch.Tensor=None, hh_weight: float=1.0) -> dict:
        idx = self.idx
        SBP = self._get(x, 'SBP')
        DBP = self._get(x, 'DBP')
        MAP = self._get(x, 'MAP')
        HR = self._get(x, 'HR')
        SpO2 = self._get(x, 'SpO2')
        pH = self._get(x, 'pH')
        HCO3 = self._get(x, 'HCO3')
        pCO2 = self._get(x, 'pCO2')
        PaO2 = self._get(x, 'PaO2')
        losses: dict = {}
        active: dict = {}
        SBP_d = SBP * (300 - 40) + 40
        DBP_d = DBP * (200 - 20) + 20
        MAP_d = MAP * (200 - 20) + 20
        HR_d = HR * (300 - 20) + 20
        m0 = self._constraint_mask(c_mask[:, :, 0], pretrain_mask, [idx['SBP'], idx['DBP'], idx['MAP']])
        predicted_map = DBP_d + (SBP_d - DBP_d) / 3.0
        predicted_map_norm = (predicted_map - 20) / (200 - 20 + self.eps)
        losses['MAP'] = self._l1(MAP, predicted_map_norm, m0)
        active['MAP'] = m0.sum().item()
        m1 = self._constraint_mask(c_mask[:, :, 1], pretrain_mask, [idx['SBP'], idx['DBP'], idx['MAP']])
        PP_direct = SBP_d - DBP_d
        PP_via_map = 3.0 * (MAP_d - DBP_d)
        losses['PP'] = self._mse(PP_direct / 260.0, PP_via_map / 260.0, m1)
        active['PP'] = m1.sum().item()
        m2 = self._constraint_mask(c_mask[:, :, 2], pretrain_mask, [idx['HR'], idx['SBP']])
        SBP_floor = torch.clamp(SBP_d, min=40.0)
        pred_si = HR_d / (SBP_floor + self.eps)
        target_si = torch.full_like(pred_si, 0.7)
        losses['SI'] = self._mse(pred_si / 3.0, target_si / 3.0, m2)
        active['SI'] = m2.sum().item()
        m3 = self._constraint_mask(c_mask[:, :, 3], pretrain_mask, [idx['pH'], idx['HCO3'], idx['pCO2']])
        HCO3_raw = HCO3 * (60 - 5) + 5
        pCO2_raw = pCO2 * (150 - 10) + 10
        H_conc = HCO3_raw / (0.0307 * pCO2_raw + self.eps)
        H_conc_safe = torch.clamp(H_conc, min=0.01, max=1000.0)
        ph_pred = 6.1 + torch.log10(H_conc_safe)
        ph_pred_norm = (ph_pred - 6.5) / (7.9 - 6.5 + self.eps)
        ph_pred_norm_clamped = torch.clamp(ph_pred_norm, 0.0, 1.0)
        hh_raw_norm = self._mse(pH, ph_pred_norm_clamped, m3)
        if isinstance(hh_weight, torch.Tensor):
            hh_w = hh_weight.mean()
        else:
            hh_w = float(hh_weight)
        losses['HH'] = hh_raw_norm * hh_w
        active['HH'] = m3.sum().item()
        m4 = self._constraint_mask(c_mask[:, :, 4], pretrain_mask, [idx['SpO2'], idx['PaO2']])
        PaO2_raw = torch.clamp(PaO2 * (700 - 20) + 20, min=1.0)
        sao2 = severinghaus_sao2_fraction(PaO2_raw, eps=self.eps)
        spo2_pred_pct = sao2 * 100.0
        spo2_pred_norm = torch.clamp((spo2_pred_pct - 50.0) / 50.0, 0.0, 1.0)
        losses['SpO2'] = self._mse(SpO2, spo2_pred_norm, m4)
        active['SpO2'] = m4.sum().item()
        L_PCL = (self.map_weight * losses['MAP'] +
                 self.pp_weight * losses['PP'] +
                 self.si_weight * losses['SI'] +
                 self.hh_weight * losses['HH'] +
                 self.spo2_weight * losses['SpO2'])
        if torch.isnan(L_PCL):
            L_PCL = torch.tensor(0.0, device=x.device, requires_grad=True)
        return {'L_PCL': L_PCL, 'losses': losses, 'active': active}
PCLoss = PhysiologicalConstraintLoss
