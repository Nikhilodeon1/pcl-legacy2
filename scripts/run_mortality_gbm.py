"""
Compute exact HistGradientBoostingClassifier baselines for:
1. In-Hospital Mortality under fixed T=48 on MIMIC and eICU (bidirectional transfer)
2. Sepsis on PhysioNet A and B under true labels
3. LOS continuous prediction under fixed T=24
Saves verified output to results/revision/verified_gbm_baselines.json.
"""
import json
import logging
import os
import pickle
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error, brier_score_loss
from sklearn.model_selection import StratifiedKFold, KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
SEEDS = [42, 43, 44, 45, 46]


def extract_summary_features(samples, max_T=48):
    X, y_mort, y_sep, y_los = [], [], [], []
    for s in samples:
        vals = s["values"][:max_T]
        mask = s["mask"][:max_T]
        m_h = s.get("mortality_hospital", s.get("mortality", 0))
        if hasattr(m_h, "item"): m_h = m_h.item()
        y_mort.append(float(m_h))

        l_h = s.get("los_h", 0)
        if hasattr(l_h, "item"): l_h = l_h.item()
        y_los.append(float(l_h))

        if "label" in s:
            sep = float(np.sum(s["label"][:max_T]) > 0)
        elif "sepsis" in s:
            sep = float(s["sepsis"].item() if hasattr(s["sepsis"], "item") else s["sepsis"])
        else:
            sep = 0.0
        y_sep.append(sep)

        feats = []
        for col in range(vals.shape[1]):
            col_m = mask[:, col]
            col_v = vals[:, col]
            cnt = float(np.sum(col_m))
            if cnt > 0:
                obs = col_v[col_m > 0]
                feats.extend([float(np.mean(obs)), float(np.min(obs)), float(np.max(obs)), float(obs[-1]), cnt])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        X.append(feats)
    return np.array(X), np.array(y_mort), np.array(y_sep), np.array(y_los)


def main():
    out = {}
    cache_dir = os.path.join(_ROOT, "results", "revision", "cache")

    # 1. Mortality (Fixed T=48)
    mim_f = os.path.join(cache_dir, "mimic_frac0.25_s42.pkl")
    eicu_f = os.path.join(cache_dir, "eicu_frac0.25_s42.pkl")
    if os.path.exists(mim_f) and os.path.exists(eicu_f):
        logging.info("Loading MIMIC and eICU caches for Mortality GBM...")
        mim = pickle.load(open(mim_f, "rb"))
        eicu = pickle.load(open(eicu_f, "rb"))
        X_mim, y_mim_mort, _, _ = extract_summary_features(mim, max_T=48)
        X_eicu, y_eicu_mort, _, _ = extract_summary_features(eicu, max_T=48)

        mort_res = {"mimic_in_domain": [], "eicu_in_domain": [], "mimic_to_eicu_ood": [], "eicu_to_mimic_ood": []}
        for s in SEEDS:
            # MIMIC -> eICU
            clf_m = HistGradientBoostingClassifier(max_iter=200, random_state=s)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
            tr_idx, va_idx = next(skf.split(X_mim, y_mim_mort))
            clf_m.fit(X_mim[tr_idx], y_mim_mort[tr_idx])
            va_p = clf_m.predict_proba(X_mim[va_idx])[:, 1]
            ood_p = clf_m.predict_proba(X_eicu)[:, 1]
            mort_res["mimic_in_domain"].append(float(roc_auc_score(y_mim_mort[va_idx], va_p)))
            mort_res["mimic_to_eicu_ood"].append(float(roc_auc_score(y_eicu_mort, ood_p)))

            # eICU -> MIMIC
            clf_e = HistGradientBoostingClassifier(max_iter=200, random_state=s)
            skf_e = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
            tr_e, va_e = next(skf_e.split(X_eicu, y_eicu_mort))
            clf_e.fit(X_eicu[tr_e], y_eicu_mort[tr_e])
            va_pe = clf_e.predict_proba(X_eicu[va_e])[:, 1]
            ood_pe = clf_e.predict_proba(X_mim)[:, 1]
            mort_res["eicu_in_domain"].append(float(roc_auc_score(y_eicu_mort[va_e], va_pe)))
            mort_res["eicu_to_mimic_ood"].append(float(roc_auc_score(y_mim_mort, ood_pe)))

        out["mortality_T48"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in mort_res.items()}
        logging.info(f"Mortality GBM: MIMIC In-domain={out['mortality_T48']['mimic_in_domain']['mean']:.4f}, MIMIC->eICU OOD={out['mortality_T48']['mimic_to_eicu_ood']['mean']:.4f}")

    # 2. Sepsis on PhysioNet A & B
    pn_f = os.path.join(cache_dir, "physionet_frac1.0.pkl")
    if os.path.exists(pn_f):
        logging.info("Loading PhysioNet cache for Sepsis & LOS GBM...")
        pn = pickle.load(open(pn_f, "rb"))
        site_a = [s for s in pn if s["site_id"] == 0]
        site_b = [s for s in pn if s["site_id"] == 1]
        Xa, _, ya_sep, ya_los = extract_summary_features(site_a, max_T=24)
        Xb, _, yb_sep, yb_los = extract_summary_features(site_b, max_T=24)

        sep_res = {"site_a_val_auc": [], "site_b_ood_auc": [], "site_b_ood_brier": [], "site_b_ood_auprc": []}
        los_res = {"site_a_r2": [], "site_b_r2": [], "site_b_mae": []}
        for s in SEEDS:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
            tr, va = next(skf.split(Xa, ya_sep))

            # Sepsis
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=s, class_weight="balanced")
            clf.fit(Xa[tr], ya_sep[tr])
            va_p = clf.predict_proba(Xa[va])[:, 1]
            sb_p = clf.predict_proba(Xb)[:, 1]
            sep_res["site_a_val_auc"].append(float(roc_auc_score(ya_sep[va], va_p)))
            sep_res["site_b_ood_auc"].append(float(roc_auc_score(yb_sep, sb_p)))
            sep_res["site_b_ood_auprc"].append(float(average_precision_score(yb_sep, sb_p)))
            sep_res["site_b_ood_brier"].append(float(brier_score_loss(yb_sep, sb_p)))

            # LOS (log-transformed)
            reg = HistGradientBoostingRegressor(max_iter=200, random_state=s)
            reg.fit(Xa[tr], np.log1p(ya_los[tr]))
            pred_b = np.expm1(reg.predict(Xb))
            pred_a = np.expm1(reg.predict(Xa[va]))
            los_res["site_a_r2"].append(float(r2_score(ya_los[va], pred_a)))
            los_res["site_b_r2"].append(float(r2_score(yb_los, pred_b)))
            los_res["site_b_mae"].append(float(mean_absolute_error(yb_los, pred_b)))

        out["sepsis_T24"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in sep_res.items()}
        out["los_T24"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in los_res.items()}
        out["physionet_b_sepsis_prevalence"] = float(np.mean(yb_sep))
        out["physionet_b_constant_brier"] = float(np.mean(yb_sep) * (1.0 - np.mean(yb_sep)))

    out_file = os.path.join(_ROOT, "results", "revision", "verified_gbm_baselines.json")
    with open(out_file, "w") as fp:
        json.dump(out, fp, indent=2)
    logging.info(f"Saved verified GBM baselines to {out_file}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
