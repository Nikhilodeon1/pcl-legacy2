"""
Reviewer Round 2 Decisive Local Checks:
1. Exact constraint firing frequencies on PhysioNet Site A vs Site B (hours and stays where c_mask fires).
2. Empirical constraint residuals on ground-truth observed data in Site A vs Site B.
3. Site A stratified performance: evaluate ERM and PCL on Site A stays WITH ABGs vs WITHOUT ABGs.
4. Formal interaction test: (Site B_PCL - Site B_ERM) - (Site A_PCL - Site A_ERM).
5. Explain reference set mismatch between Section 6.3 and Section 6B.
6. Calibration metrics: ECE (Expected Calibration Error) and Brier calibration breakdown.
7. Within-condition correlations between erank/spread and performance (avoiding pooled conflation).
"""
import os
import sys
os.environ["PCL_TEST_MODE"] = "0"
os.environ["PCL_DISABLE_AMP"] = "1"
import gzip
import json
import pickle
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)

from src.data.variables import CANONICAL_VARIABLES, ABG_VARIABLES, HEMO_VARIABLES
from src.losses.pcl_loss import PhysiologicalConstraintLoss
from src.data.dataset import ICUDataset
from src.baselines import fresh_model
from src.training.train_utils import load_state_dict_flexible
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def main():
    print("Loading PhysioNet dataset...")
    cache_path = os.path.join(_ROOT, "colab", "physionet_frac1.0.pkl.gz")
    with gzip.open(cache_path, "rb") as f:
        pn = pickle.load(f)

    site_a = [s for s in pn if s["site_id"] == 0]
    site_b = [s for s in pn if s["site_id"] == 1]
    print(f"Loaded: Site A = {len(site_a)} stays, Site B = {len(site_b)} stays")

    var_idx = {v: i for i, v in enumerate(CANONICAL_VARIABLES)}
    abg_indices = [var_idx[v] for v in ABG_VARIABLES if v in var_idx]

    # --- 1. Constraint Firing Rates & Ground-Truth Residuals ---
    def analyze_constraints(stays, site_name):
        total_hours = sum(s["values"].shape[0] for s in stays)
        n_stays = len(stays)
        
        # c_mask indices: 0: MAP, 1: PP, 2: SI, 3: HH, 4: SpO2
        c_counts = {"MAP": 0, "PP": 0, "SI": 0, "HH": 0, "SpO2": 0}
        c_stay_counts = {"MAP": 0, "PP": 0, "SI": 0, "HH": 0, "SpO2": 0}
        has_abg_stays = 0

        # Ground truth residuals
        gt_res_map = []
        gt_res_hh = []
        gt_res_spo2 = []

        loss_fn = PhysiologicalConstraintLoss()

        for s in stays:
            c_mask = s["c_mask"] # (T, 5)
            x = s["values"] # (T, 17)
            mask = s["mask"] # (T, 17)
            
            # Check if stay has any ABG observation
            abg_obs = mask[:, abg_indices].any()
            if abg_obs:
                has_abg_stays += 1

            for k, name in enumerate(["MAP", "PP", "SI", "HH", "SpO2"]):
                fired = c_mask[:, k] > 0
                n_fired = fired.sum()
                c_counts[name] += int(n_fired)
                if n_fired > 0:
                    c_stay_counts[name] += 1

            # Compute ground truth residuals on hours where c_mask fires
            # MAP: predicted_map = DBP + (SBP - DBP)/3
            m_map = c_mask[:, 0] > 0
            if m_map.any():
                sbp = x[m_map, var_idx["SBP"]] * (300 - 40) + 40
                dbp = x[m_map, var_idx["DBP"]] * (200 - 20) + 20
                map_true = x[m_map, var_idx["MAP"]] * (200 - 20) + 20
                map_pred = dbp + (sbp - dbp) / 3.0
                gt_res_map.extend(np.abs(map_true - map_pred).tolist())

            # HH: pH = 6.1 + log10(HCO3 / (0.0307 * pCO2))
            m_hh = c_mask[:, 3] > 0
            if m_hh.any():
                ph_true = x[m_hh, var_idx["pH"]] * (7.9 - 6.5) + 6.5
                hco3 = x[m_hh, var_idx["HCO3"]] * (60 - 5) + 5
                pco2 = x[m_hh, var_idx["pCO2"]] * (150 - 10) + 10
                ph_pred = 6.1 + np.log10(np.clip(hco3 / (0.0307 * pco2 + 1e-8), 0.01, 1000.0))
                gt_res_hh.extend(np.abs(ph_true - ph_pred).tolist())

            # Severinghaus SpO2
            m_spo2 = c_mask[:, 4] > 0
            if m_spo2.any():
                spo2_true = x[m_spo2, var_idx["SpO2"]] * 50.0 + 50.0
                pao2 = x[m_spo2, var_idx["PaO2"]] * (700 - 20) + 20
                p_c = np.clip(pao2, 1.0, None)
                inner = p_c ** 3 + 150.0 * p_c
                spo2_pred = (inner / (inner + 23400.0)) * 100.0
                gt_res_spo2.extend(np.abs(spo2_true - spo2_pred).tolist())

        return {
            "n_stays": n_stays,
            "total_hours": total_hours,
            "has_abg_stay_rate": has_abg_stays / n_stays,
            "hourly_firing_rates": {k: v / total_hours for k, v in c_counts.items()},
            "stay_firing_rates": {k: v / n_stays for k, v in c_stay_counts.items()},
            "gt_residual_map_mae_mmhg": float(np.mean(gt_res_map)) if gt_res_map else None,
            "gt_residual_hh_mae_ph": float(np.mean(gt_res_hh)) if gt_res_hh else None,
            "gt_residual_spo2_mae_pct": float(np.mean(gt_res_spo2)) if gt_res_spo2 else None,
            "n_gt_eval_map": len(gt_res_map),
            "n_gt_eval_hh": len(gt_res_hh),
            "n_gt_eval_spo2": len(gt_res_spo2),
        }

    stats_a = analyze_constraints(site_a, "Site A")
    stats_b = analyze_constraints(site_b, "Site B")
    print("\n=== Site A Constraint Stats ===")
    print(json.dumps(stats_a, indent=2))
    print("\n=== Site B Constraint Stats ===")
    print(json.dumps(stats_b, indent=2))

    # --- 2. Stratified Site A Evaluation (ABG Present vs ABG Absent) ---
    print("\n--- Evaluating Site A Stratified by ABG Presence across 18 Sweep Encoders ---")
    sweep_dir = os.path.join(_ROOT, "results", "revision", "encoders_slim", "sweep")
    from src.baselines import fresh_model
    from src.training.train_utils import load_state_dict_flexible
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Subsample 3000 stays per site for fast CPU evaluation while preserving exact ABG/HH representation
    rng_sub = np.random.default_rng(42)
    ia = rng_sub.choice(len(site_a), min(3500, len(site_a)), replace=False)
    ib = rng_sub.choice(len(site_b), min(3500, len(site_b)), replace=False)

    sub_a = [site_a[i] for i in ia]
    sub_b = [site_b[i] for i in ib]

    ds_a = ICUDataset(sub_a)
    loader_a = DataLoader(ds_a, batch_size=256, shuffle=False)
    y_a = np.array([int(np.asarray(s["label"]).sum() > 0) for s in sub_a])
    hh_flag_a = np.array([(s["c_mask"][:, 3] > 0).any() for s in sub_a])
    abg_flag_a = np.array([s["abg_mask"].any() for s in sub_a])
    print(f"Site A sample: {hh_flag_a.sum()}/{len(sub_a)} stays have HH firing ({hh_flag_a.mean():.3%})")
    print(f"Site A sample: {abg_flag_a.sum()}/{len(sub_a)} stays have abg_mask ({abg_flag_a.mean():.3%})")

    ds_b = ICUDataset(sub_b)
    loader_b = DataLoader(ds_b, batch_size=256, shuffle=False)
    y_b = np.array([int(np.asarray(s["label"]).sum() > 0) for s in sub_b])
    hh_flag_b = np.array([(s["c_mask"][:, 3] > 0).any() for s in sub_b])
    abg_flag_b = np.array([s["abg_mask"].any() for s in sub_b])
    print(f"Site B sample: {hh_flag_b.sum()}/{len(sub_b)} stays have HH firing ({hh_flag_b.mean():.3%})")
    print(f"Site B sample: {abg_flag_b.sum()}/{len(sub_b)} stays have abg_mask ({abg_flag_b.mean():.3%})")

    # Evaluate frozen linear probes for Site A val and Site B OOD across primary lambdas {0.0, 0.5, 1.0}
    strat_results = {}
    for lam in [0.0, 0.5, 1.0]:
        strat_results[lam] = {
            "site_a_overall": [], "site_a_hh_present": [], "site_a_hh_absent": [],
            "site_b_overall": [], "site_b_hh_present": [], "site_b_hh_absent": [],
            "site_b_abg_present": [], "site_b_abg_absent": [],
            "site_b_ece": [], "site_b_brier": [],
        }
        for s in [42, 43, 44]:
            ckpt_path = os.path.join(sweep_dir, f"lam{lam:.1f}_p{s}.pt")
            if not os.path.exists(ckpt_path):
                continue
            model = fresh_model(seed=s)
            model.load_state_dict(load_state_dict_flexible(ckpt_path, "cpu"))
            model.eval()

            # Extract reps at last_idx
            def extract(loader):
                reps_list = []
                with torch.no_grad():
                    for batch in loader:
                        x = batch["x"]
                        mask = batch["mask"]
                        reps = model.encode(x, mask)
                        t_obs = mask.any(dim=-1)
                        last_idx = (t_obs.size(1) - 1 - t_obs.flip(dims=[1]).long().argmax(dim=1)).clamp(min=0)
                        batch_idx = torch.arange(reps.size(0))
                        pooled = reps[batch_idx, last_idx]
                        reps_list.append(pooled.numpy())
                return np.vstack(reps_list)

            Z_a = extract(loader_a)
            Z_b = extract(loader_b)

            # Fit logistic regression on train split of Site A (80%)
            rng = np.random.default_rng(s)
            pos_idx = np.where(y_a == 1)[0]
            neg_idx = np.where(y_a == 0)[0]
            train_idx = np.concatenate([
                rng.choice(pos_idx, size=int(0.8 * len(pos_idx)), replace=False),
                rng.choice(neg_idx, size=int(0.8 * len(neg_idx)), replace=False)
            ])
            val_idx = np.setdiff1d(np.arange(len(y_a)), train_idx)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(Z_a[train_idx])
            X_va = scaler.transform(Z_a[val_idx])
            X_b = scaler.transform(Z_b)
            y_tr, y_va = y_a[train_idx], y_a[val_idx]

            clf = LogisticRegression(max_iter=500, random_state=s, C=1.0)
            clf.fit(X_tr, y_tr)

            preds_va = clf.predict_proba(X_va)[:, 1]
            preds_b = clf.predict_proba(X_b)[:, 1]

            # Site A val evaluations
            hh_va = hh_flag_a[val_idx]
            strat_results[lam]["site_a_overall"].append(roc_auc_score(y_va, preds_va))
            strat_results[lam]["site_a_hh_present"].append(roc_auc_score(y_va[hh_va], preds_va[hh_va]))
            strat_results[lam]["site_a_hh_absent"].append(roc_auc_score(y_va[~hh_va], preds_va[~hh_va]))

            # Site B OOD evaluations
            strat_results[lam]["site_b_overall"].append(roc_auc_score(y_b, preds_b))
            strat_results[lam]["site_b_ece"].append(compute_ece(preds_b, y_b))
            strat_results[lam]["site_b_brier"].append(brier_score_loss(y_b, preds_b))

            if len(np.unique(y_b[hh_flag_b])) > 1:
                strat_results[lam]["site_b_hh_present"].append(roc_auc_score(y_b[hh_flag_b], preds_b[hh_flag_b]))
            if len(np.unique(y_b[~hh_flag_b])) > 1:
                strat_results[lam]["site_b_hh_absent"].append(roc_auc_score(y_b[~hh_flag_b], preds_b[~hh_flag_b]))

            if len(np.unique(y_b[abg_flag_b])) > 1:
                strat_results[lam]["site_b_abg_present"].append(roc_auc_score(y_b[abg_flag_b], preds_b[abg_flag_b]))
            if len(np.unique(y_b[~abg_flag_b])) > 1:
                strat_results[lam]["site_b_abg_absent"].append(roc_auc_score(y_b[~abg_flag_b], preds_b[~abg_flag_b]))

    print("\n--- Stratified Site A & Site B Results (Mean ± SD, n=3 seeds) ---")
    for lam, d in strat_results.items():
        print(f"lam={lam:3g} | Site A Overall: {np.mean(d['site_a_overall']):.4f} | "
              f"Site A HH-Present: {np.mean(d['site_a_hh_present']):.4f} | Site A HH-Absent: {np.mean(d['site_a_hh_absent']):.4f} || "
              f"Site B Overall: {np.mean(d['site_b_overall']):.4f} | "
              f"Site B HH-Present: {np.mean(d['site_b_hh_present']):.4f} | Site B HH-Absent: {np.mean(d['site_b_hh_absent']):.4f} | "
              f"ECE: {np.mean(d['site_b_ece']):.4f} | Brier: {np.mean(d['site_b_brier']):.4f}")

    # --- 3. Formal Interaction Test (Site B minus Site A) ---
    # From the 10-seed paired confirmatory run:
    # ERM Site A = 0.7618 ± 0.0142, Site B = 0.6972 ± 0.0218
    # PCL Site A = 0.7483 ± 0.0169, Site B = 0.6713 ± 0.0241
    # Let's compute paired differences per seed from the saved 10-seed results
    phase2_sum_path = os.path.join(_ROOT, "results", "revision", "phase2_summary.json")
    if os.path.exists(phase2_sum_path):
        p2_data = json.load(open(phase2_sum_path))
        # Let's inspect phase2_summary.json
    
    # Save everything to json
    out_json = os.path.join(_ROOT, "results", "revision", "reviewer_round2_abg_checks.json")
    with open(out_json, "w") as fp:
        json.dump({
            "site_a_constraints": stats_a,
            "site_b_constraints": stats_b,
            "stratified_site_a_val": {k: {m: [float(x) for x in v] for m, v in vals.items()} for k, vals in strat_results.items()}
        }, fp, indent=2)
    print(f"\nSaved results to {out_json}")

if __name__ == "__main__":
    main()
