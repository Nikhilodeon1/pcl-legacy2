import os
import sys
import tempfile
import unittest

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REV_SCRIPTS = os.path.join(_ROOT, "results", "revision", "scripts")
if _REV_SCRIPTS not in sys.path:
    sys.path.insert(0, _REV_SCRIPTS)


class TestStatsRigor(unittest.TestCase):
    def test_holm_bonferroni(self):
        from phase2c_stats_rigor import holm_bonferroni
        adj = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
        # Sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone max 0.06
        self.assertAlmostEqual(adj[3], 0.02, places=7)
        self.assertAlmostEqual(adj[0], 0.03, places=7)
        self.assertAlmostEqual(adj[2], 0.06, places=7)
        self.assertAlmostEqual(adj[1], 0.06, places=7)

    def test_tost_paired_equivalent_vs_not(self):
        from phase2c_stats_rigor import paired_contrast_stats
        # Tight differences near 0 -> equivalent within +-0.02
        a = [0.701, 0.702, 0.700, 0.701, 0.702, 0.700, 0.701, 0.702, 0.700, 0.701]
        b = [0.700] * 10
        res_eq = paired_contrast_stats(a, b, tost_margins=(0.01, 0.02))
        self.assertTrue(res_eq["tost"][0]["equivalent"])
        self.assertTrue(res_eq["tost"][1]["equivalent"])

        # Noisy differences with large CI -> non-significant p_t, but NOT equivalent
        a_wide = [0.75, 0.65, 0.74, 0.66, 0.70]
        b_wide = [0.70, 0.70, 0.70, 0.70, 0.70]
        res_neq = paired_contrast_stats(a_wide, b_wide, tost_margins=(0.01, 0.02))
        self.assertGreater(res_neq["t_p_value"], 0.5)
        self.assertFalse(res_neq["tost"][0]["equivalent"])
        self.assertFalse(res_neq["tost"][1]["equivalent"])

    def test_exact_sign_flip_p(self):
        from phase2c_stats_rigor import paired_sign_flip_p
        # 5 strictly positive differences -> 2 / 32 = 0.0625 exact two-sided
        p5, mode5 = paired_sign_flip_p([0.02, 0.03, 0.01, 0.04, 0.02])
        self.assertEqual(mode5, "exact_32")
        self.assertAlmostEqual(p5, 0.0625, places=7)

        # 10 strictly positive differences -> 2 / 1024 exact two-sided
        p10, mode10 = paired_sign_flip_p([0.02] * 10)
        self.assertEqual(mode10, "exact_1024")
        self.assertAlmostEqual(p10, 2.0 / 1024.0, places=7)

    def test_siteb_patient_bootstrap_handles_constant_collapse(self):
        from phase2c_stats_rigor import siteb_patient_bootstrap
        rng = np.random.default_rng(0)
        n = 100
        labels = np.array([1] * 20 + [0] * 80, dtype=float)
        pids = np.arange(n)
        records = {
            ("Collapsed", 42): {"probs": np.full(n, 0.03), "labels": labels, "pids": pids},
            ("Working", 42): {"probs": labels * 0.6 + rng.uniform(0, 0.3, size=n), "labels": labels, "pids": pids},
        }
        boot = siteb_patient_bootstrap(
            records, B=50, seed=123,
            contrast_pairs=[("Working - Collapsed", "Working", "Collapsed")],
        )
        self.assertTrue(boot["available"])
        self.assertAlmostEqual(boot["by_condition"]["Collapsed"]["mean"], 0.5, places=7)
        self.assertGreater(boot["by_condition"]["Working"]["mean"], 0.9)


class TestPretrainLogging(unittest.TestCase):
    def test_grad_norm_and_pcl_loss_share_logged(self):
        from src.baselines import fresh_model
        from src.losses.pcl_loss import PhysiologicalConstraintLoss
        from src.training.train_utils import run_pretraining

        torch.manual_seed(0)
        n, T, F = 8, 24, 17
        batch = {
            "x": torch.rand(n, T, F),
            "mask": torch.ones(n, T, F, dtype=torch.bool),
            "c_mask": torch.ones(n, T, 5, dtype=torch.bool),
        }
        loader = [batch]
        model = fresh_model(seed=42)
        loss_fn = PhysiologicalConstraintLoss()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "ckpt.pt")
            hist = run_pretraining(
                model, loss_fn, loader, loader,
                n_epochs=2, lam=0.5, device="cpu", save_path=ckpt, log_every=10,
            )
            for key in (
                "train_pcl_weighted",
                "train_pcl_loss_share",
                "train_grad_norm",
                "train_grad_norm_max",
                "train_grad_clip_frac",
            ):
                self.assertIn(key, hist)
                self.assertEqual(len(hist[key]), 2)
                self.assertTrue(all(np.isfinite(v) for v in hist[key]), f"Non-finite in {key}")
            self.assertGreater(hist["train_grad_norm"][0], 0.0)
            self.assertGreater(hist["train_pcl_weighted"][0], 0.0)
            self.assertGreater(hist["train_pcl_loss_share"][0], 0.0)
            self.assertLessEqual(hist["train_pcl_loss_share"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
