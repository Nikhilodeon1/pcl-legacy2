# Surprises (one page, ranked by consequence for the paper's claims)

Status as of 2026-10-06. Items marked (pending) await the MIMIC-IV / eICU cache or the pod.

1. **LOS performance is explained by the observation mask / stay length, not physiology (G1 flag, PREREG threshold R2 > 0.05).**
   A gradient-boosted model fed only mask/length features reaches R2 = 0.244 (PhysioNet-A val) and 0.216 (PhysioNet-B);
   the fine-tuned transformers get 0.26-0.27 and 0.21-0.23. Randomly permuting all observed values across stays leaves
   transformer R2 unchanged (ERM 0.263 -> 0.263, PCL 0.253 -> 0.253, DRO 0.268 -> 0.268); uniform-noise values change it by <= 0.014.
   Formal probe (g1_probe.csv; PN full, MIMIC/eICU 25% stay subsamples): LOS mask+length R2 = +0.244 / +0.216 / -0.115 / -0.121 on PN-A val / PN-B / MIMIC / eICU
   (transformers: 0.26 / 0.22 / -0.12 / -0.10..-0.11), i.e. the whole LOS result including the far-target collapse is reproduced without any physiology.
   Mortality mask+length AUROC = 0.787 (MIMIC val), 0.661 (MIMIC->eICU), 0.762 (eICU val), 0.713 (eICU->MIMIC) vs fine-tuned 0.85 / 0.75 / 0.84 / 0.81; length-only
   gives 0.627 on MIMIC val. Both flags (LOS R2>0.05, mortality AUROC>0.60) tripped on source val AND zero-shot targets.
   Mechanism: the head pools the encoder state at the LAST OBSERVED hour (backbone.py ClassificationHead) and the encoder adds an
   absolute sinusoidal position at that hour; for the 66% of PhysioNet stays shorter than 48h the last observed hour equals
   min(LOS, 48) within 6h for 99.9% of stays (MIMIC: 38% of stays <48h, Spearman(last-obs, min(LOS,48)) = 0.65).
   Still pending: values-shuffled test on mortality and on MIMIC/eICU LOS targets (pod / predict step).
2. **"DRO" is not distributionally robust.** The `dro_pretrained.pt` encoder is produced by `run_erm_pretraining` (plain masked
   prediction, `_archive/run_paper_experiments.py` L472-479). Group-DRO only exists in the sepsis fine-tuning stage
   (`DROFinetuner`), with groups = `env_id` = site id, i.e. ONE group on PhysioNet-A ("DRO degenerates to ERM (expected for
   single-site training)", L489-490). pcl-legacy2 never runs that stage. So ERM vs DRO here is a pair of replicate encoders.
   The paper's Methods ("group-distributionally-robust masked prediction") is wrong.
3. **One pretrained encoder per method; the 3 seeds are fine-tune seeds.** `results_lambda17` came from `run_lambda_sweep.sh`
   (single seed 42). The prior study itself reports 4-7 AUROC points of run-to-run variability across full-pipeline seeds
   (URTC text L78), far above the 0.01-0.02 method differences this paper calls significant. Independently pretrained encoders
   (`results_final_s{42,43,44}`) may exist on the pod; inventory script provided. This makes B2 (independent pretraining) possibly cheap.
4. **PCL penalty has three active constraints, not five.** Defaults `pp_weight=0`, `si_weight=0` (`pcl_loss.py` L38, L184-187): only
   MAP, Henderson-Hasselbalch and Severinghaus SpO2-PaO2. The paper's Methods says five. SI was dropped because "ablation showed it
   hurt OOD performance" (an OOD-informed design choice).
5. **The PCL encoder used in this paper is lambda = 1.0, not 0.5 (corrects my earlier G0 note).** `results_lambda17/ckpt/pcl_pretrained.pt` carries effective lambda 1.0 (checkpoint history, epoch 16) and `lambda.log` L18 says `lambda=1.0`; it is a separate run from the sweep's `lambda_1.0` / `lambda_0.5` checkpoints (weights differ). The same run's sepsis lambda sweep (results_lambda17/paper_results.json) shows lambda=1.0 is a BAD setting: OOD AUROC PN-B/MIMIC/eICU = 0.675/0.645/0.585 vs lambda=0.5 = 0.750/0.695/0.662 vs lambda=0 (ERM) = 0.692/0.601/0.548. The sweep script's own header calls lambda=1.0 'an unjustified config default'. So the 'PCL is worst' result of this paper is for one arbitrary, known-poor lambda at one pretraining seed. `config.py` default (0.5) is NOT what was run.
   Good news: **independent pretraining seeds exist.** `IMPORTED/pclCodebase/results_lambda17{,_s43,_s44}/ckpt/lambda_{0.0,0.1,0.5,1.0,2.0,5.0}_pretrained.pt` (lambda_0.0 is bit-identical to `erm_pretrained.pt` at seed 42, so lambda_0.0 = ERM) give ERM and 5 PCL weights at 3 independent pretraining seeds. B2 and B3 are essentially free.
6. **LOS ranking is ERM > DRO > PCL at all three targets, not DRO >= ERM > PCL.** ERM is significantly above DRO at eICU
   (t = +7.04, d = 4.1). The paper's "same ranking as the corrected sepsis result" holds only for PCL-last (8/9 LOS cells).
7. **The pooling explanation is backwards.** Representation distance and MMD are LARGEST at near-domain PhysioNet-B (6.96 / 0.294)
   and smallest at MIMIC (3.74 / 0.100). The paper says they are "larger at the far targets". Pooled rho = +0.896 / +0.888 lies
   inside the within-domain shuffle null band (A6, H5 met): it is domain structure, not signal. For repr_dist the pooled sign equals
   the within-target sign at all three targets, so "opposite sign from every within-target value" is false (MMD: false at 2 of 3).
8. **At n = 9 checkpoints per target the criteria are significant only at PhysioNet-B** (permutation p = 0.03-0.04 for violation / recon
   error); at MIMIC p = 0.41 / 0.13 and eICU p = 0.23 / 0.098. Source-validation R2 gets the same selection regret as violation at
   MIMIC and eICU (0.0118, 0.0049). The 9 checkpoints are 3 encoders x 3 head seeds.
9. **Splits are by stay, not patient.** `make_patient_split_loaders` stratifies stays; MIMIC `patient_id` = `stay_id`, eICU `patient_id` =
   `patientunitstayid`; no one-stay-per-patient rule. Affects source-validation / early stopping only (targets are disjoint
   databases). The LOS fine-tune split also differs from the pretraining split, so some LOS validation stays were unlabeled
   pretraining inputs.
10. **Numbers**: "20-30x" is 17-31x by target; seed SD of AUROC is 0.0024-0.0170 (not 0.002-0.014); source AUROC 0.831-0.858; PN-A max LOS is 336.0 (Table 7: 335.0). **Table 7 tail statistics do not match the loaded cohorts**: paper MIMIC max 492.7 h / P99 387.9 / median 67.8 vs loaded cohort max 3,069 h / P99 676 / median 59.5; eICU paper max 1,108.3 / P99 469.9 / median 52.8 vs loaded max 2,540 / P99 539 / median 55.4 (25% subsample; FULL cohorts: MIMIC max 5,434 h, eICU 12,153 h, so the paper's values are off by ~11x). The "1.5-3x tail" claim is therefore wrong in the other direction (real max ratios vs PhysioNet-A are ~16x and ~36x); provenance of Table 7 is unknown. Recompute on the full cache (pod). `results/mortality/selection_criteria.json` is not in the local copy, so Table 6 / ATC rho cannot be regenerated here.
11. **Infrastructure**: `.git` is empty and cannot be restored from here (no GitHub auth), so PREREG.md could not be committed; a SHA-256 +
    timestamp stand-in is in `PREREG.sha256`. Local machine is CPU-only; MIMIC / eICU caches are 25% stay-level subsamples (the
    numbers reproduce the saved JSONs exactly on PhysioNet).
