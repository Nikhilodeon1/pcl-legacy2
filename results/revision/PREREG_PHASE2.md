# PREREG, phase 2 (written 2026-11-03, before any phase-2 experiment was run)

Scope: the new pretraining experiments in `pod/run_phase2.sh` and `pod/run_semisynth.sh` (PhysioNet 2019 only). Same rules as `PREREG.md`: every result is reported, adverse
ones included; any change after this point is logged under "Deviations" below with its date; the file hash is recorded in `PREREG_PHASE2.sha256` when it is written.
Seeds 42-46 (the seed sets the initialisation, the Site A 80/20 split and the fine-tune head); lambda in {0, 0.5, 1}; fine-tuning always fp32.
Prior knowledge that motivates the predictions (so they are not post hoc): the existing PCL encoders (pretrained with fp16 autocast on) have mean pooled-representation L2 norm
about 165-178 vs about 103 for ERM and a relative across-stay spread of 0.07 vs 0.29; 4 of 12 PCL mortality fine-tunes failed vs 0 of 6 for ERM.

## Experiment A: pretraining precision (amp vs fp32), lambda x seeds
- A1. If fp16 pretraining causes the near-constant PCL representation, then in fp32 the PCL encoders' hour-47 `mean_l2` is within 20% of the matched ERM encoders' and
  `const_frac` is within 0.10 of ERM's (means over the 5 seeds). Otherwise the pattern is a property of the objective (or of both precisions).
- A2. PCL minus ERM (paired by seed) in sepsis AUROC at Site B and in LOS R2 at Site B (T=24), per precision: reported with the paired difference, SD and t (df = 4); no
  claim of "better" or "worse" unless |t| > 2.776 AND the sign agrees in at least 4 of 5 seeds. Equivalence is NOT claimed from non-significance.
- A3. fp32 minus amp (paired by seed) for each lambda on the same two metrics, same rule.

## Experiment B: controlled pretraining leak
Pretrain with and without the Site B stays (inputs only, no labels) in the pretraining set, fp32, lambda in {0, 0.5}; evaluate zero-shot on Site B (sepsis AUROC, LOS R2 at T=24).
- B1. The leak is material if the paired-by-seed Site B gain (leak minus no leak) is at least +0.02 AUROC (or +0.02 R2) with sign agreement in at least 4 of 5 seeds.
- B2. Differential leak: the gain for lambda 0.5 exceeds the gain for lambda 0 by at least 0.01 (paired by seed). If the gain is similar for ERM, the leak is not specific to PCL.

## Experiment C: semi-synthetic shifts on Site A validation stays (source = the unshifted stays), fp32 encoders, lambda in {0, 0.5, 1}
Verdict rule (`semisynth_summary.py`): a quantity "responds" if mean |Spearman rho vs magnitude| >= 0.9 and the relative change from the smallest to the largest magnitude is >= 5%.
- C1 (label scale, LOS only): performance (R2) responds (down); representation distance, MMD, reconstruction MSE and violation are flat. If any of them responds, that criterion is not blind to label shift.
- C2 (lab composition removal, rate 0 to 1 of blood-gas labs): the violation score responds (up) while performance (AUROC) is flat or changes by less than 0.02. This would reproduce the composition artifact under control.
- C3 (physiology bias: MAP, pH, PaO2): violation responds (up) in PCL encoders (lambda > 0) more than in ERM encoders; performance is flat. If violation is flat for PCL too, the constraint score does not detect broken physiology at this magnitude.
- C4 (length-only synthetic outcome): original window model AUROC >= 0.85; fixed-T=24 window model AUROC <= 0.60 (mask/length probe reported alongside).

## Deviations
(none yet)
