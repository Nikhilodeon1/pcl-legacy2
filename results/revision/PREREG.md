# PREREG — pcl-legacy2 revision (ML4H 2026)

Written 2026-10-06T16:08Z, before any Phase A analysis was run.

## Provenance and disclosure

- Reviews received (scores 2,2,2,3,3). This file fixes the analyses and decision
  thresholds *before* the revision experiments are run.
- Before this file was written, only source code and the already-published
  result JSONs (`results/mortality/*.json`, `results/los/*.json`) were read.
  No revision analysis output existed yet.
- Git was not available when this was written (`.git` empty, no remote auth),
  so the commit required by the plan could not be made. Integrity stand-in:
  the SHA-256 of this file is recorded in `PREREG.sha256` with the same
  timestamp. **Commit both files as soon as git is restored.**

## Rules

1. Every result is reported, adverse ones included. A failed prediction is
   stated as failed. No post-hoc tuning of analyses.
2. Seeds 42/43/44 are reused for reruns so before/after is paired.
3. Any change to an analysis after this point is logged in a "Deviations"
   section at the bottom of this file with date and reason; the original text is
   not edited.
4. Cost gate: estimate before any run over $5; stop at $25 cumulative.

## Phase 0 flags

- **G1 leakage flag** is raised if any of:
  - mask/length-only probe: LOS R2 > 0.05 (source val or any target), or
    mortality AUROC > 0.60 (source val or any target);
  - values-shuffled test (real masks, randomized values) performance stays
    clearly above chance. "Clearly above" is fixed here as: LOS R2 within 0.05
    of the real-values model, or mortality AUROC > 0.60.
- If flagged: fixed-observation protocol (LOS: first T=24h, keep LOS >= 24h;
  mortality: first T=48h, keep LOS >= 48h, identical mask pattern for every
  stay), rerun all fine-tunes (LOS 9, mortality 18) and all downstream analyses,
  and report original and fixed-protocol side by side.

## Phase A hypotheses (thresholds fixed now)

- **H2 (tail-only mechanism, LOS).** On the <=336h subset and on the
  label-matched resample, far-target (MIMIC-IV, eICU-CRD) R2 >= 0 AND within
  0.15 of PhysioNet-B R2, for every method (mean over seeds). Met only if both
  conditions hold at both far targets under both subsets. If missed, the report
  states that the shift also sits in the bulk of the distribution.
- **H2b.** Spearman(pred, true) >= 0.30 on the full far target while R2 < 0
  favors a scale problem over a representation problem. Evaluated per
  method x far target (mean over seeds); reported as the fraction of 6 cells
  satisfying it.
- **H3 (covariate isolation).** On the label-matched resamples, representation
  distance and MMD are both correctly signed (negative within-target Spearman)
  at BOTH far targets. If not, the claim is restricted in the writing.
- **H4 (within-method disentangling, Phase B).** Recon error / violation track
  performance beyond method identity if the sign is correct in >= 75% of
  method x target cells at n >= 10 seeds.
- **H5 (pooling null).** Pooled Spearman for repr dist and MMD lies inside the
  95% band of the within-domain method-label-shuffle null (10,000 shuffles).
  Observed pooled value outside the band = H5 not met.

## Analysis definitions

- **A1.** Per target, method, seed: R2, MAE, RMSE, Spearman(pred,true) on
  (i) full target, (ii) stays <=336h, (iii) label-matched resample to
  PhysioNet-A's LOS decile histogram (200 resamples, mean + 95% interval over
  resamples). Constant baselines (source-train mean, target median),
  affine-recalibrated R2 (fit on a random half of the target, evaluate on the
  other half; oracle diagnostic, labeled as such), mean signed error and MAE by
  true-LOS decile, predicted-vs-true scatter data.
- **A2.** On resamples from A1(iii), recompute violation, recon error, repr
  distance, MMD from the resampled unlabeled target inputs; within-target
  Spearman against true R2 at the checkpoint level (one point per method x
  seed).
- **A3.** Patient-level paired bootstrap, 10,000 resamples of test patients,
  same resample across methods; per seed and target/direction, then averaged
  across seeds; 95% percentile intervals. Covers test-sampling variance only.
- **A4.** Unit = one checkpoint (method x fine-tune seed). Print n with every
  Spearman. Permutation p (100k), Kendall tau-b, leave-one-seed-out range.
  Selection regret and top-1 hit rate for source-validation metric, each
  criterion, random (expected), oracle. Both tasks, within-domain and pooled.
- **A5.** In-domain baselines per method and seed.
- **A6.** (a) shuffle performance across methods within each domain, recompute
  pooled Spearman 10,000 times; (b) simulation sweeping between/within domain
  variance ratio.

## Outcomes that would change the paper's claims (stated in advance)

- G1 flagged -> all headline numbers are re-derived under the fixed protocol;
  the original numbers become a sensitivity analysis.
- H2 not met -> "label-support" explanation is demoted to a partial
  explanation; the bulk shift is reported.
- H3 not met -> "repr-distance fails because of label shift specifically" is
  withdrawn or restricted.
- H5 not met -> the pooling claim is restated as a caution rather than a
  demonstrated artifact.

## Deviations

(none yet)
