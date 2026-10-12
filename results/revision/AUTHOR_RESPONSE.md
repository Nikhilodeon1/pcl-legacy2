# Author response DRAFT (ML4H 2026, submission 384). Close: Oct 12.

## Notes for the author (delete before posting)
1. **Every number below comes from `results/revision/` (script + CSV/JSON).** Items in [BRACKETS] are not yet available or need your decision. Nothing is posted; you post it.
2. **Generative-AI question (3P1U Q1-Q3): you must answer truthfully.** The text below says an AI assistant was used for code/analysis scripting and text editing. Edit it to match what actually happened for the *original* submission as well as the revision; do not post it unedited if it is not accurate. Do not argue with the Pangram score.
3. **OpenReview comment length:** check the limit (commonly 5,000 characters); each block below is written to fit one comment, split further if needed.
4. **Sweep results (independent pretraining seeds, lambda 0.5 / 1.0 / ERM at seeds 42/43/44) are placeholders** until `pod/run_sweep_parallel.sh` finishes and `results/revision/scripts/` summarises it. If they are not ready by the deadline, delete the [SWEEP] sentences; they are the only promised-not-delivered numbers.
5. **Not done, so not claimed:** A2 (covariate-only isolation on label-matched resamples), H4 (within-method disentangling, needs >= 10 seeds), mortality bootstrap CIs and mortality selection analysis under the fixed protocol. The response says so.
6. Anonymity: the response mentions "the prior study" only as the anonymised earlier work. Rename the anonymous repo slug before posting any link.

---

## 0. Global response (post as the first comment, visible to all)

We thank the reviewers; their reading was careful and most of the main concerns were correct. We treated them as an audit request: every claim was checked against the code, a pre-registered analysis plan (`PREREG.md`, fixed before the analyses) was run, and the experiments were re-run. The result is a **narrower but better-supported paper**. We report all results, including those that weaken our earlier claims. Everything is in the anonymous repository under `results/revision/` (script and CSV/JSON for every number).

**Revised contributions.** (1) *A shortcut audit for ICU outcome models.* Mask/length-only probes and a fixed-observation protocol show that last-observed-hour pooling lets a transformer read stay length, which fully explains the LOS result; removing it changes the conclusion. The audit is cheap and transferable to any model that pools at the last observed step. (2) *A cleaner replication of the PCL comparison* on mortality and LOS, now with independently pretrained encoders and explicit statements about what n = 3 can support. (3) *A corrected pooling analysis for label-free OOD selection*, with a null-band test showing when pooled correlations are domain structure.

**Where each concern is addressed.**

| Concern (reviewers) | What we did | Where |
|---|---|---|
| One encoder per method; fine-tune seeds only (3P1U, QFzr, EeBD, W6ye) | Independently pretrained encoders, 3 pretraining seeds x {ERM, PCL lambda 0.5, 1.0} | point 4 |
| LOS window, padding, length leak (3P1U, QFzr) | Mask/length probe, value-permutation test, fixed-observation re-run of all fine-tunes | point 1 |
| Statistics: n = 3, no CIs (3P1U, G9vk, W6ye, EeBD) | Patient-cluster bootstrap, per-seed values, df = 2 limits stated, no "significant" without support | R1 W2, R2 W3 |
| Cohort and preprocessing details (3P1U, W6ye) | Full cohort table, inclusion, prediction time, normalisation, missing values | R1 Q4 |
| Paper not self-contained (G9vk, W6ye) | Pretraining recipe and data in the appendix; code released | R2 W1/W2 |
| "Zero-shot" definition (QFzr) | Defined once; renamed "zero-shot cross-site transfer" | R3 W2 |
| Label scale vs representation (QFzr, EeBD) | Label-matched resample, ranking-vs-scale test | R3 W4 |
| Criteria not disentangled; Spearman on 9 values (3P1U, EeBD, W6ye) | Permutation p-values, validation-selection baseline, limits stated | R1 Q7, R5 W3 |
| Presentation (G9vk, W6ye, EeBD) | Appendix moved into 8 pages, setup figure, shorter abstract, plain prose | R2, R4 |

**1. A stay-length shortcut explains the LOS result (3P1U W3/Q6, QFzr W4).** The classification head pools the encoder state at the last *observed* hour and the encoder adds an absolute position at that hour, so stay length (< 48 h for 66% of PhysioNet, 38% of MIMIC-IV, 42% of eICU stays) is readable by the model. A gradient-boosted model that sees *only* the observation mask and stay length reproduces the transformers' LOS R² to the second decimal (0.244 / 0.216 / -0.118 / -0.103 on PhysioNet-A val / PhysioNet-B / MIMIC / eICU, versus 0.264 / 0.229 / -0.122 / -0.103), and permuting the observed values across stays leaves transformer R² unchanged (ERM 0.263 -> 0.263). We therefore re-ran all fine-tunes under a **fixed-observation protocol**: every stay is shown exactly the first T hours (T = 24 for LOS, T = 48 for mortality, which drops stays shorter than 48 h) with identical padding, and the head reads index T-1. **Under it, the LOS result disappears:** R² is +0.04 in-domain and about 0 at PhysioNet-B for all three methods, MIMIC-IV / eICU sit at -0.19 / -0.17, about where a constant prediction sits (-0.179 / -0.170), and no method difference survives (largest paired |t| = 2.93, df = 2). **We withdraw the claim that PCL is significantly worse on LOS.** The original LOS numbers will be reported next to the fixed-protocol ones as a documented shortcut, not as a finding about PCL.

**2. Mortality: PCL is not better, and is unstable.** Fixed protocol, OOD AUROC mean (SD) over 3 fine-tune seeds, fp32: ERM 0.725 (0.002) / 0.750 (0.006), DRO 0.721 (0.002) / 0.742 (0.012), PCL 0.626 (0.111) / 0.614 (0.114) for MIMIC->eICU / eICU->MIMIC. PCL has 2 of 6 collapsed fine-tunes (AUROC 0.50 from epoch 1) and a third degraded run; ERM/DRO have none. Non-collapsed PCL runs average about 0.68. With n = 3 and df = 2, no paired PCL-vs-baseline difference clears the threshold in fp32 (|t| <= 1.98), so we will not call PCL "significantly worse" for mortality either; the supported statement is "no benefit, and unstable". A mask-and-length-only model still reaches AUROC 0.63-0.73 under the fixed protocol (per-variable observation counts), but ERM/DRO exceed it by about 0.1, so mortality retains real signal.

**3. Three corrections to our own paper (W6ye, QFzr, G9vk).** (a) "DRO" is not group-DRO: that encoder was pretrained with the same masked-prediction recipe as ERM and a different initialisation (group-DRO needs more than one training site; PhysioNet-A is one). ERM-vs-DRO is a replicate pair and will be labelled so. (b) PCL has three active constraints (MAP, Henderson-Hasselbalch, Severinghaus SpO2-PaO2), not five; pulse-pressure and shock-index have weight 0. (c) The PCL encoder used is lambda = 1.0, a setting the earlier study's own sepsis sweep ranked below lambda = 0.5 (OOD AUROC 0.675 / 0.645 / 0.585 vs 0.750 / 0.695 / 0.662). Our PCL conclusions therefore apply to that configuration; see 4.

**4. Independent pretraining (3P1U W1/Q5, QFzr, EeBD W2, W6ye).** The paper used one pretrained encoder per method; the three "seeds" were fine-tune seeds, so they could not measure pretraining variance. We located independently pretrained encoders for ERM (lambda = 0) and PCL (lambda = 0.5 and 1.0) at three pretraining seeds and are re-running the fixed-protocol fine-tunes on them. [SWEEP: insert mean (SD) over the 3 pretraining seeds for LOS R² and mortality AUROC, and the paired t.]

**5. Other corrections.** Table 7's tail statistics are wrong: the real maxima are 5,434 h (MIMIC-IV) and 12,153 h (eICU) (P99 693 / 531 h), 16x / 36x PhysioNet's 336 h cap, not 1.5-3x. Our pooling explanation was backwards for representation distance and MMD: they are *largest* at the near-domain PhysioNet-B (6.96 / 0.294) and smallest at MIMIC-IV (3.74 / 0.100), and the pooled correlations (+0.896, +0.888) lie inside the null band from shuffling method labels within domain ([+0.77, +0.90], [+0.84, +0.93]), i.e. they reflect domain structure. Splits are by stay, not patient: 38% (MIMIC-IV) and 31% (eICU) of source-validation stays have a stay from the same patient in training, which makes source-side numbers optimistic (targets are separate databases and unaffected). We will rewrite the abstract, contributions and Section 5 accordingly and move cohort details and the appendix tables into the main text.

---

## 1. Reviewer 3P1U (rating 2)

**W1 / Q5 (one encoder per method; seeds are fine-tune seeds).** Correct, and a real limitation: fine-tune seeds do not measure pretraining variance, and the earlier study itself saw several AUROC points of full-pipeline variability. We now fine-tune independently pretrained encoders (3 pretraining seeds x {ERM, PCL lambda 0.5, PCL lambda 1.0}) [SWEEP numbers]. In the revised paper the three fixed-protocol fine-tune seeds are described as fine-tune seeds, all significance statements are restricted to what df = 2 can support, and the PCL conclusion is stated per lambda. "PCL" is Physiology-Constrained Learning; the abbreviation in our text ("physiological-consistency constraints") was inconsistent and will be fixed.

**W2 (statistics).** We added a patient-cluster bootstrap (10,000 resamples, one resample shared across methods and seeds). For LOS, mean over seeds, PCL-ERM was -0.020 [-0.022, -0.019] (PhysioNet-B), -0.012 [-0.013, -0.011] (MIMIC), -0.013 [-0.014, -0.012] (eICU). This interval covers **test-sampling variance only**; the ERM-DRO replicate difference (+0.004 to +0.008, also excluding zero) shows that it is too narrow to read as method evidence. We therefore stopped using these intervals to support method claims. Under the fixed protocol the method differences are within seed noise (above).

**W3 / Q6 (LOS window and padding).** Prediction is made at the end of hour 48 (or the end of the stay if shorter); stays shorter than 48 h are padded after discharge, and the padding/mask pattern leaks stay length (point 1 of the global response). Fixed protocol: LOS uses the first 24 h for stays >= 24 h (all stays kept; the minimum is 24 h by inclusion), mortality uses the first 48 h and drops stays shorter than 48 h (MIMIC-IV 74,607 -> 46,158 stays; eICU 130,446 -> 76,453). Deaths or discharges before the window end are therefore excluded in the mortality analysis; this is a selection effect we state explicitly.

**Q1-Q3 (generative AI).** [AUTHOR TO CONFIRM BEFORE POSTING] An AI assistant (Claude) was used to write and run analysis code and to edit and reorganise text; all reported numbers are produced by scripts in the repository, and the authors take responsibility for the content. The "X, not Y" constructions you point out are a real style defect and we will rewrite the text in plain declarative sentences. We cannot comment on the detector's score.

**Q4 (cohort details).** MIMIC-IV: 74,607 ICU stays from 54,427 patients (ICU LOS >= 24 h, adults, at least one hour with all hemodynamic variables observed), mortality 11.9% (8,870 deaths, in-hospital). eICU-CRD: 130,446 stays, 103,031 patients, mortality 9.1% (11,867; hospital-level discharge status). PhysioNet 2019: 10,381 (Site A) / 14,779 (Site B) stays, one per patient. Hourly median binning, forward-fill for at most 6 h, remaining gaps set to 0 with the observation mask as an input; normalisation is a fixed affine map from clinical plausibility bounds shared by all datasets (no fitted statistics). 17 variables. Prediction time: end of hour 48 (original) / hour T (fixed protocol). We will put this table in the main text.

**Q7 (what is one observation in Tables 3 and 6).** One fine-tuned checkpoint: 3 pretrained encoders x 3 fine-tune seeds = 9 per target (best-validation-loss epoch), and Spearman is computed over those 9. Only 3 encoders are independent, so these correlations are weak evidence; the permutation p-values are 0.03-0.04 at PhysioNet-B and 0.10-0.41 at MIMIC/eICU. [SWEEP: with independently pretrained candidates, n = 9 encoders per lambda grid, report rho and permutation p.]

**What we ask.** Your two main objections, one encoder per method and the unclear LOS input, are now answered with new experiments (independent pretraining; fixed-observation protocol), and the cohort details you requested are supplied. We hope this supports a higher soundness score.

## 2. Reviewer G9vk (rating 2)

**W1/W2 (uncited prior study; paper not self-contained).** The prior study is anonymised for double-blind review, so it cannot be cited without breaking anonymity. We agree the paper must stand alone and will add the pretraining recipe in an appendix: training data PhysioNet-A only (no MIMIC/eICU/Site B), 17 variables x 48 hourly bins, 6-layer d = 256 transformer (3.2M parameters), 30% of observed cells masked, AdamW, 30 epochs, constraint weight warm-up over 40% of epochs, PCL loss rescaled to the reconstruction loss by a running average. The code release includes the pretraining code. Page budget: we will use the 8 pages, with the appendix tables brought into the main text.

**W3 (confidence intervals; why 3 seeds).** The limitation was compute budget for the original study. We now report per-seed values and SDs everywhere, bootstrap intervals for LOS (test-sampling variance), paired t with df = 2 flagged as underpowered, and add independent pretraining seeds [SWEEP]. Under the fixed protocol the mortality fp32 SDs are 0.002-0.012 for ERM/DRO and 0.11 for PCL.

**W4 (LOS contribution vs MAE).** MAE of 50-63 h at MIMIC/eICU is close to those cohorts' median LOS (60 h / 56 h), consistent with predictions on the PhysioNet scale rather than the target scale; after we removed the stay-length shortcut the models are at the constant-prediction line there (-0.196 / -0.178 vs -0.179 / -0.170), so the contribution is restated: *no method* achieves positive LOS transfer to MIMIC/eICU, and the earlier in-domain R² of 0.26 was largely stay length.

**What we ask.** The paper will be self-contained, use the full page budget, and carry intervals, per-seed values and a corrected LOS claim. We hope this resolves the soundness and presentation concerns.

## 3. Reviewer QFzr (rating 3)

**W1 (reused models).** Same as above; the pretraining recipe goes into the appendix.

**W2 ("zero-shot").** We use "zero-shot" for: trained on source-site labels, evaluated on the target with no target data, labels or adaptation used for training or model selection. The models are fine-tuned on the source task, so this is cross-site transfer without target data, not zero-shot learning in the foundation-model sense. We will define it once and use "zero-shot cross-site transfer".

**W3 (recent methods).** Our scope is the pretraining objective (physiology constraint vs plain masked prediction vs a domain-agnostic robust alternative), not a state-of-the-art claim, and we state this. We now also report in-domain performance of the fine-tuned models (mortality AUROC 0.853-0.856 MIMIC, 0.838-0.841 eICU) and a gradient-boosted mask/length-only baseline, which is the reference that exposed the shortcut. We will add recent label-free selection and DG baselines to Related Work; we did not run new training-time DG methods and say so.

**W4 (label scale).** Agreed, and we ran it: on a resample of MIMIC/eICU whose LOS distribution matches PhysioNet's, R² becomes positive (MIMIC +0.127, eICU +0.072, ERM), but the eICU value is still more than 0.15 below PhysioNet-B (+0.229), so the label scale explains much but not all of the far-target drop, and our pre-registered "tail-only" hypothesis was not met. Spearman(pred, true) at the far targets is 0.38-0.72 while R² < 0, i.e. ranking is largely right and scale is wrong (6/6 method-target cells). Limitation we will state: a stay-length shortcut also ranks stays (see global point 1).

**What we ask.** The zero-shot definition, label-scale analysis and pretraining details you requested are added; the scope (pretraining objective, not state of the art) is stated. We hope this supports acceptance.

## 4. Reviewer W6ye (rating 3)

**Novelty.** The contribution is a replication-plus-audit: it tests whether the earlier findings survive a change of task, and the audit found they mostly do not survive removal of a shortcut.

**Table 4 (MIMIC->eICU seed 42: PCL and ERM differ by 0.001).** Correct; with the stay-length-fixed protocol we report per-seed values and stop describing small differences as a ranking. fp32 PCL / ERM / DRO MIMIC->eICU: 0.626 / 0.725 / 0.721.

**LOS "collapse" and Spearman on 9 values.** Agreed (global point 1; Q7 above). **Bootstrapping**: done for LOS (test sampling only). **Validation-set selection baseline:** source-validation R² achieves the same selection regret as violation at MIMIC (0.0118) and eICU (0.0049) for LOS; we add it as a baseline row. **Mortality in-domain AUROC:** 0.853-0.856 (MIMIC), 0.838-0.841 (eICU).

**Why five physiological aspects.** Our text was wrong: three are active (MAP, Henderson-Hasselbalch, Severinghaus). The identities are used because they are physical relations that should hold in every hospital; in the original code the shock-index term was dropped after an ablation showed it hurt OOD performance, and pulse pressure as algebraically redundant with MAP. We will say that this selection was made using OOD results of the earlier study.

**Was DRO in the original comparison?** DRO appears in the earlier study's corrected comparison; as explained above, it is an ERM-recipe replicate (not group-DRO), so ERM-DRO differences measure replicate noise. **Observations per Spearman:** 9 (3 encoders x 3 fine-tune seeds), not independent. **Independently trained candidates only:** [SWEEP].

**Overview figure; abstract length; cohorts; Table 3 vs 6 placement.** Accepted; we will add a setup figure and the cohort table, shorten the abstract, and bring the mortality selection table into the main text.

**What we ask.** We believe the corrections (per-seed reporting, validation-selection baseline, cohort table, constraint count, independent pretraining) address the statistical and reproducibility concerns behind a marginal reject.

## 5. Reviewer EeBD (rating 2)

**W1 (three seeds; significance).** Accepted. We will not use "significantly" except where df = 2 allows it, never interpret non-significance as equivalence, and report per-seed values, SDs and bootstrap intervals. [SWEEP] adds independent pretraining seeds. An external dataset is out of reach before camera-ready; we say so as a limitation.

**W2 (reused encoders confound PCL with checkpoint).** Agreed, and our own check supports the concern: the PCL encoder used had lambda = 1.0, which the earlier sepsis lambda sweep ranked below lambda = 0.5. [SWEEP: lambda 0.5 and 1.0 across 3 pretraining seeds.] The paper's claim is restated as "for these checkpoints (and, if [SWEEP] holds, for lambda 0.5/1.0 across 3 seeds)".

**W3 (criteria not disentangled).** Agreed. At n = 9 checkpoints (3 encoders), violation and reconstruction error are significantly correct only at PhysioNet-B (permutation p = 0.037 / 0.032), with correct sign everywhere else but p = 0.10-0.41. Violation and reconstruction error also track generic masked-prediction quality; we do not claim physiological specificity. We did **not** run the within-method (H4) analysis needed to separate method identity from performance; that needs at least 10 seeds, and [SWEEP] provides more checkpoints but not the full analysis. We will say so.

**W4 (covariate vs label shift; Simpson's paradox).** We did not isolate covariate shift from label shift in a controlled way and we will remove the claim. The pooling point is now tested: the pooled correlations of representation distance (+0.896) and MMD (+0.888) lie inside a within-domain shuffle null band (10,000 shuffles), and a simulation shows that a criterion with correct within-domain sign (rho about -0.44) produces pooled rho +0.5 to +0.8 once between-domain variance is at least three times within-domain variance. In our data the pooled sign for representation distance equals its within-target sign, so we describe it as "pooled correlations reflect domain structure", not as a sign reversal.

**What we ask.** The remaining concerns (three seeds, reused encoders, criteria not disentangled) are now addressed with independent pretraining seeds, explicit df = 2 limits, and removal of the unsupported covariate-vs-label and general-pooling claims. We hope this supports a higher rating.
