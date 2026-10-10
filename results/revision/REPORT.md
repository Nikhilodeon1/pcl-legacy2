# Revision report (interim, 2026-10-07). Phase 0 + the Phase A items that need no GPU.

**Spend so far: $0** (local CPU only). All numbers regenerate from raw files via `results/revision/scripts/`; collected in `numbers.json`.
Caveat on data: the A1/A3/A4/A6 LOS analyses below ran on **25% stay-level subsamples** of MIMIC-IV/eICU (G1 and G2 were re-run on the FULL cohorts, see the update at the end); rerun A1/A3 on full data with pod/run_audits.sh. Original statement: built with the original loaders (the pipeline reproduces
the saved JSON R2 values on PhysioNet exactly: ERM s42 0.268 / 0.230 vs 0.2677 / 0.2302). Full-cohort reruns are in `pod/run_audits.sh`.
Git: `.git` was empty when PREREG was written; the hash stand-in is `PREREG.sha256` (the repo `pcl-legacy2/` has its own working git; commit PREREG there).

## Verdicts against PREREG
| Item | Verdict | Evidence |
|---|---|---|
| G1 flag, LOS (probe R2 > 0.05) | **FLAGGED** | mask+length probe R2 +0.244 / +0.216 / -0.115 / -0.121 (PN-A val / PN-B / MIMIC / eICU); transformers 0.26 / 0.22 / -0.12 / -0.10..-0.11 |
| G1 flag, mortality (probe AUROC > 0.60) | **FLAGGED** | mask+length AUROC 0.787 (MIMIC val), 0.661 (MIMIC->eICU), 0.762 (eICU val), 0.713 (eICU->MIMIC) |
| G1 values-shuffled test | LOS (PhysioNet only): leak confirmed; MIMIC/eICU LOS and mortality: **pending (pod)** | permuted values change R2 by 0.000 (ERM/PCL/DRO); uniform noise by <= 0.014 |
| H2 tail-only mechanism | **NOT MET** | on <=336h stays, far-target R2 stays negative (ERM MIMIC -0.128, eICU -0.149). On the label-matched resample R2 turns positive (MIMIC 0.127, eICU 0.072) but eICU is >0.15 below PN-B (0.229). The shift is not only the tail. |
| H2b scale vs representation | **MET (6/6)** | Spearman(pred,true) 0.38-0.72 on the full far targets while R2 < 0: ranking is largely right, scale is wrong. But see G1: a length decoder also ranks well. |
| H3 covariate isolation (A2) | **not run** | the representation distances are ordered PN-B > eICU > MIMIC (A6/G3b), which already contradicts the premise that they measure target difficulty; A2 needs the label-matched resamples' unlabeled inputs and will be run after the fixed-T results. |
| H5 pooling null | **MET** | pooled rho +0.896 (repr. dist.) and +0.888 (MMD) lie inside the within-domain shuffle null band [+0.772,+0.898] / [+0.838,+0.930]: pooled values are domain structure. violation / recon error pooled values sit just outside their null bands. |
| H4 within-method disentangling | **not run (Phase B)** | needs n>=10 seeds |

## G0 facts (source quoted in SURPRISES.md; items needing the pod are marked)
(a) The paper used ONE pretrained encoder per method (`results_lambda17`, seed 42, lambda=1.0 for PCL); fine-tune seeds 42/43/44 reuse it. **Found locally in `C:/Users/nikhi/Codes/IMPORTED/pclCodebase/`:** those three encoders, plus `results_lambda17_s43` and `_s44` with lambda in {0.0 (=ERM), 0.1, 0.5, 1.0, 2.0, 5.0} at two further independent pretraining seeds (no separate `dro_pretrained` at s43/s44). `results_final_s42/43/44` hold results JSON/figures only, no checkpoints. The pod inventory is no longer needed for this question.
(b) 17 variables, 48 h windows, PhysioNet-A train split only; 6 layers, d=256, 8 heads (3.2M params); AdamW lr 3e-4, wd 1e-4, cosine to 0.1 lr, batch 64, 30 epochs, 30% of observed cells masked, grad-clip 1.0; lambda linear warm-up over 40% of epochs, then constant **1.0 for the PCL arm actually used** (checkpoint history + lambda.log; `config.py` default 0.5 was not what ran); L_PCL rescaled by an EMA balancer to ~1.0 x L_masked (cap 20x). **Three active constraints (MAP L1, Henderson-Hasselbalch MSE, Severinghaus SpO2-PaO2 MSE); PP and SI have weight 0.** The sepsis sweep from the same run puts lambda=0.5 as best OOD and lambda=1.0 as poor (see SURPRISES item 5), so the PCL arm is a known-poor setting. "DRO" = `run_erm_pretraining` (plain masked prediction) with a different init seed; group-DRO exists only in the sepsis fine-tuning stage with one group.
(c) No MIMIC/eICU/PhysioNet-B inputs in pretraining (code comment L446-452 documents the removed "pretraining leak"); confirming the checkpoints' provenance needs the pod. The LOS fine-tune split differs from the pretraining split, so some LOS validation stays were unlabeled pretraining inputs.
(d) DRO appears in the corrected comparison of the prior study (urtc.txt L54, "once corrected for class imbalance"); the commit history cannot be read here (`.git` empty): `pod_inventory.sh` prints it.
(e) PCL = Physiology-Constrained Learning (prior study abstract); the submitted paper's "physiological-consistency constraints (PCL)" is inconsistent.
(f) Rationale: `_archive/plan.md` L9, L48 (physiology is shared across hospitals; HH, MAP, Severinghaus). SI was dropped because "ablation showed it hurt OOD performance" and PP as "algebraically redundant with MAP" (pcl_loss.py L180-183).

## G2 cohort table (`g2_cohort.json`, 25% subsamples for MIMIC/eICU; full counts from the result JSONs: PN-A 10,381, PN-B 14,779, MIMIC 74,607, eICU 130,446)
| dataset | stays (subsample) | patients | stays/patient max | LOS mean / median / P99 / max (h) | % < 48 h | mortality |
|---|---|---|---|---|---|---|
| PhysioNet-A | 10,381 | 10,381 | 1 | 46.2 / 43.0 / 168 / 336 | 66.5 | n/a |
| PhysioNet-B | 14,779 | 14,779 | 1 | 45.3 / 42.0 / 157 / 336 | 68.9 | n/a |
| MIMIC-IV (25%) | 18,650 | 16,755 | 10 | 104.6 / 59.5 / 676 / 3,069 | 38.4 | 11.9% |
| eICU-CRD (25%) | 32,620 | 30,265 | 7 | 90.3 / 55.4 / 539 / 2,540 | 41.7 | 9.2% |
Inclusion: adults, ICU LOS >= 24 h, >= 1 hour with all hemodynamic variables observed. Window: first 48 hourly bins from ICU admission; stays < 48 h are padded after discharge. Normalization: fixed affine map from clinical plausibility bounds, no fitted statistics (so "from source train only" holds trivially). Imputation: hourly median, forward-fill <= 6 h, then 0 with the observation mask as input. LOS definitions: PhysioNet = number of hourly rows; MIMIC = icustays.los x 24; eICU = unitdischargeoffset / 60. Prediction time: end of hour 48 (or of the stay), outcome unknown but for stays < 48 h the window already spans the whole stay.
**Splits are by stay, not patient**; 15% (MIMIC) and 12% (eICU) of source-validation stays already have a sibling stay of the same patient in train (a lower bound, because the cache is subsampled). Repeated admissions are not de-duplicated.

## G3 numbers audit (`g3_numbers_audit.md`, `g3b_text_claims.md`): 55 numeric checks, 7 not PASS; the notable ones
- Table 7 tail statistics do not match the loaded cohorts (MIMIC max 492.7 h in the paper vs 3,069 h loaded; eICU 1,108 vs 2,540; medians also differ). Provenance unknown; "1.5-3x tails" is wrong.
- "20-30x" is 17-31x; seed SD of AUROC is 0.0024-0.0170 (not 0.002-0.014); source AUROC 0.831-0.858 (not 0.84-0.86); LOS paired tests clear |t|>4.303 in 3 comparisons, not 2 (ERM > DRO at eICU, t=+7.04).
- Directional claims false: "repr. dist./MMD are larger at the far targets" (they are largest at PN-B); "pooled rho is the opposite sign from every within-target value" (not for repr. dist.); "same DRO >= ERM > PCL ranking" on LOS (ERM > DRO > PCL at all three targets). True: PCL never wins a cell; PCL lowest in 8/9 LOS cells; violation/recon correctly signed at every target.

## A3 patient-cluster bootstrap, LOS (`a3_bootstrap_los.csv`; test-sampling variance ONLY; mean over seeds; same resample for all methods)
| target | PCL-ERM | PCL-DRO | ERM-DRO |
|---|---|---|---|
| PN-B | -0.0204 [-0.0223,-0.0185] | -0.0165 [-0.0181,-0.0149] | +0.0040 [+0.0028,+0.0052] |
| MIMIC | -0.0120 [-0.0130,-0.0110] | -0.0060 [-0.0065,-0.0054] | +0.0060 [+0.0055,+0.0066] |
| eICU | -0.0133 [-0.0142,-0.0124] | -0.0055 [-0.0059,-0.0051] | +0.0078 [+0.0072,+0.0084] |
**Do not read these as evidence that PCL is worse.** ERM and DRO are the same pretraining recipe, yet their difference (+0.006 / +0.008 at the far targets) is as large as PCL-DRO (-0.006 / -0.0055) and its interval also excludes zero. The intervals exclude pretraining variance, which the prior study measured at 4-7 AUROC points across full-pipeline seeds. Only PCL-ERM (about twice the replicate gap) is distinguishable even on this crude yardstick, from one encoder pair.

## A4 selection analysis, LOS (`a4_selection_los.csv`; n = 9 checkpoints = 3 encoders x 3 fine-tune seeds; sign convention: positive = correct)
violation / recon rho: PN-B +0.72 / +0.73 (permutation p 0.037 / 0.032), MIMIC +0.32 / +0.55 (p 0.41 / 0.13), eICU +0.45 / +0.60 (p 0.23 / 0.098). repr. dist. and MMD: wrong or null. Source-validation R2 gets the same regret as violation at MIMIC (0.0118) and eICU (0.0049). Within-domain correct sign everywhere, statistical evidence only at PhysioNet-B. Mortality: **pending** (needs `selection_criteria.json` and mortality predictions).

## A5 in-domain baselines (`a5_indomain.csv`)
Mortality in-domain AUROC 0.853-0.856 (MIMIC), 0.838-0.841 (eICU); zero-shot 0.744-0.764 / 0.808-0.819. LOS PN-A validation R2 0.254-0.269.

## A6 pooling (`a6_pooling_null.csv`, `a6_pooling_sim.csv/png`)
Null-band result above. Simulation: with a criterion that has correct within-domain signal (rho about -0.44) the pooled rho becomes +0.5 to +0.8 once the between/within variance ratio >= 3, i.e. the sign flip mechanism is real in general. In THIS data it is not a sign flip for repr. dist. (wrong-signed pooled and within-target) and the paper's explanation must be rewritten.

## Not done / blocked
- Fixed-observation protocol reruns (27 fine-tunes), mortality values-shuffle, mortality A3/A4, B1-B4: need the pod, the pretrained encoders (`~/pcl_pretrained/*.pt`, not in git) and raw data or full caches. Cost estimate ~2.5-4 h on a V100; **gate: tell the strategy agent if rate x 4 h > $5**.
- A2 (covariate isolation) and H4: after the above.
- Full-cohort versions of G1/G2/A1/A3 (pod: `pod/run_audits.sh`).

## Update 2026-10-09: G1 and G2 re-run on the FULL cohorts (MIMIC 74,607 stays, eICU 130,446; counts match the saved result JSONs)
**G1 (mask/length-only probe), full data. Both flags stay tripped.**
| task | probe | PN-A val | PN-B / source->target | MIMIC | eICU |
|---|---|---|---|---|---|
| LOS R2, mask+length | | +0.244 | +0.216 | -0.118 | -0.103 |
| LOS R2, transformer ERM (saved JSON means) | | +0.264 | +0.229 | -0.122 | -0.103 |
| mortality AUROC, mask+length | MIMIC->MIMIC val 0.804; MIMIC->eICU 0.654 | | | | |
| mortality AUROC, mask+length | eICU->eICU val 0.764; eICU->MIMIC 0.704 | | | | |
| mortality AUROC, fine-tuned (JSON) | MIMIC val 0.856, MIMIC->eICU 0.744-0.764; eICU val 0.840, eICU->MIMIC 0.808-0.819 | | | | |
On eICU the mask-only LOS model matches the transformer to the third decimal (-0.103 vs -0.103).

**G2 (full cohorts)** (`g2_cohort.json`):
| dataset | stays | patients | LOS mean / median / P99 / max (h) | mortality |
|---|---|---|---|---|
| MIMIC-IV | 74,607 | 54,427 | 105.4 / 60.0 / 693 / **5,434** | 11.9% (8,870 deaths) |
| eICU-CRD | 130,446 | 103,031 | 90.4 / 55.6 / 531 / **12,153** | 9.1% (11,867 deaths) |
- **Table 7 in the paper is badly wrong for the tails**: it gives MIMIC max 492.7 h and eICU max 1,108.3 h; the real maxima are 5,434 h and 12,153 h (11x and 11x the paper's values; 16x and 36x PhysioNet's 336 h cap). The label-support mismatch is far larger than stated, but a handful of multi-hundred-day "stays" are also a data-quality issue (outlier LOS were never clipped), and R2 on raw hours is dominated by them. Suggest a robust-LOS sensitivity analysis (cap at 336 h, or log-space R2).
- **Patient leakage in the source split is large**: 38% of MIMIC and 31% of eICU in-domain validation stays belong to a patient who also has a stay in the training split (by-stay split), so in-domain AUROC and the early-stopping/checkpoint choice are optimistic. This affects only the source-side numbers (the zero-shot targets are separate databases).
