# PCL Legacy2

Tests whether a prior finding — that label-free criteria computed on
unlabeled target-site data (reconstruction error, physiology-constraint
violation) can substitute for labeled out-of-distribution (OOD) validation
when selecting which trained model to deploy, and that a physiology-
constrained pretraining objective (PCL) improves cross-hospital transfer —
generalizes beyond the single clinical task (sepsis onset) it was
originally established on.

Two further tasks, chosen for different output structure: in-hospital
mortality (static binary) and length-of-stay (LOS, continuous). Same
pretrained encoders reused as-is across both (no retraining), across
MIMIC-IV, eICU-CRD, and PhysioNet 2019.

This repo was split out of a shared PCL monorepo. `config.py`, `src/`, and
`pod_monitor.py` are vendored copies of the model/training/loss
implementation from that split (see [VENDORED.md](VENDORED.md)) — this repo
is self-contained; nothing here imports from outside it.

## Result summary

- **Mortality**: PCL is the worst-performing method (lowest AUROC in every
  one of 18 fine-tune runs — 3 methods × 2 transfer directions × 3 seeds),
  the same ranking the prior work found after correcting its own original
  evaluation confounds, not the pre-correction ranking.
- **LOS**: all three methods collapse hard zero-shot on MIMIC-IV/eICU-CRD
  (near-zero or negative $R^2$) — traced to a label-distribution mismatch
  (the pretraining source's length-of-stay range is far narrower than the
  target sites') rather than a representation-quality problem. PCL still
  trends worst, significant in some paired-by-seed comparisons at $n=3$,
  but an order of magnitude smaller than the shared collapse.
- **Selection criteria**: violation and reconstruction error remain
  reliable predictors of true OOD performance across both tasks.
  Representation-distance criteria (a representation centroid distance,
  and MMD) fail specifically when the underlying shift is a label-scale
  mismatch rather than a representation-space one — a boundary condition
  for when to trust that class of detector at all. Pooling a criterion's
  values across evaluation domains of different true difficulty before
  checking correlation can flip its apparent reliability entirely
  (Simpson's paradox); every correlation reported here is computed
  within-domain for this reason.

Full write-up: [`../ml4h_PAPERS/Legacy2PCL Paper/main.tex`](../ml4h_PAPERS/Legacy2PCL%20Paper/main.tex)
(compiled `main.pdf` alongside it). Narrative results summary:
[`results/SUMMARY.md`](results/SUMMARY.md). Raw per-run data every number in
the paper traces back to: `results/mortality/*.json`, `results/los/*.json`.

## Reproducing this

1. **Checkpoints and data.** Requires the pretrained ERM/PCL/DRO encoders
   (`results_lambda17/ckpt/{erm,pcl,dro}_pretrained.pt`, not included in
   this repo — see Data and Code Availability in the paper) and credentialed
   access to PhysioNet 2019, MIMIC-IV, and eICU-CRD. No retraining or
   re-preprocessing is done; only a task-specific head and a two-phase
   unfreezing schedule are fine-tuned per (task, method, seed).
2. **Mortality** (`scripts/finetune_mortality.py`): PhysioNet 2019 has no
   mortality label, so both directions are run — MIMIC-IV→eICU-CRD and
   eICU-CRD→MIMIC-IV, 3 seeds each. Note: eICU-CRD's raw `mortality` field
   is ICU-unit-level, not hospital-level like MIMIC-IV's; the field actually
   used here, `mortality_hospital`, corrects for that (see
   `src/data/eicu.py`).
3. **LOS** (`scripts/finetune_los.py`): PhysioNet does have a valid
   continuous LOS field, so this uses a single-source protocol instead —
   train on PhysioNet Site A, zero-shot evaluate on Site B, MIMIC-IV, and
   eICU-CRD, 3 seeds. Regresses $\log(1+\text{hours})$ with MSE loss;
   evaluation metrics are all computed back in raw hours, never log-space.
4. **Selection-criteria comparison** (`scripts/selection_criteria_mortality.py`,
   `scripts/selection_criteria_los.py`): inference-only, no training.
   Computes violation, reconstruction error, representation distance, and
   MMD for every fine-tuned checkpoint against every zero-shot target; also
   entropy and ATC for mortality (both inapplicable to LOS's continuous
   target — no standard analog, so omitted rather than forced).
5. **Significance check** (`scripts/significance_check.py`): paired-by-seed
   t-tests ($n=3$, $df=2$) for the method-difference claims in the paper.
   Pure Python/NumPy, no GPU or model needed — reads only the saved result
   JSONs.

Every script is resumable (skips a run if its result JSON already exists)
and calls `pod_monitor.watch_pod()` before loading data, which prints a
loud banner if the GPU sits idle for several minutes (data loading is
CPU-bound; don't pay GPU-pod prices for it) or resumes activity after being
idle (switch back up).

## Known limitations (stated plainly, see the paper's Limitations section
for the full list)

Three seeds per task is a small evidence base for the significance claims
made ($df=2$ throughout). The encoder-bias confound check (whether
reconstruction-error ranking merely tracks masked-prediction reliance
rather than physiology specifically) was underpowered for mortality and not
attempted for LOS. A third task (decompensation, dense time-series output,
Harutyunyan et al. definition) was scoped but never started — a deliberate
decision under time constraint, not an omitted result.
