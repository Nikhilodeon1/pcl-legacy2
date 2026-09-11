# pcl-legacy2 — Results Summary

Target: ML4H 2026 Findings. Two tasks (mortality, LOS), both complete.
Decompensation not attempted — deliberately held (see README / strategy log),
not a scope cut discovered late.

Reused pretrained ERM/PCL/DRO encoders from URTC as-is (`results_lambda17/ckpt`
on the pod) — no retraining, per the project's cost premise. Only the
classification/regression head and two-phase unfreezing schedule were
fine-tuned per task.

---

## Mortality (static binary output)

Field-definition fix made before any numbers were produced: eICU's `mortality`
field is ICU-unit-level, not hospital-level like MIMIC's. Added a separate
`mortality_hospital` field (hospital-level, comparable across sites) rather
than editing the shared field in place, so chat1_protocol's already-published
detector results stayed untouched.

No single-source protocol available (PhysioNet has no mortality label at all —
hardcoded 0 in the loader), so both directions were run: MIMIC-IV and eICU-CRD
each serve as source and as zero-shot target.

### Step 3 — fine-tune AUROC (18 runs: 3 methods x 2 directions x 3 seeds)

| Direction   | ERM   | PCL   | DRO   |
|-------------|-------|-------|-------|
| MIMIC→eICU  | 0.760 | 0.744 | 0.764 |
| eICU→MIMIC  | 0.819 | 0.808 | 0.819 |

**PCL is the worst method in both directions**, consistently, low seed
variance (std 0.002–0.014). This is the opposite of the sepsis/URTC headline
(PCL beat ERM there).

### Step 4 — selection-criteria comparison (Spearman rho vs. true OOD AUROC,
within-direction — pooling across the 2 directions initially produced
misleading signs on 3 of 6 signals; corrected)

| Signal     | eicu→mimic | mimic→eicu |
|------------|-----------:|-----------:|
| violation  | −0.733     | −0.400     |
| entropy    | −0.417     | −0.150     |
| recon_mse  | −0.683     | −0.700     |
| repr_dist  | −0.517     | −0.333     |
| mmd        | −0.767     | −0.383     |
| atc        | +0.283     | +0.150     |

(negative = good, except atc where positive = good.) `recon_mse` is the most
consistently strong signal, same strength both directions.

### Step 5 — encoder-bias confound check

Honestly inconclusive, not forced. Real test needs n=3 per (method, direction)
cell — too thin to trust a Spearman correlation.

### Step 6 — categorization

Static binary. Notably the closest structural analog to sepsis of the tasks
attempted — PCL still lost, weakening the "it's just an output-structure
thing" explanation for why sepsis worked and this didn't.

---

## LOS (continuous output)

Continuous `los_h` (hours) confirmed cleanly extractable and semantically
consistent across all three databases (all measure ICU/unit-stay length, not
hospital-stay) — no field mismatch here, unlike mortality.

PhysioNet DOES have valid LOS (unlike mortality), so this used URTC's
*original* single-source protocol: train PhysioNet-A, zero-shot on Site B +
MIMIC + eICU. New regression fine-tune/eval path built (MSE loss on
log1p(los_h), MAE/RMSE/R² eval — no AUROC equivalent for a continuous target).

### Step 3 — fine-tune (9 runs: 3 methods x 3 seeds; each run evaluated
zero-shot on all 3 targets)

| Site                        | MAE (h)     | R²                |
|-----------------------------|------------:|-------------------|
| PhysioNet-A (in-domain val) | 6.4–6.7     | 0.255–0.269       |
| PhysioNet-B (zero-shot)     | 5.9–6.0     | 0.208–0.229       |
| MIMIC (zero-shot)           | 62.6–63.1   | **−0.122 to −0.134** |
| eICU (zero-shot)            | 49.7–50.2   | **−0.103 to −0.114** |

**ERM, PCL, and DRO are statistically indistinguishable** on every metric —
differences are within seed noise (`perf_spread` on true R² is ~0.01–0.02
per target). Negative R² on MIMIC/eICU means the model does worse than
predicting the target site's own average LOS.

**Mechanism, confirmed directly** (local check against real PhysioNet
full-scale data + MIMIC/eICU demo):

| Site         | mean LOS | median | p99    | max     |
|--------------|---------:|-------:|-------:|--------:|
| PhysioNet-A  | 46.0h    | 43.0h  | 174.5h | 335.0h  |
| PhysioNet-B  | 44.6h    | 42.0h  | 137.4h | 336.0h  |
| MIMIC        | 102.7h   | 67.8h  | 387.9h | 492.7h  |
| eICU         | 83.5h    | 52.8h  | 469.9h | 1108.3h |

PhysioNet's LOS distribution is hard-capped around ~335h; MIMIC and eICU both
have real tails 1.5–3x+ longer. The model, having never seen a stay that long
during training, systematically underpredicts on the unseen tail. **This is a
label-distribution mismatch, not a representation-quality problem** — it
explains why ERM/PCL/DRO fail identically: no pretraining objective fixes a
model that's never seen a 1000-hour stay.

### Step 4 — selection-criteria comparison (27 cells: 3 methods x 3 seeds x
3 targets; both pooled and within-target Spearman computed from the start,
learning from mortality's pooling mistake)

| Signal     | pooled | physionet_b | mimic  | eicu  |
|------------|-------:|------------:|-------:|------:|
| violation  | −0.198 | −0.717      | −0.317 | −0.450|
| recon_mse  | −0.374 | −0.733      | −0.550 | −0.600|
| repr_dist  | +0.896 | +0.333      | +0.633 | +0.333|
| mmd        | +0.888 | −0.300      | +0.350 | +0.017|

`entropy`/`atc` not computed — both are classification-specific (defined on a
probability / a binary correct-incorrect notion), no clean continuous-target
analog. Flagged as N/A rather than forced.

**violation and recon_mse are reliable — correctly signed in every target,
not just pooled.** **repr_dist is wrong in every single target** (not a
pooling artifact this time — a real, consistent failure). **mmd is
inconsistent** — only correctly signed for the near-domain target
(physionet_b).

**This independently cross-validates the label-distribution-shift mechanism
from a second angle:** repr_dist and MMD measure representation drift, but
LOS's failure isn't a representation problem — it's a label-scale problem.
A representation-drift detector has nothing to detect here, so it fails as a
selection signal; violation/recon_mse reflect model/data quality more
directly and stay informative regardless of the label-scale issue.

### Step 5 — skipped

Same reasoning as mortality's Step 5, stronger: methods are already
statistically tied (Step 3), so there's even less real between-method
variance to test a confound against. Would very likely repeat "inconclusive,
n=3" for no new information.

### Step 6 — categorization

Continuous. Fails via a distribution-shift mechanism entirely different from
mortality's method-specific failure — this is the throughline finding of the
paper: different output structures don't just perform differently, they fail
for different, identifiable reasons.

---

## The paper's throughline

Not "does PCL generalize" (it doesn't, uniformly) — it's **"when do
label-free selection criteria actually work, and why."** Violation and
reconstruction-error are reliable across both tasks. Representation-distance
signals (repr_dist, MMD) only work when the actual failure mode is
representation drift — they fail specifically and identifiably when the real
problem is a label-scale mismatch instead (LOS). That is a contribution about
the selection-criteria class of methods generally, not just a report of two
null results.

## Decision log

- Both directions run for mortality (not URTC's single-source design) —
  PhysioNet has no mortality label at all.
- Single-source protocol used for LOS (matches URTC exactly) — PhysioNet does
  have valid LOS.
- Decompensation held, not attempted: diminishing marginal value (dense
  time-series is the highest-risk, least-validated output-structure bucket)
  against real risk to the Sept 10 deadline, decided with ~9 days' runway
  remaining at decision time.
- Findings track locked over Proceedings: two well-diagnosed results with a
  cross-validated mechanism beats a rushed third task.
