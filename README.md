# PCL Legacy2

Tests whether the URTC paper's finding (label-free selection criteria —
reconstruction error, physiology-constraint violation — beat OOD-blind
baselines) generalizes across clinical tasks, beyond the single sepsis task
URTC covered. Co-developed with AI club members.

Target: **ML4H 2026 Findings track**, deadline Sept 10 2026 11:59 PM AoE.
(Originally scoped as Proceedings; switched to Findings once mortality's
result came in — see Status below. This was a deliberate call, not a
downgrade: two well-diagnosed results with a cross-validated mechanism beat
a rushed third task.)

This repo was split out of a shared PCL monorepo on 2026-09-10. `config.py`,
`src/`, and `pod_monitor.py` are vendored copies from that split (see
[VENDORED.md](VENDORED.md)), not shared code — this repo is self-contained.

## Status: paper drafted, reviewed, ready for submission

Both planned tasks are complete. Decompensation was deliberately not
attempted (scope decision under time constraint, not an omission — see
Limitations in the paper).

- **Mortality** (static binary): 18 fine-tune runs (ERM/PCL/DRO × 2
  directions × 3 seeds, both directions needed since PhysioNet has no
  mortality label) + 18 selection-criteria runs. **PCL is the worst method**,
  consistently — same ranking as URTC's own corrected sepsis result, not
  the pre-correction one.
- **LOS** (continuous): 9 fine-tune runs (single-source PhysioNet-A
  protocol, matching URTC exactly) + 27 selection-criteria cells. Zero-shot
  collapses hard on MIMIC-IV/eICU-CRD (a label-distribution mismatch —
  PhysioNet's training LOS range is far narrower — not a representation
  problem); PCL trends worst there too, in the comparisons that reach
  significance at $n=3$.
- All fine-tune runs, selection-criteria runs, and significance tests are
  paired-by-seed and independently re-verified against the raw per-seed
  data (not just aggregated means) before being cited in the paper.

**Paper**: [`ml4h_PAPERS/Legacy2PCL Paper/main.tex`](../ml4h_PAPERS/Legacy2PCL%20Paper/main.tex)
(compiled `main.pdf` alongside it). Compiles clean, zero LaTeX warnings,
main body fits the 4-page Findings limit with the reference list and full
per-seed appendix tables starting on page 5+ (free, don't count against the
limit).

**Results data**: [`results/SUMMARY.md`](results/SUMMARY.md) for the
narrative version; `results/mortality/*.json` and `results/los/*.json` for
the raw per-run data every number in the paper traces back to.

## Authorship — still genuinely unresolved, flagging again

Spec originally listed Dr. Lin as co-author (same as URTC), but this became
a group effort with AI club members contributing partway through. Who is
author vs. contributor was flagged as needing a decision **before
submission, not left open until then** — and per `PROJECTS.md`, it still
hasn't been decided as of this writing, on submission day. This needs to be
settled before the OpenReview submission is created, since author list
isn't easily changed after.

## What was actually done (steps 1-6, both tasks)

1. Located + verified the real trained ERM/PCL/DRO checkpoints on the RunPod
   network volume (`results_lambda17/ckpt/` — took several rounds to
   confirm after some early false leads from a different agent's wrong
   directory map). Reused as-is, no retraining, no re-preprocessing.
2. Mortality and LOS task labels added, with two real field-definition bugs
   caught and fixed before they touched a number: eICU-CRD's `mortality`
   field is ICU-unit-level, not hospital-level like MIMIC-IV's (fixed by
   adding `mortality_hospital`, not editing the shared field in place, so
   `reboot`'s already-published detector results stayed untouched); a
   `_cached_load` fraction-kwarg collision and a stale-empty-cache trap that
   would have silently zero-filled `los_h`.
3. Fine-tuned only (frozen pretrained encoder, two-phase unfreezing head)
   per task × method × 3 seeds, matching URTC's protocol.
4. Extended URTC's selection-criteria comparison with ATC and MMD. Entropy
   and ATC turned out inapplicable to LOS's continuous target (no standard
   analog) — reported as N/A rather than forced.
5. Checked the encoder-bias confound (reconstruction-error ranking tracking
   masked-prediction reliance vs. physiology specifically): underpowered for
   mortality at $n=3$, not run for LOS where there's even less real
   between-method effect to confound — reported honestly as inconclusive/
   not attempted rather than forced.
6. Categorized by output structure: mortality (static binary) fails via a
   method-specific effect; LOS (continuous) fails via a label-distribution
   mismatch that dwarfs the method effect. That contrast is the paper's
   actual throughline, not "does PCL win."

Decompensation (dense time-series, Harutyunyan et al. definition) was never
started — correctly deferred per the original gate ("don't start it early"),
then deliberately dropped from scope once Findings was chosen.

## Pod-switch monitor

`finetune_mortality.py`/`finetune_los.py` (and every new script in this
project) call `watch_pod()` from `pod_monitor.py` (vendored at repo root)
right before the data-loading phase starts. It watches `nvidia-smi` and
prints a loud banner in either direction: sustained GPU idle (data
loading on an expensive pod — switch down) or sustained GPU activity
resuming after idle (training started — switch back up). No-ops safely
with no GPU. Add the same two lines
(`sys.path.insert(0, _LEGACY2_ROOT)` +
`from pod_monitor import watch_pod; watch_pod(verbose=True)`) to any new
script in this project, before whichever step loads the datasets.
