# Phase 2 runbook (pod, PhysioNet 2019 only)

Credentialed data (MIMIC-IV, eICU-CRD) and any patient-level derivative of them must NOT be on the pod; `PCL_PHYSIONET_ONLY=1` (the default in `pod/env.sh`) makes the scripts
refuse them (`finetune_mortality.py`, `build_cache.py`, `fetch_data.sh`, `revlib.load_cache`) and `pod/preflight.py` fails if such a cache is found. Mortality fine-tuning and
all MIMIC/eICU scoring happen on the approved local machine.

## One-time per pod (fresh `/tmp`)
```bash
cd ~ && git clone <repository URL> pcl-legacy2 && cd pcl-legacy2
nohup bash pod/safe_start.sh > ~/pcl_setup.out 2>&1 &      # venv + PhysioNet download + cache build + preflight (log: ~/pcl_setup.log)
tail -f ~/pcl_setup.out                                    # wait for PREFLIGHT PASSED
source pod/env.sh && which python                          # must be /tmp/venv/bin/python
```
Optional, to survive a pod replacement (PhysioNet cache only): `bash pod/persist_caches.sh`.

## Experiments
| What | Command | Answers |
|---|---|---|
| Smoke test (about 10 min) | `JOBS=2 SEEDS="42" EXP=amp bash pod/run_phase2.sh` | pipeline works end to end |
| fp16 vs fp32 pretraining | `JOBS=4 SEEDS="42 43 44 45 46" EXP=amp nohup bash pod/run_phase2.sh > ~/p2_amp.out 2>&1 &` | does AMP create the near-constant PCL representation / unstable fine-tunes? |
| Pretraining-leak test | `JOBS=4 SEEDS="42 43 44 45 46" EXP=leak nohup bash pod/run_phase2.sh > ~/p2_leak.out 2>&1 &` | effect of Site B unlabeled inputs in pretraining on Site B, controlled |
| Semi-synthetic shifts + length control | `bash pod/run_semisynth.sh` (after the amp jobs) | which audit responds to which known shift; shortcut positive control |

Summaries (run anywhere the `results/phase2` JSON/CSV files are): `python results/revision/scripts/phase2_summary.py`, `python results/revision/scripts/semisynth_summary.py`.
All jobs are resumable (finished stages are skipped); rerun the same command after a crash or pod refresh. Commit `results/phase2/` (small JSON/CSV), never checkpoints.

## What each result would mean
- fp32 pretraining removes the large near-constant component (const_frac, spread, L2 at hour 47 match ERM) -> the PCL fine-tune instability was a precision artifact of pretraining; if it does not -> a property of the objective.
- leak test: Site B AUROC / R2 gain from adding unlabeled Site B inputs (paired by seed) quantifies the pretraining-leak confound without any credentialed data.
- semi-synthetic: label-scale shift should leave all input-based criteria flat while performance falls (blind); lab-composition removal should move the violation score without a
  physiology change (the composition artifact); the length-only outcome should be learned under the original window and fall to chance under the fixed window.
