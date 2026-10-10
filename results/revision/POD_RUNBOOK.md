# Pod runbook (JupyterHub pod: persistent `~`, ephemeral `/tmp` and `/`)

## CURRENT ORDER OF OPERATIONS (GPU is free, so the old cost gate is moot; V100/A10-class is fine)
1. Caches: `bash pod/restore_caches.sh` (needs the 3 `.pkl.gz` in `~/pcl_cache_gz/`).  -- done once per new pod
2. Encoders: upload the folder `results/revision/encoders_slim/` (local, 274 MB, gitignored) so that on the pod you have
   `~/pcl_pretrained/{erm,pcl,dro}_pretrained.pt` and `~/pcl_pretrained/sweep/lam{L}_p{S}.pt` (21 files, 13 MB each).
3. `git pull; source pod/env.sh; python pod/preflight.py`  -> must say PREFLIGHT PASSED (CUDA, encoders, caches).
4. Main fixed-observation reruns, 27 fine-tunes: `bash pod/run_fixed_T.sh`  (LOS first: `ONLY=los bash pod/run_fixed_T.sh`).
5. Inference audits on the ORIGINAL checkpoints: `bash pod/run_audits.sh`.
6. Lambda x pretraining-seed sweep, 54 fine-tunes (B2/B3): `ONLY=los bash pod/run_sweep.sh` first (about 1 h), then the rest.
7. Commit the small JSON/CSV results (`results/los_fixed_T24*/`, `results/mortality_fixed_T48*/`, `results/revision/*.csv|json`); push from the IDE.

Why this is needed: both G1 leakage flags are tripped (SURPRISES.md), so PREREG requires the fixed-observation rerun, and the local
machine has no GPU, no pretrained encoders and no mortality checkpoints. Pod: **V100, 16 cores, 64 GB**.

Layout rule: repo + small results in `~` (persistent, 5 GB). venv, raw data, caches in `/tmp` (rebuilt per pod; caches can be persisted to `~/pcl_cache`, ~2-3 GB).

## 1. Every new pod (about 5 minutes)
```bash
cd ~ && git clone https://github.com/Nikhilodeon1/pcl-legacy2.git     # or: cd ~/pcl-legacy2 && git pull   (not while a run is going)
cd ~/pcl-legacy2
bash pod/setup_env.sh          # venv in /tmp/venv, CUDA torch (cu126: new wheels dropped V100), prints a preflight report
source pod/env.sh              # in EVERY new shell: activates the venv and exports all paths
python pod/preflight.py        # re-check anytime; read the FAIL / WARN lines
```
If the V100 smoke test fails with "no kernel image", rerun with `TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash pod/setup_env.sh`
(delete `/tmp/venv` first).

## 2. Things only you can supply (not in git)
1. **Pretrained encoders** (REQUIRED): `~/pcl_pretrained/{erm,pcl,dro}_pretrained.pt` (about 13 MB each; the old `results_lambda17/ckpt/`).
   Without them nothing can be fine-tuned.
2. **Raw data or caches.** Either (a) prebuilt full caches `{physionet,mimic,eicu}_frac1.0.pkl` in `~/pcl_cache` (or `/tmp/pcl_cache`), or (b) raw data:
```bash
export PHYSIONET_USER=<your username>; read -rs PHYSIONET_PASSWORD; export PHYSIONET_PASSWORD     # typed by you; never stored
bash pod/fetch_data.sh                 # only the 13 files the loaders read; into /tmp/pcl_data (resumable)
bash pod/build_caches.sh               # ~45-60 min, CPU only, 3 builds in parallel -> /tmp/pcl_cache
bash pod/persist_caches.sh            # gzip copies into ~/pcl_cache_gz (~8x smaller, ~300 MB); on a new pod: bash pod/restore_caches.sh
```
   `build_caches.sh` is CPU work on a GPU-priced pod. Do it first on a CPU-only pod if you can, persist to `~/pcl_cache`, then switch.
3. *Optional, for the original-protocol audits:* the original fine-tuned checkpoints in `results/los/ckpt/` (9) and `results/mortality/ckpt/` (18).
4. *Optional:* `results/mortality/selection_criteria.json` (needed for Table 6 / ATC and `a4_selection.py --task mortality`).

## 3. Read-only inventory, on whichever machine holds the old `results_lambda17` / `results_final_s*` (about 1 minute)
```bash
bash results/revision/scripts/pod_inventory.sh /workspace 2>&1 | tee pod_inventory.txt        # or the path of the old volume
```
Send back `pod_inventory.txt`. It answers whether independently pretrained encoders (`results_final_s43/s44`) still exist, which would make
B2 almost free, plus the lambda / recipe metadata and the git history for DRO.

## 4. Fixed-observation reruns (GATED: tell me your hourly rate first)
LOS T=24 (cohort unchanged), mortality T=48 (drops stays shorter than 48h; reported as a cohort change). Seeds 42/43/44. 27 fine-tunes.
```bash
source pod/env.sh
export POD_RATE_USD_PER_H=<rate>          # prints running cost
bash pod/run_fixed_T.sh                   # ONLY=los or ONLY=mortality to split; resumable; logs in results/revision/logs/
```
Estimate: about 2.2 GPU-hours on a fast GPU (190 s x 9 LOS + 340 s x 18 mortality), so **budget 2.5-4 h on a V100**.
Gate: if rate x 4 h exceeds $5, stop and tell the strategy agent. Outputs: `results/los_fixed_T24/*.json`, `results/mortality_fixed_T48/*.json`
(commit the JSONs; checkpoints are gitignored and ~350 MB total, which fits in `~`).

## 5. Inference audits on the original checkpoints (minutes)
```bash
bash pod/run_audits.sh        # skips any step whose inputs are missing; writes results/revision/{g1_*,g2_*,a1_*,a3_*}
```

## 6. Getting results back
Commit small files only (JSON/CSV/PNG under `results/`); `cache/`, `preds/`, `logs/` and every `ckpt/` are gitignored. Push from the IDE, or
download `results/revision/*.csv|json` before the pod is replaced. Never leave results only in `/tmp`.

## Rules (from the pod's storage model)
- Check `df -h ~` before long jobs. Do not `git pull`/commit while a run is going (result folder names depend on the code version).
- Re-run `setup_env.sh` + `source pod/env.sh` on every new pod; `/tmp` is wiped.
