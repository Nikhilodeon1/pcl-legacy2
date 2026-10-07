#!/usr/bin/env bash
# READ-ONLY inventory for Phase 0 (G0). Writes nothing except ./pod_inventory.txt.
# Run on the pod from /workspace (or wherever results_lambda17 / results_final_s* live):
#     bash pod_inventory.sh 2>&1 | tee pod_inventory.txt
# Then send pod_inventory.txt back. No training, no cost beyond a few seconds of CPU.
set -u
ROOT="${1:-$HOME}"
export ROOT_DIR="$ROOT"
echo "=== date / host ==="; date -u; hostname; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no gpu"

echo; echo "=== every *pretrained*.pt and fine-tuned ckpt under $ROOT (size, mtime) ==="
find "$ROOT" -xdev \( -name "*_pretrained.pt" -o -name "base_init.pt" \) -printf "%TY-%Tm-%Td %TH:%TM  %10s  %p\n" 2>/dev/null | sort -k4

echo; echo "=== results_* and cache directories ==="
ls -d "$ROOT"/*/results_* "$ROOT"/results_* "$ROOT"/*/cache* "$ROOT"/cache* 2>/dev/null
ls -la "$ROOT"/*/results/mortality/ckpt "$ROOT"/*/results/los/ckpt 2>/dev/null | head -60
ls -la "$ROOT"/*/results/mortality/ 2>/dev/null | grep -i "selection\|cache"

echo; echo "=== metadata inside each pretrained checkpoint (epoch, best val masked loss, effective lambda, config) ==="
python - <<'PY'
import glob, os, sys
try:
    import torch
except Exception as e:
    print("torch missing:", e); sys.exit()
root = os.environ.get("ROOT_DIR", "/workspace")
paths = sorted(set(glob.glob(root + "/**/*_pretrained.pt", recursive=True)))
for p in paths:
    try:
        o = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(p, "LOAD FAIL", e); continue
    if isinstance(o, dict) and "model_state" in o:
        h = o.get("history", {})
        def last(k):
            v = h.get(k, []); return round(float(v[-1]), 6) if len(v) else None
        nparams = sum(v.numel() for v in o["model_state"].values())
        print(f"{p}\n   kind=checkpoint epoch={o.get('epoch')} n_params={nparams} "
              f"train_masked_last={last('train_masked')} val_masked_last={last('val_masked')} "
              f"effective_lam_last={last('effective_lam')} balance_ratio_last={last('balance_ratio')} "
              f"train_MAP={last('train_MAP')} train_HH={last('train_HH')} train_SpO2={last('train_SpO2')} "
              f"train_PP={last('train_PP')} train_SI={last('train_SI')}")
    else:
        nparams = sum(v.numel() for v in o.values()) if isinstance(o, dict) else -1
        print(f"{p}\n   kind=raw_state_dict n_params={nparams} (no training history stored)")
PY

echo; echo "=== pretraining logs / env that fixed lambda and seed ==="
grep -rIl "Starting pretraining" "$ROOT" --include="*.log" --include="*.out" --include="nohup*" 2>/dev/null | head
grep -rIh "Starting pretraining\|Pretraining complete\|PCL_LAMBDA\|Using lambda" "$ROOT" --include="*.log" --include="*.out" --include="nohup*" 2>/dev/null | head -40
env | grep -i "PCL_"

echo; echo "=== git history (DRO origin, constraint weights, pretraining code) ==="
for d in "$ROOT"/* "$ROOT"; do
  if [ -d "$d/.git" ]; then
    echo "--- repo: $d"
    git -C "$d" log --oneline --date=short --format="%h %ad %s" -- run_paper_experiments.py src/baselines.py src/losses/pcl_loss.py | head -40
    echo "--- first commit that mentions DRO / DROFinetuner:"
    git -C "$d" log --reverse --date=short --format="%h %ad %s" -S"DROFinetuner" -- . | head -5
    echo "--- commits touching si_weight/pp_weight defaults:"
    git -C "$d" log --date=short --format="%h %ad %s" -S"pp_weight" -- src/losses/pcl_loss.py | head -5
  fi
done
echo "=== done ==="
