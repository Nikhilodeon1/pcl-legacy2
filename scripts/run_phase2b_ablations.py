"""
Driver for Phase 2b Targeted Ablations:
1. Matched-Volume Pretraining Control:
   - ERM with source-only data (10,381 stays)
   - ERM with matched total volume (25,160 stays, resampling Site A to match the 10k + 14k total)
   - ERM with target domain exposure (10,381 Site A + 14,779 Site B unlabeled)
2. Loss-Balancer Mechanics Ablation:
   - PCL lambda=0.5 with standard EMA balancer
   - PCL lambda=0.5 with fixed weight (balancer disabled, balancer_target_ratio=None)
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "results", "phase2b", "pretrain"), exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "results", "phase2b", "sepsis"), exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "results", "phase2b", "los_T24"), exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "results", "revision", "logs"), exist_ok=True)

    commands = []
    for s in args.seeds:
        # 1. Matched-Volume Pretraining Control:
        # Tag: erm_matched_vol_p{s}
        tag_mv = f"erm_matched_vol_p{s}"
        commands.append((tag_mv, s, [sys.executable, "scripts/pretrain.py", "--lam", "0.0", "--seed", str(s),
                                     "--precision", "fp32", "--matched-volume-total", "25160",
                                     "--out-dir", args.out_dir]))

        # 2. Loss-Balancer Ablation (Fixed weight without balancer):
        # Tag: pcl_fixed_nobalance_p{s}
        tag_fnb = f"pcl_fixed_nobalance_p{s}"
        commands.append((tag_fnb, s, [sys.executable, "scripts/pretrain.py", "--lam", "0.5", "--seed", str(s),
                                      "--precision", "fp32", "--no-balancer",
                                      "--out-dir", args.out_dir]))

    logging.info(f"Queued {len(commands)} ablation jobs ({len(args.seeds)} seeds x 2 conditions)")

    # Execute with worker pool
    import concurrent.futures
    def run_one(item):
        tag, seed, cmd = item
        json_path = os.path.join(args.out_dir, f"{tag}.json")
        pt_path = os.path.join(args.out_dir, f"{tag}.pt")
        log_path = os.path.join(_ROOT, "results", "revision", "logs", f"phase2b_{tag}.log")

        if os.path.exists(json_path):
            logging.info(f"[RESUME] {tag} already pretrained, skipping.")
        else:
            logging.info(f"Starting pretrain {tag}...")
            t0 = time.time()
            with open(log_path, "w") as lf:
                res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            if res.returncode != 0:
                logging.error(f"FAILED: {tag} pretrain (see {log_path})")
                return tag, False
            logging.info(f"Done pretrain {tag} in {(time.time()-t0)/60:.1f} min")

        # Copy to results
        dest_json = os.path.join(_ROOT, "results", "phase2b", "pretrain", f"{tag}.json")
        try:
            shutil.copy2(json_path, dest_json)
        except Exception as e:
            logging.warning(f"Failed to copy {json_path} to {dest_json}: {e}")

        # Sepsis fine-tune
        sep_out = os.path.join(_ROOT, "results", "phase2b", "sepsis")
        sep_json = os.path.join(sep_out, f"{tag}_s{seed}.json")
        if not os.path.exists(sep_json):
            logging.info(f"Starting sepsis FT {tag}...")
            env = os.environ.copy()
            env["PCL_DISABLE_AMP"] = "1"
            with open(log_path, "a") as lf:
                subprocess.run([sys.executable, "scripts/finetune_sepsis.py", "--pretrain-ckpt", pt_path,
                                "--encoder-id", tag, "--seed", str(seed), "--out-dir", sep_out],
                               env=env, stdout=lf, stderr=subprocess.STDOUT)

        # LOS fine-tune
        los_out = os.path.join(_ROOT, "results", "phase2b", "los_T24")
        los_json = os.path.join(los_out, f"{tag}_s{seed}.json")
        if not os.path.exists(los_json):
            logging.info(f"Starting LOS FT {tag}...")
            env = os.environ.copy()
            env["PCL_DISABLE_AMP"] = "1"
            with open(log_path, "a") as lf:
                subprocess.run([sys.executable, "scripts/finetune_los.py", "--method", "pcl", "--seed", str(seed),
                                "--fixed-T", "24", "--pretrain-ckpt", pt_path, "--encoder-id", tag,
                                "--results-subdir", "phase2b/los_T24"],
                               env=env, stdout=lf, stderr=subprocess.STDOUT)

        return tag, True

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(run_one, item) for item in commands]
        for f in concurrent.futures.as_completed(futures):
            tag, success = f.result()
            logging.info(f"Job completed: {tag} (success={success})")


if __name__ == "__main__":
    main()
