"""
Driver for Phase 2c Reviewer Experiments:
1. VICReg Anti-Collapse Intervention (pcl_varreg1_p{seed}):
   - PCL lambda=0.5 (FP32) + VICReg variance & covariance regularization (--var-reg 1.0)
   - Tests whether restoring centered effective rank during pretraining recovers Site B OOD transfer.
2. Supervised Transformer From Scratch (scratch_p{seed}):
   - Same 6-layer, d=256 Transformer trained directly on Sepsis and LOS T=24 without pretraining.
"""
import argparse
import concurrent.futures
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
    os.makedirs(os.path.join(_ROOT, "results", "phase2c", "pretrain"), exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "results", "phase2c", "sepsis"), exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "results", "phase2c", "los_T24"), exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "results", "revision", "logs"), exist_ok=True)

    jobs = []
    for s in args.seeds:
        # 1. VICReg Anti-Collapse Intervention (Pretrain + Sepsis FT + LOS FT)
        tag_vr = f"pcl_varreg1_p{s}"
        jobs.append(("varreg", tag_vr, s, [
            sys.executable, "scripts/pretrain.py", "--lam", "0.5", "--seed", str(s),
            "--precision", "fp32", "--var-reg", "1.0", "--out-dir", args.out_dir
        ]))
        # 2. Supervised Transformer From Scratch (No Pretrain, Sepsis FT + LOS FT)
        tag_sc = f"scratch_p{s}"
        jobs.append(("scratch", tag_sc, s, None))

    logging.info(f"Queued {len(jobs)} Phase 2c jobs ({len(args.seeds)} seeds x 2 conditions)")

    def run_one(item):
        kind, tag, seed, pretrain_cmd = item
        log_path = os.path.join(_ROOT, "results", "revision", "logs", f"phase2c_{tag}.log")
        pt_path = "none"

        if kind == "varreg":
            json_path = os.path.join(args.out_dir, f"{tag}.json")
            pt_path = os.path.join(args.out_dir, f"{tag}.pt")
            if os.path.exists(json_path):
                logging.info(f"[RESUME] {tag} already pretrained, skipping.")
            else:
                logging.info(f"Starting pretrain {tag}...")
                t0 = time.time()
                with open(log_path, "w") as lf:
                    res = subprocess.run(pretrain_cmd, stdout=lf, stderr=subprocess.STDOUT)
                if res.returncode != 0:
                    logging.error(f"FAILED: {tag} pretrain (see {log_path})")
                    return tag, False
                logging.info(f"Done pretrain {tag} in {(time.time() - t0) / 60:.1f} min")

            dest_json = os.path.join(_ROOT, "results", "phase2c", "pretrain", f"{tag}.json")
            try:
                shutil.copy2(json_path, dest_json)
            except Exception as e:
                logging.warning(f"Failed to copy {json_path} to {dest_json}: {e}")

        # Sepsis fine-tune
        sep_out = os.path.join(_ROOT, "results", "phase2c", "sepsis")
        sep_json = os.path.join(sep_out, f"{tag}_s{seed}.json")
        if not os.path.exists(sep_json):
            logging.info(f"Starting sepsis FT {tag}...")
            env = os.environ.copy()
            env["PCL_DISABLE_AMP"] = "1"
            cmd_sep = [
                sys.executable, "scripts/finetune_sepsis.py",
                "--pretrain-ckpt", pt_path, "--encoder-id", tag,
                "--seed", str(seed), "--out-dir", sep_out
            ]
            if kind == "scratch":
                cmd_sep.append("--scratch")
            with open(log_path, "a") as lf:
                subprocess.run(cmd_sep, env=env, stdout=lf, stderr=subprocess.STDOUT)

        # LOS T=24 fine-tune
        los_out = os.path.join(_ROOT, "results", "phase2c", "los_T24")
        los_json = os.path.join(los_out, f"{tag}_s{seed}.json")
        if not os.path.exists(los_json):
            logging.info(f"Starting LOS FT {tag}...")
            env = os.environ.copy()
            env["PCL_DISABLE_AMP"] = "1"
            cmd_los = [
                sys.executable, "scripts/finetune_los.py",
                "--method", "scratch" if kind == "scratch" else "pcl",
                "--seed", str(seed), "--fixed-T", "24",
                "--pretrain-ckpt", pt_path, "--encoder-id", tag,
                "--results-subdir", "phase2c/los_T24"
            ]
            if kind == "scratch":
                cmd_los.append("--scratch")
            with open(log_path, "a") as lf:
                subprocess.run(cmd_los, env=env, stdout=lf, stderr=subprocess.STDOUT)

        return tag, True

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(run_one, item) for item in jobs]
        for f in concurrent.futures.as_completed(futures):
            tag, success = f.result()
            logging.info(f"Job completed: {tag} (success={success})")


if __name__ == "__main__":
    main()
