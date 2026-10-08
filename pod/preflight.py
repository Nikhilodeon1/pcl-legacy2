"""
Pod preflight. Read-only except for a tiny GPU smoke test. Run after `source pod/env.sh`:
    python pod/preflight.py
Exit code 1 if something REQUIRED for the fixed-observation reruns is missing.
"""
import os
import shutil
import sys

ok_all = True


def line(status, msg):
    print(f"[{status:4s}] {msg}")


def need(cond, msg_ok, msg_bad, required=True):
    global ok_all
    if cond:
        line("OK", msg_ok)
    else:
        line("FAIL" if required else "WARN", msg_bad)
        if required:
            ok_all = False
    return cond


print("python", sys.version.split()[0], sys.executable)
need(sys.version_info >= (3, 9), "python >= 3.9", "python < 3.9 (torch>=2.3 needs 3.9+): run pod/setup_env.sh")

try:
    import torch
    line("OK", f"torch {torch.__version__}")
    need(tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 3), "torch >= 2.3",
         "torch < 2.3: torch.amp.GradScaler('cuda') will fail")
    cuda = torch.cuda.is_available()
    need(cuda, "CUDA available", "CUDA NOT available (CPU wheel installed, or no GPU/driver)")
    if cuda:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        line("OK", f"GPU {name} sm_{cap[0]}{cap[1]} {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
        try:   # a kernel-image mismatch (e.g. Volta + a CUDA-12.8 wheel) only shows up when something runs
            x = torch.randn(256, 256, device="cuda")
            with torch.amp.autocast("cuda"):
                y = (x @ x).float().sum().item()
            torch.amp.GradScaler("cuda", enabled=True)
            torch.cuda.synchronize()
            line("OK", "fp16 autocast matmul + GradScaler smoke test")
        except Exception as e:
            need(False, "", f"GPU smoke test failed: {e}  (try TORCH_INDEX=.../cu124 or cu121 and rerun setup_env.sh)")
except Exception as e:
    need(False, "", f"torch import failed: {e}")

for mod in ("numpy", "pandas", "sklearn", "scipy"):
    try:
        m = __import__(mod); line("OK", f"{mod} {m.__version__}")
    except Exception as e:
        need(False, "", f"{mod} missing: {e}")

home, tmp = os.path.expanduser("~"), "/tmp"
for p, min_gb, why in ((home, 0.3, "persistent: repo + small results only (checkpoints/preds go to /tmp)"),
                       (tmp, 40, "ephemeral: venv, raw data, caches")):
    if os.path.exists(p):
        free = shutil.disk_usage(p).free / 2**30
        need(free >= min_gb, f"{p} free {free:.1f} GB ({why})", f"{p} free {free:.1f} GB < {min_gb} GB ({why})", required=False)
try:
    mem = [l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]
    need(int(mem) / 2**20 >= 24, f"RAM available {int(mem) / 2**20:.0f} GB", "RAM available < 24 GB: full eICU/MIMIC caches may not fit")
except Exception:
    pass

env = {k: os.environ.get(k, "") for k in ("PHYSIONET_DIR", "MIMIC_DIR", "EICU_DIR", "PCL_LEGACY2_PRETRAIN_DIR", "PCL_LEGACY2_CACHE_DIR")}
print("\n-- required for fixed-T reruns --")
pre = env["PCL_LEGACY2_PRETRAIN_DIR"]
have_pre = [m for m in ("erm", "pcl", "dro") if os.path.exists(os.path.join(pre, f"{m}_pretrained.pt"))]
need(len(have_pre) == 3, f"pretrained encoders in {pre}", f"pretrained encoders missing in {pre}: have {have_pre}. Put erm/pcl/dro_pretrained.pt there "
     "(from the old results_lambda17/ckpt). Without them NOTHING can be fine-tuned.")
cd = env["PCL_LEGACY2_CACHE_DIR"]
caches = {n: os.path.exists(os.path.join(cd, f"{n}_frac1.0.pkl")) for n in ("physionet", "mimic", "eicu")}
if all(caches.values()):
    line("OK", f"full caches present in {cd}")
else:
    line("WARN", f"caches missing in {cd}: {[n for n, v in caches.items() if not v]}  -> bash pod/build_caches.sh (needs raw data)")
    raw = {"PHYSIONET_DIR": os.path.isdir(os.path.join(env["PHYSIONET_DIR"], "training_setA")),
           "MIMIC_DIR": os.path.exists(os.path.join(env["MIMIC_DIR"], "icu", "chartevents.csv.gz")),
           "EICU_DIR": os.path.exists(os.path.join(env["EICU_DIR"], "patient.csv.gz"))}
    for k, v in raw.items():
        need(v or caches[{"PHYSIONET_DIR": "physionet", "MIMIC_DIR": "mimic", "EICU_DIR": "eicu"}[k]], f"{k} raw data present", f"{k}={env[k]} not found -> bash pod/fetch_data.sh")

print("\n-- optional (original-protocol audits) --")
repo = os.environ.get("PCL_REPO", ".")
n_los = len([f for f in os.listdir(os.path.join(repo, "results/los/ckpt"))]) if os.path.isdir(os.path.join(repo, "results/los/ckpt")) else 0
n_mort = len([f for f in os.listdir(os.path.join(repo, "results/mortality/ckpt"))]) if os.path.isdir(os.path.join(repo, "results/mortality/ckpt")) else 0
line("OK" if n_los == 9 else "WARN", f"original LOS fine-tuned ckpts: {n_los}/9 (results/los/ckpt)")
line("OK" if n_mort == 18 else "WARN", f"original mortality fine-tuned ckpts: {n_mort}/18 (results/mortality/ckpt) -- without them rerun the 18 original-protocol fine-tunes")
print("\nPREFLIGHT", "PASSED" if ok_all else "FAILED (fix the FAIL lines above)")
sys.exit(0 if ok_all else 1)
