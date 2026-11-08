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
        try:
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
PHYS_ONLY = os.environ.get("PCL_PHYSIONET_ONLY", "1") == "1"
print("\n-- data (PhysioNet-only mode)" if PHYS_ONLY else "\n-- data")
cd = env["PCL_LEGACY2_CACHE_DIR"]
need(os.path.exists(os.path.join(cd, "physionet_frac1.0.pkl")) or os.path.isdir(os.path.join(env["PHYSIONET_DIR"], "training_setA")),
     "PhysioNet cache or raw data present", "PhysioNet missing -> bash pod/fetch_data.sh physionet && bash pod/build_caches.sh")
if PHYS_ONLY:
    for n in ("mimic", "eicu"):
        present = os.path.exists(os.path.join(cd, f"{n}_frac1.0.pkl")) or os.path.exists(os.path.expanduser(f"~/pcl_cache_gz/{n}_frac1.0.pkl.gz"))
        need(not present, f"no {n} cache on the pod", f"{n} cache FOUND on the pod: credentialed derivative, delete it (data use agreement)")
else:
    caches = {n: os.path.exists(os.path.join(cd, f"{n}_frac1.0.pkl")) for n in ("physionet", "mimic", "eicu")}
    line("OK" if all(caches.values()) else "WARN", f"caches in {cd}: {caches}")
pre = env["PCL_LEGACY2_PRETRAIN_DIR"]
have_pre = [m for m in ("erm", "pcl", "dro") if os.path.exists(os.path.join(pre, f"{m}_pretrained.pt"))]
line("OK" if have_pre else "INFO", f"pretrained encoders in {pre}: {have_pre or 'none (fine: the pretraining scripts make their own)'}")
print("\nPREFLIGHT", "PASSED" if ok_all else "FAILED (fix the FAIL lines above)")
sys.exit(0 if ok_all else 1)
