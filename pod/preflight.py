import os
import pickle
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
                       (tmp, 15, "ephemeral: venv, raw data, caches")):
    if os.path.exists(p):
        free = shutil.disk_usage(p).free / 2**30
        need(free >= min_gb, f"{p} free {free:.1f} GB ({why})", f"{p} free {free:.1f} GB < {min_gb} GB ({why})", required=False)

PHYS_ONLY = os.environ.get("PCL_PHYSIONET_ONLY", "1") == "1"
min_ram_gb = 4 if PHYS_ONLY else 24
try:
    mem = [l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]
    need(int(mem) / 2**20 >= min_ram_gb, f"RAM available {int(mem) / 2**20:.1f} GB",
         f"RAM available < {min_ram_gb} GB")
except Exception:
    pass

env = {k: os.environ.get(k, "") for k in ("PHYSIONET_DIR", "MIMIC_DIR", "EICU_DIR", "PCL_LEGACY2_PRETRAIN_DIR", "PCL_LEGACY2_CACHE_DIR")}
print("\n-- data (PhysioNet-only mode)" if PHYS_ONLY else "\n-- data")
cd = env["PCL_LEGACY2_CACHE_DIR"]
pn_pkl = os.path.join(cd, "physionet_frac1.0.pkl")
if os.path.exists(pn_pkl):
    try:
        pn = pickle.load(open(pn_pkl, "rb"))
        n_a = sum(1 for s in pn if s.get("site_id") == 0)
        n_b = sum(1 for s in pn if s.get("site_id") == 1)
        need(n_a == 10381 and n_b == 14779,
             f"PhysioNet cache verified: {len(pn)} stays (Site A={n_a}, Site B={n_b})",
             f"PhysioNet cache INCOMPLETE: got Site A={n_a}, Site B={n_b} (expected 10381 and 14779) -> rerun colab/setup.sh")
    except Exception as e:
        need(False, "", f"Failed to read {pn_pkl}: {e}")
else:
    need(os.path.isdir(os.path.join(env["PHYSIONET_DIR"], "training_setA")) and os.path.isdir(os.path.join(env["PHYSIONET_DIR"], "training_setB")),
         "PhysioNet raw data directories present", "PhysioNet cache and raw data missing -> run colab/setup.sh or pod/safe_start.sh")

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
