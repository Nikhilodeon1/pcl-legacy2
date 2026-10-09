"""
Build the FULL (fraction 1.0) MIMIC-IV and eICU caches locally, one after the other (keeps peak RAM low), then write
gzip copies named exactly as pod/restore_caches.sh expects, ready to upload to the pod:
    results/revision/cache_gz/{physionet,mimic,eicu}_frac1.0.pkl.gz

Run (detached is fine):  python local_full_build.py
Needs the raw data paths below (local layout) and ~6-8 GB of free RAM for the larger build; close other apps.
"""
import gzip
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
DS = os.environ.get("PCL_LOCAL_DATASETS", "C:/Users/nikhi/Codes/RES/datasets")
env = dict(os.environ)
env.setdefault("PHYSIONET_DIR", f"{DS}/physionet2019")
env.setdefault("MIMIC_DIR", f"{DS}/mimic/workspace/physionet.org/files/mimiciv/3.1")
env.setdefault("EICU_DIR", f"{DS}/eicu/workspace/physionet.org/files/eicu-crd/2.0")
env.pop("PCL_CACHE_DIR", None)
env.pop("PCL_LEGACY2_CACHE_DIR", None)

cache = os.path.join(REV, "cache")
gz = os.path.join(REV, "cache_gz")
os.makedirs(gz, exist_ok=True)
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

for name in ("physionet", "mimic", "eicu"):
    pkl = os.path.join(cache, f"{name}_frac1.0_s42.pkl")
    if not os.path.exists(pkl):
        log(f"building {name} (full)")
        r = subprocess.run([sys.executable, os.path.join(HERE, "build_cache.py"), name, "1.0"], env=env, cwd=HERE)
        if r.returncode != 0 or not os.path.exists(pkl):
            log(f"FAILED {name} (exit {r.returncode}); stopping")
            sys.exit(1)
    out = os.path.join(gz, f"{name}_frac1.0.pkl.gz")
    log(f"compressing {name} -> {out}")
    with open(pkl, "rb") as a, gzip.open(out, "wb", compresslevel=3) as b:
        shutil.copyfileobj(a, b)
    log(f"{name}: {os.path.getsize(pkl) / 1e6:.0f} MB -> {os.path.getsize(out) / 1e6:.0f} MB")
log("ALL DONE")
