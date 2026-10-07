"""
Build preprocessed-sample caches for the revision analyses, using the SAME loaders
the original fine-tune scripts used (src/data/{physionet2019,mimic4,eicu}.py), so the
tensors are byte-identical in construction to what the models saw.

Usage: python build_cache.py <dataset> <fraction> [seed]
    dataset in {physionet, mimic, eicu}
Cache path: results/revision/cache/{dataset}_frac{fraction}_s{seed}.pkl

Env overrides: PHYSIONET_DIR, MIMIC_DIR, EICU_DIR (defaults below = local RES/datasets layout).
"""
import logging
import os
import pickle
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
os.environ["PCL_TEST_MODE"] = "0"

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
sys.path.insert(0, ROOT)

# PHYSIONET_DIR / MIMIC_DIR / EICU_DIR must be exported by the caller (see pod/env.sh); config.py reads them.

name = sys.argv[1]
fraction = float(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

from config import PHYSIONET_DIR, MIMIC_DIR, EICU_DIR  # noqa: E402
from src.data.physionet2019 import load_physionet2019  # noqa: E402
from src.data.mimic4 import load_mimic4  # noqa: E402
from src.data.eicu import load_eicu  # noqa: E402

out_dir = os.environ.get("PCL_CACHE_DIR") or os.environ.get("PCL_LEGACY2_CACHE_DIR") or os.path.join(REV, "cache")
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, f"{name}_frac{fraction}_s{seed}.pkl")
if os.path.exists(path):
    print("exists", path)
    sys.exit(0)

t0 = time.time()
if name == "physionet":
    samples, _ = load_physionet2019(PHYSIONET_DIR, fraction=fraction, seed=seed)
elif name == "mimic":
    samples, _ = load_mimic4(MIMIC_DIR, fraction=fraction, seed=seed)
elif name == "eicu":
    samples, _ = load_eicu(EICU_DIR, fraction=fraction, seed=seed)
else:
    raise SystemExit("dataset must be physionet|mimic|eicu")
if len(samples) == 0:
    raise SystemExit(f"{name}: 0 samples loaded, refusing to cache")
with open(path, "wb") as fp:
    pickle.dump(samples, fp, protocol=pickle.HIGHEST_PROTOCOL)
print(f"saved {path}: {len(samples)} samples in {time.time() - t0:.0f}s")
