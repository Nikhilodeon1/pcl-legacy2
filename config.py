import os
import torch

TEST_MODE = os.environ.get("PCL_TEST_MODE", "1") == "1"

SEED = int(os.environ.get("PCL_SEED", "42"))

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PHYSIONET_DIR = os.environ.get("PHYSIONET_DIR", os.path.join(_BASE_DATA_DIR, "physionet2019"))
MIMIC_DIR     = os.environ.get("MIMIC_DIR",     os.path.join(_BASE_DATA_DIR, "mimic4-demo"))
EICU_DIR      = os.environ.get("EICU_DIR",      os.path.join(_BASE_DATA_DIR, "eICU-demo"))
DATA_DIR = MIMIC_DIR  # Legacy alias for old scripts/
RESULTS_DIR    = os.environ.get("RESULTS_DIR",    os.path.join(os.path.dirname(__file__), "results"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(RESULTS_DIR, "checkpoints"))
CACHE_DIR      = os.environ.get("CACHE_DIR", CHECKPOINT_DIR)

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_FRACTION = 0.15 if TEST_MODE else 1.0
HOURS = 48
MAX_FFILL = 6
MIN_LOS_H = 24

# ── Model ────────────────────────────────────────────────────────────────────
D_MODEL = 64 if TEST_MODE else 256
N_HEADS = 4 if TEST_MODE else 8
N_LAYERS = 2 if TEST_MODE else 6
FFN_DIM = D_MODEL * 2

from src.data.variables import CANONICAL_VARIABLES as _CANON
N_VARS = len(_CANON)
DROPOUT = 0.1

# ── Training ─────────────────────────────────────────────────────────────────
LAMBDA_PCL = float(os.environ.get("PCL_LAMBDA", "0.5"))
BATCH_SIZE = 16 if TEST_MODE else 64
MASK_PROB = 0.30
LR = 3e-4
WEIGHT_DECAY = 1e-4
PRETRAIN_EPOCHS = 5 if TEST_MODE else 30
FINETUNE_EPOCHS = 5 if TEST_MODE else 30
PRETRAIN_EPOCHS_DEMO = PRETRAIN_EPOCHS 
FINETUNE_EPOCHS_DEMO = FINETUNE_EPOCHS 

# ── GPU / Performance ────────────────────────────────────────────────────────
NUM_WORKERS = 0 if TEST_MODE else int(os.environ.get("NUM_WORKERS", "8"))
PIN_MEMORY = torch.cuda.is_available()
USE_AMP = torch.cuda.is_available() and not TEST_MODE and os.environ.get("PCL_DISABLE_AMP", "0") != "1"

# ── Noise experiment ─────────────────────────────────────────────────────────
NOISE_LEVELS = [0, 5, 10, 15]

# ── Lambda sweep ─────────────────────────────────────────────────────────────
LAMBDA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
