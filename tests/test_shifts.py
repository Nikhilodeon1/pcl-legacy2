"""Tests for src/synth/shifts.py on real PhysioNet stays.  Run:  python tests/test_shifts.py   (or pytest tests/)"""
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PCL_TEST_MODE"] = "0"
from src.data.variables import PLAUS, VAR_TO_IDX  # noqa: E402
from src.synth import shifts as S  # noqa: E402

CACHE = os.environ.get("PCL_LEGACY2_CACHE_DIR") or os.path.join(ROOT, "results", "revision", "cache")


def _stays(n=600):
    for name in ("physionet_frac1.0.pkl", "physionet_frac1.0_s42.pkl"):
        p = os.path.join(CACHE, name)
        if os.path.exists(p):
            return [s for s in pickle.load(open(p, "rb")) if s["site_id"] == 0][:n]
    raise FileNotFoundError(CACHE)


def test_recompute_masks_matches_stored():
    for s in _stays():
        abg, c = S.recompute_masks(s["mask"])
        assert np.array_equal(abg, np.asarray(s["abg_mask"], bool))
        assert np.array_equal(c, np.asarray(s["c_mask"], bool))


def test_composition_shift():
    st = _stays(200)
    before = [np.asarray(s["mask"]).copy() for s in st]
    labs = ["pH", "HCO3", "pCO2", "PaO2"]
    full = S.composition_shift(st, labs, rate=1.0)
    none = S.composition_shift(st, labs, rate=0.0)
    cols = [VAR_TO_IDX[v] for v in labs]
    for a, b, c, m0 in zip(st, full, none, before):
        assert np.array_equal(np.asarray(a["mask"]), m0), "input was mutated"
        assert not b["mask"][:, cols].any() and not b["values"][:, cols].any()
        assert not b["abg_mask"].any() and not b["c_mask"][:, 3].any() and not b["c_mask"][:, 4].any()
        assert np.array_equal(b["c_mask"][:, 0], np.asarray(a["c_mask"])[:, 0])        # hemodynamic constraint unaffected
        assert np.array_equal(np.asarray(c["mask"]), m0)
    half = S.composition_shift(st, labs, rate=0.5, seed=1)
    frac = np.mean([not s["mask"][:, cols].any() for s in half])
    assert 0.35 < frac < 0.65 + 0.1 * (1 - np.mean([np.asarray(s["mask"])[:, cols].any() for s in st]))  # at least the removed share


def test_physiology_shift():
    st = _stays(200)
    out = S.physiology_shift(st, "MAP", 10.0)
    j = VAR_TO_IDX["MAP"]
    lo, hi = PLAUS["MAP"]
    for a, b in zip(st, out):
        m = np.asarray(a["mask"], bool)[:, j]
        assert not np.asarray(b["values"])[~m, j].any()                       # unobserved stays 0
        ra = lo + np.asarray(a["values"])[m, j] * (hi - lo)
        rb = lo + np.asarray(b["values"])[m, j] * (hi - lo)
        ok = np.minimum(ra + 10.0, hi)
        assert np.allclose(rb, ok, atol=1e-3), (rb[:3], ok[:3])
        assert np.array_equal(np.asarray(b["mask"]), np.asarray(a["mask"]))


def test_label_scale_shift():
    st = _stays(50)
    out = S.label_scale_shift(st, 3.0)
    for a, b in zip(st, out):
        assert abs(b["los_h"] - 3.0 * float(a["los_h"])) < 1e-9 and np.array_equal(b["values"], a["values"])


def test_length_outcome():
    st = _stays(600)
    out, thr = S.length_outcome(st, flip=0.0, seed=3)
    lens = np.array([S.observed_length(s["mask"]) for s in st])
    y = np.array([float(np.asarray(s["label"])[0]) for s in out])
    assert np.array_equal(y, (lens >= thr).astype(float))
    assert 0.3 < y.mean() < 0.7
    for a, b in zip(st, out):
        m = np.asarray(b["mask"], bool)
        assert not np.asarray(b["values"])[~m].any()
    own = np.mean([np.allclose(a["values"], b["values"]) for a, b in zip(st, out)])
    assert own < 0.2                                                           # values come from other stays


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
