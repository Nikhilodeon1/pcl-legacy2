"""
Controlled (semi-synthetic) distribution shifts applied to REAL stays, one mechanism at a time.

The idea (answering the meta-review's request): take real Site A/B stays, apply ONE known manipulation to the evaluation set, and ask which audit
tool (performance, representation distance, MMD, reconstruction error, constraint violation, mask/length probe, fixed-window protocol) responds.
Because the manipulation is known, "the audit should (not) fire" is a ground truth, independent of dataset-specific artifacts and preprocessing.

Stay format: dict with 'values' (48 x V, min-max scaled by PLAUS, 0 where unobserved), 'mask' (48 x V bool), 'abg_mask' (48,), 'c_mask' (48 x 5),
'label' (48,) (stay-level label broadcast), 'los_h', ... exactly what src.data.* produce. All functions return NEW stay dicts and never modify the input.

Mechanisms
  composition_shift   remove all observations of some variables (e.g. the blood-gas labs) in a fraction of the stays, as a hospital that orders fewer labs
                      would. Inputs change, the outcome relation does not. The constraint-violation score is computed only where constraints are evaluable,
                      so it moves for a reason unrelated to physiology (the "composition artifact").
  physiology_shift    add a bias, in physical units, to one variable of a coupled set (MAP vs SBP/DBP, pH vs HCO3/pCO2, PaO2 vs SpO2), which breaks the
                      relation the constraints encode. The relation is violated, labels unchanged.
  label_scale_shift   multiply the LOS label by a factor (inputs unchanged): a pure label-support shift. Representation-space criteria cannot see it.
  length_outcome      positive control for the stay-length shortcut: a synthetic binary outcome that depends ONLY on observed stay length (+ label noise), with the
                      values taken from other stays so they carry no information. A model that pools at the last observed hour scores well on it; under the
                      fixed-observation window it must fall to chance.
"""
import copy

import numpy as np

from src.data.variables import ABG_VARIABLES, CANONICAL_VARIABLES, PLAUS, VAR_TO_IDX

IDX = VAR_TO_IDX


def recompute_masks(mask):
    """abg_mask (T,) and c_mask (T, 5) from the observation mask, exactly as src.data.preprocessing builds them from the raw series."""
    m = np.asarray(mask, bool)

    def allof(vs):
        out = np.ones(m.shape[0], bool)
        for v in vs:
            out &= m[:, IDX[v]]
        return out

    c = np.zeros((m.shape[0], 5), bool)
    c[:, 0] = allof(["SBP", "DBP", "MAP"])
    c[:, 1] = c[:, 0]
    c[:, 2] = allof(["HR", "SBP"])
    c[:, 3] = allof(ABG_VARIABLES[:3])
    c[:, 4] = allof(["SpO2", "PaO2"])
    return allof(ABG_VARIABLES), c


def _copy(s):
    return {k: (v.copy() if isinstance(v, np.ndarray) else copy.copy(v)) for k, v in s.items()}


def composition_shift(samples, variables, rate, seed=0):
    """Remove every observation of `variables` in a random fraction `rate` of the stays."""
    rng = np.random.default_rng(seed)
    cols = [IDX[v] for v in variables]
    out = []
    for s in samples:
        t = _copy(s)
        if rng.random() < rate:
            t["mask"] = np.asarray(t["mask"], bool).copy()
            t["values"] = np.asarray(t["values"], float).copy()
            t["mask"][:, cols] = False
            t["values"][:, cols] = 0.0
            t["abg_mask"], t["c_mask"] = recompute_masks(t["mask"])
        out.append(t)
    return out


def _to_raw(x, var):
    lo, hi = PLAUS[var]
    return lo + x * (hi - lo)


def _to_scaled(x, var):
    lo, hi = PLAUS[var]
    return (x - lo) / (hi - lo + 1e-8)


def physiology_shift(samples, var, bias_raw, seed=0):
    """Add `bias_raw` (in the variable's physical unit, e.g. mmHg for MAP) to every OBSERVED value of `var`, clipped to the plausible range.
    Use MAP (breaks MAP ~ (SBP + 2 DBP) / 3), pH (breaks Henderson-Hasselbalch with HCO3/pCO2) or PaO2 (breaks the SpO2-PaO2 relation)."""
    j = IDX[var]
    lo, hi = PLAUS[var]
    out = []
    for s in samples:
        t = _copy(s)
        v = np.asarray(t["values"], float).copy()
        m = np.asarray(t["mask"], bool)
        raw = np.clip(_to_raw(v[:, j], var) + bias_raw, lo, hi)
        v[:, j] = np.where(m[:, j], _to_scaled(raw, var), 0.0)
        t["values"] = v
        out.append(t)
    return out


def label_scale_shift(samples, factor, key="los_h"):
    out = []
    for s in samples:
        t = _copy(s)
        t[key] = float(t[key]) * factor
        if "los_3d" in t:
            t["los_3d"] = int(t[key] > 72)
        out.append(t)
    return out


def observed_length(mask):
    """Number of hours up to and including the last hour with any observation (what a last-observed-hour head can read)."""
    m = np.asarray(mask, bool).any(axis=1)
    return int(np.nonzero(m)[0].max()) + 1 if m.any() else 0


def length_outcome(samples, flip=0.05, seed=0, threshold=None):
    """Synthetic binary outcome = [observed length >= threshold] with probability 1-flip of being correct; values are replaced by those of a random OTHER stay
    (re-masked with this stay's own mask) so the values carry no information about the outcome. Returns (new_samples, threshold)."""
    rng = np.random.default_rng(seed)
    lens = np.array([observed_length(s["mask"]) for s in samples])
    if threshold is None:   # the cut that makes the classes closest to balanced (many stays are padded to the full 48 h, so the median itself would be degenerate)
        cands = np.unique(lens)
        threshold = float(cands[int(np.argmin([abs((lens >= c).mean() - 0.5) for c in cands]))])
    thr = threshold
    perm = rng.permutation(len(samples))
    out = []
    for i, s in enumerate(samples):
        t = _copy(s)
        donor = np.asarray(samples[perm[i]]["values"], float)
        m = np.asarray(t["mask"], bool)
        t["values"] = np.where(m, donor, 0.0)
        y = float(lens[i] >= thr)
        if rng.random() < flip:
            y = 1.0 - y
        t["label"] = np.full(len(np.asarray(t["label"])), y)
        out.append(t)
    return out, thr
