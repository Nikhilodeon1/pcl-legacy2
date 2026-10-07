"""
Fixed-observation protocol (revision G1 remedy).

Every stay is shown exactly the first T hours: values/masks at hours >= T are zeroed/False for
EVERY stay, so the padding pattern and window length carry no information about stay length.
Stays with LOS < min_los_h are dropped (a cohort change, reported alongside results).
The classification/regression head pools at the fixed index T-1 (ClassificationHead.fixed_index),
not at the last observed hour.
"""
import numpy as np


def fix_window(samples, T, min_los_h):
    out = []
    for s in samples:
        if "los_h" not in s:
            raise KeyError("sample lacks los_h: the cache predates the los_h field; rebuild it (see finetune_los.py docstring)")
        if float(s["los_h"]) < min_los_h:
            continue
        t = dict(s)
        v = np.array(s["values"], copy=True)
        m = np.array(s["mask"], copy=True)
        v[T:] = 0.0
        m[T:] = False
        t["values"], t["mask"] = v, m
        if "abg_mask" in s:
            a = np.array(s["abg_mask"], copy=True)
            a[T:] = False
            t["abg_mask"] = a
        if "c_mask" in s:
            c = np.array(s["c_mask"], copy=True)
            c[T:] = False
            t["c_mask"] = c
        out.append(t)
    return out
