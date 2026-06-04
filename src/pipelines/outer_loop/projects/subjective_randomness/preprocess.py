"""Featurize subjective_randomness responses for PyMC models.

Raw collected responses carry the H/T sequence strings (`sequence_a`,
`sequence_b`) plus the binary choice (`chose_left`). PyMC cognitive models read
*numeric* per-sequence features through `pm.Data` containers whose names match
CSV columns. This module derives those feature columns from the raw sequences.

The inner model loop loads `featurize_stimulus` from here (by project) to turn
pooled raw responses into the feature CSV the PyMC models are fit on; the design
step uses the same function to score candidate stimuli for EIG.

Feature columns per sequence (`a` and `b`):
    n_<x>        total length                       (int)
    h_<x>        head count                         (int)
    alts_<x>     alternation count (H/T transitions)(int)
    max_run_<x>  longest constant run               (int)
    p_<x>        head proportion                    (float)
    p_alts_<x>   alternation proportion             (float)
"""

from __future__ import annotations

from typing import Dict


def _sequence_features_int(seq: str, suffix: str) -> Dict[str, int]:
    s = seq.strip().upper()
    n = len(s)
    h = sum(1 for c in s if c == "H")
    alts = sum(1 for i in range(1, n) if s[i] != s[i - 1])
    max_run = 0
    cur = 0
    prev = ""
    for c in s:
        cur = cur + 1 if c == prev else 1
        prev = c
        max_run = max(max_run, cur)
    return {
        f"n_{suffix}": n,
        f"h_{suffix}": h,
        f"alts_{suffix}": alts,
        f"max_run_{suffix}": max_run,
    }


def _sequence_features_float(suffix: str, n: int, h: int, alts: int) -> Dict[str, float]:
    return {
        f"p_{suffix}": (h / n) if n > 0 else 0.0,
        f"p_alts_{suffix}": (alts / (n - 1)) if n > 1 else 0.0,
    }


def featurize_stimulus(sequence_a: str, sequence_b: str) -> Dict[str, float]:
    """Return the full numeric feature dict for one (sequence_a, sequence_b) pair.

    Keys match the `pm.Data` container names PyMC models use: n_a, h_a, alts_a,
    max_run_a, p_a, p_alts_a and the `_b` counterparts.
    """
    ints_a = _sequence_features_int(sequence_a, "a")
    ints_b = _sequence_features_int(sequence_b, "b")
    floats_a = _sequence_features_float("a", ints_a["n_a"], ints_a["h_a"], ints_a["alts_a"])
    floats_b = _sequence_features_float("b", ints_b["n_b"], ints_b["h_b"], ints_b["alts_b"])
    return {**ints_a, **ints_b, **floats_a, **floats_b}
