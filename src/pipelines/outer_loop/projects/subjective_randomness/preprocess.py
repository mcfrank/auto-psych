"""Featurize subjective_randomness responses for PyMC models.

Raw collected responses carry the H/T sequence strings (`sequence_a`,
`sequence_b`) plus the binary choice (`chose_left`). PyMC cognitive models read
*numeric* per-sequence features through `pm.Data` containers whose names match
CSV columns. This module derives those feature columns from the raw sequences.

The inner model loop loads `featurize_stimulus` from here (by project) to turn
pooled raw responses into the feature CSV the PyMC models are fit on; the design
step uses the same function to score candidate stimuli for EIG.

Feature columns per sequence (`a` and `b`):
    n_<x>             total length                        (int)
    h_<x>             head count                          (int)
    alts_<x>          alternation count (H/T transitions) (int)
    max_run_<x>       longest constant run                (int)
    p_<x>             head proportion                     (float)
    p_alts_<x>        alternation proportion              (float)
    max_run_norm_<x>  longest run scaled to [0, 1]        (float)
    imbalance_<x>     distance from 50/50 heads/tails     (float)
    periodicity_<x>   short repeating-template score      (float)
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


def _periodicity_score(seq: str) -> float:
    """Return how well a sequence matches a short repeating template."""
    s = seq.strip().upper()
    n = len(s)
    if n <= 2:
        return 0.0

    best_match = 0.5
    for period in range(1, (n // 2) + 1):
        template = s[:period]
        matches = sum(1 for i, c in enumerate(s) if c == template[i % period])
        best_match = max(best_match, matches / n)
    return max(0.0, min(1.0, 2.0 * (best_match - 0.5)))


def _sequence_features_float(
    seq: str,
    suffix: str,
    n: int,
    h: int,
    alts: int,
    max_run: int,
) -> Dict[str, float]:
    return {
        f"p_{suffix}": (h / n) if n > 0 else 0.0,
        f"p_alts_{suffix}": (alts / (n - 1)) if n > 1 else 0.0,
        f"max_run_norm_{suffix}": ((max_run - 1) / (n - 1)) if n > 1 else 0.0,
        f"imbalance_{suffix}": 2.0 * abs((h / n) - 0.5) if n > 0 else 0.0,
        f"periodicity_{suffix}": _periodicity_score(seq),
    }


def featurize_stimulus(sequence_a: str, sequence_b: str) -> Dict[str, float]:
    """Return the full numeric feature dict for one (sequence_a, sequence_b) pair.

    Keys match the `pm.Data` container names PyMC models use: n_a, h_a, alts_a,
    max_run_a, p_a, p_alts_a, max_run_norm_a, imbalance_a, periodicity_a and
    the `_b` counterparts.
    """
    ints_a = _sequence_features_int(sequence_a, "a")
    ints_b = _sequence_features_int(sequence_b, "b")
    floats_a = _sequence_features_float(
        sequence_a,
        "a",
        ints_a["n_a"],
        ints_a["h_a"],
        ints_a["alts_a"],
        ints_a["max_run_a"],
    )
    floats_b = _sequence_features_float(
        sequence_b,
        "b",
        ints_b["n_b"],
        ints_b["h_b"],
        ints_b["alts_b"],
        ints_b["max_run_b"],
    )
    return {**ints_a, **ints_b, **floats_a, **floats_b}
