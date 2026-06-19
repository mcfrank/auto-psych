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
    rep_motifs_<x>    repetition motifs in motif parse    (int)
    alt_motifs_<x>    alternation motifs in motif parse   (int)
    p_<x>             head proportion                     (float)
    p_alts_<x>        alternation proportion              (float)
    max_run_norm_<x>  longest run scaled to [0, 1]        (float)
    imbalance_<x>     distance from 50/50 heads/tails     (float)
    periodicity_<x>   short repeating-template score      (float)
"""

from __future__ import annotations

from typing import Dict, Tuple


def _parse_motifs(seq: str) -> Tuple[int, int]:
    """Parse an H/T sequence into Falk & Konold (1997) motifs.

    Returns ``(rep_motifs, alt_motifs)`` — the counts the statistical-inference
    model of Griffiths, Daniels, Austerweil & Tenenbaum (2018) calls n1 and n2:
    the number of repetition motifs (maximal constant runs) and alternation
    motifs (maximal alternating sub-sequences of length >= 2) in the canonical
    minimal-description parse of the sequence.

    The parse run-length-encodes the sequence, then groups any maximal stretch
    of >= 2 consecutive length-1 runs (which necessarily alternate H/T) into one
    alternation motif; every other run is a repetition motif. This is the parse
    underlying Falk & Konold's Difficulty Predictor, for which DP = n1 + 2*n2.
    For example HHTTHTHT -> runs [HH, TT, H, T, H, T] -> repetition motifs
    {HH, TT} and one alternation motif {HTHT}, giving (2, 1) and DP = 4.
    """
    s = seq.strip().upper()
    n = len(s)
    if n == 0:
        return 0, 0

    run_lengths = []
    cur = 1
    for i in range(1, n):
        if s[i] == s[i - 1]:
            cur += 1
        else:
            run_lengths.append(cur)
            cur = 1
    run_lengths.append(cur)

    rep_motifs = 0
    alt_motifs = 0
    i = 0
    n_runs = len(run_lengths)
    while i < n_runs:
        if run_lengths[i] == 1:
            # Maximal stretch of consecutive length-1 runs == an alternating block.
            j = i
            while j < n_runs and run_lengths[j] == 1:
                j += 1
            if j - i >= 2:
                alt_motifs += 1
            else:
                rep_motifs += 1  # a single isolated symbol is its own run
            i = j
        else:
            rep_motifs += 1  # a constant run of length >= 2
            i += 1
    return rep_motifs, alt_motifs


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
    rep_motifs, alt_motifs = _parse_motifs(s)
    return {
        f"n_{suffix}": n,
        f"h_{suffix}": h,
        f"alts_{suffix}": alts,
        f"max_run_{suffix}": max_run,
        f"rep_motifs_{suffix}": rep_motifs,
        f"alt_motifs_{suffix}": alt_motifs,
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
