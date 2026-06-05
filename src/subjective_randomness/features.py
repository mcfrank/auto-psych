"""Numeric features derived from subjective-randomness H/T sequence pairs.

Takes an "old"-style responses.csv with raw H/T sequence strings:
    participant_id, trial_index, sequence_a, sequence_b, chose_left, chose_right, model

and produces numeric feature columns derived from each sequence. The numeric
columns are what theorist PyMC models pull into ``pm.Data`` containers (one
container per numeric column name).

Feature columns per sequence (``a`` and ``b``):
    n_<x>             total length
    h_<x>             head count
    p_<x>             head proportion (heads / length)
    alts_<x>          alternation count (transitions between H and T)
    p_alts_<x>        alternation proportion (alts / (length - 1))
    max_run_<x>       length of the longest constant run
    max_run_norm_<x>  max_run scaled to [0, 1]
    imbalance_<x>     distance from 50/50 heads/tails
    periodicity_<x>   simple repeating-template score
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

PASSTHROUGH_COLS = ["participant_id", "trial_index", "sequence_a", "sequence_b", "chose_left"]
INT_FEATURE_COLS = ["n_a", "h_a", "alts_a", "max_run_a", "n_b", "h_b", "alts_b", "max_run_b"]
FLOAT_FEATURE_COLS = [
    "p_a", "p_alts_a", "max_run_norm_a", "imbalance_a", "periodicity_a",
    "p_b", "p_alts_b", "max_run_norm_b", "imbalance_b", "periodicity_b",
]
REQUIRED_INPUT_COLS = {"sequence_a", "sequence_b", "chose_left"}


def sequence_features(seq: str, suffix: str) -> Dict[str, int]:
    """Integer features derived from a single H/T sequence string."""
    s = seq.strip().upper()
    n = len(s)
    h = sum(1 for c in s if c == "H")
    alts = sum(1 for i in range(1, n) if s[i] != s[i - 1])
    # Longest run of consecutive identical characters
    max_run = 0
    cur = 0
    prev = ""
    for c in s:
        if c == prev:
            cur += 1
        else:
            cur = 1
            prev = c
        if cur > max_run:
            max_run = cur
    return {
        f"n_{suffix}": n,
        f"h_{suffix}": h,
        f"alts_{suffix}": alts,
        f"max_run_{suffix}": max_run,
    }


def sequence_features_float(seq: str, suffix: str, n: int, alts: int, h: int, max_run: int) -> Dict[str, float]:
    """Float features (proportions) derived alongside the integer features."""
    return {
        f"p_{suffix}": (h / n) if n > 0 else 0.0,
        f"p_alts_{suffix}": (alts / (n - 1)) if n > 1 else 0.0,
        f"max_run_norm_{suffix}": ((max_run - 1) / (n - 1)) if n > 1 else 0.0,
        f"imbalance_{suffix}": 2.0 * abs((h / n) - 0.5) if n > 0 else 0.0,
        f"periodicity_{suffix}": periodicity_score(seq),
    }


def periodicity_score(seq: str) -> float:
    """Degree to which a sequence matches a short repeating template."""
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


def featurize_stimulus(sequence_a: str, sequence_b: str) -> Dict[str, float]:
    """Return the full feature-column dict for a single candidate stimulus pair.

    Keys match the ``pm.Data`` container names theorist PyMC models use:
    n_a, h_a, p_a, alts_a, p_alts_a, max_run_a, max_run_norm_a,
    imbalance_a, periodicity_a and the _b counterparts.
    """
    feats_a = sequence_features(sequence_a, "a")
    feats_b = sequence_features(sequence_b, "b")
    floats_a = sequence_features_float(
        sequence_a, "a", feats_a["n_a"], feats_a["alts_a"], feats_a["h_a"], feats_a["max_run_a"]
    )
    floats_b = sequence_features_float(
        sequence_b, "b", feats_b["n_b"], feats_b["alts_b"], feats_b["h_b"], feats_b["max_run_b"]
    )
    return {**feats_a, **feats_b, **floats_a, **floats_b}


def featurize_responses_csv(input_path: Path, output_path: Path) -> int:
    """Read a raw responses.csv, add numeric feature columns, and write it out.

    Returns the number of rows written. Fails loudly if the input is missing,
    empty, or lacks the required sequence/response columns.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    with input_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"input CSV is empty: {input_path}")

    missing = REQUIRED_INPUT_COLS - set(rows[0].keys())
    if missing:
        raise ValueError(f"input CSV missing columns {sorted(missing)}: {input_path}")

    out_cols = PASSTHROUGH_COLS + INT_FEATURE_COLS + FLOAT_FEATURE_COLS
    out_rows: List[Dict[str, object]] = []
    for r in rows:
        out_rows.append({
            **{k: r.get(k, "") for k in PASSTHROUGH_COLS},
            **featurize_stimulus(r["sequence_a"], r["sequence_b"]),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)

    return len(out_rows)
