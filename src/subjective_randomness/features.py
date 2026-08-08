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
    rep_motifs_<x>    repetition motifs in the minimal-DP motif parse (n1)
    alt_motifs_<x>    alternation motifs in the minimal-DP motif parse (n2)
    imbalance_<x>     distance from 50/50 heads/tails
    periodicity_<x>   simple repeating-template score
    sym1_<x>..sym8_<x>  raw symbols as 0/1 (H=1), zero-padded past n
    occ_n10_<x>, occ_n20_<x>, occ_n50_<x>
                      probability the sequence occurs at least once within a
                      global window of 10/20/50 fair flips (Hahn & Warren 2009)
    local_imbalance_<x>
                      worst H/T imbalance over sliding windows of length
                      min(n, 4) (Kahneman & Tversky 1972 local representativeness)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

PASSTHROUGH_COLS = [
    "participant_id",
    "trial_index",
    "sequence_a",
    "sequence_b",
    "chose_left",
]
MAX_SEQ_LEN = 8

INT_FEATURE_COLS = [
    "n_a",
    "h_a",
    "alts_a",
    "max_run_a",
    "rep_motifs_a",
    "alt_motifs_a",
    *[f"sym{i}_a" for i in range(1, MAX_SEQ_LEN + 1)],
    "n_b",
    "h_b",
    "alts_b",
    "max_run_b",
    "rep_motifs_b",
    "alt_motifs_b",
    *[f"sym{i}_b" for i in range(1, MAX_SEQ_LEN + 1)],
]
# Global experienced-sequence lengths for the Hahn & Warren (2009) occurrence
# probabilities. 20 is their focal illustration; 10 and 50 bracket it.
EXPERIENCE_LENGTHS = (10, 20, 50)

FLOAT_FEATURE_COLS = [
    "p_a",
    "p_alts_a",
    "max_run_norm_a",
    "imbalance_a",
    "periodicity_a",
    *[f"occ_n{w}_a" for w in EXPERIENCE_LENGTHS],
    "p_b",
    "p_alts_b",
    "max_run_norm_b",
    "imbalance_b",
    "periodicity_b",
    *[f"occ_n{w}_b" for w in EXPERIENCE_LENGTHS],
    "local_imbalance_a",
    "local_imbalance_b",
]

# Window length over which local representativeness is judged (Kahneman &
# Tversky 1972 give no numeric value; 4 matches the short-term-memory span
# motivation used across this literature).
LOCAL_WINDOW = 4
REQUIRED_INPUT_COLS = {"sequence_a", "sequence_b", "chose_left"}


def clean_sequence(seq: str) -> str:
    """Uppercase an H/T sequence and reject empty input.

    An empty sequence is never a legitimate trial — it means upstream breakage
    (a stimulus without ``sequence_a``, a truncated responses.csv) — so every
    helper below raises rather than emitting a zero-filled feature row that
    reads like a real observation. The model families' ``clean_sequence``
    (``model_families/common.py``) makes the same call and additionally rejects
    non-H/T symbols; this module keeps its own copy so it stays importable
    without the model-family package.
    """
    s = seq.strip().upper()
    if not s:
        raise ValueError("Sequence must not be empty")
    return s


def parse_motifs(seq: str) -> tuple[int, int]:
    """Parse an H/T sequence into Falk & Konold (1997) motifs.

    Returns ``(rep_motifs, alt_motifs)`` — n1 (repetition motifs: constant-run
    chunks) and n2 (alternation motifs: strictly alternating chunks of length
    >= 2) of the Difficulty Predictor parse, for which DP = n1 + 2*n2. Falk &
    Konold (1997, p. 308) define the parse as the partition of the sequence
    into such chunks that "achieve[s] the lowest possible number" — chunk
    boundaries need not respect run boundaries (their example: XXXOXO ->
    XX|XOXO, DP 3). DP ties are broken toward the fewest chunks (the most
    compressed description), which makes (n1, n2) unique. For example
    HHTTHTHT -> {HH, TT} repetition + {HTHT} alternation -> (2, 1), DP = 4;
    HTHHTH -> {HTH, HTH} -> (0, 2), DP = 4. The model-family helper of the same
    name in ``model_families/common.py`` wraps this one.
    """
    s = clean_sequence(seq)
    n = len(s)

    # best[i] = lexicographically minimal (DP cost, chunk count) over all
    # partitions of s[:i] into constant-run chunks (cost 1) and strictly
    # alternating chunks of length >= 2 (cost 2).
    unreachable = (n * 2 + 1, n + 1)
    best = [(0, 0)] + [unreachable] * n
    for i in range(1, n + 1):
        for j in range(i - 1, -1, -1):
            chunk = s[j:i]
            if all(c == chunk[0] for c in chunk):
                cost = 1
            elif all(a != b for a, b in zip(chunk, chunk[1:])):
                cost = 2
            else:
                continue
            candidate = (best[j][0] + cost, best[j][1] + 1)
            if candidate < best[i]:
                best[i] = candidate
    dp, chunks = best[n]
    rep_motifs = 2 * chunks - dp
    alt_motifs = dp - chunks
    return rep_motifs, alt_motifs


def sequence_features(seq: str, suffix: str) -> Dict[str, int]:
    """Integer features derived from a single H/T sequence string."""
    s = clean_sequence(seq)
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
    rep_motifs, alt_motifs = parse_motifs(s)
    if n > MAX_SEQ_LEN:
        raise ValueError(
            f"sequence {s!r} is longer than the supported maximum of "
            f"{MAX_SEQ_LEN} symbols; the per-symbol sym1..sym{MAX_SEQ_LEN} "
            f"columns cannot represent it"
        )
    symbols = {
        f"sym{i}_{suffix}": (1 if s[i - 1] == "H" else 0) if i <= n else 0
        for i in range(1, MAX_SEQ_LEN + 1)
    }
    return {
        f"n_{suffix}": n,
        f"h_{suffix}": h,
        f"alts_{suffix}": alts,
        f"max_run_{suffix}": max_run,
        f"rep_motifs_{suffix}": rep_motifs,
        f"alt_motifs_{suffix}": alt_motifs,
        **symbols,
    }


def sequence_features_float(
    seq: str, suffix: str, n: int, alts: int, h: int, max_run: int
) -> Dict[str, float]:
    """Float features (proportions) derived alongside the integer features.

    ``n >= 1`` here: ``sequence_features`` has already rejected empty input, so
    only the length-1 case (no transitions to average over) needs a guard.
    """
    return {
        f"p_{suffix}": h / n,
        f"p_alts_{suffix}": (alts / (n - 1)) if n > 1 else 0.0,
        f"max_run_norm_{suffix}": ((max_run - 1) / (n - 1)) if n > 1 else 0.0,
        f"imbalance_{suffix}": 2.0 * abs((h / n) - 0.5),
        f"periodicity_{suffix}": periodicity_score(seq),
        **{
            f"occ_n{w}_{suffix}": occurrence_probability(seq, w)
            for w in EXPERIENCE_LENGTHS
        },
        f"local_imbalance_{suffix}": local_imbalance(seq),
    }


def local_imbalance(seq: str) -> float:
    """Worst H/T imbalance over sliding windows of length min(n, LOCAL_WINDOW).

    2*|prop_heads - 0.5| of the most imbalanced window — 0 when every window
    is balanced, 1 when some window is a single symbol repeated. The
    model-family helper of the same name in ``model_families/common.py`` wraps
    this one.
    """
    s = clean_sequence(seq)
    n = len(s)
    window = min(n, LOCAL_WINDOW)
    worst = 0.0
    for start in range(n - window + 1):
        chunk = s[start : start + window]
        heads = sum(1 for c in chunk if c == "H")
        worst = max(worst, 2.0 * abs(heads / window - 0.5))
    return worst


def occurrence_probability(pattern: str, n_global: int) -> float:
    """P(``pattern`` occurs as a contiguous substring of ``n_global`` fair flips).

    The quantity of Hahn & Warren (2009), computed exactly by evolving the
    distribution over KMP prefix-automaton states (state = length of the
    longest pattern prefix matching the current suffix; reaching the full
    pattern absorbs). The model-family helper of the same name in
    ``model_families/common.py`` wraps this one.
    """
    p = clean_sequence(pattern)
    k = len(p)
    if n_global < 0:
        raise ValueError(f"n_global must be >= 0, got {n_global}")
    if n_global < k:
        return 0.0

    failure = [0] * k
    for i in range(1, k):
        j = failure[i - 1]
        while j > 0 and p[i] != p[j]:
            j = failure[j - 1]
        failure[i] = j + 1 if p[i] == p[j] else 0

    def next_state(state: int, symbol: str) -> int:
        while True:
            if symbol == p[state]:
                return state + 1
            if state == 0:
                return 0
            state = failure[state - 1]

    transitions = [
        {symbol: next_state(state, symbol) for symbol in "HT"} for state in range(k)
    ]

    dist = [0.0] * k
    dist[0] = 1.0
    absorbed = 0.0
    for _ in range(n_global):
        new_dist = [0.0] * k
        for state, mass in enumerate(dist):
            if mass == 0.0:
                continue
            for symbol in "HT":
                target = transitions[state][symbol]
                if target == k:
                    absorbed += 0.5 * mass
                else:
                    new_dist[target] += 0.5 * mass
        dist = new_dist
    return absorbed


def periodicity_score(seq: str) -> float:
    """Degree to which a sequence matches a short repeating template.

    The model-family helper of the same name in ``model_families/common.py``
    wraps this one.
    """
    s = clean_sequence(seq)
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
        sequence_a,
        "a",
        feats_a["n_a"],
        feats_a["alts_a"],
        feats_a["h_a"],
        feats_a["max_run_a"],
    )
    floats_b = sequence_features_float(
        sequence_b,
        "b",
        feats_b["n_b"],
        feats_b["alts_b"],
        feats_b["h_b"],
        feats_b["max_run_b"],
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
        out_rows.append(
            {
                **{k: r.get(k, "") for k in PASSTHROUGH_COLS},
                **featurize_stimulus(r["sequence_a"], r["sequence_b"]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)

    return len(out_rows)
