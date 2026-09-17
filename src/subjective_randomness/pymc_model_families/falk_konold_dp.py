"""PyMC adapter for the Falk & Konold (1997) Difficulty Predictor family.

Randomness = DP = rep_motifs + 2*alt_motifs (the minimal-DP parse computed by
the featurizer), unnormalised by length; harder-to-encode sequences seem more
random. The theory has no free cognitive parameters — only the choice rule's
``beta`` and ``side_bias`` are inferred. See the pure-Python twin in
``model_families/falk_konold_dp.py`` for the full rationale.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    rep_motifs_a = pm.Data("rep_motifs_a", np.zeros(1, dtype="int64"))
    alt_motifs_a = pm.Data("alt_motifs_a", np.zeros(1, dtype="int64"))
    rep_motifs_b = pm.Data("rep_motifs_b", np.zeros(1, dtype="int64"))
    alt_motifs_b = pm.Data("alt_motifs_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    score_a = pt.cast(rep_motifs_a + 2 * alt_motifs_a, "float64")
    score_b = pt.cast(rep_motifs_b + 2 * alt_motifs_b, "float64")

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)


# ─────────────────────────────────────────────────────────────────────────────
# Self-contained feature computation
# ─────────────────────────────────────────────────────────────────────────────
# The agents' responses CSV carries only raw H/T sequences, so every model
# derives the columns its `pm.Data` containers read from the sequences
# themselves, through the `compute_features(sequence_a, sequence_b)` hook
# that `src/models/pymc_inference.py` applies wherever rows become model
# data — fitting, the EIG design and the held-out prediction path alike.
#
# The helpers below are copied VERBATIM from
# `src/subjective_randomness/features.py` (extracted programmatically, not
# retyped) so this file is self-contained: it imports no project featurizer,
# and it leaves nothing for a candidate agent to import either.
# `tests/test_raw_features_seeds.py` pins each copy to its original and pins
# `compute_features` to the featurizer's values for these columns, so the
# copies cannot silently drift.


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


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """`rep_motifs_{a,b}` / `alt_motifs_{a,b}`: the minimal-DP motif parse counts."""
    features: dict = {}
    for seq, suffix in ((sequence_a, "a"), (sequence_b, "b")):
        rep_motifs, alt_motifs = parse_motifs(seq)
        features[f"rep_motifs_{suffix}"] = rep_motifs
        features[f"alt_motifs_{suffix}"] = alt_motifs
    return features
