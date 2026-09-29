"""Impossible ground-truth theory: longer runs => more random-looking.

The opposite of the gambler's-fallacy intuition (and of the compressibility
seed model, which *penalizes* long runs): this generator judges a sequence more
random the longer its longest streak of identical outcomes. NOT a plausible
model of human subjective-randomness judgment; a deliberately weird ground-truth
generator for the impossible-theory holdout analysis.

Same PyMC structure as the seed models; the score is the normalized maximum run
length alone (no free shape parameters), so the generating params are exactly
{beta, side_bias}.
"""

import numpy as np
import pymc as pm

with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    score_a = max_run_norm_a
    score_b = max_run_norm_b

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)


# ─────────────────────────────────────────────────────────────────────────────
# Self-contained feature computation
# ─────────────────────────────────────────────────────────────────────────────
# Data rows carry only the raw H/T sequences, so the model derives the columns
# its `pm.Data` containers read through the `compute_features(sequence_a,
# sequence_b)` hook, as the literature models in `pymc_model_families/` do.
# The values are the old featurizer's (`src/subjective_randomness/features.py`,
# `sequence_features` / `sequence_features_float`), which fed these containers
# before the raw-only pipeline; `clean_sequence` is copied from it verbatim.
# `tests/test_impossible_models_raw.py` pins both.


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


def max_run_norm(seq: str) -> float:
    """The longest run of identical outcomes, scaled to [0, 1]:
    ``(max_run - 1) / (n - 1)``, and 0 for a single flip."""
    s = clean_sequence(seq)
    n = len(s)
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
    return ((max_run - 1) / (n - 1)) if n > 1 else 0.0


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """`max_run_norm_{a,b}`: each sequence's normalised longest run."""
    return {
        "max_run_norm_a": max_run_norm(sequence_a),
        "max_run_norm_b": max_run_norm(sequence_b),
    }
