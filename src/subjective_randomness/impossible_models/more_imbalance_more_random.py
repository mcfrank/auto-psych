"""Impossible ground-truth theory: more imbalance => more random-looking.

The opposite of the representativeness heuristic: this generator judges a
sequence more random the more lopsided its heads/tails split (further from
50/50). NOT a plausible model of human subjective-randomness judgment; a
deliberately weird ground-truth generator for the impossible-theory holdout
analysis.

Same PyMC structure as the seed models; the score is the head/tail imbalance
alone (no free shape parameters), so the generating params are exactly
{beta, side_bias}.
"""

import numpy as np
import pymc as pm

with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    score_a = imbalance_a
    score_b = imbalance_b

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


def imbalance(seq: str) -> float:
    """Distance from a 50/50 heads/tails split, ``2 * |heads / n - 0.5|``:
    0 when balanced, 1 for a single symbol repeated."""
    s = clean_sequence(seq)
    return 2.0 * abs((sum(1 for c in s if c == "H") / len(s)) - 0.5)


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """`imbalance_{a,b}`: each sequence's heads/tails imbalance."""
    return {"imbalance_a": imbalance(sequence_a), "imbalance_b": imbalance(sequence_b)}
