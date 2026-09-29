"""Impossible ground-truth theory: more heads => more random-looking.

NOT a plausible model of how humans judge subjective randomness — a deliberately
weird ground-truth generator for the impossible-theory holdout analysis. The
loop's models (alternation / balance / compressibility / diagnosticity) should
never recover it, so the held-out correlation should stay low.

Same PyMC structure as the seed models (feature pm.Data containers, free
beta/side_bias, deterministic score, sigmoid p_left, Bernoulli response). The
score is the head count alone — no free shape parameters — so the generating
params are exactly {beta, side_bias}. Mirrors the head-count process in
projects/subjective_randomness/ground_truth_models.py::prefer_more_heads, ported
to the PyMC fixed-param generator interface.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    score_a = pt.cast(h_a, "float64")
    score_b = pt.cast(h_b, "float64")

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


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """`h_{a,b}`: the head count of each sequence."""
    return {
        f"h_{suffix}": sum(1 for c in clean_sequence(seq) if c == "H")
        for seq, suffix in ((sequence_a, "a"), (sequence_b, "b"))
    }
