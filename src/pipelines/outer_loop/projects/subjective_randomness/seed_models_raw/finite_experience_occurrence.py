"""PyMC adapter for the Hahn & Warren (2009) finite-experience family.

Randomness is the log probability that the sequence occurs at least once in
the paper's focal finite stream of 20 fair flips. The occurrence probability
is precomputed exactly by the featurizer. The source analysis compares only
equal-length strings, so an explicit graph assertion rejects cross-length data
rather than silently introducing a new theory of length comparison.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.raise_op import Assert

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    occ_n20_a = pm.Data("occ_n20_a", np.zeros(1, dtype="float64"))
    occ_n20_b = pm.Data("occ_n20_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    checked_n_a = Assert(
        "finite_experience_occurrence requires same-length alternatives"
    )(n_a, pt.all(pt.eq(n_a, n_b)))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    length_check = pt.sum(pt.cast(checked_n_a - n_a, "float64"))
    score_a = pm.math.log(occ_n20_a) + length_check
    score_b = pm.math.log(occ_n20_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)


# ─────────────────────────────────────────────────────────────────────────────
# Raw-features (arm C) support: this model computes the columns it binds
# ─────────────────────────────────────────────────────────────────────────────
# In a `raw_features` run the agents' responses CSV carries only the raw H/T
# sequences, so no model may depend on harness-provided feature columns (see
# docs/raw_features_arm.md). This model therefore derives the columns its
# `pm.Data` containers read from the sequences themselves, through the
# `compute_features(sequence_a, sequence_b)` hook that
# `src/models/pymc_inference.py` applies wherever rows become model data —
# fitting, the EIG design and the held-out prediction path alike.
#
# The helpers below are copied VERBATIM from
# `src/subjective_randomness/features.py` (extracted programmatically, not
# retyped) so this file is self-contained: it imports no project featurizer,
# and it leaves nothing for a candidate agent to import either.
# `tests/test_raw_features_seeds.py` pins each copy to its original and pins
# `compute_features` to the featurizer's values for these columns, so the
# copies cannot silently drift.
#
# Column collision is deliberate and fatal: the hook raises if a name it
# returns is already present, so these models must NOT be fitted on a
# featurized CSV. That is what makes a raw-features run all-or-nothing.


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


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """`n_{a,b}` (length) and `occ_n20_{a,b}` (Hahn & Warren occurrence probability
    over a 20-flip experience window)."""
    features: dict = {}
    for seq, suffix in ((sequence_a, "a"), (sequence_b, "b")):
        features[f"n_{suffix}"] = len(clean_sequence(seq))
        features[f"occ_n20_{suffix}"] = occurrence_probability(seq, 20)
    return features
