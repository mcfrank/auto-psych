"""Running-tally excursion (goldilocks) hypothesis.

People keep a running tally of heads minus tails while reading a sequence and
expect a fair coin's tally to wander from balance by a moderate, characteristic
amount. A sequence looks random to the extent its tally's typical excursion
(squared lead at each point, relative to the fair-coin expectation at that
point) matches that expected wandering: a tally pinned at balance (strict
alternation) and one that runs far to one side (streaks) both look non-random.
"""
import math

import numpy as np
import pymc as pm


def _log_excursion(seq):
    seq = seq.strip().upper()
    lead = 0
    total = 0.0
    for t, c in enumerate(seq, start=1):
        lead += 1 if c == "H" else -1
        # a fair coin's expected squared lead after t flips is t
        total += lead * lead / t
    return math.log(total / len(seq) + 0.05)


def compute_features(sequence_a, sequence_b):
    return {"exc_a": _log_excursion(sequence_a), "exc_b": _log_excursion(sequence_b)}


with pm.Model() as model:
    exc_a = pm.Data("exc_a", np.zeros(1, dtype="float64"))
    exc_b = pm.Data("exc_b", np.zeros(1, dtype="float64"))

    # Expected (ideal) log excursion level of a fair coin's tally.
    mu = pm.Normal("mu", mu=-0.5, sigma=1.0)
    # Sensitivity of choice to mismatch from the expected wandering.
    beta = pm.HalfNormal("beta", sigma=2.0)

    mismatch_a = (exc_a - mu) ** 2
    mismatch_b = (exc_b - mu) ** 2
    p = pm.math.sigmoid(beta * (mismatch_b - mismatch_a))
    p_left = pm.Deterministic("p_left", pm.math.clip(p, 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
