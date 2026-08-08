"""Compression cost: people judge a sequence as more random when it is harder
to compress.

The index of randomness is the description cost of the sequence's finest
run-length encoding. A string with very few runs (long streaks, periodic
structure) is highly compressible and judged non-random; so too is a string
with the maximum possible number of runs (perfectly alternating HT), because
it admits a one-line description. Genuinely random strings fall in between,
resisting compression. On a forced choice, the person picks as more random the
sequence whose run-length description cost is larger.
"""

import math

import numpy as np
import pymc as pm


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Return the run-length-encoding description cost of each sequence.

    For a binary string of length n with r runs (each run = maximal streak of
    identical tosses), describing the string requires specifying r run-lengths
    that sum to n, costing about r * log2(n/r) bits. This cost is near zero
    both when r is small (all same / long streaks) and when r=n (perfectly
    alternating), and largest for strings with an intermediate, irregular run
    structure -- exactly the compression-resisting middle.
    """
    def cost(seq: str) -> float:
        s = seq.strip().upper()
        n = len(s)
        if n == 0:
            return 0.0
        runs = 1
        for i in range(1, n):
            if s[i] != s[i - 1]:
                runs += 1
        if runs <= 1 or runs >= n:
            return 0.0
        mean_run = n / runs
        return runs * math.log2(mean_run)

    return {
        "rle_cost_a": cost(sequence_a),
        "rle_cost_b": cost(sequence_b),
    }


with pm.Model() as model:
    # Run-length description costs of the two sequences (derived feature).
    rle_a = pm.Data("rle_cost_a", np.zeros(1, dtype="float64"))
    rle_b = pm.Data("rle_cost_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter: sensitivity to compressibility (non-negative,
    # so more incompressibility can only increase judged randomness).
    kappa = pm.HalfNormal("kappa", sigma=2.0)

    # A person prefers the sequence with the larger run-length description cost.
    score_a = kappa * rle_a
    score_b = kappa * rle_b
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))

    # Observed forced choice: 1 = chose sequence A (left).
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
