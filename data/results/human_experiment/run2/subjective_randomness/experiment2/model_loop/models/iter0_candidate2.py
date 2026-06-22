"""
People evaluate randomness by assessing the structural diversity of a sequence's run-lengths. They compute a positive cognitive score based on the combinatorial diversity of the sequence (the log number of ways to arrange the observed counts of short, medium, and long runs) combined with a weighted preference for the sheer number of those runs. Because this mechanism accumulates positive structural evidence, longer balanced sequences naturally achieve higher scores than shorter ones, while highly regular sequences (zero diversity) and extremely imbalanced sequences (very few runs) are inherently penalized.
"""

import math
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Return run-length structural diversity features for one stimulus pair."""

    def extract(s):
        if not s:
            return 0.0, 0.0, 0.0, 0.0
        runs = []
        current_run = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)

        c1, c2, c3 = 0, 0, 0
        for r in runs:
            if r == 1:
                c1 += 1
            elif r == 2:
                c2 += 1
            else:
                c3 += 1

        R = c1 + c2 + c3
        div = (
            math.lgamma(R + 1)
            - math.lgamma(c1 + 1)
            - math.lgamma(c2 + 1)
            - math.lgamma(c3 + 1)
        )
        return float(c1), float(c2), float(c3), float(div)

    c1_a, c2_a, c3_a, div_a = extract(sequence_a)
    c1_b, c2_b, c3_b, div_b = extract(sequence_b)

    return {
        "c1_a": c1_a,
        "c2_a": c2_a,
        "c3_plus_a": c3_a,
        "div_a": div_a,
        "c1_b": c1_b,
        "c2_b": c2_b,
        "c3_plus_b": c3_b,
        "div_b": div_b,
    }


with pm.Model() as model:
    # Stimulus inputs
    c1_a = pm.Data("c1_a", np.zeros(1, dtype="float64"))
    c2_a = pm.Data("c2_a", np.zeros(1, dtype="float64"))
    c3_plus_a = pm.Data("c3_plus_a", np.zeros(1, dtype="float64"))
    div_a = pm.Data("div_a", np.zeros(1, dtype="float64"))

    c1_b = pm.Data("c1_b", np.zeros(1, dtype="float64"))
    c2_b = pm.Data("c2_b", np.zeros(1, dtype="float64"))
    c3_plus_b = pm.Data("c3_plus_b", np.zeros(1, dtype="float64"))
    div_b = pm.Data("div_b", np.zeros(1, dtype="float64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free parameters for the weights of the structural features
    # (Using Normal priors as weights can be positive or negative)
    w_div = pm.Normal("w_div", mu=1.0, sigma=5.0)
    w_c1 = pm.Normal("w_c1", mu=0.0, sigma=5.0)
    w_c2 = pm.Normal("w_c2", mu=0.0, sigma=5.0)
    w_c3 = pm.Normal("w_c3", mu=0.0, sigma=5.0)

    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)

    # Compute subjective randomness score for each sequence
    score_a = w_div * div_a + w_c1 * c1_a + w_c2 * c2_a + w_c3 * c3_plus_a
    score_b = w_div * div_b + w_c1 * c1_b + w_c2 * c2_b + w_c3 * c3_plus_b

    # Sigmoid link to probability with numerical safety clipping
    p_left_raw = pm.math.sigmoid(score_a - score_b + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Likelihood
    pm.Bernoulli("response", p=p_left, observed=chose_left)
