"""Digit-feature exemplar similarity.

People judge a target by its featural resemblance to the examples in decimal
notation: each example's similarity to the target falls off exponentially with
whether the two differ in their ones digit and whether they differ in their
tens digit, and the probability of saying yes grows with the target's averaged
similarity to the shown examples (an exemplar model over digit features, not a
search over explicit rules).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4


def _digits(n):
    return n // 10, n % 10


def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")]
    t_tens, t_ones = _digits(int(target))
    out = {}
    for i in range(MAX_EX):
        if i < len(examples):
            e_tens, e_ones = _digits(examples[i])
            out[f"ones_mis_{i}"] = float(e_ones != t_ones)
            out[f"tens_mis_{i}"] = float(e_tens != t_tens)
            out[f"mask_{i}"] = 1.0
        else:
            out[f"ones_mis_{i}"] = 0.0
            out[f"tens_mis_{i}"] = 0.0
            out[f"mask_{i}"] = 0.0
    return out


with pm.Model() as model:
    ones_mis = pt.stack([pm.Data(f"ones_mis_{i}", np.zeros(1)) for i in range(MAX_EX)], axis=1)
    tens_mis = pt.stack([pm.Data(f"tens_mis_{i}", np.zeros(1)) for i in range(MAX_EX)], axis=1)
    mask = pt.stack([pm.Data(f"mask_{i}", np.zeros(1)) for i in range(MAX_EX)], axis=1)

    # Attention weights on the two digit features (dissimilarity per mismatch).
    w_ones = pm.HalfNormal("w_ones", sigma=3.0)
    w_tens = pm.HalfNormal("w_tens", sigma=3.0)
    # Mapping from mean similarity to yes-probability.
    alpha = pm.Normal("alpha", mu=-2.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=5.0)

    sim = pt.exp(-w_ones * ones_mis - w_tens * tens_mis) * mask
    mean_sim = pt.sum(sim, axis=1) / pt.maximum(pt.sum(mask, axis=1), 1.0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(alpha + beta * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
