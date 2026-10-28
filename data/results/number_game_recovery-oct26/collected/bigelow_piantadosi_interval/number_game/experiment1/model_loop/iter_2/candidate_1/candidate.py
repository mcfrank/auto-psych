"""Digit-grid exemplar similarity.

People judge a target by its similarity to the examples in the written-digit
form of the numbers: each number is a point on a (tens digit, ones digit) grid,
and similarity to an example falls off exponentially with the attention-weighted
city-block distance on that grid (inferred attention share on the ones vs the
tens digit). The higher the target's mean similarity to the examples, the more
likely a "yes" -- no rules are entertained.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4


def _digits(n):
    return n // 10, n % 10


def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")][:MAX_EX]
    tt, to = _digits(int(target))
    feats = {}
    for i in range(MAX_EX):
        if i < len(examples):
            et, eo = _digits(examples[i])
            feats[f"dtens_{i}"] = float(abs(tt - et))
            feats[f"dones_{i}"] = float(abs(to - eo))
            feats[f"present_{i}"] = 1.0
        else:
            feats[f"dtens_{i}"] = 0.0
            feats[f"dones_{i}"] = 0.0
            feats[f"present_{i}"] = 0.0
    return feats


with pm.Model() as model:
    dtens = [pm.Data(f"dtens_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)]
    dones = [pm.Data(f"dones_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)]
    present = [pm.Data(f"present_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)]

    # Attention share on the ones digit (vs the tens digit).
    w_ones = pm.Beta("w_ones", alpha=2.0, beta=2.0)
    # Specificity: how fast similarity decays with grid distance.
    c = pm.HalfNormal("c", sigma=1.0)
    # Mapping from mean similarity to a yes/no response.
    alpha = pm.Normal("alpha", mu=-2.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=5.0)

    D = pt.stack([w_ones * dones[i] + (1.0 - w_ones) * dtens[i] for i in range(MAX_EX)])
    M = pt.stack(present)
    sim = pt.exp(-c * D) * M
    mean_sim = pt.sum(sim, axis=0) / pt.sum(M, axis=0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(alpha + beta * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
