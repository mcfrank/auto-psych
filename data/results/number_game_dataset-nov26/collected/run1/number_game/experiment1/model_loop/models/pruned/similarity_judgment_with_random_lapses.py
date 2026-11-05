"""Exemplar similarity judgment with random guessing lapses.

People judge a target by its summed similarity to the remembered examples on
magnitude and final digit, but on a fraction of trials they lapse - they do not
consult the examples and guess yes or no at random (a coin flip). The single
decision-rule mechanism is this lapse rate, which sets a floor and a ceiling on
the probability of answering yes regardless of similarity.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")][:MAXK]
    t = int(target)
    out = {}
    for k in range(MAXK):
        if k < len(ex):
            out[f"dist_{k}"] = abs(t - ex[k]) / 10.0
            out[f"unit_diff_{k}"] = 0.0 if t % 10 == ex[k] % 10 else 1.0
            out[f"present_{k}"] = 1.0
        else:
            out[f"dist_{k}"] = 0.0
            out[f"unit_diff_{k}"] = 0.0
            out[f"present_{k}"] = 0.0
    return out


with pm.Model() as model:
    dist = pt.stack([pm.Data(f"dist_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    udiff = pt.stack([pm.Data(f"unit_diff_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    present = pt.stack([pm.Data(f"present_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)

    # Similarity: magnitude gradient (per 10 units) and final-digit mismatch penalty.
    c_mag = pm.HalfNormal("c_mag", sigma=2.0)
    c_unit = pm.HalfNormal("c_unit", sigma=3.0)
    # Response mapping from log summed similarity.
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)
    # Probability of a lapse trial (random 50/50 guess).
    lapse = pm.Beta("lapse", alpha=1.5, beta=6.0)

    log_sim = -c_mag * dist - c_unit * udiff
    log_sim = pt.switch(present > 0.5, log_sim, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten()

    p_engaged = pm.math.sigmoid(alpha + beta * log_summed)
    p = (1.0 - lapse) * p_engaged + lapse * 0.5
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
