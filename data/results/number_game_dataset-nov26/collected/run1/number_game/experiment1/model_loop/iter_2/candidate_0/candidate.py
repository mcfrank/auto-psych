"""Serial-position-weighted exemplar similarity.

People judge a target by its summed similarity to the remembered examples (on
magnitude and final digit, with attention tuned to the examples' variability),
but each example's contribution is weighted by its memory strength, which
changes geometrically with serial position: recency (later examples count more)
or primacy (earlier examples count more), with direction and strength inferred.
"""
from collections import Counter

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")][:MAXK]
    t = int(target)
    n = len(ex)
    out = {}
    for k in range(MAXK):
        if k < n:
            out[f"dist_{k}"] = abs(t - ex[k]) / 10.0
            out[f"unit_diff_{k}"] = 0.0 if t % 10 == ex[k] % 10 else 1.0
            out[f"present_{k}"] = 1.0
            # lag from the last example (0 = most recent, -1 = one before, ...)
            out[f"lag_{k}"] = float(k - (n - 1))
        else:
            out[f"dist_{k}"] = 0.0
            out[f"unit_diff_{k}"] = 0.0
            out[f"present_{k}"] = 0.0
            out[f"lag_{k}"] = 0.0
    if n > 1:
        out["mag_spread"] = (max(ex) - min(ex)) / 100.0
        out["digit_homog"] = (max(Counter(e % 10 for e in ex).values()) - 1) / (n - 1)
    else:
        out["mag_spread"] = 0.5
        out["digit_homog"] = 0.5
    out["log_n_ex"] = float(np.log(n))
    return out


with pm.Model() as model:
    dist = pt.stack([pm.Data(f"dist_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    udiff = pt.stack([pm.Data(f"unit_diff_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    present = pt.stack([pm.Data(f"present_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    lag = pt.stack([pm.Data(f"lag_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    spread = pm.Data("mag_spread", np.zeros(1, dtype="float64"))
    homog = pm.Data("digit_homog", np.zeros(1, dtype="float64"))
    log_n = pm.Data("log_n_ex", np.zeros(1, dtype="float64"))

    log_c_mag = pm.Normal("log_c_mag", mu=0.0, sigma=1.5)
    log_c_unit = pm.Normal("log_c_unit", mu=1.0, sigma=1.5)
    spread_tuning = pm.Normal("spread_tuning", mu=0.0, sigma=4.0)
    homog_tuning = pm.Normal("homog_tuning", mu=0.0, sigma=2.0)
    # Serial-position gradient: >0 recency, <0 primacy.
    recency = pm.Normal("recency", mu=0.0, sigma=1.0)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)

    c_mag = pt.exp(log_c_mag - spread_tuning * spread)[:, None]
    c_unit = pt.exp(log_c_unit + homog_tuning * homog)[:, None]

    # Memory weights, normalized so they sum to the number of examples.
    raw_w = pt.switch(present > 0.5, recency * lag, -1e3)
    log_w = raw_w - pt.logsumexp(raw_w, axis=1, keepdims=True) + log_n[:, None]

    log_sim = -c_mag * dist - c_unit * udiff + log_w
    log_sim = pt.switch(present > 0.5, log_sim, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten()

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(alpha + beta * log_summed), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
