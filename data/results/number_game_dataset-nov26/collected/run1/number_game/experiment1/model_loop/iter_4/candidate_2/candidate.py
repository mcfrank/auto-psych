"""Capacity-limited attention exemplar model.

People judge a target by its similarity to the remembered examples on
magnitude, final digit, parity and shared factors, but attention is a fixed,
limited budget split across these dimensions (weights sum to one): whatever the
examples are consistent on pulls attention onto that dimension and necessarily
away from the others. After a set sharing a final digit, wrong-digit targets
are rejected and right-digit targets are judged more tolerantly in magnitude,
because attention spent on the digit is taken from magnitude.
"""
from collections import Counter
from math import gcd

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
            out[f"par_diff_{k}"] = 0.0 if t % 2 == ex[k] % 2 else 1.0
            out[f"coprime_{k}"] = 1.0 if gcd(t, ex[k]) == 1 else 0.0
            out[f"present_{k}"] = 1.0
        else:
            out[f"dist_{k}"] = 0.0
            out[f"unit_diff_{k}"] = 0.0
            out[f"par_diff_{k}"] = 0.0
            out[f"coprime_{k}"] = 0.0
            out[f"present_{k}"] = 0.0
    if n > 1:
        out["mag_spread"] = (max(ex) - min(ex)) / 100.0
        out["digit_homog"] = (max(Counter(e % 10 for e in ex).values()) - 1) / (n - 1)
        out["parity_homog"] = (max(Counter(e % 2 for e in ex).values()) - 1) / (n - 1)
    else:
        out["mag_spread"] = 0.5
        out["digit_homog"] = 0.5
        out["parity_homog"] = 0.5
    return out


def _stack(name):
    return pt.stack([pm.Data(f"{name}_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)


with pm.Model() as model:
    dist = _stack("dist")
    udiff = _stack("unit_diff")
    pdiff = _stack("par_diff")
    cop = _stack("coprime")
    present = _stack("present")
    spread = pm.Data("mag_spread", np.zeros(1, dtype="float64"))
    dhomog = pm.Data("digit_homog", np.zeros(1, dtype="float64"))
    phomog = pm.Data("parity_homog", np.zeros(1, dtype="float64"))

    # Baseline attention logits (shared-factor dimension is the reference, 0).
    a_mag = pm.Normal("a_mag", mu=0.0, sigma=2.0)
    a_unit = pm.Normal("a_unit", mu=0.0, sigma=2.0)
    a_par = pm.Normal("a_par", mu=0.0, sigma=2.0)
    # How the examples' consistency pulls attention toward each dimension.
    spread_pull = pm.Normal("spread_pull", mu=0.0, sigma=3.0)
    digit_pull = pm.Normal("digit_pull", mu=0.0, sigma=2.0)
    parity_pull = pm.Normal("parity_pull", mu=0.0, sigma=2.0)
    # Overall sensitivity (fixed attentional capacity) and response mapping.
    log_sens = pm.Normal("log_sens", mu=3.0, sigma=1.0)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    logits = pt.stack(
        [a_mag - spread_pull * spread,
         a_unit + digit_pull * dhomog,
         a_par + parity_pull * phomog,
         pt.zeros_like(spread)],
        axis=1,
    )
    w = pt.special.softmax(logits, axis=1)  # attention budget sums to one
    wdist = (w[:, 0:1] * dist + w[:, 1:2] * udiff + w[:, 2:3] * pdiff + w[:, 3:4] * cop)
    sens = pt.exp(log_sens)
    log_sim = pt.switch(present > 0.5, -sens * wdist, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten()

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(alpha + gain * log_summed / sens), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
