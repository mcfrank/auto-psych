"""Digit-run sharpened exemplar similarity (refines shared_factor_parity_exemplar).

People judge a target by its summed similarity to the remembered examples in a
space of magnitude, final digit, parity and shared factors, with attention
tuned to how variable the examples are. The one change: when three or more
examples all end in the same digit, people take the final digit as the rule
and attend to it far more sharply, so a target with a different final digit
loses much more similarity to every example; matching targets are unaffected.
"""
from collections import Counter
from math import gcd

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")][:MAXK]
    t = int(float(target))
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
    # A run of >= 3 examples all ending in one digit reads as a digit rule.
    out["digit_run"] = 1.0 if (n >= 3 and len({e % 10 for e in ex}) == 1) else 0.0
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
    digit_run = pm.Data("digit_run", np.zeros(1, dtype="float64"))

    log_c_mag = pm.Normal("log_c_mag", mu=0.0, sigma=1.5)
    log_c_unit = pm.Normal("log_c_unit", mu=1.0, sigma=1.5)
    spread_tuning = pm.Normal("spread_tuning", mu=0.0, sigma=4.0)
    homog_tuning = pm.Normal("homog_tuning", mu=0.0, sigma=2.0)
    c_par_max = pm.HalfNormal("c_par_max", sigma=2.0)
    c_factor = pm.HalfNormal("c_factor", sigma=2.0)
    # The new component: extra (log-scale) final-digit attention after a digit run.
    run_sharpening = pm.Normal("run_sharpening", mu=0.0, sigma=1.5)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)

    c_mag = pt.exp(log_c_mag - spread_tuning * spread)[:, None]
    c_unit = pt.exp(log_c_unit + homog_tuning * dhomog + run_sharpening * digit_run)[:, None]
    c_par = (c_par_max * phomog)[:, None]

    log_sim = -c_mag * dist - c_unit * udiff - c_par * pdiff - c_factor * cop
    log_sim = pt.switch(present > 0.5, log_sim, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten()

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(alpha + beta * log_summed), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
