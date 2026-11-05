"""Common-step lattice exemplar (refines shared_factor_parity_exemplar).

People judge a target by its summed similarity to the remembered examples in a
space of magnitude, final digit, parity and shared factors, with attention set
by how variable the examples are. The one change: people read the examples'
spacing. When >=3 examples have gaps that are all multiples of a common step
<= 20, a target not reachable from the examples in whole steps loses
similarity to every example by a fixed factor.
"""
from collections import Counter
from functools import reduce
from math import gcd

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
MAX_STEP = 20


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
    # Common step: gcd of all gaps between distinct examples (needs >=3 examples).
    step = reduce(gcd, [abs(a - b) for a in ex for b in ex if a != b], 0)
    off = 0.0
    if n >= 3 and 2 <= step <= MAX_STEP and (t - ex[0]) % step != 0:
        off = 1.0
    out["off_step"] = off
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
    off_step = pm.Data("off_step", np.zeros(1, dtype="float64"))

    log_c_mag = pm.Normal("log_c_mag", mu=0.0, sigma=1.5)
    log_c_unit = pm.Normal("log_c_unit", mu=1.0, sigma=1.5)
    spread_tuning = pm.Normal("spread_tuning", mu=0.0, sigma=4.0)
    homog_tuning = pm.Normal("homog_tuning", mu=0.0, sigma=2.0)
    c_par_max = pm.HalfNormal("c_par_max", sigma=2.0)
    c_factor = pm.HalfNormal("c_factor", sigma=2.0)
    # The new dimension: fixed penalty for being off the examples' common step.
    c_step = pm.HalfNormal("c_step", sigma=2.0)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)

    c_mag = pt.exp(log_c_mag - spread_tuning * spread)[:, None]
    c_unit = pt.exp(log_c_unit + homog_tuning * dhomog)[:, None]
    c_par = (c_par_max * phomog)[:, None]

    log_sim = -c_mag * dist - c_unit * udiff - c_par * pdiff - c_factor * cop
    log_sim = pt.switch(present > 0.5, log_sim, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten() - c_step * off_step

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(alpha + beta * log_summed), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
