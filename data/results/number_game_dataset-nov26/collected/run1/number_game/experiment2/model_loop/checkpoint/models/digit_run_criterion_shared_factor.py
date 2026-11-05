"""Digit-run criterion + shared-factor exemplar (refines digit_run_strict_criterion_exemplar).

People judge a target by its summed similarity to the remembered examples
(magnitude and final-digit attention tuned to example variability, a fixed
parity penalty) and raise their response criterion when >=3 examples all end
in the same digit. The one change, grafted from shared_factor_parity_exemplar:
a fixed shared-factor dimension -- an example contributes less similarity to a
target coprime with it (no common divisor > 1), so factor-sharing targets far
from the examples (36 after "3, 63") feel like they belong.
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
            out[f"parity_diff_{k}"] = 0.0 if t % 2 == ex[k] % 2 else 1.0
            out[f"coprime_{k}"] = 1.0 if gcd(t, ex[k]) == 1 else 0.0
            out[f"present_{k}"] = 1.0
        else:
            out[f"dist_{k}"] = 0.0
            out[f"unit_diff_{k}"] = 0.0
            out[f"parity_diff_{k}"] = 0.0
            out[f"coprime_{k}"] = 0.0
            out[f"present_{k}"] = 0.0
    if n > 1:
        out["mag_spread"] = (max(ex) - min(ex)) / 100.0
        out["digit_homog"] = (max(Counter(e % 10 for e in ex).values()) - 1) / (n - 1)
    else:
        out["mag_spread"] = 0.5
        out["digit_homog"] = 0.5
    # A run of >= 3 examples all sharing one final digit signals a narrow rule.
    out["digit_run"] = 1.0 if (n >= 3 and len({e % 10 for e in ex}) == 1) else 0.0
    return out


with pm.Model() as model:
    dist = pt.stack([pm.Data(f"dist_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    udiff = pt.stack([pm.Data(f"unit_diff_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    pdiff = pt.stack([pm.Data(f"parity_diff_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    cop = pt.stack([pm.Data(f"coprime_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    present = pt.stack([pm.Data(f"present_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    spread = pm.Data("mag_spread", np.zeros(1, dtype="float64"))
    homog = pm.Data("digit_homog", np.zeros(1, dtype="float64"))
    digit_run = pm.Data("digit_run", np.zeros(1, dtype="float64"))

    log_c_mag = pm.Normal("log_c_mag", mu=0.0, sigma=1.5)
    log_c_unit = pm.Normal("log_c_unit", mu=1.0, sigma=1.5)
    log_c_parity = pm.Normal("log_c_parity", mu=0.0, sigma=1.5)
    spread_tuning = pm.Normal("spread_tuning", mu=0.0, sigma=4.0)
    homog_tuning = pm.Normal("homog_tuning", mu=0.0, sigma=2.0)
    # Grafted dimension: fixed penalty for sharing no divisor with an example.
    c_factor = pm.HalfNormal("c_factor", sigma=2.0)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)
    # Criterion shift after a same-final-digit run (negative = stricter).
    run_shift = pm.Normal("run_shift", mu=0.0, sigma=2.0)

    c_mag = pt.exp(log_c_mag - spread_tuning * spread)[:, None]
    c_unit = pt.exp(log_c_unit + homog_tuning * homog)[:, None]
    c_parity = pt.exp(log_c_parity)

    log_sim = -c_mag * dist - c_unit * udiff - c_parity * pdiff - c_factor * cop
    log_sim = pt.switch(present > 0.5, log_sim, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten()

    p_yes = pm.Deterministic(
        "p_yes",
        pt.clip(pm.math.sigmoid(alpha + run_shift * digit_run + beta * log_summed), 1e-6, 1 - 1e-6),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
