"""Running-guess coincidence exemplar.

People read the examples in order and keep a running guess about the rule
(shared final digit, multiples of 5, shared parity). Each example that fits the
guess strengthens it; an example that breaks it resets the guess, so earlier
evidence is forgotten. Generalization is summed similarity (magnitude, final
digit, parity, shared factors) sharpened on every dimension by how suspicious a
coincidence the final running guess is -- i.e. the run of consecutive examples
ending with the last one that share the property. Order therefore matters:
"83, 38, 58, 68" ends on a run of three 8s and generalizes narrowly.
"""
from collections import Counter
from math import gcd, log

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def _coincidence(ex):
    """Evidence of the running guess at the end of the sequence: for each
    property, the number of consecutive examples ending with the last one that
    share it (a disconfirming example resets the run)."""
    last = ex[-1]
    tests = [
        (log(2), lambda e: e % 2 == last % 2, True),
        (log(5), lambda e: e % 5 == 0, last % 5 == 0),
        (log(10), lambda e: e % 10 == last % 10, True),
    ]
    best = 0.0
    for lr, fits, applies in tests:
        if not applies:
            continue
        run = 1
        for e in reversed(ex[:-1]):
            if fits(e):
                run += 1
            else:
                break
        best = max(best, (run - 1) * lr)
    return best / (3 * log(10))


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
            for nm in ("dist", "unit_diff", "par_diff", "coprime", "present"):
                out[f"{nm}_{k}"] = 0.0
    if n > 1:
        out["mag_spread"] = (max(ex) - min(ex)) / 100.0
        out["digit_homog"] = (max(Counter(e % 10 for e in ex).values()) - 1) / (n - 1)
        out["parity_homog"] = (max(Counter(e % 2 for e in ex).values()) - 1) / (n - 1)
    else:
        out["mag_spread"] = 0.5
        out["digit_homog"] = 0.5
        out["parity_homog"] = 0.5
    out["coincidence"] = _coincidence(ex)
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
    coinc = pm.Data("coincidence", np.zeros(1, dtype="float64"))

    log_c_mag = pm.Normal("log_c_mag", mu=0.0, sigma=1.5)
    log_c_unit = pm.Normal("log_c_unit", mu=1.0, sigma=1.5)
    spread_tuning = pm.Normal("spread_tuning", mu=0.0, sigma=4.0)
    homog_tuning = pm.Normal("homog_tuning", mu=0.0, sigma=2.0)
    c_par_max = pm.HalfNormal("c_par_max", sigma=2.0)
    c_factor = pm.HalfNormal("c_factor", sigma=2.0)
    # The mechanism: sharpening by the coincidence of the final running guess.
    sharpening = pm.Normal("sharpening", mu=0.0, sigma=1.5)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)

    c_mag = pt.exp(log_c_mag - spread_tuning * spread)[:, None]
    c_unit = pt.exp(log_c_unit + homog_tuning * dhomog)[:, None]
    c_par = (c_par_max * phomog)[:, None]
    gain = pt.exp(sharpening * coinc)[:, None]

    log_sim = gain * (-c_mag * dist - c_unit * udiff - c_par * pdiff - c_factor * cop)
    log_sim = pt.switch(present > 0.5, log_sim, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten()

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(alpha + beta * log_summed), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
