"""Times-table kinship exemplar.

People judge a target by summed similarity to the examples (magnitude, final
digit, parity, divisibility by the set's common divisor >= 3), sharpened on
every dimension by how suspicious the examples' shared coincidence is. The
distance-free arithmetic route is membership of the same familiar times table:
a target sharing a factor >= 3 with an example (other than the example itself)
gains a fixed amount of similarity to it, at any distance, and that gain falls
off with the size of the smallest shared factor >= 3 (small, well-known tables
count most). So 12 after "15" and 49 after "7" are endorsed, but 94 after "47"
gets little boost.
"""
from collections import Counter
from functools import reduce
from math import gcd, log

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def _coincidence(ex):
    n = len(ex)
    if n < 2:
        return 0.0
    best = 0.0
    if len({e % 2 for e in ex}) == 1:
        best = max(best, (n - 1) * log(2))
    if all(e % 5 == 0 for e in ex):
        best = max(best, (n - 1) * log(5))
    if len({e % 10 for e in ex}) == 1:
        best = max(best, (n - 1) * log(10))
    return best / (3 * log(10))


def _smallest_factor_ge3(g):
    for d in range(3, g + 1):
        if g % d == 0:
            return d
    return 0


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")][:MAXK]
    t = int(target)
    n = len(ex)
    g = reduce(gcd, ex)
    out = {}
    for k in range(MAXK):
        if k < n:
            e = ex[k]
            out[f"dist_{k}"] = abs(t - e) / 10.0
            out[f"unit_diff_{k}"] = 0.0 if t % 10 == e % 10 else 1.0
            out[f"par_diff_{k}"] = 0.0 if t % 2 == e % 2 else 1.0
            out[f"off_divisor_{k}"] = 1.0 if (n > 1 and g >= 3 and t % g != 0) else 0.0
            d = _smallest_factor_ge3(gcd(t, e)) if t != e else 0
            out[f"kin_{k}"] = 1.0 if d >= 3 else 0.0
            out[f"kin_logd_{k}"] = log(d / 3.0) if d >= 3 else 0.0
            out[f"present_{k}"] = 1.0
        else:
            for nm in ("dist", "unit_diff", "par_diff", "off_divisor", "kin", "kin_logd", "present"):
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
    cop = _stack("off_divisor")
    present = _stack("present")
    kin = _stack("kin")
    kin_logd = _stack("kin_logd")
    spread = pm.Data("mag_spread", np.zeros(1, dtype="float64"))
    dhomog = pm.Data("digit_homog", np.zeros(1, dtype="float64"))
    phomog = pm.Data("parity_homog", np.zeros(1, dtype="float64"))
    coinc = pm.Data("coincidence", np.zeros(1, dtype="float64"))

    log_c_mag = pm.Normal("log_c_mag", mu=0.0, sigma=1.5)
    log_c_unit = pm.Normal("log_c_unit", mu=1.0, sigma=1.5)
    spread_tuning = pm.Normal("spread_tuning", mu=0.0, sigma=4.0)
    homog_tuning = pm.Normal("homog_tuning", mu=0.0, sigma=2.0)
    c_par_max = pm.HalfNormal("c_par_max", sigma=2.0)
    c_divisor = pm.HalfNormal("c_divisor", sigma=2.0)
    sharpening = pm.Normal("sharpening", mu=0.0, sigma=1.5)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)
    log_w_kin = pm.Normal("log_w_kin", mu=-3.0, sigma=2.0)
    # The mechanism: kinship strength falls with the size of the shared table.
    table_decay = pm.HalfNormal("table_decay", sigma=1.5)

    c_mag = pt.exp(log_c_mag - spread_tuning * spread)[:, None]
    c_unit = pt.exp(log_c_unit + homog_tuning * dhomog)[:, None]
    c_par = (c_par_max * phomog)[:, None]
    gain = pt.exp(sharpening * coinc)[:, None]

    log_sim = gain * (-c_mag * dist - c_unit * udiff - c_par * pdiff - c_divisor * cop)
    kin_route = pt.exp(log_w_kin * gain - table_decay * kin_logd) * kin
    sim = pt.exp(log_sim) + kin_route
    sim = pt.switch(present > 0.5, sim, 0.0)
    log_summed = pt.log(pt.sum(sim, axis=1) + 1e-12)

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(alpha + beta * log_summed), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
