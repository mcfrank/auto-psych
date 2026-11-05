"""Times-table multiple route exemplar (refinement of multiple_route_exemplar).

People judge a target by summed similarity to the examples (magnitude, final
digit, parity, divisibility by the set's common divisor), sharpened by how
suspicious the examples' shared coincidence is, plus a separate, distance-free
route for targets that are multiples of an example. The one change: that route
comes from times-table knowledge, so it is strong for one-digit examples
(49 or 98 after "7") and weakened by a fitted factor for two-digit examples,
whose multiples are not memorised facts (94 after "47").
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
            if t == e:
                kin, fac = 0.0, 0.0
            elif e > 1 and t % e == 0:
                kin, fac = 1.0, float(e)
            else:
                kin, fac = 0.0, 0.0
            out[f"kin_{k}"] = kin
            # 1 when the multiple is of a two-digit example (outside the times tables).
            out[f"kin_factor_{k}"] = 1.0 if (kin and fac >= 10) else 0.0
            out[f"present_{k}"] = 1.0
        else:
            for nm in ("dist", "unit_diff", "par_diff", "off_divisor", "kin", "kin_factor", "present"):
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
    kfac = _stack("kin_factor")
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
    # The change: the multiples route is weaker for two-digit examples.
    factor_decay = pm.HalfNormal("factor_decay", sigma=1.0)

    c_mag = pt.exp(log_c_mag - spread_tuning * spread)[:, None]
    c_unit = pt.exp(log_c_unit + homog_tuning * dhomog)[:, None]
    c_par = (c_par_max * phomog)[:, None]
    gain = pt.exp(sharpening * coinc)[:, None]

    log_sim = gain * (-c_mag * dist - c_unit * udiff - c_par * pdiff - c_divisor * cop)
    route = pt.exp(log_w_kin * gain - factor_decay * kfac) * kin
    sim = pt.exp(log_sim) + route
    sim = pt.switch(present > 0.5, sim, 0.0)
    log_summed = pt.log(pt.sum(sim, axis=1) + 1e-12)

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(alpha + beta * log_summed), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
