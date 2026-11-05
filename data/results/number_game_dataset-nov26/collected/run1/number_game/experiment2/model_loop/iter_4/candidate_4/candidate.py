"""Times-table round-number kinship exemplar (refinement of round_number_kinship_exemplar).

People judge a target by summed similarity to the examples (magnitude, final
digit, parity, the set's common divisor), sharpened by suspicious
coincidences, with distance-free routes for factor-kin of an example and for
round targets after 0/5-ending examples. The one change: the multiples route
is retrieved times-table knowledge, so only examples 2-12 give their
multiples a distance-free boost; multiples of two-digit examples above 12
(94 after "47") get none.
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
            out[f"dist_{k}"] = abs(t - ex[k]) / 10.0
            out[f"unit_diff_{k}"] = 0.0 if t % 10 == ex[k] % 10 else 1.0
            out[f"par_diff_{k}"] = 0.0 if t % 2 == ex[k] % 2 else 1.0
            out[f"off_divisor_{k}"] = 1.0 if (n > 1 and g >= 3 and t % g != 0) else 0.0
            # The one change: only times-table examples (2-12) give a multiples route.
            out[f"multiple_of_{k}"] = (1.0 if t % ex[k] == 0 and t != ex[k] and 2 <= ex[k] <= 12 else 0.0)
            out[f"factor_kin_{k}"] = (1.0 if gcd(t, ex[k]) >= 3 and t % ex[k] != 0 and ex[k] % t != 0 else 0.0)
            out[f"round_both_{k}"] = 1.0 if (t % 10 == 0 and ex[k] % 5 == 0) else 0.0
            out[f"present_{k}"] = 1.0
        else:
            for nm in ("dist", "round_both", "unit_diff", "par_diff", "off_divisor", "multiple_of", "factor_kin", "present"):
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
    kin = _stack("multiple_of")
    fkin = _stack("factor_kin")
    round_both = _stack("round_both")
    # Round targets resemble round-ish examples at any distance.
    log_w_round = pm.Normal("log_w_round", -3.0, 2.0)

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
    # The mechanism: global sharpening of generalization by suspicious coincidence.
    sharpening = pm.Normal("sharpening", mu=0.0, sigma=1.5)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)
    log_w_multiple = pm.Normal("log_w_multiple", mu=-3.0, sigma=2.0)
    # A weaker distance-free route for factor-kin of an example.
    log_w_factor_kin = pm.Normal("log_w_factor_kin", mu=-3.0, sigma=2.0)

    c_mag = pt.exp(log_c_mag - spread_tuning * spread)[:, None]
    c_unit = pt.exp(log_c_unit + homog_tuning * dhomog)[:, None]
    c_par = (c_par_max * phomog)[:, None]
    gain = pt.exp(sharpening * coinc)[:, None]

    log_sim = gain * (-c_mag * dist - c_unit * udiff - c_par * pdiff - c_divisor * cop)
    # The mechanism: an additive similarity route for multiples of an example.
    sim = (pt.exp(log_sim) + pt.exp(log_w_multiple * gain) * kin
           + pt.exp(log_w_factor_kin * gain) * fkin
           + pt.exp(log_w_round * gain) * round_both)
    sim = pt.switch(present > 0.5, sim, 0.0)
    log_summed = pt.log(pt.sum(sim, axis=1) + 1e-12)

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(alpha + beta * log_summed), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
