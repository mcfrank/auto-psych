"""Prime-example multiple block (refinement of coincidence_rule_violation_route).

People judge a target by its partly averaged similarity to the stored examples
(magnitude, final digit, parity, the set's common divisor) plus distance-free
factor-kin, whole-set round and parity-preserving multiples routes, with a
fixed loss for breaking the set's coincidence rule and a decision that steepens
after suspicious coincidences. The one change: an example brings its multiples
to mind only if people know it as a factor; a two-digit prime (13, 17, 23, 31,
43, 47, ...) is never multiplied in everyday arithmetic, so its multiples route
is weakened by a fixed factor (34 after "17", 86 after "43" mostly rejected,
while 14 after "7", 64 after "16", 99 after "33" keep the full route).
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


def _isprime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


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
            out[f"multiple_of_{k}"] = (1.0 if t % ex[k] == 0 and t != ex[k] and ex[k] > 1 else 0.0)
            out[f"factor_kin_{k}"] = (1.0 if gcd(t, ex[k]) >= 3 and t % ex[k] != 0 and ex[k] % t != 0 else 0.0)
            out[f"round_both_{k}"] = 1.0 if (t % 10 == 0 and all(e % 5 == 0 for e in ex)) else 0.0
            out[f"present_{k}"] = 1.0
            out[f"prime_example_{k}"] = (1.0 if (_isprime(ex[k]) and ex[k] > 10) else 0.0)
        else:
            for nm in ("dist", "round_both", "unit_diff", "par_diff", "off_divisor", "multiple_of", "factor_kin", "present", "prime_example"):
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
    viol = 0.0
    if n > 1:
        if len({e % 10 for e in ex}) == 1:
            viol = 1.0 if t % 10 != ex[0] % 10 else 0.0
        elif all(e % 5 == 0 for e in ex):
            viol = 1.0 if t % 5 != 0 else 0.0
        elif len({e % 2 for e in ex}) == 1:
            viol = 1.0 if t % 2 != ex[0] % 2 else 0.0
    out["breaks_coincidence"] = viol
    out["log_n_examples"] = log(n)
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
    # The one change: two-digit prime examples weaken their multiples route.
    prime_ex = _stack("prime_example")
    c_prime = pm.HalfNormal("c_prime", sigma=2.0)
    fkin = _stack("factor_kin")
    round_both = _stack("round_both")
    # Round targets resemble round-ish examples at any distance.
    log_w_round = pm.Normal("log_w_round", -3.0, 2.0)

    spread = pm.Data("mag_spread", np.zeros(1, dtype="float64"))
    dhomog = pm.Data("digit_homog", np.zeros(1, dtype="float64"))
    phomog = pm.Data("parity_homog", np.zeros(1, dtype="float64"))
    coinc = pm.Data("coincidence", np.zeros(1, dtype="float64"))
    viol = pm.Data("breaks_coincidence", np.zeros(1, dtype="float64"))
    # The one change: breaking the set's coincidence rule costs a fixed share of support.
    c_viol = pm.HalfNormal("c_viol", sigma=2.0)
    log_n = pm.Data("log_n_examples", np.zeros(1, dtype="float64"))
    # Degree of averaging (0 = pure summing, 1 = pure averaging).
    kappa = pm.Beta("kappa", alpha=2.0, beta=2.0)

    log_c_mag = pm.Normal("log_c_mag", mu=0.0, sigma=1.5)
    log_c_unit = pm.Normal("log_c_unit", mu=1.0, sigma=1.5)
    spread_tuning = pm.Normal("spread_tuning", mu=0.0, sigma=4.0)
    homog_tuning = pm.Normal("homog_tuning", mu=0.0, sigma=2.0)
    c_par_max = pm.HalfNormal("c_par_max", sigma=2.0)
    c_divisor = pm.HalfNormal("c_divisor", sigma=2.0)
    # The one change: suspicious coincidence steepens the decision, not the similarity gradient.
    sharpening = pm.Normal("sharpening", mu=0.0, sigma=1.5)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)
    log_w_multiple = pm.Normal("log_w_multiple", mu=-3.0, sigma=2.0)
    # A weaker distance-free route for factor-kin of an example.
    log_w_factor_kin = pm.Normal("log_w_factor_kin", mu=-3.0, sigma=2.0)

    c_mag = pt.exp(log_c_mag - spread_tuning * spread)[:, None]
    c_unit = pt.exp(log_c_unit + homog_tuning * dhomog)[:, None]
    c_par = (c_par_max * phomog)[:, None]
    # Parity switch cuts the multiples route (from parity_preserving_multiple_route).
    c_par_route = pm.HalfNormal("c_par_route", sigma=2.0)

    log_sim = (-c_mag * dist - c_unit * udiff - c_par * pdiff - c_divisor * cop)
    # The mechanism: an additive similarity route for multiples of an example.
    ppen = pt.exp(-c_par_route * pdiff)
    sim = (pt.exp(log_sim) + ppen * pt.exp(log_w_multiple - c_prime * prime_ex) * kin
           + pt.exp(log_w_factor_kin) * fkin
           + pt.exp(log_w_round) * round_both)
    sim = pt.switch(present > 0.5, sim, 0.0)
    log_summed = pt.log(pt.sum(sim, axis=1) + 1e-12) - kappa * log_n - c_viol * viol

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(alpha + beta * pt.exp(sharpening * coinc) * log_summed), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
