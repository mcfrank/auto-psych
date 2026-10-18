"""Conjunctive example-by-example verification.

People check the target against each example in turn and say "yes" only if it
passes every check. Each check is passed with a probability that rises with the
target's resemblance to that example (same final digit, same parity, nearby
decade, multiple of the examples' common factor). Each extra example is one more
check a rule violator can fail, so violators of a shared final digit are
rejected more firmly as consistent examples accumulate, while targets fitting
every example remain accepted.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXN = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")][:MAXN]
    t = int(target)
    g = reduce(math.gcd, xs)
    out = {"cf_multiple": 1.0 if (len(xs) >= 2 and g >= 3 and t % g == 0) else 0.0}
    for k in range(MAXN):
        if k < len(xs):
            x = xs[k]
            out[f"mask_{k}"] = 1.0
            out[f"digit_{k}"] = 1.0 if x % 10 == t % 10 else 0.0
            out[f"parity_{k}"] = 1.0 if (x % 2 == t % 2 and x % 10 != t % 10) else 0.0
            # Decade proximity: 1 in the same decade, halving per decade apart.
            out[f"dprox_{k}"] = 0.5 ** abs(x // 10 - t // 10)
        else:
            out[f"mask_{k}"] = 0.0
            out[f"digit_{k}"] = 0.0
            out[f"parity_{k}"] = 0.0
            out[f"dprox_{k}"] = 0.0
    return out


with pm.Model() as model:
    cf = pm.Data("cf_multiple", np.zeros(1, dtype="float64"))
    masks = [pm.Data(f"mask_{k}", np.zeros(1, dtype="float64")) for k in range(MAXN)]
    digits = [pm.Data(f"digit_{k}", np.zeros(1, dtype="float64")) for k in range(MAXN)]
    paritys = [pm.Data(f"parity_{k}", np.zeros(1, dtype="float64")) for k in range(MAXN)]
    dproxs = [pm.Data(f"dprox_{k}", np.zeros(1, dtype="float64")) for k in range(MAXN)]

    # Check-passing propensity for a target unlike the example, and the
    # additive gains from each kind of resemblance to that example.
    base = pm.Normal("base", mu=0.0, sigma=1.5)
    digit_gain = pm.HalfNormal("digit_gain", sigma=3.0)
    parity_gain = pm.HalfNormal("parity_gain", sigma=2.0)
    decade_gain = pm.HalfNormal("decade_gain", sigma=2.0)
    factor_gain = pm.Normal("factor_gain", mu=0.0, sigma=1.5)

    log_p = 0.0
    for k in range(MAXN):
        z = (base + digit_gain * digits[k] + parity_gain * paritys[k]
             + decade_gain * dproxs[k] + factor_gain * cf)
        # log sigmoid(z): probability the check against example k is passed.
        log_p = log_p + masks[k] * (-pt.softplus(-z))

    p_yes = pm.Deterministic("p_yes", pt.clip(pt.exp(log_p), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
