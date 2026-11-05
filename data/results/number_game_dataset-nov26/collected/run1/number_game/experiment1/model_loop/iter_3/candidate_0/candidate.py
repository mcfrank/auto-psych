"""Shared-regularity exemplar model (concept-space change on graded_parity_variability_exemplar).

People judge a target by summed similarity to the remembered examples
(magnitude, final digit, parity; magnitude and final-digit attention tuned by
how variable the examples are). In addition, people notice the explicit
arithmetic/digit regularities that ALL examples share (multiples of 3-12,
powers of 2,3,4,5,10, squares, cubes, primes, odd, even, ending in d), weighting
each by its surprisingness log(100/size). A target conforming to the most
specific shared regularity gains, and one breaking the most specific shared
regularity loses, endorsement in proportion to that surprisingness.
"""
import math
from collections import Counter

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
_R = range(1, 101)


def _isprime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _concepts():
    c = {}
    for k in range(3, 13):
        c[f"mult{k}"] = {x for x in _R if x % k == 0}
    c["square"] = {x * x for x in range(1, 11)}
    c["cube"] = {x ** 3 for x in range(1, 5)}
    for b in (2, 3, 4, 5, 10):
        c[f"pow{b}"] = {b ** i for i in range(1, 8) if b ** i <= 100}
    c["prime"] = {x for x in _R if _isprime(x)}
    for d in range(10):
        c[f"end{d}"] = {x for x in _R if x % 10 == d}
    c["even"] = {x for x in _R if x % 2 == 0}
    c["odd"] = {x for x in _R if x % 2 == 1}
    return c


CONCEPTS = _concepts()
SURPRISE = {k: math.log(100.0 / len(v)) for k, v in CONCEPTS.items()}


def compute_features(number_set, target):
    ex_all = [int(x) for x in number_set.split(",")]
    ex = ex_all[:MAXK]
    t = int(target)
    n = len(ex_all)
    out = {}
    for k in range(MAXK):
        if k < len(ex):
            out[f"dist_{k}"] = abs(t - ex[k]) / 10.0
            out[f"unit_diff_{k}"] = 0.0 if t % 10 == ex[k] % 10 else 1.0
            out[f"parity_diff_{k}"] = 0.0 if t % 2 == ex[k] % 2 else 1.0
            out[f"present_{k}"] = 1.0
        else:
            out[f"dist_{k}"] = 0.0
            out[f"unit_diff_{k}"] = 0.0
            out[f"parity_diff_{k}"] = 0.0
            out[f"present_{k}"] = 0.0
    if n > 1:
        out["mag_spread"] = (max(ex_all) - min(ex_all)) / 100.0
        out["digit_homog"] = (max(Counter(e % 10 for e in ex_all).values()) - 1) / (n - 1)
        shared = [c for c, s in CONCEPTS.items() if all(e in s for e in ex_all)]
        conform = max([SURPRISE[c] for c in shared if t in CONCEPTS[c]], default=0.0)
        violate = max([SURPRISE[c] for c in shared if t not in CONCEPTS[c]], default=0.0)
        out["regularity_fit"] = conform - violate
    else:
        out["mag_spread"] = 0.5
        out["digit_homog"] = 0.5
        out["regularity_fit"] = 0.0  # one example reveals no shared regularity
    return out


with pm.Model() as model:
    dist = pt.stack([pm.Data(f"dist_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    udiff = pt.stack([pm.Data(f"unit_diff_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    pdiff = pt.stack([pm.Data(f"parity_diff_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    present = pt.stack([pm.Data(f"present_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    spread = pm.Data("mag_spread", np.zeros(1, dtype="float64"))
    homog = pm.Data("digit_homog", np.zeros(1, dtype="float64"))
    reg = pm.Data("regularity_fit", np.zeros(1, dtype="float64"))

    log_c_mag = pm.Normal("log_c_mag", mu=0.0, sigma=1.5)
    log_c_unit = pm.Normal("log_c_unit", mu=1.0, sigma=1.5)
    log_c_parity = pm.Normal("log_c_parity", mu=0.0, sigma=1.5)
    spread_tuning = pm.Normal("spread_tuning", mu=0.0, sigma=4.0)
    homog_tuning = pm.Normal("homog_tuning", mu=0.0, sigma=2.0)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)
    # Weight of the shared-regularity evidence (per nat of surprisingness).
    rule_weight = pm.HalfNormal("rule_weight", sigma=1.0)

    c_mag = pt.exp(log_c_mag - spread_tuning * spread)[:, None]
    c_unit = pt.exp(log_c_unit + homog_tuning * homog)[:, None]
    c_parity = pt.exp(log_c_parity)

    log_sim = -c_mag * dist - c_unit * udiff - c_parity * pdiff
    log_sim = pt.switch(present > 0.5, log_sim, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten()

    p_yes = pm.Deterministic(
        "p_yes",
        pt.clip(pm.math.sigmoid(alpha + beta * log_summed + rule_weight * reg), 1e-6, 1 - 1e-6),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
