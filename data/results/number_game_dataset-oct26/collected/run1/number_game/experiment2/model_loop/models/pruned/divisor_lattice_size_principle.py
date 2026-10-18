"""Final-digit similarity + nearest-decade familiarity + divisor-lattice concepts.

A refinement of nearest_decade_digit_common_factor that changes its space of
arithmetic concepts: rather than the single rule "multiples of the examples'
gcd (>= 3)", people entertain every "multiples of d" concept that all examples
obey (every common divisor d >= 2), weighted by the size principle
(1/|concept|)^(strength * n_examples) with an inferred strength. The arithmetic
boost toward "yes" is the graded posterior belief that the target lies in the
inferred multiples concept; final-digit similarity and nearest-decade
familiarity are unchanged.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 12  # max number of common divisors >= 2 of any gcd <= 100 (100 has 8)


def _stimulus(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = float(len(xs))
    same_digit = sum(1 for x in xs if x % 10 == t % 10) / n
    same_par = sum(1 for x in xs if x % 2 == t % 2 and x % 10 != t % 10) / n
    dec_gap = float(min(abs(x // 10 - t // 10) for x in xs))
    g = reduce(math.gcd, xs)
    ds = [d for d in range(2, 101) if g % d == 0][:K]
    logsize = np.zeros(K)
    hit = np.zeros(K)
    mask = np.zeros(K)
    for j, d in enumerate(ds):
        # |multiples of d in 1..100| = 100 // d; size-principle log-likelihood per example
        logsize[j] = -n * math.log(100 // d)
        hit[j] = 1.0 if t % d == 0 else 0.0
        mask[j] = 1.0
    return same_digit, same_par, dec_gap, logsize, hit, mask


def prepare_observed(rows):
    cache = {}
    sd, sp, dg, ls, hi, mk, y = [], [], [], [], [], [], []
    for r in rows:
        key = (str(r["number_set"]), str(r["target"]))
        if key not in cache:
            cache[key] = _stimulus(*key)
        a, b, c, l, h, m = cache[key]
        sd.append(a); sp.append(b); dg.append(c); ls.append(l); hi.append(h); mk.append(m)
        y.append(int(float(r["rating"])))
    mk = np.array(mk, dtype="float64")
    return {
        "frac_same_digit": np.array(sd, dtype="float64"),
        "frac_same_parity_other": np.array(sp, dtype="float64"),
        "min_decade_gap": np.array(dg, dtype="float64"),
        "div_logsize": np.array(ls, dtype="float64"),
        "div_hit": np.array(hi, dtype="float64"),
        "div_mask": mk,
        "has_divisor": (mk.sum(axis=1) > 0).astype("float64"),
        "rating": np.array(y, dtype="int64"),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    dec_gap = pm.Data("min_decade_gap", np.zeros(1, dtype="float64"))
    div_logsize = pm.Data("div_logsize", np.zeros((1, K), dtype="float64"))
    div_hit = pm.Data("div_hit", np.zeros((1, K), dtype="float64"))
    div_mask = pm.Data("div_mask", np.zeros((1, K), dtype="float64"))
    has_div = pm.Data("has_divisor", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    # Size-principle posterior over the lattice of common-divisor concepts.
    size_strength = pm.LogNormal("size_strength", mu=np.log(1.5), sigma=0.75)
    logits = size_strength * div_logsize + (div_mask - 1.0) * 1e3
    post = pt.special.softmax(logits, axis=1)
    p_in_concept = pt.sum(post * div_hit, axis=1) * has_div
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)

    decade_boost = pm.HalfNormal("decade_boost", sigma=2.0)
    decade_width = pm.LogNormal("decade_width", mu=np.log(1.5), sigma=0.75)
    decade_sim = pm.math.exp(-dec_gap / decade_width)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim + factor_boost * p_in_concept
            + decade_boost * decade_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
