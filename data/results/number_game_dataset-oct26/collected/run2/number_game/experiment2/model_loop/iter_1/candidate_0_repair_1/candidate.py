"""Region-distance generalization.

People do not compare the target with each example separately; they judge its
similarity to the region of number space the examples jointly occupy. Distance
to that region is the sum of how far the target lies outside the examples' size
range (relative to how widely the examples are spread, so clustered examples
give a steep gradient and spread-out ones a shallow one) plus a mismatch on each
property for which the target takes a value none of the examples has (final
digit, parity) or lacks the factor all the examples share; the probability of
"yes" falls off exponentially with that distance.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    lo, hi = min(ex), max(ex)
    outside = max(lo - t, 0, t - hi) / 100.0
    spread = (hi - lo) / 100.0
    digit_miss = 0.0 if t % 10 in {x % 10 for x in ex} else 1.0
    parity_miss = 0.0 if t % 2 in {x % 2 for x in ex} else 1.0
    g = reduce(math.gcd, ex) if len(ex) >= 2 else 1
    factor_miss = 1.0 if (g >= 3 and t % g != 0) else 0.0
    return {"outside": outside, "spread": spread, "digit_miss": digit_miss,
            "parity_miss": parity_miss, "factor_miss": factor_miss}


with pm.Model() as model:
    outside = pm.Data("outside", np.zeros(1, dtype="float64"))
    spread = pm.Data("spread", np.zeros(1, dtype="float64"))
    digit_miss = pm.Data("digit_miss", np.zeros(1, dtype="float64"))
    parity_miss = pm.Data("parity_miss", np.zeros(1, dtype="float64"))
    factor_miss = pm.Data("factor_miss", np.zeros(1, dtype="float64"))

    # Weight of each region-mismatch dimension in psychological distance.
    w_size = pm.HalfNormal("w_size", sigma=3.0)
    w_digit = pm.HalfNormal("w_digit", sigma=2.0)
    w_parity = pm.HalfNormal("w_parity", sigma=2.0)
    w_factor = pm.HalfNormal("w_factor", sigma=2.0)
    # Baseline width of the size gradient when the examples have no spread.
    width0 = pm.HalfNormal("width0", sigma=0.3)
    # Guessing floor and ceiling lapse.
    guess = pm.Beta("guess", alpha=1.5, beta=6.0)
    lapse = pm.Beta("lapse", alpha=1.5, beta=6.0)

    dist = (w_size * outside / (spread + width0 + 0.01)
            + w_digit * digit_miss + w_parity * parity_miss + w_factor * factor_miss)
    sim = pt.exp(-dist)
    p = guess + (1.0 - guess) * (1.0 - lapse) * sim
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
