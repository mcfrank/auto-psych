"""Digit-wise common-feature matching.

People read each number as a string of two digit features, a tens digit (its
decade) and a units digit together with that digit's parity, and judge the
target by Tversky-style common-feature matching: every feature the target
shares with an example (same decade, same final digit, same parity) adds its
own independent increment of support for "yes", averaged over the examples.
Features add rather than multiply, so a same-decade target gains credit even
with a different final digit, and a final-digit match earns credit however
many decades away it lies.
"""
import numpy as np
import pymc as pm


def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = len(examples)
    tens = sum(1.0 for x in examples if x // 10 == t // 10) / n
    units = sum(1.0 for x in examples if x % 10 == t % 10) / n
    parity = sum(1.0 for x in examples if x % 2 == t % 2) / n
    return {"share_decade": tens, "share_units": units, "share_parity": parity}


with pm.Model() as model:
    share_decade = pm.Data("share_decade", np.zeros(1, dtype="float64"))
    share_units = pm.Data("share_units", np.zeros(1, dtype="float64"))
    share_parity = pm.Data("share_parity", np.zeros(1, dtype="float64"))

    # Baseline log-odds of "yes" with no shared feature, and the support each
    # shared digit feature adds.
    intercept = pm.Normal("intercept", mu=-1.0, sigma=2.0)
    w_decade = pm.HalfNormal("w_decade", sigma=3.0)
    w_units = pm.HalfNormal("w_units", sigma=3.0)
    w_parity = pm.HalfNormal("w_parity", sigma=3.0)

    score = (intercept + w_decade * share_decade + w_units * share_units
             + w_parity * share_parity)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(score))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
