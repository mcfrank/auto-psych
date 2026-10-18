"""Proximity-gated final-digit similarity.

A refinement of digit_similarity_with_range_extension: "yes" still grows with
the target's average final-digit similarity to the examples (same digit = 1,
same parity other digit = partial, opposite parity = 0), but instead of a
separate span-membership boost, each example's digit resemblance is weighted
by its numerical proximity to the target (fading with distance down to an
inferred floor), so digit matches with nearby examples count more than
matches with distant ones.
"""
import numpy as np
import pymc as pm

K = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")][:K]
    t = int(target)
    out = {}
    for k in range(K):
        if k < len(xs):
            x = xs[k]
            out[f"same_digit_{k}"] = 1.0 if x % 10 == t % 10 else 0.0
            out[f"same_par_{k}"] = 1.0 if (x % 2 == t % 2 and x % 10 != t % 10) else 0.0
            out[f"dist_{k}"] = abs(t - x) / 100.0
            out[f"mask_{k}"] = 1.0 / len(xs)
        else:
            out[f"same_digit_{k}"] = 0.0
            out[f"same_par_{k}"] = 0.0
            out[f"dist_{k}"] = 0.0
            out[f"mask_{k}"] = 0.0
    return out


with pm.Model() as model:
    same = [pm.Data(f"same_digit_{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    par = [pm.Data(f"same_par_{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    dist = [pm.Data(f"dist_{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    mask = [pm.Data(f"mask_{k}", np.zeros(1, dtype="float64")) for k in range(K)]

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    # Weight of a maximally distant example relative to an adjacent one, and
    # the distance scale (on 0-1 = 0-100) over which proximity fades.
    floor = pm.Beta("floor", alpha=2.0, beta=2.0)
    width = pm.LogNormal("width", mu=np.log(0.15), sigma=1.0)

    score = 0.0
    for k in range(K):
        prox = floor + (1.0 - floor) * pm.math.exp(-dist[k] / width)
        score = score + mask[k] * (same[k] + s_parity * par[k]) * prox

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * score))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
