"""Proximity-weighted final-digit similarity.

A refinement of final_digit_similarity_gradient: "yes" still grows with the
target's average final-digit similarity to the examples (same digit = 1, same
parity other digit = partial, opposite parity = 0), but the one change is that
each example's digit resemblance is weighted by its numerical closeness to the
target: full weight when the example is near, decaying exponentially with
distance down to an inferred floor when it is far. So a shared ending counts
most when the matching example is nearby.
"""
import numpy as np
import pymc as pm

MAX_K = 4


def prepare_observed(rows):
    n = len(rows)
    same_digit = np.zeros((n, MAX_K))
    same_par = np.zeros((n, MAX_K))
    dist = np.zeros((n, MAX_K))
    weight = np.zeros((n, MAX_K))
    for i, r in enumerate(rows):
        xs = [int(x) for x in str(r["number_set"]).split(",")][:MAX_K]
        t = int(r["target"])
        for k, x in enumerate(xs):
            same_digit[i, k] = 1.0 if x % 10 == t % 10 else 0.0
            same_par[i, k] = 1.0 if (x % 2 == t % 2 and x % 10 != t % 10) else 0.0
            dist[i, k] = abs(t - x) / 100.0
            weight[i, k] = 1.0 / len(xs)
    rating = np.array([int(r.get("rating", 0) or 0) for r in rows], dtype="int64")
    return {
        "ex_same_digit": same_digit,
        "ex_same_parity_other": same_par,
        "ex_distance": dist,
        "ex_weight": weight,
        "rating": rating,
    }


with pm.Model() as model:
    ex_digit = pm.Data("ex_same_digit", np.zeros((1, MAX_K), dtype="float64"))
    ex_par = pm.Data("ex_same_parity_other", np.zeros((1, MAX_K), dtype="float64"))
    ex_dist = pm.Data("ex_distance", np.zeros((1, MAX_K), dtype="float64"))
    ex_w = pm.Data("ex_weight", np.zeros((1, MAX_K), dtype="float64"))

    # Final-digit similarity of a same-parity, different-digit ending.
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    digit_sim = ex_digit + s_parity * ex_par

    # Proximity weighting: far examples keep a fraction `floor` of their
    # digit resemblance; closeness decays over `width` (in units of 100).
    floor = pm.Beta("floor", alpha=2.0, beta=2.0)
    width = pm.LogNormal("width", mu=np.log(0.15), sigma=0.8)
    proximity = floor + (1.0 - floor) * pm.math.exp(-ex_dist / width)

    mean_sim = pm.math.sum(ex_w * digit_sim * proximity, axis=1)

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
