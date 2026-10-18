"""Proximity-weighted final-digit similarity.

A refinement of final_digit_similarity_gradient: "yes" still grows with the
target's average final-digit similarity to the examples (same digit = 1, same
parity other digit = partial, opposite parity = 0), but the one change is that
each example's digit similarity is discounted with its numerical distance from
the target (inferred depth and width), so digit resemblance to a nearby example
counts more than resemblance to a distant one.
"""
import numpy as np
import pymc as pm

K = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")][:K]
    t = int(target)
    feats = {}
    for k in range(K):
        if k < len(xs):
            x = xs[k]
            feats[f"m{k}"] = 1.0
            feats[f"dig{k}"] = 1.0 if x % 10 == t % 10 else 0.0
            feats[f"par{k}"] = 1.0 if (x % 2 == t % 2 and x % 10 != t % 10) else 0.0
            feats[f"dist{k}"] = abs(t - x) / 100.0
        else:
            feats[f"m{k}"] = 0.0
            feats[f"dig{k}"] = 0.0
            feats[f"par{k}"] = 0.0
            feats[f"dist{k}"] = 0.0
    return feats


with pm.Model() as model:
    m = [pm.Data(f"m{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    dig = [pm.Data(f"dig{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    par = [pm.Data(f"par{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    dist = [pm.Data(f"dist{k}", np.zeros(1, dtype="float64")) for k in range(K)]

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    # Share of an example's digit similarity that depends on proximity, and
    # the distance (fraction of 1-100) over which that share decays.
    prox_share = pm.Beta("prox_share", alpha=2.0, beta=2.0)
    prox_width = pm.LogNormal("prox_width", mu=np.log(0.15), sigma=1.0)

    n = m[0] + m[1] + m[2] + m[3]
    total = 0.0
    for k in range(K):
        sim_k = dig[k] + s_parity * par[k]
        weight_k = 1.0 - prox_share + prox_share * pm.math.exp(-dist[k] / prox_width)
        total = total + m[k] * sim_k * weight_k
    mean_sim = total / pm.math.maximum(n, 1.0)

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
