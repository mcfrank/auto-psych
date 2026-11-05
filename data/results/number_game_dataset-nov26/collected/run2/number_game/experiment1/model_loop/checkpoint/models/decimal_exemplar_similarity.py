"""Decimal exemplar similarity (GCM-style).

People do not induce rules or concepts; they judge a target by its summed
similarity to the remembered examples. Numbers are represented in decimal
written form, so similarity decays exponentially with a weighted city-block
distance combining magnitude difference (in tens) and ones-digit mismatch,
with a learned attention weight between the two. Mean similarity relative to
a response criterion (Luce choice rule) gives the probability of "yes".
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4


def compute_features(number_set, target):
    xs = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    out = {}
    for i in range(K):
        if i < len(xs):
            x = xs[i]
            out[f"mag_{i}"] = abs(t - x) / 10.0
            out[f"ones_{i}"] = 1.0 if (t % 10) != (x % 10) else 0.0
            out[f"mask_{i}"] = 1.0
        else:
            out[f"mag_{i}"] = 0.0
            out[f"ones_{i}"] = 0.0
            out[f"mask_{i}"] = 0.0
    return out


with pm.Model() as model:
    mags = [pm.Data(f"mag_{i}", np.zeros(1, dtype="float64")) for i in range(K)]
    ones = [pm.Data(f"ones_{i}", np.zeros(1, dtype="float64")) for i in range(K)]
    masks = [pm.Data(f"mask_{i}", np.zeros(1, dtype="float64")) for i in range(K)]

    c = pm.HalfNormal("sensitivity", sigma=2.0)          # similarity gradient
    w = pm.Beta("attn_ones", alpha=2.0, beta=2.0)          # attention to ones digit
    log_k = pm.Normal("log_criterion", mu=-1.0, sigma=1.5)  # response criterion
    eps = pm.Beta("lapse", alpha=1.0, beta=10.0)

    sim_sum = 0.0
    n = 0.0
    for i in range(K):
        d = (1.0 - w) * mags[i] + w * ones[i]
        sim_sum = sim_sum + masks[i] * pt.exp(-c * d)
        n = n + masks[i]
    S = sim_sum / pt.maximum(n, 1.0)
    k = pt.exp(log_k)
    p_core = S / (S + k)
    p_yes = pm.Deterministic("p_yes", pt.clip(eps / 2.0 + (1.0 - eps) * p_core, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
