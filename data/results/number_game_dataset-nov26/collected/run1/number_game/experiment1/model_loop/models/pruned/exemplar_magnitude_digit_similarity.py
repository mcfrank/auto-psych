"""Exemplar (GCM) similarity hypothesis.

People judge a target by its summed similarity to the remembered example
numbers rather than by inferring rules. Each number is represented by its
magnitude and its final digit; similarity to an example decays exponentially
with attention-weighted city-block distance on these two dimensions, and
p(yes) = S / (S + k), the Luce ratio of summed similarity S to a criterion k.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXN = 4


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",") if x.strip() != ""]
    t = int(target)
    out = {}
    for i in range(MAXN):
        if i < len(ex):
            out[f"mag_dist_{i}"] = abs(t - ex[i]) / 100.0
            out[f"digit_mis_{i}"] = 0.0 if (t % 10) == (ex[i] % 10) else 1.0
            out[f"ex_mask_{i}"] = 1.0
        else:
            out[f"mag_dist_{i}"] = 0.0
            out[f"digit_mis_{i}"] = 0.0
            out[f"ex_mask_{i}"] = 0.0
    return out


with pm.Model() as model:
    mags = [pm.Data(f"mag_dist_{i}", np.zeros(1, dtype="float64")) for i in range(MAXN)]
    mis = [pm.Data(f"digit_mis_{i}", np.zeros(1, dtype="float64")) for i in range(MAXN)]
    masks = [pm.Data(f"ex_mask_{i}", np.zeros(1, dtype="float64")) for i in range(MAXN)]

    # Sensitivity (specificity of generalization), attention to magnitude vs digit,
    # and the response criterion of the Luce rule.
    c = pm.HalfNormal("c", sigma=10.0)
    w_mag = pm.Beta("w_mag", alpha=2.0, beta=2.0)
    log_k = pm.Normal("log_k", mu=0.0, sigma=2.0)

    S = 0.0
    for i in range(MAXN):
        d = w_mag * mags[i] + (1.0 - w_mag) * mis[i]
        S = S + masks[i] * pt.exp(-c * d)

    p = S / (S + pt.exp(log_k))
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
