"""Digit-space exemplar similarity (GCM-style).

People judge a target by its exemplar similarity to the examples in a
digit-based representation: each number is coded by its tens digit and ones
digit, distance to an example is a weighted city-block distance (graded in
tens-digit steps, all-or-none for a differing ones digit, with fitted
attention weights), similarity decays exponentially with distance, and the
probability of "yes" rises with the mean similarity to the examples.
No rule inference, no size principle.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4


def _digits(n):
    return n // 10, n % 10  # 100 -> tens 10, ones 0


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    t_tens, t_ones = _digits(int(target))
    out = {}
    for k in range(K):
        if k < len(ex):
            e_tens, e_ones = _digits(ex[k])
            out[f"tens_gap_{k}"] = float(abs(t_tens - e_tens))
            out[f"ones_diff_{k}"] = 1.0 if t_ones != e_ones else 0.0
            out[f"present_{k}"] = 1.0
        else:
            out[f"tens_gap_{k}"] = 0.0
            out[f"ones_diff_{k}"] = 0.0
            out[f"present_{k}"] = 0.0
    return out


with pm.Model() as model:
    tg = [pm.Data(f"tens_gap_{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    od = [pm.Data(f"ones_diff_{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    pr = [pm.Data(f"present_{k}", np.zeros(1, dtype="float64")) for k in range(K)]

    # Attention-weighted sensitivity on each digit dimension.
    w_tens = pm.HalfNormal("w_tens", sigma=2.0)
    w_ones = pm.HalfNormal("w_ones", sigma=3.0)
    # Response mapping from mean similarity to yes.
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    sim_sum = 0.0
    n_ex = 0.0
    for k in range(K):
        sim_sum = sim_sum + pr[k] * pt.exp(-(w_tens * tg[k] + w_ones * od[k]))
        n_ex = n_ex + pr[k]
    mean_sim = sim_sum / pt.maximum(n_ex, 1.0)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(intercept + gain * mean_sim), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
