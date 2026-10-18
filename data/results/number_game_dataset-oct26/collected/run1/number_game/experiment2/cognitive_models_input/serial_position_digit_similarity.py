"""Serial-position-weighted final-digit similarity.

People compare the target's final digit with the final digits of the examples
(same digit most similar, same parity partly, opposite parity least), but each
example's weight in the comparison depends on its serial position, changing
steadily from first to last shown (recency or primacy, direction inferred).
"Yes" grows with a position-weighted average of final-digit similarity, so the
order in which the same numbers are shown changes which digit matches count.
"""
import numpy as np
import pymc as pm

K = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")][-K:]
    t = int(target)
    rev = xs[::-1]  # lag 0 = last-shown example
    out = {}
    for k in range(K):
        if k < len(rev):
            x = rev[k]
            same = 1.0 if x % 10 == t % 10 else 0.0
            par = 1.0 if (x % 2 == t % 2 and same == 0.0) else 0.0
            m = 1.0
        else:
            same, par, m = 0.0, 0.0, 0.0
        out[f"rw_same_lag{k}"] = same
        out[f"rw_par_lag{k}"] = par
        out[f"rw_mask_lag{k}"] = m
    return out


with pm.Model() as model:
    same = [pm.Data(f"rw_same_lag{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    par = [pm.Data(f"rw_par_lag{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    mask = [pm.Data(f"rw_mask_lag{k}", np.zeros(1, dtype="float64")) for k in range(K)]

    # Similarity of a same-parity, different-digit ending (identical = 1).
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    # Memory decay per position back from the last-shown example
    # (positive = recency; negative would mean primacy).
    decay = pm.Normal("decay", mu=0.0, sigma=1.5)

    total = 0.0
    wsum = 0.0
    for k in range(K):
        w = mask[k] * pm.math.exp(-decay * k)
        total = total + w * (same[k] + s_parity * par[k])
        wsum = wsum + w
    mean_sim = total / wsum

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
