"""Proximity-cued retrieval of examples for final-digit comparison.

People compare the target's final digit with the examples' final digits (same
digit most similar, same parity partly, opposite parity least), but the target
acts as a retrieval cue: examples close to it in magnitude come to mind
readily, while numerically distant examples are only sometimes recalled for the
comparison. "Yes" grows with the final-digit similarity to the retrieved
examples, so a digit match with a nearby example counts for more than the same
match with a far-away one.
"""
import numpy as np
import pymc as pm

K = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = float(len(xs))
    out = {}
    for k in range(K):
        if k < len(xs):
            x = xs[k]
            out[f"w{k}"] = 1.0 / n
            out[f"dist{k}"] = abs(t - x) / 100.0
            out[f"dig{k}"] = 1.0 if x % 10 == t % 10 else 0.0
            out[f"par{k}"] = 1.0 if (x % 2 == t % 2 and x % 10 != t % 10) else 0.0
        else:
            out[f"w{k}"] = 0.0
            out[f"dist{k}"] = 0.0
            out[f"dig{k}"] = 0.0
            out[f"par{k}"] = 0.0
    return out


with pm.Model() as model:
    w = [pm.Data(f"w{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    dist = [pm.Data(f"dist{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    dig = [pm.Data(f"dig{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    par = [pm.Data(f"par{k}", np.zeros(1, dtype="float64")) for k in range(K)]

    # Final-digit similarity of a same-parity, different-digit ending.
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    # Retrieval: a distant example is still recalled with probability rho;
    # recall of nearer examples rises toward 1 over a magnitude scale tau.
    rho = pm.Beta("rho", alpha=2.0, beta=2.0)
    tau = pm.LogNormal("tau", mu=np.log(0.15), sigma=0.7)

    evidence = 0.0
    for k in range(K):
        recall = rho + (1.0 - rho) * pm.math.exp(-dist[k] / tau)
        evidence = evidence + w[k] * recall * (dig[k] + s_parity * par[k])

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * evidence))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
