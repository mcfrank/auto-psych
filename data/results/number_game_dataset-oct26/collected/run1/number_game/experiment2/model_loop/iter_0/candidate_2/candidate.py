"""Recency-faded magnitude anchoring alongside final-digit gist.

People keep the final digit of every example as durable gist and say "yes" in
proportion to the target's average final-digit similarity to the examples
(same digit most similar, same parity partly, opposite parity least); but their
memory of each example's exact magnitude fades with every later example shown,
so the most recent example acts as a magnitude anchor. Targets numerically
close to the remembered magnitudes, above all to the last example, get an
extra push toward "yes".
"""
import numpy as np
import pymc as pm

K = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = len(xs)
    out = {}
    for k in range(K):
        if k < n:
            x = xs[k]
            out[f"pres{k}"] = 1.0
            out[f"lag{k}"] = float(n - 1 - k)
            out[f"dist{k}"] = abs(t - x) / 100.0
            out[f"dig{k}"] = (1.0 if x % 10 == t % 10 else 0.0) / n
            out[f"par{k}"] = (1.0 if (x % 2 == t % 2 and x % 10 != t % 10) else 0.0) / n
        else:
            out[f"pres{k}"] = 0.0
            out[f"lag{k}"] = 0.0
            out[f"dist{k}"] = 0.0
            out[f"dig{k}"] = 0.0
            out[f"par{k}"] = 0.0
    return out


with pm.Model() as model:
    pres = [pm.Data(f"pres{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    lag = [pm.Data(f"lag{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    dist = [pm.Data(f"dist{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    dig = [pm.Data(f"dig{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    par = [pm.Data(f"par{k}", np.zeros(1, dtype="float64")) for k in range(K)]

    # Final-digit gist similarity of a same-parity, different-digit ending.
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    # Magnitude trace retained per later example (1 = no fading, 0 = last only).
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    # Width of magnitude generalization around a remembered example.
    tau = pm.LogNormal("tau", mu=np.log(0.15), sigma=0.7)

    digit_sim = 0.0
    wsum = 0.0
    mag_sum = 0.0
    for k in range(K):
        digit_sim = digit_sim + dig[k] + s_parity * par[k]
        wk = pres[k] * retention ** lag[k]
        wsum = wsum + wk
        mag_sum = mag_sum + wk * pm.math.exp(-dist[k] / tau)
    mag_sim = mag_sum / wsum

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    mag_gain = pm.HalfNormal("mag_gain", sigma=3.0)
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + gain * digit_sim + mag_gain * mag_sim)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
