"""Place-value grid similarity.

People judge whether the program would produce a target by how close it looks
to the examples on a decimal place-value grid (tens digit x ones digit) rather
than on the number line: similarity to each example falls off exponentially
with the city-block grid distance, with separately fitted sensitivities to
tens-digit and ones-digit differences, and the yes probability grows with the
target's average grid similarity to the examples.
"""
import numpy as np
import pymc as pm

MAX_K = 4


def _grid(n):
    n = int(n)
    if n == 100:
        return 10, 0
    return n // 10, n % 10


def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")]
    tt, to = _grid(target)
    out = {}
    for k in range(MAX_K):
        if k < len(examples):
            et, eo = _grid(examples[k])
            out[f"dt{k}"] = float(abs(tt - et))
            out[f"do{k}"] = float(abs(to - eo))
            out[f"m{k}"] = 1.0
        else:
            out[f"dt{k}"] = 0.0
            out[f"do{k}"] = 0.0
            out[f"m{k}"] = 0.0
    return out


with pm.Model() as model:
    dts = [pm.Data(f"dt{k}", np.zeros(1, dtype="float64")) for k in range(MAX_K)]
    dos = [pm.Data(f"do{k}", np.zeros(1, dtype="float64")) for k in range(MAX_K)]
    ms = [pm.Data(f"m{k}", np.zeros(1, dtype="float64")) for k in range(MAX_K)]

    w_tens = pm.HalfNormal("w_tens", sigma=2.0)
    w_ones = pm.HalfNormal("w_ones", sigma=2.0)
    intercept = pm.Normal("intercept", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    sim_sum = 0.0
    n_ex = 0.0
    for k in range(MAX_K):
        sim_sum = sim_sum + ms[k] * pm.math.exp(-(w_tens * dts[k] + w_ones * dos[k]))
        n_ex = n_ex + ms[k]
    mean_sim = sim_sum / pm.math.maximum(n_ex, 1.0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
