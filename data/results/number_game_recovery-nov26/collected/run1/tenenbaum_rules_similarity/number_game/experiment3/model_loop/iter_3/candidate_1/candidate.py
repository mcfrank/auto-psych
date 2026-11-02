"""Place-value grid similarity.

People do not infer a rule; they represent each number by its written
place-value form (tens digit, units digit) as a point on a 10x10 grid and judge
a target by its average similarity to the examples, where similarity decays
exponentially with the attention-weighted city-block distance between target
and example on that grid (separate attention to tens and units).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _tu(n):
    n = int(n)
    return (n // 10, n % 10) if n < 100 else (10, 0)


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",") if x.strip()]
    tt, tu = _tu(target)
    out = {}
    for k in range(4):
        if k < len(ex):
            et, eu = _tu(ex[k])
            out[f"dt{k}"] = float(abs(tt - et))
            out[f"du{k}"] = float(abs(tu - eu))
            out[f"m{k}"] = 1.0
        else:
            out[f"dt{k}"] = 0.0
            out[f"du{k}"] = 0.0
            out[f"m{k}"] = 0.0
    return out


with pm.Model() as model:
    dt = pt.stack([pm.Data(f"dt{k}", np.zeros(1, dtype="float64")) for k in range(4)], axis=1)
    du = pt.stack([pm.Data(f"du{k}", np.zeros(1, dtype="float64")) for k in range(4)], axis=1)
    m = pt.stack([pm.Data(f"m{k}", np.zeros(1, dtype="float64")) for k in range(4)], axis=1)

    sensitivity = pm.HalfNormal("sensitivity", sigma=2.0)
    w_units = pm.Beta("w_units", alpha=2.0, beta=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    bias = pm.Normal("bias", mu=0.0, sigma=2.0)

    dist = (1.0 - w_units) * dt + w_units * du
    sim = pt.exp(-sensitivity * dist) * m
    mean_sim = pt.sum(sim, axis=1) / pt.maximum(pt.sum(m, axis=1), 1.0)

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(bias + gain * mean_sim), 1e-6, 1 - 1e-6))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
