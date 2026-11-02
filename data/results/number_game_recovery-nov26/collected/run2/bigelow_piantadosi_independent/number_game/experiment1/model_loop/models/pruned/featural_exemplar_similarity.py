"""Featural exemplar similarity (GCM-style, no rule induction).

People judge a target by its average similarity to the remembered examples,
where similarity to each example decays exponentially with the (attention-
weighted) number of surface features the two do not share: parity, final
digit, and decade. The yes probability follows a Luce choice rule between the
summed similarity and a free response criterion.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4  # maximum number of examples in a set
DIMS = ("par", "dig", "dec")


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",") if x.strip()]
    t = int(target)
    out = {}
    for k in range(K):
        if k < len(ex):
            e = ex[k]
            out[f"mask_{k}"] = 1.0
            out[f"par_{k}"] = float(e % 2 != t % 2)
            out[f"dig_{k}"] = float(e % 10 != t % 10)
            out[f"dec_{k}"] = float(min(e, 99) // 10 != min(t, 99) // 10)
        else:
            out[f"mask_{k}"] = 0.0
            for d in DIMS:
                out[f"{d}_{k}"] = 0.0
    return out


with pm.Model() as model:
    mask = pt.stack([pm.Data(f"mask_{k}", np.zeros(1)) for k in range(K)], axis=1)
    mism = {d: pt.stack([pm.Data(f"{d}_{k}", np.zeros(1)) for k in range(K)], axis=1)
            for d in DIMS}

    # Attention-weighted sensitivity to each featural mismatch.
    w_par = pm.HalfNormal("w_par", sigma=3.0)
    w_dig = pm.HalfNormal("w_dig", sigma=3.0)
    w_dec = pm.HalfNormal("w_dec", sigma=3.0)
    # Response criterion (log scale) of the Luce choice rule.
    log_k = pm.Normal("log_k", mu=-1.0, sigma=2.0)

    dist = w_par * mism["par"] + w_dig * mism["dig"] + w_dec * mism["dec"]
    sim = pt.sum(pt.exp(-dist) * mask, axis=1) / pt.maximum(pt.sum(mask, axis=1), 1.0)
    p = sim / (sim + pt.exp(log_k))
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
