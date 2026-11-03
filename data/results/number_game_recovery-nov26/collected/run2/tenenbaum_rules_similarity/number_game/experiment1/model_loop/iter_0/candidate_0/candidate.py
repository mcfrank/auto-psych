"""Exemplar-based generalization (GCM-style).

People do not infer a rule; they judge the target by its summed similarity to
the stored examples. Each number is represented on magnitude, parity, final
digit and tens digit, each with its own attention weight; similarity to an
example decays exponentially with attention-weighted mismatch, and the
probability of "yes" rises with the target's average similarity to the
examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4  # maximum number of examples per set


def compute_features(number_set, target):
    xs = [int(v) for v in number_set.split(",")][:K]
    t = int(target)
    out = {}
    for k in range(K):
        if k < len(xs):
            x = xs[k]
            out[f"dmag_{k}"] = abs(t - x) / 100.0
            out[f"dpar_{k}"] = float((t % 2) != (x % 2))
            out[f"dlast_{k}"] = float((t % 10) != (x % 10))
            out[f"dtens_{k}"] = float((t // 10) != (x // 10))
            out[f"mask_{k}"] = 1.0
        else:
            for n in ("dmag", "dpar", "dlast", "dtens"):
                out[f"{n}_{k}"] = 0.0
            out[f"mask_{k}"] = 0.0
    return out


with pm.Model() as model:
    z = lambda: np.zeros(1, dtype="float64")
    dmag = pt.stack([pm.Data(f"dmag_{k}", z()) for k in range(K)], axis=1)
    dpar = pt.stack([pm.Data(f"dpar_{k}", z()) for k in range(K)], axis=1)
    dlast = pt.stack([pm.Data(f"dlast_{k}", z()) for k in range(K)], axis=1)
    dtens = pt.stack([pm.Data(f"dtens_{k}", z()) for k in range(K)], axis=1)
    mask = pt.stack([pm.Data(f"mask_{k}", z()) for k in range(K)], axis=1)

    # Attention weights on each psychological dimension.
    w_mag = pm.HalfNormal("w_mag", sigma=10.0)
    w_par = pm.HalfNormal("w_par", sigma=3.0)
    w_last = pm.HalfNormal("w_last", sigma=3.0)
    w_tens = pm.HalfNormal("w_tens", sigma=3.0)
    # Response bias (the similarity scale is set by the attention weights).
    bias = pm.Normal("bias", mu=0.0, sigma=2.0)

    dist = w_mag * dmag + w_par * dpar + w_last * dlast + w_tens * dtens
    # log of mean similarity over present examples (stable log-sum-exp)
    logsim = pt.where(mask > 0, -dist, -1e6)
    m = pt.max(logsim, axis=1, keepdims=True)
    log_sum = pt.log(pt.sum(pt.exp(logsim - m), axis=1)) + m[:, 0]
    log_mean_sim = log_sum - pt.log(pt.sum(mask, axis=1))

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + log_mean_sim), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
