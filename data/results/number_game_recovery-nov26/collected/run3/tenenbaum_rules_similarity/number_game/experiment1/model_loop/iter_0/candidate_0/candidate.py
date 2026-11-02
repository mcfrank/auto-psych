"""Exemplar similarity (generalized context model) over number dimensions.

People do not infer a rule: they judge the target by its average similarity
to the examples shown. Each example is compared with the target on magnitude,
final digit, tens digit and parity, with attention shared among these
dimensions; similarity decays exponentially with attention-weighted mismatch,
and higher mean similarity makes "yes" more likely.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXN = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    out = {}
    for i in range(MAXN):
        if i < len(xs):
            x = xs[i]
            out[f"dmag_{i}"] = abs(t - x) / 100.0
            out[f"dones_{i}"] = float(t % 10 != x % 10)
            out[f"dtens_{i}"] = float(t // 10 != x // 10)
            out[f"dpar_{i}"] = float(t % 2 != x % 2)
            out[f"mask_{i}"] = 1.0
        else:
            for k in ("dmag", "dones", "dtens", "dpar"):
                out[f"{k}_{i}"] = 0.0
            out[f"mask_{i}"] = 0.0
    return out


with pm.Model() as model:
    dims = {}
    for k in ("dmag", "dones", "dtens", "dpar", "mask"):
        dims[k] = pt.stack(
            [pm.Data(f"{k}_{i}", np.zeros(1, dtype="float64")) for i in range(MAXN)],
            axis=1,
        )

    # Attention over the four dimensions and overall specificity.
    w = pm.Dirichlet("attention", a=np.ones(4) * 2.0)
    c = pm.HalfNormal("specificity", sigma=5.0)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    dist = (
        w[0] * dims["dmag"]
        + w[1] * dims["dones"]
        + w[2] * dims["dtens"]
        + w[3] * dims["dpar"]
    )
    sim = pt.exp(-c * dist) * dims["mask"]
    mean_sim = sim.sum(axis=1) / dims["mask"].sum(axis=1)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(intercept + gain * mean_sim), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
