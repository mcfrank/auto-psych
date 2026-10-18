"""Place-value digit similarity.

People represent each number by its decimal place values (tens digit, units
digit) and judge the target by its digit-wise similarity to each example: the
product of units-digit similarity (identical most, same parity partly,
opposite parity least) and tens-digit similarity (decaying with the number of
decades between them). "Yes" grows with the target's average place-value
similarity to the examples, so a final-digit match counts strongly in the same
or a nearby decade and much less many decades away.
"""
import numpy as np
import pymc as pm

K = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")][:K]
    t = int(target)
    out = {}
    for k in range(K):
        if k < len(xs):
            x = xs[k]
            same = 1.0 if x % 10 == t % 10 else 0.0
            par = 1.0 if (x % 2 == t % 2 and same == 0.0) else 0.0
            dt = abs(t // 10 - x // 10) / 10.0
            m = 1.0
        else:
            same, par, dt, m = 0.0, 0.0, 0.0, 0.0
        out[f"pv_same_{k}"] = same
        out[f"pv_par_{k}"] = par
        out[f"pv_dtens_{k}"] = dt
        out[f"pv_mask_{k}"] = m
    return out


with pm.Model() as model:
    same = [pm.Data(f"pv_same_{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    par = [pm.Data(f"pv_par_{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    dt = [pm.Data(f"pv_dtens_{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    mask = [pm.Data(f"pv_mask_{k}", np.zeros(1, dtype="float64")) for k in range(K)]

    # Units-digit similarity of a same-parity, different ending.
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    # Decay of tens-digit similarity per decade (dtens is in units of 10 decades).
    decade_decay = pm.HalfNormal("decade_decay", sigma=3.0)

    total = 0.0
    n = 0.0
    for k in range(K):
        units_sim = same[k] + s_parity * par[k]
        tens_sim = pm.math.exp(-decade_decay * dt[k])
        total = total + mask[k] * units_sim * tens_sim
        n = n + mask[k]
    mean_sim = total / pm.math.maximum(n, 1.0)

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
