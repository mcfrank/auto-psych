"""Place-value grid similarity.

People represent each number by its place values (tens digit x units digit)
and judge the target by its similarity to each example on that grid: the
units dimension gives same final digit = 1, same parity different digit =
partial, opposite parity = 0, and this is multiplied by a similarity that
decays exponentially with the number of decades separating target and
example. "Yes" grows with the average place-value similarity to the examples,
so a final-digit match counts mostly when it lies in a nearby decade.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")][:K]
    t = int(target)
    out = {}
    for k in range(K):
        if k < len(xs):
            x = xs[k]
            out[f"pv_tens_diff_{k}"] = abs(t // 10 - x // 10) / 9.0
            out[f"pv_same_digit_{k}"] = 1.0 if x % 10 == t % 10 else 0.0
            out[f"pv_same_par_{k}"] = 1.0 if (x % 2 == t % 2 and x % 10 != t % 10) else 0.0
            out[f"pv_mask_{k}"] = 1.0
        else:
            out[f"pv_tens_diff_{k}"] = 0.0
            out[f"pv_same_digit_{k}"] = 0.0
            out[f"pv_same_par_{k}"] = 0.0
            out[f"pv_mask_{k}"] = 0.0
    return out


with pm.Model() as model:
    td = [pm.Data(f"pv_tens_diff_{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    sd = [pm.Data(f"pv_same_digit_{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    sp = [pm.Data(f"pv_same_par_{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    mk = [pm.Data(f"pv_mask_{k}", np.zeros(1, dtype="float64")) for k in range(K)]

    # Units-digit similarity of a same-parity, different-digit ending.
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    # Decay of similarity per decade apart (in units of the full 9-decade span).
    tens_decay = pm.HalfNormal("tens_decay", sigma=3.0)

    sim_sum = 0.0
    n_ex = 0.0
    for k in range(K):
        units_sim = sd[k] + s_parity * sp[k]
        sim_sum = sim_sum + mk[k] * units_sim * pt.exp(-tens_decay * td[k])
        n_ex = n_ex + mk[k]
    mean_sim = sim_sum / pt.maximum(n_ex, 1.0)

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
