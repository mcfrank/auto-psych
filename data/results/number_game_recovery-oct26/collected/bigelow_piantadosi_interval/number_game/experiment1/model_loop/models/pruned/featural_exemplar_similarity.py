"""Featural exemplar similarity.

People do not entertain candidate rules; they judge a target by its featural
similarity to each example seen, treating numbers as bundles of salient
properties (parity, divisibility by 3/4/5/10, last digit, decade, prime,
square, power of two). Each example's similarity decays exponentially with the
number of mismatching properties, and P(yes) rises with the mean similarity
across examples (GCM-style, no size principle).
"""
import numpy as np
import pymc as pm

MAX_EX = 4
_PRIMES = {n for n in range(2, 101) if all(n % k for k in range(2, int(n ** 0.5) + 1))}
_SQUARES = {k * k for k in range(1, 11)}
_POW2 = {2 ** k for k in range(0, 7)}


def _props(n):
    return (
        n % 2 == 0, n % 3 == 0, n % 4 == 0, n % 5 == 0, n % 10 == 0,
        n in _PRIMES, n in _SQUARES, n in _POW2,
    )


def _mismatch(t, x):
    a, b = _props(t), _props(x)
    d = sum(1 for u, v in zip(a, b) if u != v)
    d += 0 if t % 10 == x % 10 else 1          # same last digit
    d += 0 if min(t, 99) // 10 == min(x, 99) // 10 else 1  # same decade
    return float(d)


def compute_features(number_set, target):
    xs = [int(v) for v in number_set.split(",")][:MAX_EX]
    t = int(target)
    out = {}
    for i in range(MAX_EX):
        if i < len(xs):
            out[f"mm_{i}"] = _mismatch(t, xs[i])
            out[f"mask_{i}"] = 1.0
        else:
            out[f"mm_{i}"] = 0.0
            out[f"mask_{i}"] = 0.0
    return out


with pm.Model() as model:
    mm = [pm.Data(f"mm_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)]
    mask = [pm.Data(f"mask_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)]

    specificity = pm.HalfNormal("specificity", sigma=1.5)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    sim_sum = sum(m * pm.math.exp(-specificity * d) for d, m in zip(mm, mask))
    n_ex = sum(mask)
    mean_sim = sim_sum / n_ex

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
