"""Tversky ratio-model feature similarity.

People judge the target by its feature-based similarity to each example
separately (Tversky's ratio model): a number is a bundle of salient properties
(parity, final digit, decade, contained digits, prime, square, cube,
power-of-k, multiple-of-k), and the target's similarity to an example is
common / (common + alpha * example-only + beta * target-only) features, with
the two discount weights inferred. The higher the target's average similarity
to the examples, the more likely people say "yes"; distance on the number line
plays no role beyond the shared decade feature.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4
_PRIMES = {n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))}


def _props(n):
    p = {"even" if n % 2 == 0 else "odd", f"last{n % 10}", f"dec{min(n // 10, 9)}"}
    p |= {f"has{d}" for d in set(str(n))}
    if n in _PRIMES:
        p.add("prime")
    if round(n ** 0.5) ** 2 == n:
        p.add("square")
    if round(n ** (1 / 3)) ** 3 == n:
        p.add("cube")
    for k in range(2, 11):
        if n in {k ** e for e in range(1, 8)}:
            p.add(f"pow{k}")
    for k in range(3, 13):
        if n % k == 0:
            p.add(f"mult{k}")
    return p


_P = {n: _props(n) for n in range(1, 101)}


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")][:MAX_EX]
    t = int(target)
    out = {}
    for i in range(MAX_EX):
        if i < len(ex):
            x = ex[i]
            out[f"common{i}"] = float(len(_P[x] & _P[t]))
            out[f"ex_only{i}"] = float(len(_P[x] - _P[t]))
            out[f"tg_only{i}"] = float(len(_P[t] - _P[x]))
            out[f"mask{i}"] = 1.0
        else:
            out[f"common{i}"] = 0.0
            out[f"ex_only{i}"] = 1.0
            out[f"tg_only{i}"] = 0.0
            out[f"mask{i}"] = 0.0
    return out


with pm.Model() as model:
    common = pt.stack([pm.Data(f"common{i}", np.zeros(1)) for i in range(MAX_EX)])
    ex_only = pt.stack([pm.Data(f"ex_only{i}", np.ones(1)) for i in range(MAX_EX)])
    tg_only = pt.stack([pm.Data(f"tg_only{i}", np.zeros(1)) for i in range(MAX_EX)])
    mask = pt.stack([pm.Data(f"mask{i}", np.ones(1)) for i in range(MAX_EX)])

    # Discount weights for the example's unmatched features and the target's
    # extra features, and the mapping from mean similarity to "yes" log-odds.
    alpha = pm.LogNormal("alpha", mu=np.log(0.5), sigma=0.75)
    beta = pm.LogNormal("beta", mu=np.log(0.25), sigma=0.75)
    intercept = pm.Normal("intercept", mu=-3.0, sigma=2.0)
    slope = pm.HalfNormal("slope", sigma=10.0)

    sim = common / (common + alpha * ex_only + beta * tg_only + 1e-6)
    mean_sim = (sim * mask).sum(axis=0) / mask.sum(axis=0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
