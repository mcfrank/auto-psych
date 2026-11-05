"""Log-scale feature prototype.

People do not infer a rule; they abstract a prototype from the examples: the
average profile of mathematical properties the examples have (parity, primes,
squares/cubes, powers, multiples, multiples of 5/10, final digit, contained
digit) and a central magnitude on a compressed (logarithmic) number line. They
say "yes" in proportion to the target's similarity to that prototype, which
falls off with the attention-weighted property mismatch and the log-scale
distance from the prototype centre (relative to the examples' log spread).
The log line makes upward extension cheaper than downward extension.
"""
import math
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_N = range(1, 101)
N_FAM = 8


def _props():
    props = []
    props.append((0, {n for n in _N if n % 2 == 1}))
    props.append((0, {n for n in _N if n % 2 == 0}))
    props.append((1, {n for n in _N if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))}))
    props.append((2, {k * k for k in range(1, 11)}))
    props.append((2, {k ** 3 for k in range(1, 5)}))
    for b in range(2, 11):
        props.append((3, {b ** k for k in range(1, 8) if b ** k <= 100}))
    for m in range(3, 13):
        props.append((7 if m in (5, 10) else 4, {n for n in _N if n % m == 0}))
    for d in range(10):
        props.append((5, {n for n in _N if n % 10 == d}))
    for d in range(10):
        props.append((6, {n for n in _N if str(d) in str(n)}))
    return props


_PROPS = _props()


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    out = {}
    for f in range(N_FAM):
        mism = 0.0
        for fam, p in _PROPS:
            if fam != f:
                continue
            proto = sum(1.0 for e in ex if e in p) / len(ex)
            mism += abs((1.0 if t in p else 0.0) - proto)
        out[f"mis_{f}"] = mism
    logs = [math.log(e) for e in ex]
    c = sum(logs) / len(logs)
    out["logdist"] = abs(math.log(t) - c)
    out["logspread"] = math.sqrt(sum((l - c) ** 2 for l in logs) / len(logs))
    return out


with pm.Model() as model:
    mis = [pm.Data(f"mis_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    logdist = pm.Data("logdist", np.zeros(1, dtype="float64"))
    logspread = pm.Data("logspread", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=1.0, sigma=2.0)
    sensitivity = pm.HalfNormal("sensitivity", sigma=10.0)
    attention = pm.Dirichlet("attention", a=np.full(N_FAM + 1, 1.5))
    width = pm.HalfNormal("width", sigma=1.0)

    dist = attention[N_FAM] * logdist / (logspread + width + 0.05)
    for f in range(N_FAM):
        dist = dist + attention[f] * mis[f]
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(intercept - sensitivity * dist), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
