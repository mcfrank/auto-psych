"""Property-based exemplar similarity (GCM over mathematical properties).

People do not infer a concept from the examples; they store each example as a
separate exemplar, described by its mathematical properties (parity, primes,
squares/cubes, powers, multiples, base-ten multiples, final digit, contained
digits) and its magnitude, and judge a target by its average similarity to the
stored examples. Similarity to an example decays exponentially with the
attention-weighted count of properties on which target and example differ plus
their numerical distance; higher mean similarity -> more likely "yes".
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_N = range(1, 101)
N_FAM = 8          # property families
N_DIM = N_FAM + 1  # + magnitude
MAX_EX = 4


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
        fam = 7 if m in (5, 10) else 4
        props.append((fam, {n for n in _N if n % m == 0}))
    for d in range(10):
        props.append((5, {n for n in _N if n % 10 == d}))
    for d in range(10):
        props.append((6, {n for n in _N if str(d) in str(n)}))
    return props


_PROPS = _props()


def _mismatch(a, b):
    """Per-family count of properties on which a and b differ, plus distance."""
    out = [0.0] * N_DIM
    for fam, p in _PROPS:
        if (a in p) != (b in p):
            out[fam] += 1.0
    out[N_FAM] = abs(a - b) / 10.0
    return out


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    feats = {}
    for s in range(MAX_EX):
        if s < len(ex):
            mm = _mismatch(t, ex[s])
            on = 1.0
        else:
            mm = [0.0] * N_DIM
            on = 0.0
        feats[f"ex_on_{s}"] = on
        for d in range(N_DIM):
            feats[f"mm_{s}_{d}"] = mm[d]
    return feats


with pm.Model() as model:
    on = pt.stack([pm.Data(f"ex_on_{s}", np.zeros(1, dtype="float64"))
                   for s in range(MAX_EX)], axis=1)                       # (N, 4)
    mm = pt.stack([pt.stack([pm.Data(f"mm_{s}_{d}", np.zeros(1, dtype="float64"))
                             for d in range(N_DIM)], axis=1)
                   for s in range(MAX_EX)], axis=1)                       # (N, 4, D)

    attention = pm.Dirichlet("attention", a=np.full(N_DIM, 1.5))
    specificity = pm.HalfNormal("specificity", sigma=3.0)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)

    dist = pt.sum(mm * attention, axis=2)                                 # (N, 4)
    sim = pt.exp(-specificity * dist) * on
    mean_sim = pt.sum(sim, axis=1) / pt.sum(on, axis=1)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
