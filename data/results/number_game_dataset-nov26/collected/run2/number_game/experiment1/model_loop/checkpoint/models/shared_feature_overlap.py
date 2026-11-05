"""Shared-feature overlap (no size principle).

People list every salient mathematical property that all the examples share
(parity, prime, square, cube, powers of 2-10, multiples of 3-12, final digit,
contained digit, and lying within the examples' span) and say "yes" in
proportion to the fraction of those shared properties the target also has.
All rule properties count equally (no preference for narrow rules); the
"within span" property gets its own salience weight.
"""
import numpy as np
import pymc as pm

_N = range(1, 101)


def _sets():
    props = []
    props.append({n for n in _N if n % 2 == 1})
    props.append({n for n in _N if n % 2 == 0})
    props.append({n for n in _N if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))})
    props.append({k * k for k in range(1, 11)})
    props.append({k ** 3 for k in range(1, 5)})
    for b in range(2, 11):
        props.append({b ** k for k in range(1, 8) if b ** k <= 100})
    for m in range(3, 13):
        props.append({n for n in _N if n % m == 0})
    for d in range(10):
        props.append({n for n in _N if n % 10 == d})
    for d in range(10):
        props.append({n for n in _N if str(d) in str(n)})
    return props


_PROPS = _sets()


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    shared = [p for p in _PROPS if all(e in p for e in ex)]
    hits = sum(1 for p in shared if t in p)
    in_span = 1.0 if min(ex) <= t <= max(ex) else 0.0
    return {"n_shared": float(len(shared)), "n_hit": float(hits), "in_span": in_span}


with pm.Model() as model:
    n_shared = pm.Data("n_shared", np.zeros(1, dtype="float64"))
    n_hit = pm.Data("n_hit", np.zeros(1, dtype="float64"))
    in_span = pm.Data("in_span", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    log_w_span = pm.Normal("log_w_span", mu=0.0, sigma=1.0)
    w_span = pm.math.exp(log_w_span)

    overlap = (n_hit + w_span * in_span) / (n_shared + w_span)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
