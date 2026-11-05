"""Violation-penalty shared-feature test.

Refinement of family_salience_feature_overlap: people note the salient
mathematical properties all examples share (parity, primes, squares/cubes,
powers, multiples, final digit, contained digit, within-span), each family
with its own salience, no size principle. The one change: the judgment is
non-compensatory (noisy-AND) rather than a share of the profile matched — each
shared property the target violates lowers the log-odds of "yes" by its
family's salience (scaled by the fraction of that family's shared properties
violated), and matching other properties cannot make up for it.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_N = range(1, 101)
N_FAM = 7


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
        props.append((4, {n for n in _N if n % m == 0}))
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
        shared = [p for fam, p in _PROPS if fam == f and all(e in p for e in ex)]
        miss = sum(1 for p in shared if t not in p)
        out[f"miss_{f}"] = float(miss) / len(shared) if shared else 0.0
    out["out_span"] = 0.0 if min(ex) <= t <= max(ex) else 1.0
    return out


with pm.Model() as model:
    miss = pt.stack(
        [pm.Data(f"miss_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
        + [pm.Data("out_span", np.zeros(1, dtype="float64"))],
        axis=0,
    )
    intercept = pm.Normal("intercept", mu=1.0, sigma=1.5)
    penalty = pm.HalfNormal("penalty", sigma=3.0, shape=N_FAM + 1)

    logit = intercept - pt.dot(penalty, miss)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(logit))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
