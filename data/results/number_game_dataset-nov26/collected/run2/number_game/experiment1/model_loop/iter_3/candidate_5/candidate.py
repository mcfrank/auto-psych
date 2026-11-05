"""Parity-direction salience feature overlap.

Refinement of family_salience_feature_overlap (in the form of its successor
decimal_multiple_salience_overlap, with multiples of 5/10 as their own salience
family): people note every salient mathematical property all the examples
share and say "yes" in proportion to the salience-weighted fraction of that
shared profile the target also has, with no size principle. The one new
change (grafted from asymmetric_decimal_contrast_similarity's directional
parity contrast): "all odd" and "all even" each get their own salience instead
of sharing one parity-family weight, so one parity pattern can dominate the
judgment far more than the other.
"""
import numpy as np
import pymc as pm

_N = range(1, 101)
N_FAM = 9  # 0 odd, 1 primes, 2 squares/cubes, 3 powers, 4 other multiples,
# 5 final digit, 6 contained digit, 7 multiples of 5/10, 8 even; span is 9th


def _props():
    props = []  # (family index, set)
    props.append((0, {n for n in _N if n % 2 == 1}))
    props.append((8, {n for n in _N if n % 2 == 0}))
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


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    out = {}
    for f in range(N_FAM):
        shared = [p for fam, p in _PROPS if fam == f and all(e in p for e in ex)]
        out[f"sh_{f}"] = float(len(shared))
        out[f"hit_{f}"] = float(sum(1 for p in shared if t in p))
    out["in_span"] = 1.0 if min(ex) <= t <= max(ex) else 0.0
    return out


with pm.Model() as model:
    sh = [pm.Data(f"sh_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    hit = [pm.Data(f"hit_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    in_span = pm.Data("in_span", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 1, 1.5))

    num = salience[N_FAM] * in_span
    den = salience[N_FAM] + 0.0 * in_span
    for f in range(N_FAM):
        num = num + salience[f] * hit[f]
        den = den + salience[f] * sh[f]
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
