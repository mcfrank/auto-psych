"""Surface-digit search termination in shared-feature overlap.

People search the examples for shared properties in order of visibility,
starting with the written surface of the numbers. When that first glance finds
a shared final digit, they largely stop searching, so the arithmetic properties
the examples also share (primes, squares/cubes, powers, multiples) are only
partly noticed (their salience scaled down by an inferred search-termination
probability). "Yes" is given in proportion to the salience-weighted share of
the noticed shared profile the target has (one-sided magnitude bounds, no size
principle).
"""
import numpy as np
import pymc as pm

_N = range(1, 101)
N_FAM = 8
ARITH = (1, 2, 3, 4)  # deep arithmetic families masked by a surface pattern


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


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    out = {}
    for f in range(N_FAM):
        shared = [p for fam, p in _PROPS if fam == f and all(e in p for e in ex)]
        out[f"sh_{f}"] = float(len(shared))
        out[f"hit_{f}"] = float(sum(1 for p in shared if t in p))
    out["at_least_min"] = 1.0 if t >= min(ex) else 0.0
    out["at_most_max"] = 1.0 if t <= max(ex) else 0.0
    out["surface_found"] = 1.0 if (len(ex) >= 2 and len({e % 10 for e in ex}) == 1) else 0.0
    return out


with pm.Model() as model:
    sh = [pm.Data(f"sh_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    hit = [pm.Data(f"hit_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    ge_min = pm.Data("at_least_min", np.zeros(1, dtype="float64"))
    le_max = pm.Data("at_most_max", np.zeros(1, dtype="float64"))
    surface = pm.Data("surface_found", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 2, 1.5))
    # Probability the search stops once a shared final digit is seen.
    stop = pm.Beta("stop", alpha=1.5, beta=1.5)
    noticed = 1.0 - stop * surface

    num = salience[N_FAM] * ge_min + salience[N_FAM + 1] * le_max
    den = salience[N_FAM] + salience[N_FAM + 1] + 0.0 * ge_min
    for f in range(N_FAM):
        w = salience[f] * noticed if f in ARITH else salience[f]
        num = num + w * hit[f]
        den = den + w * sh[f]
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
