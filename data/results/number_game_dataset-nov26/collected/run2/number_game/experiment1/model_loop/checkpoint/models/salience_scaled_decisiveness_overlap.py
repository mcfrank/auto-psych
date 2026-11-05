"""Salience-scaled decisiveness feature overlap.

Refinement of one_sided_magnitude_bound_overlap: people note every salient
property all the examples share (family-specific salience, base-ten multiples
their own family, two one-sided magnitude bounds, no size principle) and judge
a target by the salience-weighted share of that profile it also has. The one
change: how decisively that share drives a "yes" grows with the total salience
of the shared profile, so a target fitting a highly noticeable common pattern
(a shared final digit, powers of one base) is endorsed more confidently than a
target fully fitting a profile of weak properties (merely even).
"""
import numpy as np
import pymc as pm

_N = range(1, 101)
N_FAM = 8  # rule families; the two one-sided bounds are salience weights 8, 9


def _props():
    props = []  # (family index, set)
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
    return out


with pm.Model() as model:
    sh = [pm.Data(f"sh_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    hit = [pm.Data(f"hit_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    ge_min = pm.Data("at_least_min", np.zeros(1, dtype="float64"))
    le_max = pm.Data("at_most_max", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    # How much decisiveness grows per unit of shared-profile salience
    # (salience mass measured relative to the average weight, 1/10).
    mass_gain = pm.HalfNormal("mass_gain", sigma=1.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 2, 1.5))

    num = salience[N_FAM] * ge_min + salience[N_FAM + 1] * le_max
    den = salience[N_FAM] + salience[N_FAM + 1] + 0.0 * ge_min
    for f in range(N_FAM):
        num = num + salience[f] * hit[f]
        den = den + salience[f] * sh[f]
    overlap = num / den
    decisiveness = slope + mass_gain * (N_FAM + 2) * den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + decisiveness * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
