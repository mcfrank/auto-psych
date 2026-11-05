"""Shared final digit read as a rule, on top of one-sided magnitude bound overlap.

Refinement of one_sided_magnitude_bound_overlap: people note every salient
property that all the examples share (family-specific salience, base-ten
multiples their own family, two one-sided magnitude bounds, no size principle)
and say "yes" in proportion to the salience-weighted share of that profile the
target also has. The one change: when two or more examples share a final digit,
people read it as a self-evident rule, so a target that ends in that digit gets
an extra, undiluted boost toward "yes", whatever other coincidental properties
the examples share.
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
    fd = len(ex) >= 2 and len({e % 10 for e in ex}) == 1
    out["fd_match"] = 1.0 if fd and t % 10 == ex[0] % 10 else 0.0
    out["at_least_min"] = 1.0 if t >= min(ex) else 0.0
    out["at_most_max"] = 1.0 if t <= max(ex) else 0.0
    return out


with pm.Model() as model:
    sh = [pm.Data(f"sh_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    hit = [pm.Data(f"hit_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    fd_match = pm.Data("fd_match", np.zeros(1, dtype="float64"))
    ge_min = pm.Data("at_least_min", np.zeros(1, dtype="float64"))
    le_max = pm.Data("at_most_max", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    fd_boost = pm.HalfNormal("fd_boost", sigma=2.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 2, 1.5))

    # Both one-sided bounds are always shared by every example.
    num = salience[N_FAM] * ge_min + salience[N_FAM + 1] * le_max
    den = salience[N_FAM] + salience[N_FAM + 1] + 0.0 * ge_min
    for f in range(N_FAM):
        num = num + salience[f] * hit[f]
        den = den + salience[f] * sh[f]
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap + fd_boost * fd_match))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
