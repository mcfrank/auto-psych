"""Seed-multiple salience overlap.

People notice whether the smallest example divides every other example
(7, 63; 9, 81; 8, 64). If it does, they read that smallest number as the
program's seed, and "multiples of the seed" becomes a salient property of the
set with its own salience. Otherwise they judge as in the salience-weighted
feature-overlap account: they note every salient property all the examples
share (family-specific salience, base-ten multiples their own family, one-sided
magnitude bounds, no size principle) and say "yes" in proportion to the
salience-weighted share of that profile the target also has. So a multiple of
the seed is endorsed even when it breaks other shared properties such as
parity.
"""
import numpy as np
import pymc as pm

_N = range(1, 101)
N_FAM = 8  # rule families; salience slots 8, 9 = one-sided bounds, 10 = seed


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
    seed = min(ex)
    is_seed = seed >= 2 and all(e % seed == 0 for e in ex)
    out["seed_shared"] = 1.0 if is_seed else 0.0
    out["seed_hit"] = 1.0 if (is_seed and t % seed == 0) else 0.0
    return out


with pm.Model() as model:
    sh = [pm.Data(f"sh_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    hit = [pm.Data(f"hit_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    ge_min = pm.Data("at_least_min", np.zeros(1, dtype="float64"))
    le_max = pm.Data("at_most_max", np.zeros(1, dtype="float64"))
    seed_shared = pm.Data("seed_shared", np.zeros(1, dtype="float64"))
    seed_hit = pm.Data("seed_hit", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 3, 1.5))

    # Both one-sided bounds are always shared; the seed property only when
    # the smallest example divides all the others.
    num = (salience[N_FAM] * ge_min + salience[N_FAM + 1] * le_max
           + salience[N_FAM + 2] * seed_hit)
    den = salience[N_FAM] + salience[N_FAM + 1] + salience[N_FAM + 2] * seed_shared
    for f in range(N_FAM):
        num = num + salience[f] * hit[f]
        den = den + salience[f] * sh[f]
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
