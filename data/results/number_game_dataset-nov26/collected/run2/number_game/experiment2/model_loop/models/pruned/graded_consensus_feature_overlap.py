"""Graded-consensus feature overlap.

Refinement of final_digit_overshadowing_overlap by one change: people do not
restrict the profile to properties shared by *all* the examples. Every
property held by some of the examples enters the profile, with a weight that
grows with the fraction of examples that hold it (fraction raised to an
inferred consensus exponent; a very large exponent recovers the unanimous-only
incumbent). People still say "yes" in proportion to the salience-weighted share
of that profile the target has (family saliences, base-ten multiples their own
family, two one-sided magnitude bounds, a visible shared final digit
overshadowing the other properties). So a property held by most but not all
examples (three of 69, 9, 39, 21 end in 9) or by half (13, 91 end in 1 or 3)
still pulls generalization toward targets that have it.
"""
import numpy as np
import pymc as pm

_N = range(1, 101)
N_FAM = 8  # family 5 = final digit; bounds are salience entries 8, 9
FD = 5
LEVELS = [(1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (1, 1)]
LOGLEV = np.log(np.array([a / b for a, b in LEVELS]))
N_LEV = len(LEVELS)


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


def _level(k, n):
    for j, (a, b) in enumerate(LEVELS):
        if k * b == a * n:
            return j
    raise ValueError


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    n = len(ex)
    t = int(target)
    out = {}
    for f in range(N_FAM):
        sh = [0.0] * N_LEV
        hit = [0.0] * N_LEV
        for fam, p in _PROPS:
            if fam != f:
                continue
            k = sum(1 for e in ex if e in p)
            if k == 0:
                continue
            j = _level(k, n)
            sh[j] += 1.0
            if t in p:
                hit[j] += 1.0
        for j in range(N_LEV):
            out[f"sh_{f}_{j}"] = sh[j]
            out[f"hit_{f}_{j}"] = hit[j]
    out["at_least_min"] = 1.0 if t >= min(ex) else 0.0
    out["at_most_max"] = 1.0 if t <= max(ex) else 0.0
    out["visible_digit_pattern"] = 1.0 if (n >= 2 and len({e % 10 for e in ex}) == 1) else 0.0
    return out


with pm.Model() as model:
    sh = [[pm.Data(f"sh_{f}_{j}", np.zeros(1, dtype="float64")) for j in range(N_LEV)] for f in range(N_FAM)]
    hit = [[pm.Data(f"hit_{f}_{j}", np.zeros(1, dtype="float64")) for j in range(N_LEV)] for f in range(N_FAM)]
    ge_min = pm.Data("at_least_min", np.zeros(1, dtype="float64"))
    le_max = pm.Data("at_most_max", np.zeros(1, dtype="float64"))
    vis = pm.Data("visible_digit_pattern", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 2, 1.5))
    retain = pm.Beta("retain", alpha=2.0, beta=2.0)
    # consensus exponent: how steeply a property's weight falls as fewer examples hold it
    consensus = pm.Gamma("consensus", alpha=2.0, beta=0.5)

    lw = pm.math.exp(consensus * LOGLEV)  # weight per consensus level, 1 for unanimous
    other_w = 1.0 - (1.0 - retain) * vis
    num = other_w * (salience[N_FAM] * ge_min + salience[N_FAM + 1] * le_max)
    den = other_w * (salience[N_FAM] + salience[N_FAM + 1])
    for f in range(N_FAM):
        w = salience[f] if f == FD else salience[f] * other_w
        hs = sum(lw[j] * hit[f][j] for j in range(N_LEV))
        ss = sum(lw[j] * sh[f][j] for j in range(N_LEV))
        num = num + w * hs
        den = den + w * ss
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
