"""Directional span salience overlap.

Refinement of decimal_multiple_salience_overlap: people note every salient
mathematical property all examples share (multiples of 5/10 as their own
salience family) and say "yes" in proportion to the salience-weighted share of
that profile the target also has, with no size principle. The one change: the
"within the examples' span" property is graded, not all-or-none — a target
outside the span keeps credit that decays with its distance from the span, at
separate inferred rates above the largest example and below the smallest, so
people may extend upward more readily than downward.
"""
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
    out["dist_above"] = max(0, t - max(ex)) / 100.0
    out["dist_below"] = max(0, min(ex) - t) / 100.0
    return out


with pm.Model() as model:
    sh = [pm.Data(f"sh_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    hit = [pm.Data(f"hit_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    dist_above = pm.Data("dist_above", np.zeros(1, dtype="float64"))
    dist_below = pm.Data("dist_below", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 1, 1.5))
    # decay rates of span credit per unit distance (distance scaled to 0-1)
    decay_up = pm.HalfNormal("decay_up", sigma=20.0)
    decay_down = pm.HalfNormal("decay_down", sigma=20.0)

    span_credit = pt.exp(-decay_up * dist_above - decay_down * dist_below)
    num = salience[N_FAM] * span_credit
    den = salience[N_FAM] + 0.0 * span_credit
    for f in range(N_FAM):
        num = num + salience[f] * hit[f]
        den = den + salience[f] * sh[f]
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
