"""Directional span salience overlap.

Refinement of decimal_multiple_salience_overlap: people note every salient
mathematical property all the examples share (multiples of 5/10 a separate
salience family) and say "yes" in proportion to the salience-weighted share of
that profile the target also has, with no size principle. The one change: the
"within the examples' span" property is graded and directional - its match
decays with the target's distance beyond the examples, at separately inferred
rates above the largest example and below the smallest, so people extend
upward more readily than downward.
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
    out["d_above"] = max(0, t - max(ex)) / 100.0
    out["d_below"] = max(0, min(ex) - t) / 100.0
    return out


with pm.Model() as model:
    sh = [pm.Data(f"sh_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    hit = [pm.Data(f"hit_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    d_above = pm.Data("d_above", np.zeros(1, dtype="float64"))
    d_below = pm.Data("d_below", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 1, 1.5))
    log_rate_up = pm.Normal("log_rate_up", mu=2.0, sigma=1.5)
    log_rate_down = pm.Normal("log_rate_down", mu=2.0, sigma=1.5)

    span_match = pt.exp(-pt.exp(log_rate_up) * d_above - pt.exp(log_rate_down) * d_below)
    num = salience[N_FAM] * span_match
    den = salience[N_FAM] + 0.0 * span_match
    for f in range(N_FAM):
        num = num + salience[f] * hit[f]
        den = den + salience[f] * sh[f]
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
