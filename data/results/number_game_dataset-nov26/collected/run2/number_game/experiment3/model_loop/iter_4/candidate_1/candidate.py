"""Conjunctive (product-rule) exemplar similarity.

People do not infer a rule; they keep each example shown as a stored exemplar
and ask whether the target resembles every one of them at once. Similarity to
each stored example falls off exponentially with the salience-weighted fraction
of that example's own properties (parity, primality, squares/cubes, powers,
multiples, round multiples of 5/10, final digit, contained digits) the target
lacks, plus their magnitude distance. The similarities to the examples are
multiplied (geometric mean, so the scale does not depend on set size): one
badly mismatched example vetoes the target, a property every example carries
counts fully, one carried by only some counts in proportion.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_N = list(range(1, 101))
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
        props.append((7 if m in (5, 10) else 4, {n for n in _N if n % m == 0}))
    for d in range(10):
        props.append((5, {n for n in _N if n % 10 == d}))
    for d in range(10):
        props.append((6, {n for n in _N if str(d) in str(n)}))
    return props


_PROPS = _props()
_FAM = np.array([f for f, _ in _PROPS])
_F = np.zeros((101, len(_PROPS)))
for _j, (_f, _s) in enumerate(_PROPS):
    for _n in _s:
        _F[_n, _j] = 1.0
_OH = np.zeros((len(_PROPS), N_FAM))
_OH[np.arange(len(_PROPS)), _FAM] = 1.0


def prepare_observed(rows):
    n_tr = len(rows)
    lack = np.zeros((n_tr, 4, N_FAM))   # example's properties the target lacks
    has = np.zeros((n_tr, 4, N_FAM))    # example's properties
    mag = np.zeros((n_tr, 4))
    wgt = np.zeros((n_tr, 4))           # 1/n for present slots (geometric mean)
    rating = np.zeros(n_tr, dtype="int64")
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"])
        for k, e in enumerate(ex):
            has[i, k] = _F[e] @ _OH
            lack[i, k] = (_F[e] * (1.0 - _F[t])) @ _OH
            mag[i, k] = abs(t - e) / 100.0
            wgt[i, k] = 1.0 / len(ex)
        rating[i] = int(float(r["rating"]))
    return {"ex_lack": lack.reshape(n_tr, 4 * N_FAM), "ex_has": has.reshape(n_tr, 4 * N_FAM),
            "mag_dist": mag, "slot_weight": wgt, "rating": rating}


with pm.Model() as model:
    lack = pm.Data("ex_lack", np.zeros((1, 4 * N_FAM), dtype="float64"))
    has = pm.Data("ex_has", np.ones((1, 4 * N_FAM), dtype="float64"))
    mag = pm.Data("mag_dist", np.zeros((1, 4), dtype="float64"))
    wgt = pm.Data("slot_weight", np.full((1, 4), 0.25, dtype="float64"))

    salience = pm.Dirichlet("salience", a=np.full(N_FAM, 1.5))
    feat_scale = pm.HalfNormal("feature_scale", sigma=3.0)
    mag_scale = pm.HalfNormal("magnitude_scale", sigma=3.0)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)

    lack3 = lack.reshape((lack.shape[0], 4, N_FAM))
    has3 = has.reshape((has.shape[0], 4, N_FAM))
    w_lack = pt.dot(lack3, salience)
    w_has = pt.dot(has3, salience) + 1e-9
    mismatch = w_lack / w_has                       # fraction of example profile missed
    dist = feat_scale * mismatch + mag_scale * mag  # per-example distance
    log_sim = -(wgt * dist).sum(axis=1)             # log of geometric-mean product similarity
    sim = pt.exp(log_sim)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
