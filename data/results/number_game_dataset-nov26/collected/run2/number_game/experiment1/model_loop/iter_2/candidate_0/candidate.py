"""Position-weighted feature tally.

People build up their sense of which mathematical properties matter by reading
the examples in order and keeping a running tally of each property (parity,
primes, squares/cubes, powers, multiples, final digit, contained digit), in
which examples at certain positions count more than others (recent ones if
memory fades, the first ones if the first example anchors the guess). They say
"yes" in proportion to how much of that order-weighted, family-salience-weighted
property profile (plus lying within the examples' span) the target also has.

Position weight of example j (0-based, n examples): exp(gamma * (j - (n-1)));
gamma > 0 is recency, gamma < 0 primacy. A property's strength is the
position-weighted fraction of examples having it, sharpened by a power k
(large k -> only properties shared by all examples count, as in the
incumbent family_salience_feature_overlap, where order is irrelevant).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_N = range(1, 101)
N_FAM = 7
MAX_EX = 4


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
N_P = len(_PROPS)
FAM_OH = np.zeros((N_P, N_FAM))
for _i, (_f, _s) in enumerate(_PROPS):
    FAM_OH[_i, _f] = 1.0


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    sidx = {s: i for i, s in enumerate(unique)}
    S, T = len(unique), len(rows)
    has = np.zeros((S, MAX_EX, N_P))
    lag = np.zeros((S, MAX_EX))
    present = np.zeros((S, MAX_EX))
    for i, s in enumerate(unique):
        n = len(s)
        for j, e in enumerate(s):
            present[i, j] = 1.0
            lag[i, j] = float(n - 1 - j)
            for p, (_, ps) in enumerate(_PROPS):
                has[i, j, p] = 1.0 if e in ps else 0.0
    onehot = np.zeros((T, S))
    hit = np.zeros((T, N_P))
    in_span = np.zeros(T)
    for t, (s, r) in enumerate(zip(sets, rows)):
        onehot[t, sidx[s]] = 1.0
        tg = int(r["target"])
        for p, (_, ps) in enumerate(_PROPS):
            hit[t, p] = 1.0 if tg in ps else 0.0
        in_span[t] = 1.0 if min(s) <= tg <= max(s) else 0.0
    return {
        "ex_has": has.reshape(S, MAX_EX * N_P),
        "ex_lag": lag,
        "ex_present": present,
        "set_onehot": onehot,
        "target_hit": hit,
        "in_span_t": in_span,
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    ex_has = pm.Data("ex_has", np.zeros((1, MAX_EX * N_P)))
    ex_lag = pm.Data("ex_lag", np.zeros((1, MAX_EX)))
    ex_present = pm.Data("ex_present", np.ones((1, MAX_EX)))
    set_onehot = pm.Data("set_onehot", np.ones((1, 1)))
    target_hit = pm.Data("target_hit", np.zeros((1, N_P)))
    in_span = pm.Data("in_span_t", np.zeros(1))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 1, 1.5))
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)
    k = pm.LogNormal("k", mu=np.log(3.0), sigma=0.7)

    w = ex_present * pt.exp(-gamma * ex_lag)  # (S, 4)
    w = w / pt.sum(w, axis=1, keepdims=True)
    has3 = ex_has.reshape((ex_has.shape[0], MAX_EX, N_P))
    strength = pt.sum(w[:, :, None] * has3, axis=1)  # (S, P)
    s_eff = pt.switch(strength > 1e-9, pt.exp(k * pt.log(pt.maximum(strength, 1e-9))), 0.0)
    sal_p = pt.dot(FAM_OH, salience[:N_FAM])  # (P,)
    wprop = s_eff * sal_p[None, :]  # (S, P)
    wprop_t = pt.dot(set_onehot, wprop)  # (T, P)

    num = pt.sum(wprop_t * target_hit, axis=1) + salience[N_FAM] * in_span
    den = pt.sum(wprop_t, axis=1) + salience[N_FAM]
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
