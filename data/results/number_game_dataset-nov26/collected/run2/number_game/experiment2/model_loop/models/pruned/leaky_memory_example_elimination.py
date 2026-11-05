"""Leaky-memory example elimination.

People generalize like a weak-sampling Bayesian learner over salient number
properties (salience-weighted overlap; a visible shared final digit
overshadows the other properties; two one-sided magnitude bounds), with one
resource limit: each earlier example is remembered -- and so eliminates the
properties it lacks -- only with probability retention**lag, where lag is the
number of examples shown after it (the last example is always remembered).
A property's expected weight is its salience times the probability that no
remembered example contradicts it, so properties of the last example that an
earlier example breaks keep partial weight: a recency effect.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_N = range(1, 101)
N_FAM = 8
FD = 5


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
J = len(_PROPS)
FAM_ONEHOT = np.zeros((J, N_FAM))
for _j, (_f, _p) in enumerate(_PROPS):
    FAM_ONEHOT[_j, _f] = 1.0
IS_FD = FAM_ONEHOT[:, FD]


def prepare_observed(rows):
    sets, set_idx = {}, []
    for r in rows:
        set_idx.append(sets.setdefault(str(r["number_set"]), len(sets)))
    S = len(sets)
    alive = np.zeros((S, J))          # property consistent with last example
    viol = np.zeros((S, J, 3))        # violated by example at lag 1, 2, 3
    vis = np.zeros(S)
    for s, (k, i) in enumerate(sets.items()):
        ex = [int(x) for x in k.split(",")]
        n = len(ex)
        vis[i] = 1.0 if (n >= 2 and len({e % 10 for e in ex}) == 1) else 0.0
        for j, (_f, p) in enumerate(_PROPS):
            alive[i, j] = 1.0 if ex[-1] in p else 0.0
            for pos, e in enumerate(ex[:-1]):
                lag = n - 1 - pos
                if e not in p:
                    viol[i, j, lag - 1] += 1.0
    T = len(rows)
    hit = np.zeros((T, J))
    ge = np.zeros(T)
    le = np.zeros(T)
    for t, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        tg = int(r["target"])
        for j, (_f, p) in enumerate(_PROPS):
            hit[t, j] = 1.0 if tg in p else 0.0
        ge[t] = 1.0 if tg >= min(ex) else 0.0
        le[t] = 1.0 if tg <= max(ex) else 0.0
    return {
        "set_idx": np.asarray(set_idx, dtype="int64"),
        "alive": alive, "viol": viol.reshape(S, J * 3), "visible_digit_pattern": vis,
        "target_hit": hit, "at_least_min": ge, "at_most_max": le,
        "rating": np.asarray([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    alive = pm.Data("alive", np.zeros((1, J)))
    viol = pm.Data("viol", np.zeros((1, J * 3)))
    vis = pm.Data("visible_digit_pattern", np.zeros(1))
    hit = pm.Data("target_hit", np.zeros((1, J)))
    ge_min = pm.Data("at_least_min", np.zeros(1))
    le_max = pm.Data("at_most_max", np.zeros(1))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 2, 1.5))
    retain = pm.Beta("retain", alpha=2.0, beta=2.0)
    # probability an example at lag 1, 2, 3 is still remembered
    memory = pm.Beta("memory", alpha=3.0, beta=1.5)

    v = viol.reshape((viol.shape[0], J, 3))
    lags = np.array([1.0, 2.0, 3.0])
    forget = pt.clip(1.0 - memory ** lags, 1e-9, 1.0)           # (3,)
    survive = alive * pt.exp(pt.sum(v * pt.log(forget), axis=2))  # (S, J)
    other_w = 1.0 - (1.0 - retain) * vis                          # (S,)
    fam_sal = pt.dot(FAM_ONEHOT, salience[:N_FAM])               # (J,)
    w_prop = fam_sal[None, :] * (IS_FD[None, :] + (1.0 - IS_FD[None, :]) * other_w[:, None])
    w_set = w_prop * survive                                      # (S, J)

    w_t = w_set[set_idx]                                          # (T, J)
    ow_t = other_w[set_idx]
    num = pt.sum(w_t * hit, axis=1) + ow_t * (salience[N_FAM] * ge_min + salience[N_FAM + 1] * le_max)
    den = pt.sum(w_t, axis=1) + ow_t * (salience[N_FAM] + salience[N_FAM + 1])
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
