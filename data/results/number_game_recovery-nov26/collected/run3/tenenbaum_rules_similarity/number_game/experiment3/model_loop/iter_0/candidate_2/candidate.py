"""Conjunctive shared-feature exemplar similarity.

People do not infer a hidden concept; they compare the target with each stored
example by additive-clustering similarity: the summed weight of familiar number
properties (parity, multiples, powers, squares/cubes, primes, final digit,
digits contained) the two numbers share, each property weighted by the salience
of its kind and by its distinctiveness (properties held by few numbers count
more, to an inferred degree), plus a magnitude-closeness term. Each similarity
is normalised by the example's self-similarity, and the target is accepted to
the extent it resembles every example: the log-odds of "yes" grows with the
mean log normalised similarity (a geometric mean over examples).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4


def property_sets():
    props = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
    }
    props.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    props.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    props.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    props.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return props


def kind(name):
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers"):
        return 3
    if name.startswith("multiples"):
        return 4
    if name.startswith("ends in"):
        return 5
    return 6


_P = property_sets()
NAMES = list(_P)
MEMBER = np.array([np.isin(NUMBERS, _P[n]) for n in NAMES], dtype="float64")  # (props, 100)
N_PROPS = len(NAMES)
KIND = np.array([kind(n) for n in NAMES])
N_KINDS = KIND.max() + 1
KIND_SIZE = np.bincount(KIND)
LOG_PROP_SIZE = np.log(MEMBER.sum(axis=1) / 100.0)

with pm.Model() as model:
    shared = pm.Data("shared", np.zeros((1, MAX_EX, N_PROPS)))  # (cells, ex, props): target & example share
    self_props = pm.Data("self_props", np.zeros((1, MAX_EX, N_PROPS)))  # (cells, ex, props): example's own
    distance = pm.Data("distance", np.zeros((1, MAX_EX)))  # |target - example| / 100
    ex_mask = pm.Data("ex_mask", np.zeros((1, MAX_EX)))  # 1 for real examples
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # salience of each kind of property, plus magnitude as one more dimension
    salience = pm.Dirichlet("salience", a=np.full(N_KINDS + 1, 2.0))
    distinctiveness = pm.Normal("distinctiveness", mu=1.0, sigma=0.5)
    closeness_width = pm.LogNormal("closeness_width", mu=np.log(0.1), sigma=0.7)
    intercept = pm.Normal("intercept", mu=1.0, sigma=1.5)
    sensitivity = pm.HalfNormal("sensitivity", sigma=2.0)

    w_prop = salience[KIND] / KIND_SIZE[KIND] * pt.exp(-distinctiveness * LOG_PROP_SIZE)
    w_prop = w_prop / w_prop.sum()
    w_mag = salience[N_KINDS]
    sim = (shared * w_prop).sum(axis=2) + w_mag * pt.exp(-distance / closeness_width)
    self_sim = (self_props * w_prop).sum(axis=2) + w_mag
    log_rel = pt.log(sim + 1e-9) - pt.log(self_sim)
    mean_log_sim = (log_rel * ex_mask).sum(axis=1) / ex_mask.sum(axis=1)

    logit = intercept + sensitivity * mean_log_sim
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(logit[row_cell]), 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    keys = [(tuple(int(x) for x in str(r["number_set"]).split(",")), int(r["target"])) for r in rows]
    cells = list(dict.fromkeys(keys))
    index = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    shared = np.zeros((n, MAX_EX, N_PROPS))
    self_props = np.zeros((n, MAX_EX, N_PROPS))
    distance = np.zeros((n, MAX_EX))
    mask = np.zeros((n, MAX_EX))
    for i, (s, t) in enumerate(cells):
        for j, x in enumerate(s[:MAX_EX]):
            shared[i, j] = MEMBER[:, x - 1] * MEMBER[:, t - 1]
            self_props[i, j] = MEMBER[:, x - 1]
            distance[i, j] = abs(t - x) / 100.0
            mask[i, j] = 1.0
    return {
        "shared": shared,
        "self_props": self_props,
        "distance": distance,
        "ex_mask": mask,
        "row_cell": np.array([index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
