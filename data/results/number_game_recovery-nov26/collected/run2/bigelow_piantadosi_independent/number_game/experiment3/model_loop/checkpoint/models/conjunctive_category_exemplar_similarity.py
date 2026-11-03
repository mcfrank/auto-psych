"""Exemplar comparison through shared stored number categories (additive-clustering similarity).

People do not infer a single hidden concept. They compare the target with each
remembered example in turn, and two numbers feel similar to the extent that they
are co-members of the number categories stored in memory -- arithmetic
categories (odd, even, primes, squares/cubes, powers, multiples, final digit,
contained digit, multiples of 5) and stretches of the number line -- with small,
distinctive shared categories counting more than broad ones (Shepard & Arabie's
additive clustering: sim(t, x) = sum over stored categories containing both of
weight_kind * size ** (-s)). Because every example must be matched, the
comparisons combine conjunctively: the yes-probability rises with the target's
geometric-mean similarity to the examples, p_yes = sigmoid(b0 + b1 * mean_i log sim(t, x_i)).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4


def rule_categories():
    """(kind, members) for each stored arithmetic category."""
    primes = [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))]
    cats = [
        (0, [n for n in NUMBERS if n % 2 == 1]),
        (1, list(range(2, 101, 2))),
        (2, primes),
        (3, [n * n for n in range(1, 11)]),
        (3, [n**3 for n in range(1, 5)]),
    ]
    cats += [(8 if m == 5 else 5, list(range(m, 101, m))) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    cats += [(4, [m**k for k in range(1, 8) if m**k <= 100]) for m in range(2, 11)]
    cats += [(6, [n for n in NUMBERS if n % 10 == d]) for d in range(10)]
    cats += [(7, [n for n in NUMBERS if str(d) in str(n)]) for d in range(10)]
    return cats


_CATS = rule_categories()
N_KINDS = 9
CAT_KIND = np.array([k for k, _ in _CATS], dtype="int64")
CAT_MEMBER = np.array([np.isin(NUMBERS, c) for _, c in _CATS], dtype="float64")  # (cats, 100)
CAT_LOG_SIZE = np.log(CAT_MEMBER.sum(axis=1))
LENGTHS = np.arange(1, 101, dtype="float64")


def _interval_counts(t, x):
    """Number of stretches of 1-100 of each length that contain both t and x."""
    a, b = min(t, x), max(t, x)
    lo = np.maximum(1, b - LENGTHS + 1)
    hi = np.minimum(a, 101 - LENGTHS)
    return np.maximum(0.0, hi - lo + 1)


with pm.Model() as model:
    # (cells, examples, categories): 1 if target and example share the category.
    shared_cat = pm.Data("shared_cat", np.zeros((1, MAX_N, len(_CATS))))
    # (cells, examples, lengths): number of shared stretches of each length.
    shared_int = pm.Data("shared_int", np.zeros((1, MAX_N, 100)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAX_N)))  # real (1) vs padding (0) example slots
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Salience of each kind of stored arithmetic category, relative to stretches of the number line.
    kind_weight = pm.Normal("kind_weight", mu=0.0, sigma=2.0, shape=N_KINDS)
    # How much more a small shared category counts than a broad one.
    distinct = pm.HalfNormal("distinct", sigma=1.5)
    b0 = pm.Normal("b0", mu=0.0, sigma=3.0)
    b1 = pm.HalfNormal("b1", sigma=2.0)

    cat_w = pt.exp(kind_weight[CAT_KIND] - distinct * CAT_LOG_SIZE)  # (cats,)
    len_w = pt.exp(-distinct * np.log(LENGTHS))  # (lengths,)
    sim = pt.dot(shared_cat, cat_w) + pt.dot(shared_int, len_w)  # (cells, examples)
    log_sim = pt.log(sim + 1e-12) * ex_mask
    mean_log_sim = log_sim.sum(axis=1) / ex_mask.sum(axis=1)

    p_cell = pm.math.sigmoid(b0 + b1 * mean_log_sim)
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Per distinct (set, target) cell, the categories and stretches the target shares with each example."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(float(r["target"])) for r in rows]
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: k for k, c in enumerate(cells)}
    shared_cat = np.zeros((len(cells), MAX_N, len(_CATS)))
    shared_int = np.zeros((len(cells), MAX_N, 100))
    mask = np.zeros((len(cells), MAX_N))
    for k, (s, t) in enumerate(cells):
        for i, x in enumerate(s[:MAX_N]):
            shared_cat[k, i] = CAT_MEMBER[:, t - 1] * CAT_MEMBER[:, x - 1]
            shared_int[k, i] = _interval_counts(t, x)
            mask[k, i] = 1.0
    return {
        "shared_cat": shared_cat,
        "shared_int": shared_int,
        "ex_mask": mask,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
