"""Similarity by the distance between the most specific shared categories.

People judge a target by its similarity to the examples on a category-distance
metric: they find the most specific familiar number category holding all the
examples (an arithmetic rule, or the stretch of the number line they span) and
the most specific one still holding the examples together with the target. The
distance is how much larger (in log size) the second category is than the
first, where each kind of category (odd, even, primes, squares/cubes, powers,
multiples, final digit, contained digit, multiples of 5) carries its own cost of
being called to mind, added to its log size. "Most specific" is a near-hard
minimum (a soft-min with a small fixed temperature, for smooth sampling). The
yes probability falls off as a logistic gradient of this distance, steeper the
more examples are shown: p = sigmoid(a - b * n**g * distance).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
TEMPERATURE = 0.3  # soft-min temperature, in nats (near the hard minimum)


def rule_categories():
    """(kind, members) for each stored arithmetic category; kinds index KINDS."""
    primes = [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))]
    cats = [
        (0, [n for n in NUMBERS if n % 2 == 1]),
        (1, list(range(2, 101, 2))),
        (2, primes),
        (3, [n * n for n in range(1, 11)]),
        (3, [n**3 for n in range(1, 5)]),
        (4, [2, 4, 8, 16, 64]),
        (4, [2, 4, 8, 16, 32, 37, 64]),
    ]
    cats += [(8 if m == 5 else 5, list(range(m, 101, m))) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    cats += [(4, [m**k for k in range(1, 8) if m**k <= 100]) for m in range(2, 11)]
    cats += [(6, [n for n in NUMBERS if n % 10 == d]) for d in range(10)]
    cats += [(7, [n for n in NUMBERS if str(d) in str(n)]) for d in range(10)]
    return cats


KINDS = ["odd", "even", "primes", "squares_cubes", "powers", "multiples", "final_digit", "contains_digit", "multiples_of_5"]
_CATS = rule_categories()
CAT_KIND = np.array([k for k, _ in _CATS], dtype="int64")
N_CATS = len(_CATS)
CAT_MEMBER = np.array([np.isin(NUMBERS, c) for _, c in _CATS])  # (cats, 100)
CAT_LOG_SIZE = np.log(CAT_MEMBER.sum(axis=1)).astype("float64")


def _soft_min(cover, span_log_size, cost):
    """Soft-min over the covering rules (log size + kind cost) and the spanning stretch."""
    rule_cost = pt.switch(cover > 0.5, CAT_LOG_SIZE[None, :] + cost[CAT_KIND][None, :], 1e4)
    v = pt.concatenate([rule_cost, span_log_size[:, None]], axis=1)
    return -TEMPERATURE * pt.logsumexp(-v / TEMPERATURE, axis=1)


with pm.Model() as model:
    set_cover = pm.Data("set_cover", np.zeros((1, N_CATS)))  # (cells, cats): rule holds all examples
    joint_cover = pm.Data("joint_cover", np.zeros((1, N_CATS)))  # (cells, cats): ... and the target
    set_span = pm.Data("set_span", np.zeros(1))  # log size of the examples' stretch
    joint_span = pm.Data("joint_span", np.zeros(1))  # log size of the stretch with the target
    cell_n = pm.Data("cell_n", np.ones(1))  # number of examples
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Cost of calling each kind of category to mind (log units, relative to a stretch).
    kind_cost = pm.Normal("kind_cost", mu=0.0, sigma=3.0, shape=len(KINDS))
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    slope = pm.LogNormal("slope", mu=0.0, sigma=1.0)
    set_size_exp = pm.Normal("set_size_exp", mu=0.5, sigma=0.5)

    distance = _soft_min(joint_cover, joint_span, kind_cost) - _soft_min(set_cover, set_span, kind_cost)
    cell_p = pm.math.sigmoid(intercept - slope * cell_n**set_size_exp * distance)
    p_yes = pm.Deterministic("p_yes", pt.clip(cell_p, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Per distinct (set, target) cell: which rules cover the examples (and the target)."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(float(r["target"])) for r in rows]
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: k for k, c in enumerate(cells)}
    set_cover = np.zeros((len(cells), N_CATS))
    joint_cover = np.zeros((len(cells), N_CATS))
    set_span = np.zeros(len(cells))
    joint_span = np.zeros(len(cells))
    for k, (s, t) in enumerate(cells):
        xs = np.array(s) - 1
        set_cover[k] = CAT_MEMBER[:, xs].all(axis=1)
        joint_cover[k] = set_cover[k] * CAT_MEMBER[:, t - 1]
        set_span[k] = np.log(xs.max() - xs.min() + 1)
        joint_span[k] = np.log(max(xs.max(), t - 1) - min(xs.min(), t - 1) + 1)
    return {
        "set_cover": set_cover,
        "joint_cover": joint_cover,
        "set_span": set_span,
        "joint_span": joint_span,
        "cell_n": np.array([float(len(s)) for s, _ in cells]),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(float(r.get("rating", 0) or 0)) for r in rows], dtype="int64"),
    }
