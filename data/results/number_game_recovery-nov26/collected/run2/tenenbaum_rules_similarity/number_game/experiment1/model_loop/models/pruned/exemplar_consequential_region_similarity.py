"""Exemplar generalization through consequential regions (Shepard 1987).

People store each shown example as a separate exemplar and judge the target by
its average similarity to the exemplars. Similarity of target x to one exemplar
y is the probability that a concept containing y also contains x, where the
concepts are the 45 arithmetic/digit rules (independently weighted) and the
intervals of 1..100 (Erlang prior on width), each exemplar sampled from its
concept by the size principle with an inferred share of random examples.
Generalization is computed from each exemplar alone and then averaged, so the
size principle never accumulates jointly over the examples.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (rules, 100)
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Intervals by size s: BOTH[s, y, x] = intervals of size s containing y and x;
# HAS[s, y] = intervals of size s containing y.
SIZES_F = np.arange(1, 101).astype("float64")
N_PER_SIZE = 101.0 - SIZES_F
_lo = np.minimum.outer(NUMBERS, NUMBERS)
_hi = np.maximum.outer(NUMBERS, NUMBERS)
BOTH = np.zeros((100, 100, 100))
for _s in range(1, 101):
    # start m with max(1, hi - s + 1) <= m <= min(lo, 101 - s)
    BOTH[_s - 1] = np.clip(np.minimum(_lo, 101 - _s) - np.maximum(1, _hi - _s + 1) + 1, 0, None)
HAS = np.stack([np.diag(BOTH[s]) for s in range(100)])  # (sizes, 100)
BOTH_FLAT = BOTH.reshape(100, -1)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    cell_ex_weight = pm.Data("cell_ex_weight", np.zeros((1, 100)))  # (cells, 100): 1/n per exemplar
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ])
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_erlang = pt.log(SIZES_F) - SIZES_F / sigma
    per_interval = pt.exp(log_erlang - pt.logsumexp(log_erlang + np.log(N_PER_SIZE)))  # (sizes,)

    l_out = noise / 100.0
    # Rules: weight of each rule given single exemplar y, (rules, y).
    rule_w = (lambda_0 * rule_weights)[:, None] * (
        RULE_MEMBERSHIP * ((1 - noise) / RULE_SIZE)[:, None] + l_out
    )
    rule_num = pt.dot(rule_w.T, RULE_MEMBERSHIP)  # (y, x): mass of rules containing x
    rule_den = rule_w.sum(axis=0)  # (y,)
    # Intervals.
    int_prior = (1 - lambda_0) * per_interval  # (sizes,)
    l_in = (1 - noise) / SIZES_F + l_out
    both = pt.dot(int_prior * (l_in - l_out), BOTH_FLAT).reshape((100, 100))  # (y, x)
    has_x = pt.dot(int_prior * l_out, HAS)  # (x,)
    int_num = both + has_x[None, :]
    int_den = pt.dot(int_prior * (l_in - l_out), HAS) + (int_prior * l_out * N_PER_SIZE).sum()
    gen = (rule_num + int_num) / (rule_den + int_den)[:, None]  # (y, x) generalization

    sim = (cell_ex_weight * gen[:, cell_target].T).sum(axis=1)  # average over exemplars
    p_yes = pm.Deterministic("p_yes", pt.clip(sim[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    keys = [(tuple(int(x) for x in str(r["number_set"]).split(",")), int(r["target"])) for r in rows]
    cells = list(dict.fromkeys(keys))
    index = {k: i for i, k in enumerate(cells)}
    w = np.zeros((len(cells), 100))
    for i, (s, _) in enumerate(cells):
        for x in s:
            w[i, x - 1] += 1.0 / len(s)
    return {
        "cell_ex_weight": w,
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
