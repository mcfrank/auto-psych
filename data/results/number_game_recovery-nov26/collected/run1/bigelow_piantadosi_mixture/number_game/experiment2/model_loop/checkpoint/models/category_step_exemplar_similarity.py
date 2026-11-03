"""Category-step exemplar similarity.

People do not infer a concept; they retrieve familiar number categories stored
in memory as ordered lists of their members (45 familiar rules and 210 ranges
bounded by multiples of five), each activated by how similar the examples are
to its stored members (GCM similarity exp(-c |x - y|), averaged over members,
10% chance an example is arbitrary). They then endorse a target by its exemplar
similarity to the nearest example measured *along* each activated category:
the number of steps through the category's member list from an example to the
target, as a fraction of the category's size, with similarity
exp(-s * fraction) (and exp(-c * distance) for a non-member's distance to the
category's nearest member). A member a step or two from an example is endorsed
readily; one many steps away (96 after a lone 46 in "ends in 6") much less.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
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
RULE_NAMES = list(_RULES)
RULES = [sorted(set(int(x) for x in v)) for v in _RULES.values()]
N_RULES = len(RULES)
IS_DIGIT = np.array([n.startswith("ends in") or n.startswith("contains") for n in RULE_NAMES], dtype="float64")
RANGES = [list(range(5 * a + 1, 5 * b + 1)) for a in range(20) for b in range(a + 1, 21)]
N_RANGES = len(RANGES)
CATEGORIES = RULES + RANGES
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in CATEGORIES], dtype="float64")  # (K, 100)
SIZE = MEMBERSHIP.sum(axis=1)
ABS_DIFF = np.abs(NUMBERS[:, None] - NUMBERS[None, :]).astype("float64")

# For each category and number: distance to, and rank of, the nearest member.
_DMIN = np.zeros((len(CATEGORIES), 100))
_RANK = np.zeros((len(CATEGORIES), 100), dtype="int64")
for _k, _c in enumerate(CATEGORIES):
    _arr = np.array(_c)
    for _t in NUMBERS:
        _j = int(np.argmin(np.abs(_arr - _t)))
        _DMIN[_k, _t - 1] = abs(_arr[_j] - _t)
        _RANK[_k, _t - 1] = _j


def prepare_observed(rows):
    """Distinct sets, distinct (set, target) cells, and per-cell step fractions."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    sidx = {s: i for i, s in enumerate(unique)}
    counts = np.zeros((len(unique), 100))
    for i, s in enumerate(unique):
        for x in s:
            counts[i, x - 1] += 1
    cells = list(dict.fromkeys((s, int(r["target"])) for s, r in zip(sets, rows)))
    cidx = {c: i for i, c in enumerate(cells)}
    steps = np.zeros((len(cells), len(CATEGORIES)))
    dmin = np.zeros((len(cells), len(CATEGORIES)))
    for i, (s, t) in enumerate(cells):
        ex = np.array(s) - 1
        rt = _RANK[:, t - 1][:, None]
        re_ = _RANK[:, ex]
        steps[i] = np.abs(rt - re_).min(axis=1) / SIZE
        dmin[i] = _DMIN[:, t - 1]
    return {
        "example_count": counts,
        "cell_set": np.array([sidx[s] for s, _ in cells], dtype="int64"),
        "cell_steps": steps,
        "cell_dmin": dmin,
        "row_cell": np.array([cidx[(s, int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    example_count = pm.Data("example_count", np.zeros((1, 100)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_steps = pm.Data("cell_steps", np.zeros((1, len(CATEGORIES))))
    cell_dmin = pm.Data("cell_dmin", np.zeros((1, len(CATEGORIES))))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior share of rules vs ranges, and relative weight of digit rules.
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_digit_w = pm.Normal("log_digit_w", 0.0, 1.0)
    # Number-line similarity specificity and along-category step decay.
    log_c = pm.Normal("log_c", 0.0, 1.0)
    log_s = pm.Normal("log_s", 0.0, 1.0)
    c = pt.exp(log_c)
    s = pt.exp(log_s)

    rule_logw = log_digit_w * IS_DIGIT
    rule_logp = rule_logw - pt.logsumexp(rule_logw)
    log_prior = pt.concatenate([pt.log(lambda_0) + rule_logp,
                                pt.log1p(-lambda_0) + pt.zeros(N_RANGES) - np.log(N_RANGES)])

    # Retrieval: examples' mean similarity to each category's stored members.
    sim = pt.exp(-c * pt.as_tensor(ABS_DIFF))
    match = pt.dot(sim, pt.as_tensor(MEMBERSHIP.T)) / SIZE[None, :]  # (100, K)
    log_lik = pt.dot(example_count, pt.log(ALPHA * match + (1 - ALPHA) / 100))  # (sets, K)
    log_act = log_prior[None, :] + log_lik
    act = pt.exp(log_act - pt.logsumexp(log_act, axis=1, keepdims=True))

    # Endorsement: similarity to the nearest example along each category.
    endorse_k = pt.exp(-c * cell_dmin - s * cell_steps)  # (cells, K)
    endorse = (act[cell_set] * endorse_k).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(endorse[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
