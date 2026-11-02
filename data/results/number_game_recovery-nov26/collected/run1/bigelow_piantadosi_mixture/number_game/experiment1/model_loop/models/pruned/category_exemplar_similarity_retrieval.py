"""Category-exemplar similarity retrieval.

People retrieve familiar number categories stored as sets of remembered member
numbers: 45 familiar rules and 210 familiar ranges bounded by multiples of five
([5a+1, 5b], e.g. 21-30, 1-50, 61-75). Each example's match to a category is
its mean similarity to the category's stored members (GCM: similarity
exp(-c |x - y|) on the number line, c fitted), mixed with a 10% chance that the
example is an arbitrary number. A category's activation is its prior weight
times the product of its examples' matches, normalised over categories. The
target is endorsed in proportion to its similarity to the nearest stored member
of each activated category, weighted by activation. Averaging similarity over a
category's members makes small categories match their members more strongly,
and the finite similarity gradient gives near-misses partial credit both in
retrieval and in endorsement.
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
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RANGES = [list(range(5 * a + 1, 5 * b + 1)) for a in range(20) for b in range(a + 1, 21)]
N_RANGES = len(RANGES)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + RANGES], dtype="float64")  # (categories, 100)
SIZE = MEMBERSHIP.sum(axis=1)
ABS_DIFF = np.abs(NUMBERS[:, None] - NUMBERS[None, :]).astype("float64")
# Distance from each target to the nearest stored member of each category.
DMIN = np.array([[np.abs(t - np.array(c)).min() for t in NUMBERS] for c in RULES + RANGES], dtype="float64")

# "ends in 0" and "contains digit 0" are one set: their weights share a
# Dirichlet component split by a Beta (as in the other rule-based models).
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    example_count = pm.Data("example_count", np.zeros((1, 100)))  # (sets, numbers): times each number is an example
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    # Similarity gradient on the number line (specificity c).
    log_c = pm.Normal("log_c", mu=0.0, sigma=1.0)
    c = pm.Deterministic("c", pt.exp(log_c))
    sim = pt.exp(-c * pt.as_tensor(ABS_DIFF))  # (x, y)

    # Mean similarity of each number to each category's stored members.
    match = pt.dot(sim, pt.as_tensor(MEMBERSHIP.T)) / SIZE[None, :]  # (100, categories)
    log_lik = pt.dot(example_count, pt.log(ALPHA * match + (1 - ALPHA) / 100))  # (sets, categories)
    log_prior = pt.concatenate([pt.log(lambda_0) + pt.log(rule_weights),
                                pt.log1p(-lambda_0) + pt.zeros(N_RANGES) - np.log(N_RANGES)])
    log_act = log_prior[None, :] + log_lik
    act = pt.exp(log_act - pt.logsumexp(log_act, axis=1, keepdims=True))

    # Endorsement: activation-weighted similarity of the target to each
    # category's nearest stored member (1 for a member).
    endorse = pt.dot(act, pt.exp(-c * pt.as_tensor(DMIN)))  # (sets, 100)

    p_yes = pm.Deterministic("p_yes", pt.clip(endorse.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    counts = np.zeros((len(unique), 100))
    for i, s in enumerate(unique):
        for x in s:
            counts[i, x - 1] += 1
    return {
        "example_count": counts,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
