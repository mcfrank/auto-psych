"""Weak-sampling concept learning (no size principle).

People entertain the same kinds of number concepts (45 arithmetic rules and
every interval of 1-100) but treat the examples only as consistent with the
hidden concept rather than as random samples from it: each example counts as
evidence only through whether the concept contains it (with a small chance
eps that an example falls outside the concept), never through how many other
numbers the concept contains. Every concept containing the examples therefore
stays plausible in proportion to its prior, so the model predicts broad "yes"
to targets only a larger concept contains (other evens for 16,32,2,8, wide
ranges for 20,32,92), where the size-principle model predicts narrow "yes".
"Yes" is the posterior probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4


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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
LO, HI = np.meshgrid(NUMBERS, NUMBERS, indexing="ij")
VALID = LO <= HI
INT_LO, INT_HI = LO[VALID], HI[VALID]  # the 5,050 intervals
INT_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")

# "ends in 0" and "contains digit 0" are one set; as in the size-principle
# model, sample one weight per distinct set and a Beta(2, 2) split of it
# (exactly a Dirichlet(2, ..., 2) prior on the 45 weights by aggregation).
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Weak sampling: P(D | h) = prod_x [(1 - eps) 1(x in h) + eps], which depends
# only on k, the number of examples outside h. Intervals share one prior, so
# per set they reduce to a count of intervals by k (and, per target, a count
# of those containing the target), computed once in prepare_observed.

with pm.Model() as model:
    rule_outside = pm.Data("rule_outside", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules): k
    set_size = pm.Data("set_size", np.ones(1, dtype="int64"))  # (sets,)
    interval_k_count = pm.Data("interval_k_count", np.zeros((1, MAX_N + 1)))  # (sets, k)
    interval_k_target = pm.Data("interval_k_target", np.zeros((1, MAX_N + 1, 100)))  # (sets, k, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    eps = pm.Beta("eps", 1, 9)  # chance an example lies outside the concept
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ])

    log_eps, log_keep = pt.log(eps), pt.log1p(-eps)
    k_vals = pt.arange(MAX_N + 1)
    # log P(D | h) for k outside of n examples: (sets, k)
    log_lik_k = k_vals[None, :] * log_eps + (set_size[:, None] - k_vals[None, :]) * log_keep
    rule_ll = rule_outside * log_eps + (set_size[:, None] - rule_outside) * log_keep

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll
    log_int_terms = pt.log(interval_k_count + 1e-300) + log_lik_k
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.logsumexp(log_int_terms, axis=1)
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_z = pt.logsumexp(log_joint, axis=1, keepdims=True)
    post_rules = pt.exp(log_rules - log_z)
    # interval posterior mass on each target: sum_k count_k(t) * lik_k * prior / Z
    int_w = pt.exp(pt.log1p(-lambda_0) - 2 * np.log(100) + log_lik_k - log_z)  # (sets, k)
    int_in = (int_w[:, :, None] * interval_k_target).sum(axis=1)  # (sets, targets)
    in_concept = post_rules @ pt.as_tensor(RULE_MEMBERSHIP) + int_in

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_out, sizes, k_count, k_target = [], [], [], []
    for s in unique:
        cols = np.array(s) - 1
        rule_out.append(len(s) - RULE_MEMBERSHIP[:, cols].sum(axis=1))
        sizes.append(len(s))
        k_int = (len(s) - INT_MEMBERSHIP[:, cols].sum(axis=1)).astype(int)
        onehot = np.eye(MAX_N + 1)[k_int]  # (intervals, k)
        k_count.append(onehot.sum(axis=0))
        k_target.append(onehot.T @ INT_MEMBERSHIP)
    return {
        "rule_outside": np.array(rule_out, dtype="int64"),
        "set_size": np.array(sizes, dtype="int64"),
        "interval_k_count": np.array(k_count),
        "interval_k_target": np.array(k_target),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
