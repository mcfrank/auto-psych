"""Serial-position weighting of evidence in Bayesian concept learning.

People do Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11
and 12, ending in 0-9, containing the digit 0-9; rule prior organised by
family, with inferred family weights) and the 5,050 intervals of 1..100
(share lambda_0 for rules), with the size principle (10% of examples random).
The single change from `rule_family_prior_concepts`: the examples are read
and integrated sequentially, and each example's log-likelihood is weighted by
exp(-delta * (number of examples shown after it)). delta > 0 is recency
(earlier examples partly forgotten, longer sets narrow generalisation less);
delta < 0 is primacy (early examples overcounted, longer sets narrow more);
delta = 0 recovers the incumbent. "Yes" is the posterior probability that
the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
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
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
N_INTERVALS = len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
# log P(x | h) of one example x (column x - 1) under each hypothesis.
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


def rule_family(name):
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


FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_SIZE = MEMBERSHIP[N_RULES:].sum(axis=1).astype(int)

# Within one example set, intervals with the same size that contain the same
# examples have the same (position-weighted) likelihood for every delta, so
# prepare_observed pools each such group into one hypothesis carrying its
# interval count and summed membership. The posterior is exactly the one over
# all 5,095 hypotheses, at a fraction of the cost per gradient.

with pm.Model() as model:
    rule_pos_log_lik = pm.Data("rule_pos_log_lik", np.zeros((1, MAX_N, N_RULES)))  # log P(x_k | rule)
    group_pos_log_lik = pm.Data("group_pos_log_lik", np.zeros((1, MAX_N, 1)))  # log P(x_k | interval group)
    group_log_count = pm.Data("group_log_count", np.zeros((1, 1)))  # log #intervals; -1e4 for padding
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, 1, 100)))  # share of group containing target
    lag = pm.Data("lag", np.zeros((1, MAX_N)))  # examples shown after position k
    present = pm.Data("present", np.zeros((1, MAX_N)))  # 1 if position k exists
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    delta = pm.Normal("delta", mu=0.0, sigma=0.5)  # serial-position decay rate

    weights = present * pt.exp(-delta * lag)  # (sets, positions)
    rule_log_lik = (weights[:, :, None] * rule_pos_log_lik).sum(axis=1)
    group_log_lik = (weights[:, :, None] * group_pos_log_lik).sum(axis=1)

    rule_log_prior = pt.log(lambda_0) + pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    log_rules = rule_log_prior[None, :] + rule_log_lik
    log_groups = pt.log1p(-lambda_0) - 2 * np.log(100) + group_log_count + group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + (
        posterior[:, N_RULES:, None] * group_in_concept
    ).sum(axis=1)  # (sets, targets)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    interval_member = MEMBERSHIP[N_RULES:]
    groups = []
    for s in unique:
        contains = interval_member[:, np.array(s) - 1]  # (intervals, examples)
        keys = np.concatenate([INTERVAL_SIZE[:, None], contains], axis=1)
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        inverse = np.ravel(inverse)
        counts = np.bincount(inverse)
        member_sum = np.zeros((len(first), 100))
        np.add.at(member_sum, inverse, interval_member)
        groups.append((first, counts, member_sum / counts[:, None]))
    n_groups = max(len(g[0]) for g in groups)
    n_sets = len(unique)
    rule_pos = np.zeros((n_sets, MAX_N, N_RULES))
    group_pos = np.zeros((n_sets, MAX_N, n_groups))
    log_count = np.full((n_sets, n_groups), -1e4)
    in_concept = np.zeros((n_sets, n_groups, 100))
    lag = np.zeros((n_sets, MAX_N))
    present = np.zeros((n_sets, MAX_N))
    for i, (s, (first, counts, share)) in enumerate(zip(unique, groups)):
        g = len(first)
        log_count[i, :g] = np.log(counts)
        in_concept[i, :g] = share
        for k, x in enumerate(s):
            rule_pos[i, k] = EXAMPLE_LOG_LIK[:N_RULES, x - 1]
            group_pos[i, k, :g] = EXAMPLE_LOG_LIK[N_RULES + first, x - 1]
            lag[i, k] = len(s) - 1 - k
            present[i, k] = 1.0
    return {
        "rule_pos_log_lik": rule_pos,
        "group_pos_log_lik": group_pos,
        "group_log_count": log_count,
        "group_in_concept": in_concept,
        "lag": lag,
        "present": present,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
