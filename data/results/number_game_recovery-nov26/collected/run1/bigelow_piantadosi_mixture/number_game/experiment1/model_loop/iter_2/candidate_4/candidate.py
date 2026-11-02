"""Refinement of erlang_interval_prior_concepts: Bayesian concept learning over
the same 45 rules and all 5,050 intervals of 1..100 (size principle, examples
drawn from the concept with probability 0.9 and uniformly otherwise, a free
prior weight per rule and a rule share lambda_0, Erlang prior on interval size
with fitted characteristic size sigma), with one change: intervals bounded by
round numbers (lower endpoint 1 or ending in 0/1, upper endpoint ending in 0/9,
e.g. 1-10, 20-29, 11-50, 51-100) get a fitted log prior boost `round_boost`.
People thinking of a range thus favour decade-aligned ranges and generalise out
to decade boundaries. "Yes" is the posterior probability that the concept
contains the target.
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
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
INTERVAL_SIZE = (HI - LO + 1).astype(int)
# Decade-aligned interval: lower end 1 or x0/x1, upper end x0/x9.
ROUND = (((LO == 1) | (LO % 10 == 0) | (LO % 10 == 1)) & ((HI % 10 == 0) | (HI % 10 == 9))).astype(int)
MAX_K = 5  # examples per set inside an interval: 0..4
N_GROUPS = 100 * MAX_K * 2

# Intervals grouped by (size s, examples inside k, round r):
# g = ((s - 1) * MAX_K + k) * 2 + r. Likelihood depends only on (s, k) and
# prior only on (s, r), so the grouped posterior is exact.
_g = np.arange(N_GROUPS)
GROUP_SIZE = (_g // (2 * MAX_K) + 1).astype("float64")
GROUP_K = ((_g // 2) % MAX_K).astype("float64")
GROUP_ROUND = (_g % 2).astype("float64")
EMPTY_LOG_COUNT = -1e4

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS, 100)))  # (sets, groups, targets)
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

    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
    # Log prior boost for decade-aligned intervals (0 recovers the incumbent).
    round_boost = pm.Normal("round_boost", mu=0.0, sigma=2.0)

    size = pt.as_tensor(INTERVAL_SIZE.astype("float64"))
    log_w = pt.log(size) - size / sigma + round_boost * pt.as_tensor(ROUND.astype("float64"))
    log_norm = pt.logsumexp(log_w)
    gsize = pt.as_tensor(GROUP_SIZE)
    group_log_prior = pt.log(gsize) - gsize / sigma + round_boost * pt.as_tensor(GROUP_ROUND) - log_norm
    in_log, out_log = np.log(ALPHA / GROUP_SIZE + (1 - ALPHA) / 100), np.log((1 - ALPHA) / 100)
    group_log_lik = (GROUP_K * in_log)[None, :] + (n_examples[:, None] - GROUP_K[None, :]) * out_log

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_intervals = pt.log1p(-lambda_0) + group_log_prior[None, :] + group_log_count + group_log_lik
    log_joint = pt.concatenate([log_rules, log_intervals], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
                  + pt.batched_dot(posterior[:, N_RULES:], group_in_concept))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    log_count = np.full((len(unique), N_GROUPS), EMPTY_LOG_COUNT)
    in_concept = np.zeros((len(unique), N_GROUPS, 100))
    for i, s in enumerate(unique):
        k = sum(((LO <= x) & (HI >= x)).astype(int) for x in s)
        group = ((INTERVAL_SIZE - 1) * MAX_K + k) * 2 + ROUND
        count = np.bincount(group, minlength=N_GROUPS).astype("float64")
        members = np.zeros((N_GROUPS, 100))
        np.add.at(members, group, INTERVAL_MEMBERSHIP)
        filled = count > 0
        log_count[i, filled] = np.log(count[filled])
        in_concept[i, filled] = members[filled] / count[filled, None]
    return {
        "rule_log_lik": set_log_lik,
        "n_examples": np.array([float(len(s)) for s in unique]),
        "group_log_count": log_count,
        "group_in_concept": in_concept,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
