"""Concept selection with a sharpened posterior: a refinement of
`family_weighted_concept_prior`.

People weigh the incumbent's 45 rules and 5,050 intervals of 1..100 by a
family-level prior and the size principle (10% of examples uniform noise), but
they do not probability-match over the posterior: they select concepts with
probability proportional to posterior ** gamma, an inferred decision exponent
(gamma > 1 sharpens toward the most convincing concepts, gamma < 1 flattens).
A clear winner such as "ends in 3" then crowds out the many loosely fitting
intervals, so in-range targets breaking the shared final digit get "no".

For speed, intervals are grouped by (number of examples inside, size): every
interval in a group has the same likelihood, so the model only needs each
group's interval count and summed membership vector, precomputed per set.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_K = 5  # examples inside an interval: 0..4


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


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in RULE_NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EXAMPLE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
N_INTERVALS = 5050
SIZES = np.arange(1, 101)
# Log likelihood of one interval with k of the examples inside and n examples total, size s.
LOG_IN = np.log(ALPHA / SIZES + (1 - ALPHA) / 100)  # (100,)
LOG_OUT = np.log((1 - ALPHA) / 100)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, MAX_K * 100)))  # per-interval log lik of each group
    group_log_count = pm.Data("group_log_count", np.zeros((1, MAX_K * 100)))  # log #intervals (-1e4 if empty)
    group_membership = pm.Data("group_membership", np.zeros((1, MAX_K * 100, 100)))  # summed / count
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(6))
    log_w = pt.log(family_weight)
    # Decision exponent on the posterior; log-normal prior centred on Bayes (gamma = 1).
    log_gamma = pm.Normal("log_gamma", mu=0.0, sigma=0.7)
    gamma = pm.Deterministic("gamma", pt.exp(log_gamma))

    log_rules = gamma * ((log_w[:5] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik)
    log_groups = gamma * (log_w[5] - np.log(N_INTERVALS) + group_log_lik) + group_log_count
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    weights = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = weights[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + pt.batched_dot(
        weights[:, None, N_RULES:], group_membership
    )[:, 0, :]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _interval_groups(examples):
    ex = np.array(examples)
    n = len(ex)
    log_lik = np.zeros((MAX_K, 100))
    count = np.zeros((MAX_K, 100))
    msum = np.zeros((MAX_K, 100, 100))
    for k in range(MAX_K):
        log_lik[k] = k * LOG_IN + (n - k) * LOG_OUT
    for lo in range(1, 101):
        for hi in range(lo, 101):
            k = int(np.sum((ex >= lo) & (ex <= hi)))
            s = hi - lo
            count[k, s] += 1
            msum[k, s, lo - 1:hi] += 1
    mean_member = np.where(count[:, :, None] > 0, msum / np.maximum(count, 1)[:, :, None], 0.0)
    log_count = np.where(count > 0, np.log(np.maximum(count, 1)), -1e4)
    return log_lik.ravel(), log_count.ravel(), mean_member.reshape(MAX_K * 100, 100)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    groups = [_interval_groups(s) for s in unique]
    return {
        "rule_log_lik": np.stack([RULE_EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "group_log_lik": np.stack([g[0] for g in groups]),
        "group_log_count": np.stack([g[1] for g in groups]),
        "group_membership": np.stack([g[2] for g in groups]),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
