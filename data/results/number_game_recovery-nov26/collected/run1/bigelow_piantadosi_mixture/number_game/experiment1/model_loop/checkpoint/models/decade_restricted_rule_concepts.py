"""Decade-restricted rules: Bayesian concept learning over the 45 rules and all
intervals of 1..100 (size principle, examples drawn from the concept with
probability 0.9 and uniformly otherwise, a free prior weight per rule, an
Erlang prior on interval size), with one change: each rule can also be held in
a version confined to a decade-bounded region of the number line (from 1, 11,
..., 91 up to 10, 20, ..., 100, e.g. "ends in 3 between 11 and 50"). A fitted
share c of each rule's prior is spread equally over its restricted versions,
the rest stays on the unrestricted rule. Restricted rules are smaller, so the
size principle favours them when the examples sit in one region, curbing
generalisation of "ends in d"-type rules to far-away targets. "Yes" is the
posterior probability that the concept contains the target.
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
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZE = INTERVAL_MEMBERSHIP.sum(axis=1).astype(int)
MAX_K = 5

# Rule x decade-bounded range conjunctions (the unrestricted rule is range 1..100).
RANGES = [(lo, hi) for lo in range(1, 92, 10) for hi in range(lo + 9, 101, 10)]
_rc_rule, _rc_full, _rc_mem = [], [], []
for _r in range(N_RULES):
    for _lo, _hi in RANGES:
        _m = RULE_MEMBERSHIP[_r] * ((NUMBERS >= _lo) & (NUMBERS <= _hi))
        if _m.sum() == 0:
            continue
        _rc_rule.append(_r)
        _rc_full.append(1.0 if (_lo, _hi) == (1, 100) else 0.0)
        _rc_mem.append(_m)
RC_RULE = np.array(_rc_rule, dtype="int64")
RC_FULL = np.array(_rc_full)
RC_MEMBERSHIP = np.array(_rc_mem)
N_RESTRICTED = np.bincount(RC_RULE, weights=1 - RC_FULL, minlength=N_RULES)
RC_LOG_SHARE = np.where(RC_FULL > 0, 0.0, -np.log(np.maximum(N_RESTRICTED[RC_RULE], 1)))
RC_EXAMPLE_LOG_LIK = np.log(ALPHA * RC_MEMBERSHIP / RC_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
N_RC = len(RC_RULE)  # examples per set inside an interval: 0..4

# An interval's likelihood depends only on its size s and on how many examples
# k it contains, and its prior only on s, so for each example set the 5,050
# intervals are grouped by (s, k): group g = (s - 1) * MAX_K + k. Per set,
# prepare_observed stores each group's log count and the share of its
# intervals containing each target; the posterior below is exactly the one
# over all intervals, at ~1/10 of the cost.
GROUP_SIZE = np.repeat(np.arange(1, 101), MAX_K).astype("float64")
GROUP_K = np.tile(np.arange(MAX_K), 100).astype("float64")
EMPTY_LOG_COUNT = -1e4

# As in the incumbent: "ends in 0" and "contains digit 0" are one set, so
# their weights are sampled as a shared Dirichlet component and a Beta split.
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RC)))  # (sets, rule x range concepts)
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    group_log_count = pm.Data("group_log_count", np.zeros((1, 100 * MAX_K)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, 100 * MAX_K, 100)))  # (sets, groups, targets)
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

    # Erlang prior on interval size with characteristic size sigma (1..100 scale).
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
    size = pt.as_tensor(INTERVAL_SIZE.astype("float64"))
    log_erlang = pt.log(size) - size / sigma
    log_norm = pt.logsumexp(log_erlang)
    gsize = pt.as_tensor(GROUP_SIZE)
    group_log_prior = pt.log(gsize) - gsize / sigma - log_norm  # per interval of the group
    in_log, out_log = np.log(ALPHA / GROUP_SIZE + (1 - ALPHA) / 100), np.log((1 - ALPHA) / 100)
    group_log_lik = (GROUP_K * in_log)[None, :] + (n_examples[:, None] - GROUP_K[None, :]) * out_log

    # Share of each rule's prior that goes to its decade-restricted versions.
    restrict = pm.Beta("restrict_share", 1.5, 3.0)
    rc_log_prior = (pt.log(rule_weights)[RC_RULE] + RC_LOG_SHARE
                    + RC_FULL * pt.log1p(-restrict) + (1 - RC_FULL) * pt.log(restrict))
    log_rules = pt.log(lambda_0) + rc_log_prior[None, :] + rule_log_lik
    log_intervals = pt.log1p(-lambda_0) + group_log_prior[None, :] + group_log_count + group_log_lik
    log_joint = pt.concatenate([log_rules, log_intervals], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[:, :N_RC] @ pt.as_tensor(RC_MEMBERSHIP)
                  + pt.batched_dot(posterior[:, N_RC:], group_in_concept))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([RC_EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    lo = np.array([m for m in NUMBERS for n in range(m, 101)])
    hi = np.array([n for m in NUMBERS for n in range(m, 101)])
    log_count = np.full((len(unique), 100 * MAX_K), EMPTY_LOG_COUNT)
    in_concept = np.zeros((len(unique), 100 * MAX_K, 100))
    for i, s in enumerate(unique):
        k = sum(((lo <= x) & (hi >= x)).astype(int) for x in s)
        group = (INTERVAL_SIZE - 1) * MAX_K + k
        count = np.bincount(group, minlength=100 * MAX_K).astype("float64")
        members = np.zeros((100 * MAX_K, 100))
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
