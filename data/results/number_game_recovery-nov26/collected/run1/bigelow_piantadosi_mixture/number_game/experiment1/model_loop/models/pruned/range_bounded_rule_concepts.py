"""People's hypothesis space contains whole-number-line rules (the 45 rules) and
numerical ranges (all intervals of 1..100, Erlang size prior), and also rules
confined to a range ("even numbers in the 20s"): every rule intersected with
every interval, which takes a fitted share of the prior (split across rules by
the same rule weights, uniformly across intervals). The size principle favours
these smaller range-bounded rules, so a rule is generalised mainly near the
examples and endorsement of far targets that fit a rule (e.g. by last digit)
falls off with distance. Examples drawn from the concept with probability 0.9,
otherwise uniformly; "yes" is the posterior probability the concept contains
the target.
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
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
MAX_K = 5
GROUP_SIZE = np.repeat(np.arange(1, 101), MAX_K).astype("float64")
GROUP_K = np.tile(np.arange(MAX_K), 100).astype("float64")
EMPTY_LOG_COUNT = -1e4

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Range-bounded rules: rule r restricted to interval I, for every I with
# nonempty r ∩ I. Their likelihood and membership involve no free parameters,
# so per set and rule prepare_observed marginalises over intervals (uniform
# within rule): CONJ sizes and per-rule interval counts are fixed here.
CONJ_SIZE = INTERVAL_MEMBERSHIP @ RULE_MEMBERSHIP.T  # (intervals, rules)
CONJ_VALID = CONJ_SIZE > 0
CONJ_LOG_N = np.log(CONJ_VALID.sum(axis=0))  # (rules,)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    conj_log_lik = pm.Data("conj_log_lik", np.zeros((1, N_RULES)))  # log mean lik over intervals
    conj_in_concept = pm.Data("conj_in_concept", np.zeros((1, N_RULES, 100)))
    n_examples = pm.Data("n_examples", np.ones(1))
    group_log_count = pm.Data("group_log_count", np.zeros((1, 100 * MAX_K)))
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, 100 * MAX_K, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior shares of whole rules, range-bounded rules and plain ranges.
    family = pm.Dirichlet("family", a=np.array([2.0, 2.0, 2.0]))
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    size = pt.as_tensor(INTERVAL_SIZE.astype("float64"))
    log_norm = pt.logsumexp(pt.log(size) - size / pt.exp(log_sigma))
    gsize = pt.as_tensor(GROUP_SIZE)
    group_log_prior = pt.log(gsize) - gsize / pt.exp(log_sigma) - log_norm
    in_log, out_log = np.log(ALPHA / GROUP_SIZE + (1 - ALPHA) / 100), np.log((1 - ALPHA) / 100)
    group_log_lik = (GROUP_K * in_log)[None, :] + (n_examples[:, None] - GROUP_K[None, :]) * out_log

    log_w = pt.log(rule_weights)[None, :]
    log_rules = pt.log(family[0]) + log_w + rule_log_lik
    log_conj = pt.log(family[1]) + log_w + conj_log_lik
    log_intervals = pt.log(family[2]) + group_log_prior[None, :] + group_log_count + group_log_lik
    log_joint = pt.concatenate([log_rules, log_conj, log_intervals], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
                  + pt.batched_dot(posterior[:, N_RULES:2 * N_RULES], conj_in_concept)
                  + pt.batched_dot(posterior[:, 2 * N_RULES:], group_in_concept))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _conjunctions(s):
    """Per rule: log mean likelihood over its range-bounded versions, and the
    posterior (within rule) probability that each target is in the concept."""
    x = np.array(s) - 1
    in_int = (LO[:, None] <= x + 1) & (HI[:, None] >= x + 1)  # (intervals, examples)
    log_lik = np.zeros(CONJ_SIZE.shape)
    safe = np.maximum(CONJ_SIZE, 1)
    for j, xi in enumerate(x):
        inside = in_int[:, j][:, None] & (RULE_MEMBERSHIP[:, xi][None, :] > 0)
        log_lik += np.log(ALPHA * inside / safe + (1 - ALPHA) / 100)
    log_lik = np.where(CONJ_VALID, log_lik, -np.inf)
    m = log_lik.max(axis=0)
    w = np.exp(log_lik - m)  # (intervals, rules)
    total = w.sum(axis=0)
    log_mean = m + np.log(total) - CONJ_LOG_N
    # P(target in r ∩ I) = [t in r] * sum_I w_I [t in I] / sum_I w_I
    in_range = (w.T @ INTERVAL_MEMBERSHIP) / total[:, None]  # (rules, targets)
    return log_mean, in_range * RULE_MEMBERSHIP


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    conj = [_conjunctions(s) for s in unique]
    log_count = np.full((len(unique), 100 * MAX_K), EMPTY_LOG_COUNT)
    in_concept = np.zeros((len(unique), 100 * MAX_K, 100))
    for i, s in enumerate(unique):
        k = sum(((LO <= x) & (HI >= x)).astype(int) for x in s)
        group = (INTERVAL_SIZE - 1) * MAX_K + k
        count = np.bincount(group, minlength=100 * MAX_K).astype("float64")
        members = np.zeros((100 * MAX_K, 100))
        np.add.at(members, group, INTERVAL_MEMBERSHIP)
        filled = count > 0
        log_count[i, filled] = np.log(count[filled])
        in_concept[i, filled] = members[filled] / count[filled, None]
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "conj_log_lik": np.stack([c[0] for c in conj]),
        "conj_in_concept": np.stack([c[1] for c in conj]),
        "n_examples": np.array([float(len(s)) for s in unique]),
        "group_log_count": log_count,
        "group_in_concept": in_concept,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
