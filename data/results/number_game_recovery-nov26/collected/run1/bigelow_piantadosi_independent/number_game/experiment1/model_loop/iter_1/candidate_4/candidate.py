"""Tempered size-principle mixture with disjunctive last-digit concepts.

Refinement of `tempered_size_principle_mixture`. The learner keeps the same 45
rules, all intervals of 1..100, the same mixed likelihood (90% from the
concept, 10% uniform) tempered by an exponent beta, and lambda_0 for the rule
share of the prior. The one change: the rule vocabulary also contains every
union of two to four final digits ("ends in 2 or 7", "ends in 0, 1 or 6"),
each with the prior of a single rule times exp(-gamma * (digits - 1)). When
examples' final digits differ, a final-digit rule covering all of them is
still available, so targets sharing a final digit with any example are
endorsed far from the examples. "Yes" is the posterior probability that the
concept contains the target.
"""


import itertools

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
BASE_RULES = list(_RULES.values())
# Disjunctive last-digit concepts: unions of 2-4 final digits; their prior
# penalty grows with the number of extra digits.
DIGIT_UNIONS = [c for k in range(2, 5) for c in itertools.combinations(range(10), k)]
RULES = BASE_RULES + [[n for n in NUMBERS if n % 10 in c] for c in DIGIT_UNIONS]
RULE_EXTRA_DIGITS = np.array([0.0] * len(BASE_RULES) + [len(c) - 1.0 for c in DIGIT_UNIONS])
N_BASE_RULES = len(BASE_RULES)
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
MAX_EXAMPLES = 4
# An interval's likelihood depends only on its size s and how many examples k
# it contains, so the 5,050 intervals are grouped by (s, k) per example set:
# the group's evidence is beta * log P(D | s, k) plus the log number of its
# intervals, and its in-concept probability for a target is the share of its
# intervals containing that target. This is exact and ~10x cheaper.
SIZES = np.arange(1, 101)
N_GROUPS = 100 * (MAX_EXAMPLES + 1)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | rule)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))  # (sets, groups): log P(D | s, k)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # log number of intervals
    # Only the (set, target) cells asked about: the cell's set, whether each
    # rule contains its target, and each group's share of intervals containing it.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_group_in_concept = pm.Data("cell_group_in_concept", np.zeros((1, N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # each trial's cell
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    gamma = pm.LogNormal("gamma", mu=1.0, sigma=0.75)  # prior cost per extra digit in a union

    log_rules = pt.log(lambda_0) - np.log(N_BASE_RULES) - gamma * RULE_EXTRA_DIGITS + beta * rule_log_lik
    log_groups = pt.log1p(-lambda_0) - 2 * np.log(100) + beta * group_log_lik + group_log_count
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))[cell_set]  # (cells, hyps)
    in_concept = (posterior[:, :N_RULES] * cell_rule_member).sum(axis=1) + (
        posterior[:, N_RULES:] * cell_group_in_concept
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _interval_groups(examples):
    n_groups = N_GROUPS
    log_lik = np.zeros(n_groups)
    count = np.zeros(n_groups)
    in_concept = np.zeros((n_groups, 100))
    ex = np.array(examples)
    n = len(ex)
    for s in SIZES:
        for k in range(MAX_EXAMPLES + 1):
            log_lik[(s - 1) * (MAX_EXAMPLES + 1) + k] = k * np.log(ALPHA / s + (1 - ALPHA) / 100) + (n - k) * np.log(
                (1 - ALPHA) / 100
            )
        for m in range(1, 102 - s):
            k = int(((ex >= m) & (ex <= m + s - 1)).sum())
            g = (s - 1) * (MAX_EXAMPLES + 1) + k
            count[g] += 1
            in_concept[g, m - 1 : m + s - 1] += 1
    nonzero = count > 0
    in_concept[nonzero] /= count[nonzero, None]
    log_count = np.where(nonzero, np.log(np.maximum(count, 1)), -1e4)
    return log_lik, log_count, in_concept


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    groups = [_interval_groups(s) for s in unique]
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "rule_log_lik": np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "group_log_lik": np.stack([g[0] for g in groups]),
        "group_log_count": np.stack([g[1] for g in groups]),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_group_in_concept": np.stack([groups[i][2][:, t - 1] for i, t in cells]),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
