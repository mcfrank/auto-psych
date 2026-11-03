"""Refinement of `rule_family_prior_concepts`: Bayesian concept learning over
rules (with a family-level rule prior) and intervals, where the share of
examples people believe were truly sampled from the concept is inferred.

A learner considers 45 rule concepts (odd, even, primes, squares, cubes,
powers of 2-10, powers of 2 without 32 / with 37, multiples of 3-9, 11, 12,
ending in 0-9, containing digit 0-9) and the 5,050 intervals of 1..100. A rule
is chosen with probability lambda_0 (family drawn with inferred weight, then a
rule uniformly within it), an interval otherwise (prior (1 - lambda_0)/100^2).
The single change: each example is drawn from the concept with probability
alpha, inferred from the data, rather than a fixed 0.9, and from 1..100
otherwise. "Yes" is the posterior probability that the concept contains the
target.
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
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
N_INTERVALS = len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
SIZES = MEMBERSHIP.sum(axis=1)


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
RULE_SIZES = SIZES[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZES = SIZES[N_RULES:].astype(int)
MAX_EXAMPLES = 4
# Intervals of equal size holding the same number of examples have the same
# prior and likelihood, so per set they are pooled into groups (size, n_in):
# group log-count and, for each asked (set, target) cell, the share of the
# group's intervals containing the target are computed once.
GROUP_SIZE = np.repeat(np.arange(1, 101), MAX_EXAMPLES + 1).astype("float64")
GROUP_N_IN = np.tile(np.arange(MAX_EXAMPLES + 1), 100).astype("float64")
N_GROUPS = len(GROUP_SIZE)

# log P(D | h) = n_in * log(alpha/|h| + (1-alpha)/100) + n_out * log((1-alpha)/100),
# with n_in / n_out the examples inside / outside h.

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,): set of each asked cell
    cell_rule_membership = pm.Data("cell_rule_membership", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_group_membership = pm.Data("cell_group_membership", np.zeros((1, N_GROUPS)))  # (cells, groups)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,): cell of each response
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    # Share of examples believed sampled from the concept (was fixed at 0.9).
    alpha = pm.Beta("alpha", 4, 1)

    log_noise = pt.log1p(-alpha) - np.log(100)
    n = n_examples[:, None]
    rule_log_prior = pt.log(lambda_0) + pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    rule_log_in = pt.log(alpha / RULE_SIZES + (1 - alpha) / 100)
    log_rules = rule_log_prior[None, :] + rule_n_in * rule_log_in[None, :] + (n - rule_n_in) * log_noise
    group_log_in = pt.log(alpha / GROUP_SIZE + (1 - alpha) / 100)
    log_groups = (pt.log1p(-lambda_0) - 2 * np.log(100) + group_log_count
                  + GROUP_N_IN[None, :] * group_log_in[None, :] + (n - GROUP_N_IN[None, :]) * log_noise)
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_posterior = posterior[cell_set]  # (cells, rules + groups)
    in_concept = pt.sum(cell_posterior[:, :N_RULES] * cell_rule_membership, axis=1) + pt.sum(
        cell_posterior[:, N_RULES:] * cell_group_membership, axis=1)  # (cells,)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_n_in, log_count, membership = [], [], []
    for s in unique:
        cols = np.array(s) - 1
        rule_n_in.append(RULE_MEMBERSHIP[:, cols].sum(axis=1))
        group = (INTERVAL_SIZES - 1) * (MAX_EXAMPLES + 1) + INTERVAL_MEMBERSHIP[:, cols].sum(axis=1).astype(int)
        count = np.bincount(group, minlength=N_GROUPS).astype("float64")
        summed = np.zeros((N_GROUPS, 100))
        np.add.at(summed, group, INTERVAL_MEMBERSHIP)
        log_count.append(np.where(count > 0, np.log(np.maximum(count, 1)), -1e4))
        membership.append(summed / np.maximum(count, 1)[:, None])
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "rule_n_in": np.stack(rule_n_in),
        "group_log_count": np.stack(log_count),
        "n_examples": np.array([float(len(s)) for s in unique]),
        "cell_set": np.array([c[0] for c in cells], dtype="int64"),
        "cell_rule_membership": np.stack([RULE_MEMBERSHIP[:, t] for _, t in cells]),
        "cell_group_membership": np.stack([membership[i][:, t] for i, t in cells]),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
