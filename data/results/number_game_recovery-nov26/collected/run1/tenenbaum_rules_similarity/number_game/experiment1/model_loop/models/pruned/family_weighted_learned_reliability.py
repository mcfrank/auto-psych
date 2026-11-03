"""Bayesian concept learning over rules and intervals with a prior weight per
concept family and a learned example reliability: a refinement of
`family_weighted_concept_prior`.

The learner considers 45 rules and the 5,050 intervals of 1..100, with a
prior mass per concept family (parity; primes/squares/cubes/powers; multiples
of k; ends in d; contains digit d; intervals) shared equally by its members,
samples examples from the concept (the size principle), and answers "yes"
with the posterior probability that the concept contains the target. The one
change: the share of examples believed drawn from the concept (rather than
uniformly from 1..100) is not fixed at 0.9 but inferred, so trusting the
examples more removes support for small intervals that explain one example
and treat the rest as noise.
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
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
N_INTERVALS = len(INTERVALS)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INTERVAL_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in INTERVALS], dtype="float64")
INTERVAL_SIZE = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
# Intervals sharing a size and a count of examples inside have the same
# likelihood, so they are grouped: group = (size - 1) * 5 + examples inside.
MAX_K = 5
GROUP_SIZE = np.repeat(np.arange(1, 101, dtype="float64"), MAX_K)
GROUP_K = np.tile(np.arange(MAX_K, dtype="float64"), 100)

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules): examples inside
    group_log_count = pm.Data("group_log_count", np.zeros((1, 100 * MAX_K)))  # log #intervals per group
    group_member = pm.Data("group_member", np.zeros((1, 100 * MAX_K, 100)))  # mean membership per group
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the five rule families and of intervals, and the share of
    # examples believed drawn from the concept rather than at random.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(6))
    reliability = pm.Beta("reliability", 18.0, 2.0)
    log_w = pt.log(family_weight)
    log_noise = pt.log1p(-reliability) - np.log(100.0)
    n = n_examples[:, None]

    rule_log_member = pt.log(reliability / RULE_SIZE + (1 - reliability) / 100.0)
    log_rules = (
        (log_w[:5] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :]
        + rule_n_in * rule_log_member[None, :]
        + (n - rule_n_in) * log_noise
    )
    group_log_member = pt.log(reliability / GROUP_SIZE + (1 - reliability) / 100.0)
    log_groups = (
        log_w[5] - np.log(N_INTERVALS)
        + group_log_count
        + GROUP_K[None, :] * group_log_member[None, :]
        + (n - GROUP_K[None, :]) * log_noise
    )
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = pt.dot(posterior[:, :N_RULES], pt.as_tensor(RULE_MEMBERSHIP)) + (posterior[:, None, N_RULES:] @ group_member)[:, 0, :]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    log_count = np.full((len(unique), 100 * MAX_K), -1e4)
    member = np.zeros((len(unique), 100 * MAX_K, 100))
    for i, s in enumerate(unique):
        k = INTERVAL_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1).astype("int64")
        group = (INTERVAL_SIZE - 1) * MAX_K + k
        count = np.bincount(group, minlength=100 * MAX_K).astype("float64")
        np.add.at(member[i], group, INTERVAL_MEMBERSHIP)
        nonzero = count > 0
        log_count[i, nonzero] = np.log(count[nonzero])
        member[i, nonzero] /= count[nonzero, None]
    return {
        "rule_n_in": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "group_log_count": log_count,
        "group_member": member,
        "n_examples": np.array([float(len(s)) for s in unique]),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
