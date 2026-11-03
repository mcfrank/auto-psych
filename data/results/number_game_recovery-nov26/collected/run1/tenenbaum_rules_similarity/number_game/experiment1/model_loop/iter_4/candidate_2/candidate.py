"""Bayesian concept learning from a leaky working memory of the examples.

People do the same Bayesian concept learning as `exception_rule_family_prior_2`
(45 rules and all intervals of 1..100, the size principle with 10% of examples
uniform noise, a prior mass per concept family with the exception rules in
their own family), but each shown example is independently retained in working
memory with an inferred probability and otherwise forgotten; a forgotten
example contributes nothing to inference. "Yes" is the posterior membership
probability averaged over which subset of examples was remembered.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

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


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in RULE_NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
N_INTERVALS = len(INTERVALS)

with pm.Model() as model:
    # One row per (distinct set, remembered subset of its examples).
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))
    n_kept = pm.Data("n_kept", np.zeros(1))  # examples remembered in the subset
    n_lost = pm.Data("n_lost", np.zeros(1))  # examples forgotten in the subset
    subset_of_set = pm.Data("subset_of_set", np.zeros((1, 1)))  # (sets, subsets) indicator
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(7))
    retain = pm.Beta("retain", alpha=4.0, beta=1.5)  # chance each example stays in memory
    log_w = pt.log(family_weight)

    log_rules = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    log_intervals = log_w[6] - np.log(N_INTERVALS) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES][:, None] * interval_in_concept

    subset_weight = pt.exp(n_kept * pt.log(retain) + n_lost * pt.log1p(-retain))
    in_concept_set = (subset_of_set * subset_weight[None, :]) @ in_concept  # (sets, 100)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept_set.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    subset_lik, kept, lost, owner = [], [], [], []
    for i, s in enumerate(unique):
        for mask in itertools.product([0, 1], repeat=len(s)):
            ex = [x for x, m in zip(s, mask) if m]
            ll = EXAMPLE_LOG_LIK[:, np.array(ex, dtype=int) - 1].sum(axis=1) if ex else np.zeros(len(MEMBERSHIP))
            subset_lik.append(ll)
            kept.append(sum(mask))
            lost.append(len(s) - sum(mask))
            owner.append(i)
    subset_lik = np.stack(subset_lik)
    interval_log_lik = subset_lik[:, N_RULES:]
    log_evidence = logsumexp(interval_log_lik, axis=1)
    indicator = np.zeros((len(unique), len(owner)))
    indicator[owner, np.arange(len(owner))] = 1.0
    return {
        "rule_log_lik": subset_lik[:, :N_RULES],
        "interval_log_evidence": log_evidence,
        "interval_in_concept": np.exp(interval_log_lik - log_evidence[:, None]) @ MEMBERSHIP[N_RULES:],
        "n_kept": np.array(kept, dtype="float64"),
        "n_lost": np.array(lost, dtype="float64"),
        "subset_of_set": indicator,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
