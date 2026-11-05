"""Bayesian concept learning with a family-structured rule prior (refinement
of `bigelow_piantadosi_mixture`).

A learner who sees the example set D considers the 45 rule concepts of
Bigelow & Piantadosi (2016) (odd, even, primes, squares, cubes, powers of
2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending
in 0-9, containing the digit 0-9) and the 5,050 intervals of 1..100. A rule is
chosen with probability lambda_0, each interval has prior
(1 - lambda_0) / 100^2. Each example is drawn from the concept with
probability 0.9 (size principle), uniformly from 1..100 otherwise, and "yes"
is the posterior probability that the concept contains the target.

The single change from the mixture model: the rule prior is not flat. Rules
belong to seven families (parity, primes, squares/cubes, powers, multiples,
final digit, contained digit); each family has an inferred share of the rule
prior (Dirichlet(2, ..., 2)) split equally among its rules.
"""

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


def family_of(name):
    if name in ("odd", "even"):
        return 0  # parity
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
    return 6  # contains digit


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
N_FAMILIES = 7
FAMILY = np.array([family_of(n) for n in RULE_NAMES])
FAMILY_SIZE = np.bincount(FAMILY, minlength=N_FAMILIES).astype("float64")
# (rules, families): one-hot assignment divided by the family's size.
SPLIT = (FAMILY[:, None] == np.arange(N_FAMILIES)[None, :]) / FAMILY_SIZE[None, :]
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    rule_weights = pm.Deterministic("rule_weights", pt.dot(pt.as_tensor(SPLIT), family_weights))

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES][:, None] * interval_in_concept

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    log_evidence = logsumexp(interval_log_lik, axis=1)
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "interval_log_evidence": log_evidence,
        "interval_in_concept": np.exp(interval_log_lik - log_evidence[:, None]) @ MEMBERSHIP[N_RULES:],
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
