"""Bigelow & Piantadosi's rules-plus-intervals Bayesian concept learner, with
the strength of the size principle as a free parameter (a likelihood exponent).

Refines `bigelow_piantadosi_mixture`. A learner who sees the example set D
weighs 45 rule concepts (odd, even, primes, squares, cubes, powers of 2-10,
powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9,
containing the digit 0-9) and every interval of 1..100; a rule has prior
lambda_0 / 45 and each interval (1 - lambda_0) / 100^2. Examples are drawn
from the concept (90%) or uniformly from 1..100 (10%). The one change: people
do not treat the examples as fully independent strong samples -- the evidence
log P(D | h) is raised to a power gamma (gamma = 1 is the incumbent), so the
size principle can be weaker (gamma < 1) or stronger (gamma > 1) than
optimal. "Yes" is the posterior probability that the concept contains the
target.
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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE = INT_HI - INT_LO + 1
INT_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")
N_CLASS = 100 * 5  # intervals grouped by (size, number of examples inside)

# Under the exponent gamma the 5,050 intervals' likelihoods are no longer
# precomputable as one total, but an interval's likelihood depends only on its
# size and how many examples it contains. prepare_observed groups each set's
# intervals into those classes (count, base log-likelihood, summed membership),
# so the posterior over all intervals is exact at a fraction of the cost.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    class_log_count = pm.Data("class_log_count", np.zeros((1, N_CLASS)))  # -1e4 for empty classes
    class_log_lik = pm.Data("class_log_lik", np.zeros((1, N_CLASS)))
    class_in_concept = pm.Data("class_in_concept", np.zeros((1, N_CLASS, 100)))  # mean membership
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.LogNormal("gamma", mu=0.0, sigma=0.5)

    log_rules = pt.log(lambda_0) - np.log(N_RULES) + gamma * rule_log_lik
    log_classes = pt.log1p(-lambda_0) - 2 * np.log(100) + class_log_count + gamma * class_log_lik
    log_joint = pt.concatenate([log_rules, log_classes], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + pt.matmul(
        posterior[:, None, N_RULES:], class_in_concept
    )[:, 0, :]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    rule_ll = np.zeros((n_sets, N_RULES))
    log_count = np.full((n_sets, N_CLASS), -1e4)
    class_ll = np.zeros((n_sets, N_CLASS))
    in_concept = np.zeros((n_sets, N_CLASS, 100))
    for i, s in enumerate(unique):
        x = np.array(s)
        rule_ll[i] = RULE_LOG_LIK[:, x - 1].sum(axis=1)
        k = INT_MEMBERSHIP[:, x - 1].sum(axis=1).astype(int)
        cls = (INT_SIZE - 1) * 5 + k
        counts = np.bincount(cls, minlength=N_CLASS)
        sums = np.zeros((N_CLASS, 100))
        np.add.at(sums, cls, INT_MEMBERSHIP)
        ok = counts > 0
        log_count[i, ok] = np.log(counts[ok])
        in_concept[i, ok] = sums[ok] / counts[ok, None]
        size = np.arange(N_CLASS) // 5 + 1
        kk = np.arange(N_CLASS) % 5
        inside = np.log(ALPHA / size + (1 - ALPHA) / 100)
        class_ll[i] = np.minimum(kk, len(x)) * inside + np.maximum(len(x) - kk, 0) * np.log((1 - ALPHA) / 100)
    return {
        "rule_log_lik": rule_ll,
        "class_log_count": log_count,
        "class_log_lik": class_ll,
        "class_in_concept": in_concept,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
