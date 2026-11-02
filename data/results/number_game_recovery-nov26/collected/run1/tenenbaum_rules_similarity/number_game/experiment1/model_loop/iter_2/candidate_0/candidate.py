"""Order-weighted evidence in Bayesian concept learning: a refinement of
`family_weighted_concept_prior`.

People do Bayesian concept learning over 45 rules and the 5,050 intervals of
1..100 (size principle, 10% of examples uniform noise, a prior mass per concept
family), but each example's evidence is weighted by its position in the
sequence: weight is proportional to exp(order_slope * position), position
running from -0.5 (first example) to +0.5 (last), normalised so the weights
average one within a set. order_slope > 0 is recency, < 0 is primacy, 0 recovers
the equal-weight learner. "Yes" is the posterior probability that the concept
contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_EXAMPLES = 4


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
RULE_FAMILY = np.array([family(n) for n in RULE_NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
N_INTERVALS = len(INTERVALS)
CONCEPT_FAMILY = np.concatenate([RULE_FAMILY, np.full(N_INTERVALS, 5)])
WITHIN_FAMILY_LOG = -np.log(np.concatenate([FAMILY_SIZE, [N_INTERVALS]]))[CONCEPT_FAMILY]
N_CONCEPTS = len(CONCEPT_FAMILY)

with pm.Model() as model:
    # Per-set, per-position example log-likelihoods under every concept.
    pos_log_lik = pm.Data("pos_log_lik", np.zeros((1, MAX_EXAMPLES, N_CONCEPTS)))
    position = pm.Data("position", np.zeros((1, MAX_EXAMPLES)))  # -0.5 first .. +0.5 last
    present = pm.Data("present", np.ones((1, MAX_EXAMPLES)))  # 1 if that slot holds an example
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(6))
    # Order gradient on evidence weight: >0 recency, <0 primacy.
    order_slope = pm.Normal("order_slope", mu=0.0, sigma=1.5)

    raw_w = pt.exp(order_slope * position) * present
    weights = raw_w * present.sum(axis=1, keepdims=True) / raw_w.sum(axis=1, keepdims=True)
    set_log_lik = (weights[:, :, None] * pos_log_lik).sum(axis=1)

    log_prior = pt.log(family_weight)[CONCEPT_FAMILY] + WITHIN_FAMILY_LOG
    log_joint = log_prior[None, :] + set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior @ pt.as_tensor(MEMBERSHIP)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    pos_log_lik = np.zeros((len(unique), MAX_EXAMPLES, N_CONCEPTS))
    position = np.zeros((len(unique), MAX_EXAMPLES))
    present = np.zeros((len(unique), MAX_EXAMPLES))
    for i, s in enumerate(unique):
        k = len(s)
        for j, x in enumerate(s):
            pos_log_lik[i, j] = EXAMPLE_LOG_LIK[:, x - 1]
            position[i, j] = (j / (k - 1) - 0.5) if k > 1 else 0.0
            present[i, j] = 1.0
    return {
        "pos_log_lik": pos_log_lik,
        "position": position,
        "present": present,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
