"""Refinement of concept_learning_endorsement_ceiling: Bayesian concept
learning with an endorsement ceiling and a learned example reliability.

People consider 45 equally likely rules (odd, even, primes, squares, cubes,
powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12,
ending in 0-9, containing the digit 0-9), chosen with probability lambda_0, and
every interval of 1..100 with prior (1 - lambda_0) / 100^2. Each example is
drawn uniformly from the concept with probability `reliability` and uniformly
from 1..100 otherwise (size principle); "yes" is `ceiling` times the posterior
probability that the concept contains the target. The one change from
concept_learning_endorsement_ceiling: `reliability` (fixed at 0.9 there) is a
free parameter — how far people trust the examples to come from the concept.
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
RULES = list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1)
LOG_PRIOR_CONST = np.concatenate([np.full(N_RULES, -np.log(N_RULES)), np.full(len(INTERVALS), -2 * np.log(100))])
IS_RULE = np.concatenate([np.ones(N_RULES), np.zeros(len(INTERVALS))])

with pm.Model() as model:
    n_in = pm.Data("n_in", np.zeros((1, MEMBERSHIP.shape[0])))  # (sets, hyps): examples inside h
    n_out = pm.Data("n_out", np.zeros((1, MEMBERSHIP.shape[0])))  # (sets, hyps): examples outside h
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Probability of endorsing a target believed to be in the concept.
    ceiling = pm.Beta("ceiling", 4, 2)
    # Believed probability that an example comes from the concept (0.9 in the original).
    reliability = pm.Beta("reliability", 18, 2)

    log_prior = LOG_PRIOR_CONST + IS_RULE * pt.log(lambda_0) + (1 - IS_RULE) * pt.log1p(-lambda_0)
    log_in = pt.log(reliability / SIZE + (1 - reliability) / 100)  # (hyps,)
    log_out = pt.log1p(-reliability) - np.log(100)
    log_joint = log_prior + n_in * log_in + n_out * log_out  # (sets, hyps)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior @ pt.as_tensor(MEMBERSHIP)  # (sets, targets)

    membership = in_concept.reshape((-1,))[row_cell]
    p_yes = pm.Deterministic("p_yes", pt.clip(ceiling * membership, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    k_in = np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])
    k_total = np.array([len(s) for s in unique], dtype="float64")[:, None]
    return {
        "n_in": k_in,
        "n_out": k_total - k_in,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
