"""Tempered size principle in Bayesian concept learning (refines the Mixture
Model of Bigelow & Piantadosi, 2016).

A learner considers 45 equally likely rule concepts (odd, even, primes,
squares, cubes, powers of 2-10, powers of 2 without 32 / with 37, multiples of
3-9, 11, 12, ending in 0-9, containing digit 0-9) with total prior lambda_0,
and the 5,050 intervals of 1..100, each with prior (1 - lambda_0) / 100^2.
Each example is drawn from the concept with probability 0.9 (uniformly: the
size principle) and from 1..100 otherwise. The single change from the mixture
model: people weigh the examples' evidence only partially, raising the
likelihood P(D | h) to a power beta (beta < 1 = a weakened size principle,
beta = 1 = the original model). "Yes" is the posterior probability that the
concept contains the target.
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
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
N_HYP = MEMBERSHIP.shape[0]
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LOG_PRIOR_SHAPE = np.r_[np.zeros(N_RULES), np.ones(N_HYP - N_RULES)]  # 0 = rule, 1 = interval

with pm.Model() as model:
    set_log_lik = pm.Data("set_log_lik", np.zeros((1, N_HYP)))  # (sets, hypotheses): log P(D | h)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # size-principle strength

    is_int = pt.as_tensor(LOG_PRIOR_SHAPE)
    log_prior = (1 - is_int) * (pt.log(lambda_0) - np.log(N_RULES)) + is_int * (pt.log1p(-lambda_0) - 2 * np.log(100))
    log_joint = beta * set_log_lik + log_prior[None, :]
    posterior = pt.special.softmax(log_joint, axis=1)
    in_concept = posterior @ pt.as_tensor(MEMBERSHIP)  # (sets, 100)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    return {
        "set_log_lik": np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
