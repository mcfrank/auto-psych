"""Refinement of bigelow_piantadosi_mixture: conservative (tempered) evidence.

Bayesian concept learning over 45 rules and the 5,050 intervals of 1..100,
rules with prior lambda_0 / 45, each interval (1 - lambda_0) / 100^2, examples
sampled from the concept with probability 0.9 (size principle) and uniformly
otherwise. The single change: learners under-weight the examples' evidence,
raising the likelihood P(D | h) to a power beta (beta < 1 = conservative), so
they are less confident in the smallest consistent rule. "Yes" is the
posterior probability that the concept contains the target.
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
RULES = list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
N_HYP = MEMBERSHIP.shape[0]
LOG_PRIOR_BASE_RULE = -np.log(N_RULES)
LOG_PRIOR_BASE_INT = -2 * np.log(100)
IS_RULE = np.zeros(N_HYP)
IS_RULE[:N_RULES] = 1.0

with pm.Model() as model:
    set_log_lik = pm.Data("set_log_lik", np.zeros((1, N_HYP)))  # (sets, hyps): log P(D | h)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Evidence weight: 1 = ideal learner, < 1 = conservative.
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)

    is_rule = pt.as_tensor(IS_RULE)
    log_prior = is_rule * (pt.log(lambda_0) + LOG_PRIOR_BASE_RULE) + (1 - is_rule) * (
        pt.log1p(-lambda_0) + LOG_PRIOR_BASE_INT
    )
    log_joint = log_prior[None, :] + beta * set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior @ pt.as_tensor(MEMBERSHIP)  # (sets, 100)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_ll = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    return {
        "set_log_lik": set_ll,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
