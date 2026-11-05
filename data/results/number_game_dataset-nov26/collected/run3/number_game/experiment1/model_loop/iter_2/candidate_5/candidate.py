"""Refinement of sampled_hypothesis_majority_vote: sampled concepts with a free decision criterion.

People do the same tempered Bayesian concept learning over 45 individually
weighted rules and all intervals of 1..100, bring k = 1 + n concepts to mind
by sampling them from the posterior, and say "yes" only if at least a
criterion share theta of the sampled concepts contains the target. With
m = 1 + theta * n required "contains" votes, P(yes) = P(Binomial(k, q) >= m)
= I_q(m, k - m + 1) = I_q(1 + theta n, 1 + (1 - theta) n), continued to real
n >= 0, where q is the posterior probability the concept contains the target.
theta = 1/2 recovers majority vote; n = 0 (one sample) recovers the incumbent.
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
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]

# "ends in 0" and "contains digit 0" are the same set; sample their summed
# weight plus a Beta split (Dirichlet aggregation keeps the Dirichlet(2,...,2)).
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    interval_log_lik = pm.Data("interval_log_lik", np.zeros((1, len(INTERVALS))))  # (sets, intervals)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    # Evidence-strength exponent on the size-principle likelihood.
    log_beta = pm.Normal("log_beta", mu=0.0, sigma=0.75)
    beta = pm.Deterministic("beta", pt.exp(log_beta))

    t_int = beta * interval_log_lik
    int_log_evidence = pt.logsumexp(t_int, axis=1)
    int_post = pt.exp(t_int - int_log_evidence[:, None])
    interval_in_concept = pt.dot(int_post, pt.as_tensor(INTERVAL_MEMBERSHIP))

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + beta * rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + int_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES][:, None] * interval_in_concept

    # Number of extra sampled concepts n (k = 1 + n) and the criterion share theta.
    n_extra = pm.HalfNormal("n_extra", sigma=4.0)
    theta = pm.Beta("theta", 2.0, 2.0)
    q = pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6)
    majority = pt.betainc(1.0 + theta * n_extra, 1.0 + (1.0 - theta) * n_extra, q)
    p_yes = pm.Deterministic("p_yes", pt.clip(majority, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "interval_log_lik": set_log_lik[:, N_RULES:],
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
