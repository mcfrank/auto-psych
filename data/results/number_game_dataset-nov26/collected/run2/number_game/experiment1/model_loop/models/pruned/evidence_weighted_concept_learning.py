"""Evidence-weighted (conservative) Bayesian concept learning.

People do Bayesian concept learning over 45 mathematical rules and all
intervals of 1..100 with the size principle (Bigelow & Piantadosi's
Independent model: a prior weight per rule, share lambda_0 for rules, each
interval (1 - lambda_0) / 100^2, 10% uniform example noise), but they mis-weight
the evidence relative to the prior: the likelihood of the examples enters the
update raised to one inferred power beta (beta < 1: conservatism, under-updating;
beta > 1: over-commitment). "Yes" = posterior probability that the concept
contains the target.
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
    # "contains digit 0" equals "ends in 0" on 1..100; merged into one rule.
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(1, 10)})
    return rules


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEM = MEMBERSHIP[:N_RULES]
INT_MEM = MEMBERSHIP[N_RULES:]


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    ll = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    return {
        "rule_log_lik": ll[:, :N_RULES],
        "interval_log_lik": ll[:, N_RULES:],
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    interval_log_lik = pm.Data("interval_log_lik", np.zeros((1, len(INTERVALS))))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    beta = pm.LogNormal("evidence_weight", mu=0.0, sigma=0.75)  # power on the likelihood

    # Intervals share one prior, so they collapse to one tempered evidence term.
    int_scaled = beta * interval_log_lik
    int_lse = pt.logsumexp(int_scaled, axis=1)  # (S,)
    int_w = pt.exp(int_scaled - int_lse[:, None])
    int_in = pt.dot(int_w, pt.as_tensor(INT_MEM))  # (S, 100)

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + beta * rule_log_lik
    log_int = pt.log1p(-lambda_0) - 2 * np.log(100) + int_lse
    log_joint = pt.concatenate([log_rules, log_int[:, None]], axis=1)
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = pt.dot(post[:, :N_RULES], pt.as_tensor(RULE_MEM)) + post[:, N_RULES][:, None] * int_in

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
