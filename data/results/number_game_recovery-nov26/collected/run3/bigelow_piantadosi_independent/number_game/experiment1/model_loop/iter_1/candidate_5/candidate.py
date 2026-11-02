"""Refinement of concept_posterior_with_lapse_guessing: Bayesian concept
learning over 45 rules and every interval of 1..100 with a lapse-and-guess
response stage, in which parity (odd / even) is a privileged concept with a
higher prior than the other rules.

A learner who sees the example set D considers 45 rule concepts (odd, even,
primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37,
multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and the
5,050 intervals of 1..100. A rule is chosen with probability lambda_0 (rules
equally likely); each interval has prior (1 - lambda_0) / 100^2. Each example
is drawn from the concept with probability 0.9 (the size principle) and from
1..100 otherwise; the concept posterior gives P(target in concept).

The single change: on a fraction `lapse` of trials the participant ignores
the examples and answers "yes" with base rate `guess`, so
p_yes = (1 - lapse) * P(target in concept) + lapse * guess. This makes
responses to the same (set, target) less deterministic than the posterior
alone implies (inherited unchanged).

The single change in this refinement: the rule prior is no longer uniform.
Odd and even each get their rule prior multiplied by a fitted salience factor
exp(parity_boost) (rules renormalised so total rule mass stays lambda_0).
Parity is the most accessible property of a number, so it survives the size
principle more often than a uniform prior allows: people generalise to
parity-matching targets of an all-even or all-odd set far more than the plain
posterior predicts.
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


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
# log P(x | h) of one example x (column x - 1) under each hypothesis.
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
IS_PARITY = np.array([name in ("odd", "even") for name in RULE_NAMES], dtype="float64")

# Every interval has the same prior and a likelihood free of parameters, so
# for each example set the 5,050 intervals act as one hypothesis: total
# evidence sum_j P(D | interval j), and in-concept probability for each target
# the share of that evidence from intervals containing it. prepare_observed
# computes both per set; the posterior below is exactly the one over all 5,095
# hypotheses, at ~1/100 of the cost per gradient.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | rule)
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))  # (sets,)
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))  # (sets, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)

    parity_boost = pm.Normal("parity_boost", 0.0, 3.0)
    rule_log_prior = pt.as_tensor(IS_PARITY) * parity_boost
    rule_log_prior = rule_log_prior - pt.logsumexp(rule_log_prior)
    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + 1)
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES][:, None] * interval_in_concept

    # One flat index per response: half the gradient cost of 2-D indexing.
    # Rounding can put a sure membership a few ulps above 1.
    lapse = pm.Beta("lapse", 1, 9)
    guess = pm.Beta("guess", 2, 2)
    p_yes = pm.Deterministic("p_yes", (1 - lapse) * pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0) + lapse * guess)
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
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
