"""Bayesian concept learning over rules and intervals, rules equally likely:
the Mixture Model of Bigelow & Piantadosi (2016).

"Inferring priors in compositional cognitive models" (CogSci 2016), after
Tenenbaum (2000). A learner who sees the example set D considers 45 rule
concepts (odd, even, primes, squares, cubes, powers of 2-10, powers of 2
without 32, powers of 2 with 37, multiples of 3-9, 11 and 12, numbers ending
in 0-9, numbers containing the digit 0-9) and the 5,050 intervals [m, n] of
1..100. The grammar's first choice is a rule, with probability lambda_0, or an
interval; a rule is then drawn uniformly, and an interval as two numbers of
1..100, so each interval has prior (1 - lambda_0) / 100^2 (a pair with m > n
is the empty set and no concept). Each example is drawn from the concept with
probability 0.9 (uniformly: the size principle) and from 1..100 otherwise.
The probability of answering "yes" to a target is the posterior probability
that the concept contains it. lambda_0 is the only parameter; the Gamma(2, 1)
prior on each of the two grammar weights it normalises makes it Beta(2, 2).

The concepts are those of the paper's implementation, LOTlib's
`independent_grammar` (the paper's Mixture Model has the Independent model's
concepts), except that odd numbers include 1 and primes exclude it.
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

    log_rules = pt.log(lambda_0) - np.log(N_RULES) + rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + 1)
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES][:, None] * interval_in_concept

    # One flat index per response: half the gradient cost of 2-D indexing.
    # Rounding can put a sure membership a few ulps above 1.
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0))
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
