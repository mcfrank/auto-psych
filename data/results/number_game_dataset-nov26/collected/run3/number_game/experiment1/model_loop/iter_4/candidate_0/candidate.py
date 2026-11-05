"""Predictive output-probability readout of tempered Bayesian concept learning.

People do tempered Bayesian concept learning over 45 individually weighted
rules and all intervals of 1..100 (size principle, 10% noise examples), but
they answer "would the program produce this target?" by the posterior
predictive probability that the program outputs the target as its next sample
(sum over concepts of posterior x 1/|concept|), not by the probability that
the target is a member of the concept. "Yes" is a logistic function of the log
predictive probability relative to an internal criterion. This predicts that
rule members outside the examples' range, and members of very large rules, are
endorsed less than the membership readout implies.
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
N_INT = len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
# Probability that a concept outputs each number as a sample (size principle + noise).
SAMPLE_PROB = ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100
EXAMPLE_LOG_LIK = np.log(SAMPLE_PROB)
RULE_PRED = SAMPLE_PROB[:N_RULES] * 100.0      # scaled so uniform = 1
INTERVAL_PRED = SAMPLE_PROB[N_RULES:] * 100.0

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    interval_log_lik = pm.Data("interval_log_lik", np.zeros((1, N_INT)))
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
    log_beta = pm.Normal("log_beta", mu=0.0, sigma=0.75)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    # Readout: criterion on log predictive probability, and its sharpness.
    criterion = pm.Normal("criterion", mu=1.0, sigma=1.5)
    log_slope = pm.Normal("log_slope", mu=0.0, sigma=0.75)
    slope = pm.Deterministic("slope", pt.exp(log_slope))

    t_int = beta * interval_log_lik
    int_log_evidence = pt.logsumexp(t_int, axis=1)
    int_post = pt.exp(t_int - int_log_evidence[:, None])
    interval_pred = pt.dot(int_post, pt.as_tensor(INTERVAL_PRED))

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + beta * rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + int_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    pred = posterior[:, :N_RULES] @ pt.as_tensor(RULE_PRED) + posterior[:, N_RULES][:, None] * interval_pred

    score = slope * (pt.log(pred) - criterion)
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(score.reshape((-1,))[row_cell]), 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    ll = np.array([EXAMPLE_LOG_LIK[:, [x - 1 for x in s]].sum(axis=1) for s in unique])
    return {
        "rule_log_lik": ll[:, :N_RULES],
        "interval_log_lik": ll[:, N_RULES:],
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
