"""Refinement of concept_learning_with_guess_lapse: Bayesian concept learning
over the same 45 rules and every interval of 1..100 (size principle, 10% of
examples drawn uniformly), with a response lapse -- on a fraction `lapse` of
trials the participant ignores the concept and answers "yes" with probability
`guess_yes`. The one change: the prior over rules is a specificity bias, not 45
free weights. Rule r gets prior lambda_0 * |r|^-g / sum_r' |r'|^-g, so g > 0
favours narrow rules before any example is seen (g = 0 is the equal prior).
Each interval keeps prior (1 - lambda_0) / 5050.
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
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
# log P(x | h) of one example x (column x - 1) under each hypothesis.
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
LOG_RULE_SIZE = np.log(RULE_MEMBERSHIP.sum(axis=1))
SAMPLER_SETTINGS = {"target_accept": 0.9}

# Intervals keep equal priors, so prepare_observed collapses them per example
# set (total evidence and evidence-weighted membership of each target); the
# 45 rules stay separate because their prior depends on g.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | r)
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))  # (sets,)
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))  # (sets, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    g = pm.Normal("g", mu=0.0, sigma=1.0)  # specificity bias of the rule prior
    log_prior_rule = -g * LOG_RULE_SIZE
    log_prior_rule = log_prior_rule - pt.logsumexp(log_prior_rule) + pt.log(lambda_0)
    log_rules = log_prior_rule[None, :] + rule_log_lik  # (sets, rules)
    log_rule_total = pt.logsumexp(log_rules, axis=1, keepdims=True)
    log_intervals = pt.log1p(-lambda_0) - np.log(len(INTERVALS)) + interval_log_evidence[:, None]
    log_total = pt.logaddexp(log_rule_total, log_intervals)
    rule_post = pt.exp(log_rules - log_total)  # (sets, rules)
    p_interval = pt.exp(log_intervals - log_total)  # (sets, 1)
    in_concept = pt.dot(rule_post, RULE_MEMBERSHIP) + p_interval * interval_in_concept

    # Rounding can put a sure membership a few ulps above 1.
    p_concept = pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)  # share of trials answered by guessing
    guess_yes = pm.Beta("guess_yes", 2, 2)  # P(yes) when guessing
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    interval_log_evidence = logsumexp(interval_log_lik, axis=1)
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "interval_log_evidence": interval_log_evidence,
        "interval_in_concept": np.exp(interval_log_lik - interval_log_evidence[:, None]) @ MEMBERSHIP[N_RULES:],
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
