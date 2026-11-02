"""Refinement of rule_breadth_prior_concepts: Bayesian concept learning over
the same 45 rules (prior proportional to |rule| ** gamma) and every interval
of 1..100 (size principle, 10% of examples uniform, guess lapse). People also
entertain "counting by twos" runs, bounded stretches of every other number
(a, a+2, ..., b, at least two members), such as the even numbers 60..66.
All runs share one prior equally, and the family's total prior mass relative
to the rules is exp(log_run_share), which is inferred.
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


def twos_runs():
    """Every run a, a+2, ..., b inside 1..100 with at least two members."""
    return [list(range(a, b + 1, 2)) for a in NUMBERS for b in range(a + 2, 101, 2)]


RULES = rule_concepts()
RUNS = twos_runs()
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]


def membership(concepts):
    return np.array([np.isin(NUMBERS, c) for c in concepts], dtype="float64")


def example_log_lik(mem):
    """log P(x | h) of one example x (column x - 1) under each hypothesis."""
    return np.log(ALPHA * mem / mem.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


RULE_MEMBERSHIP = membership(RULES)
RUN_MEMBERSHIP = membership(RUNS)
INTERVAL_MEMBERSHIP = membership(INTERVALS)
RULE_LL = example_log_lik(RULE_MEMBERSHIP)
RUN_LL = example_log_lik(RUN_MEMBERSHIP)
INTERVAL_LL = example_log_lik(INTERVAL_MEMBERSHIP)
LOG_RULE_SIZE = np.log(RULE_MEMBERSHIP.sum(axis=1))
LOG_RULE_SIZE_C = LOG_RULE_SIZE - LOG_RULE_SIZE.mean()


def class_summary(ll_table, mem, examples):
    """Mean-prior log evidence of a class (uniform within it) and its posterior membership."""
    ll = ll_table[:, np.array(examples) - 1].sum(axis=1)
    ev = logsumexp(ll)
    return ev - np.log(len(ll)), np.exp(ll - ev) @ mem


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | r)
    run_log_evidence = pm.Data("run_log_evidence", np.zeros(1))  # (sets,): mean over runs of P(D | run)
    run_in_concept = pm.Data("run_in_concept", np.zeros((1, 100)))
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))  # mean over intervals
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)  # rules' prior share against intervals
    gamma = pm.Normal("gamma", 0.0, 1.0)  # prior preference for rule breadth
    log_run_share = pm.Normal("log_run_share", -3.0, 1.5)  # runs' total prior mass relative to the rules

    log_rule_prior = gamma * LOG_RULE_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    rule_joint = rule_log_lik + log_rule_prior[None, :]
    rule_log_evidence = pt.logsumexp(rule_joint, axis=1)
    rule_post = pt.exp(rule_joint - rule_log_evidence[:, None])
    rule_in_concept = pt.dot(rule_post, RULE_MEMBERSHIP)

    log_rules = pt.log(lambda_0) + rule_log_evidence
    log_runs = pt.log(lambda_0) + log_run_share + run_log_evidence
    log_intervals = pt.log1p(-lambda_0) + interval_log_evidence
    class_post = pt.special.softmax(pt.stack([log_rules, log_runs, log_intervals], axis=1), axis=1)
    in_concept = (
        class_post[:, 0:1] * rule_in_concept
        + class_post[:, 1:2] * run_in_concept
        + class_post[:, 2:3] * interval_in_concept
    )

    p_concept = pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    runs = [class_summary(RUN_LL, RUN_MEMBERSHIP, s) for s in unique]
    intervals = [class_summary(INTERVAL_LL, INTERVAL_MEMBERSHIP, s) for s in unique]
    return {
        "rule_log_lik": np.stack([RULE_LL[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "run_log_evidence": np.array([e for e, _ in runs]),
        "run_in_concept": np.stack([m for _, m in runs]),
        "interval_log_evidence": np.array([e for e, _ in intervals]),
        "interval_in_concept": np.stack([m for _, m in intervals]),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
