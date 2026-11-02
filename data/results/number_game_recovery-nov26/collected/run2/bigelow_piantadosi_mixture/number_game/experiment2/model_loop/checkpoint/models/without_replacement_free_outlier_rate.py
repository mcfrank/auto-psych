"""Refinement of concepts_sampled_without_replacement: Bayesian concept learning
over the same 45 arithmetic and digit rules and every interval of 1..100, with a
rule prior proportional to |rule| ** gamma, strong sampling without replacement
(each example drawn from the concept's members not yet shown) and a response
lapse that guesses "yes" at the participant's own rate. The one change (the
component taken from breadth_prior_concepts_free_outlier_rate): how often
people believe the program shows a stray random number from 1..100 is a
property of the learner, inferred from the answers, instead of being fixed at
one in ten. The k-th example's likelihood under concept h is
(1 - outlier) * [x in h] / (|h| - members of h already shown) + outlier / 100.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

NUMBERS = np.arange(1, 101)
MAX_EX = 4
# The interval part depends on the outlier rate only through smooth per-set sums;
# tabulate them on a Chebyshev grid in log(outlier) over [LO, HI].
LO, HI = np.log(0.01), np.log(0.45)
DEG = 20


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
CONCEPT_SIZE = MEMBERSHIP.sum(axis=1)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
LOG_RULE_SIZE = np.log(CONCEPT_SIZE[:N_RULES])
LOG_RULE_SIZE_C = LOG_RULE_SIZE - LOG_RULE_SIZE.mean()
NODES = np.cos(np.pi * (np.arange(DEG + 1) + 0.5) / (DEG + 1))  # Chebyshev nodes in [-1, 1]


def member_rates(examples, concept_size, membership):
    """Per example: [x in h] / (|h| - members already shown); repeats count as strays (0)."""
    rates = np.zeros((len(concept_size), MAX_EX))
    used = np.zeros(MAX_EX)
    shown = np.zeros(len(concept_size))
    seen = set()
    for k, x in enumerate(examples):
        used[k] = 1.0
        if x in seen:
            continue
        seen.add(x)
        in_h = membership[:, x - 1]
        rates[:, k] = in_h / np.maximum(concept_size - shown, 1.0)
        shown += in_h
    return rates, used


def interval_tables(examples):
    """Chebyshev coefficients (in log outlier) of the interval log evidence and of
    the log evidence of the intervals containing each target."""
    rates, used = member_rates(examples, CONCEPT_SIZE[N_RULES:], INTERVAL_MEMBERSHIP)
    vals_e, vals_t = [], []
    for z in NODES:
        eps = np.exp(LO + (z + 1) * (HI - LO) / 2)
        ll = (used[None, :] * np.log((1 - eps) * rates + eps / 100)).sum(axis=1) - np.log(len(INTERVALS))
        m = ll.max()
        w = np.exp(ll - m)
        vals_e.append(np.log(w.sum()) + m)
        vals_t.append(np.log(w @ INTERVAL_MEMBERSHIP) + m)
    coef_e = np.polynomial.chebyshev.chebfit(NODES, np.array(vals_e), DEG)
    coef_t = np.polynomial.chebyshev.chebfit(NODES, np.array(vals_t), DEG)  # (DEG+1, 100)
    return coef_e, coef_t.T


with pm.Model() as model:
    rule_rate = pm.Data("rule_rate", np.zeros((1, N_RULES, MAX_EX)))  # (sets, rules, examples)
    example_used = pm.Data("example_used", np.zeros((1, MAX_EX)))  # (sets, examples)
    int_coef_e = pm.Data("int_coef_e", np.zeros((1, DEG + 1)))  # (sets, coefs)
    int_coef_t = pm.Data("int_coef_t", np.zeros((1, 100, DEG + 1)))  # (sets, targets, coefs)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)  # prior preference for rule breadth
    outlier = pm.Beta("outlier", 2, 10)  # believed rate of stray random examples

    # Rules: exact without-replacement likelihood.
    rule_log_lik = (
        example_used[:, None, :] * pt.log((1 - outlier) * rule_rate + outlier / 100.0)
    ).sum(axis=2)
    log_rule_prior = gamma * LOG_RULE_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    rule_joint = rule_log_lik + log_rule_prior[None, :]
    rule_log_evidence = pt.logsumexp(rule_joint, axis=1)
    rule_post = pt.exp(rule_joint - rule_log_evidence[:, None])
    rule_in_concept = pt.dot(rule_post, RULE_MEMBERSHIP)  # (sets, targets)

    # Intervals: Chebyshev series in log(outlier).
    z = pt.clip(2 * (pt.log(outlier) - LO) / (HI - LO) - 1, -1.0, 1.0)
    cheb = [pt.ones_like(z), z]
    for _ in range(DEG - 1):
        cheb.append(2 * z * cheb[-1] - cheb[-2])
    cheb = pt.stack(cheb)  # (DEG+1,)
    interval_log_evidence = pt.dot(int_coef_e, cheb)  # (sets,)
    interval_in_concept = pt.exp(pt.dot(int_coef_t, cheb) - interval_log_evidence[:, None])

    log_rules = pt.log(lambda_0) + rule_log_evidence
    log_intervals = pt.log1p(-lambda_0) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_rules - log_intervals)[:, None]
    in_concept = p_rule * rule_in_concept + (1 - p_rule) * interval_in_concept

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
    rule_rate, used, coef_e, coef_t = [], [], [], []
    for s in unique:
        rr, u = member_rates(s, CONCEPT_SIZE[:N_RULES], RULE_MEMBERSHIP)
        ce, ct = interval_tables(s)
        rule_rate.append(rr)
        used.append(u)
        coef_e.append(ce)
        coef_t.append(ct)
    return {
        "rule_rate": np.stack(rule_rate),
        "example_used": np.stack(used),
        "int_coef_e": np.stack(coef_e),
        "int_coef_t": np.stack(coef_t),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
