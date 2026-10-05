"""Bayesian concept learning over rules and intervals (Tenenbaum 1999/2000),
with the grammar prior inferred from data as in Bigelow & Piantadosi (2016).

A PyMC port of Eric Bigelow's numpyro model (`number_bpl_model`,
`prior_type='basic'`; reference copy in `scripts/number_game/numpyro_reference/`,
checked by `tests/test_number_game_seed_port.py`). A learner who sees the
example set D considers 42 rule concepts (powers, multiples, final digit,
contained digit, primes, even, odd, squares, cubes) and the 5,050 intervals
[m, n] of 1..100. Each example is drawn from the concept with probability
0.9 (uniformly: the size principle) and from 1..100 otherwise. The prior puts
p_mix on the rules, shared by Dirichlet(2, ..., 2) weights, and 1 - p_mix
uniformly on the intervals. (The reference draws Gamma(2, 1) weights and
normalises them, which is the same prior on the normalised weights; their
unnormalised total entered nothing but the prior and mixed slowly.) The
probability of answering "yes" to a target is the posterior probability that
the concept contains it.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)


def concept_sets():
    """The reference's `np_utils.get_concept_sets`: 42 rules, then 5,050 intervals."""
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
# log P(x | h) of one example x (column x - 1) under each hypothesis.
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]

# Every interval has the same prior, (1 - p_mix) / N_INTERVALS, and a likelihood
# free of parameters, so for each example set the 5,050 intervals act as one
# hypothesis: total evidence sum_j P(D | interval j), and in-concept probability
# for each target the share of that evidence from intervals containing it.
# prepare_observed computes both per set; the posterior below is exactly the
# one over all 5,092 hypotheses, at ~1/100 of the cost per gradient.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | rule)
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))  # (sets,)
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))  # (sets, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_intervals = pt.log1p(-p_mix) - np.log(N_INTERVALS) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + 1)
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES:] * interval_in_concept

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
