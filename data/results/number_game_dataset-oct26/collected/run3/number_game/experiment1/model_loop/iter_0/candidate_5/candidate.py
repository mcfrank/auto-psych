"""Refinement of `bigelow_piantadosi_independent` with an Erlang size prior on intervals.

A learner who sees the example set D considers 42 rule concepts (powers,
multiples, final digit, contained digit, primes, even, odd, squares, cubes),
each with its own inferred prior weight (Dirichlet(2, ..., 2), total share
p_mix), and the 5,050 intervals [m, n] of 1..100. Each example is drawn from
the concept with probability 0.9 (uniformly: the size principle) and from
1..100 otherwise. The one change from the independent model: the intervals'
share 1 - p_mix is spread not uniformly but by an Erlang density on interval
size, p(h) ~ (|h| / sigma^2) exp(-|h| / sigma) (Tenenbaum 2000), sigma
inferred, so intervals of intermediate size are favoured. "Yes" is the
posterior probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)


def concept_sets():
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
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_SIZE = np.array([len(c) for c in INTERVALS])  # 1..100
SIZES = np.arange(1, 101, dtype="float64")
N_OF_SIZE = 101 - SIZES

# An interval's prior depends only on its size, so for each set the intervals
# of one size act as one hypothesis: their summed evidence, and per asked
# target the share of it from intervals containing the target.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    size_log_evidence = pm.Data("size_log_evidence", np.zeros((1, 100)))  # (sets, sizes)
    size_in_concept = pm.Data("size_in_concept", np.zeros((1, 100, 1)))  # (sets, sizes, targets asked)
    rule_in_concept = pm.Data("rule_in_concept", np.zeros((1, N_RULES, 1)))  # (sets, rules, targets asked)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    sigma = pm.LogNormal("sigma", mu=np.log(15.0), sigma=1.0)

    log_erlang = np.log(SIZES) - 2 * pt.log(sigma) - SIZES / sigma
    log_size_prior = log_erlang - pt.logsumexp(np.log(N_OF_SIZE) + log_erlang)  # per interval

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_sizes = pt.log1p(-p_mix) + log_size_prior[None, :] + size_log_evidence
    log_joint = pt.concatenate([log_rules, log_sizes], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    membership = pt.concatenate([rule_in_concept, size_in_concept], axis=1)
    in_concept = pt.matmul(posterior[:, None, :], membership)[:, 0, :]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    asked = {s: [] for s in unique}
    for s, t in dict.fromkeys(zip(sets, targets)):
        asked[s].append(t)
    slot = {(s, t): i for s in unique for i, t in enumerate(asked[s])}
    width = max(len(ts) for ts in asked.values())
    size_idx = INTERVAL_SIZE - 1
    interval_membership = MEMBERSHIP[N_RULES:]
    rule_ll = np.empty((len(unique), N_RULES))
    size_ev = np.empty((len(unique), 100))
    size_in = np.zeros((len(unique), 100, width))
    rule_in = np.zeros((len(unique), N_RULES, width))
    for u, s in enumerate(unique):
        ll = EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1)
        rule_ll[u] = ll[:N_RULES]
        ill = ll[N_RULES:]
        cols = np.array(asked[s]) - 1
        for k in range(100):
            mask = size_idx == k
            ev = logsumexp(ill[mask])
            size_ev[u, k] = ev
            size_in[u, k, : len(cols)] = np.exp(ill[mask] - ev) @ interval_membership[mask][:, cols]
        rule_in[u, :, : len(cols)] = RULE_MEMBERSHIP[:, cols]
    return {
        "rule_log_lik": rule_ll,
        "size_log_evidence": size_ev,
        "size_in_concept": size_in,
        "rule_in_concept": rule_in,
        "row_cell": np.array([index[s] * width + slot[(s, t)] for s, t in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
