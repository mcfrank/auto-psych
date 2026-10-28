"""Tenenbaum's rules-and-similarity model with a tempered size principle.

Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept
learning over 33 mathematical rules and the 5,050 intervals of 1..100 (rules
share a prior fraction lambda equally, intervals an Erlang density on size),
but they discount the evidence of the examples: the likelihood of the example
set under each concept is raised to a fitted power beta (each example counts
as beta of an independent draw). beta < 1 tempers the size principle and keeps
generalization broad, reaching targets far outside the examples' range; beta = 1
is the original. "Yes" is the posterior probability the concept contains the
target. Examples are drawn from the concept with probability 1 - NOISE, NOISE
fixed at 0.1 as in the original.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
NOISE = 0.1


def is_prime(n):
    return n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))


def rule_concepts():
    rules = [
        [n for n in NUMBERS if n % 2 == 1],
        [n for n in NUMBERS if n % 2 == 0],
        [n * n for n in range(1, 11)],
        [n**3 for n in range(1, 5)],
        [n for n in NUMBERS if is_prime(n)],
    ]
    rules += [list(range(m, 101, m)) for m in range(3, 13)]
    rules += [[m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)]
    rules += [[n for n in NUMBERS if n % 10 == d] for d in range(1, 10)]
    return rules


RULES = rule_concepts()
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, r) for r in RULES], dtype="float64")
RULE_SIZES = RULE_MEMBERSHIP.sum(axis=1)
INTERVAL_SIZES = np.arange(1, 101, dtype="float64")
N_INTERVALS_OF_SIZE = 101 - INTERVAL_SIZES
_STARTS, _ENDS = np.array([(a, b) for a in NUMBERS for b in NUMBERS if a <= b]).T
INTERVAL_MEMBERSHIP = (NUMBERS >= _STARTS[:, None]) & (NUMBERS <= _ENDS[:, None])  # (5050, 100)

# Intervals are grouped by (size, examples contained), as in the original:
# the likelihood depends only on those, so the posterior over groups is exact.

with pm.Model() as model:
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    rule_hits = pm.Data("rule_hits", np.zeros((1, N_RULES)))  # (sets, rules)
    interval_log_count = pm.Data("interval_log_count", np.zeros((1, 100, 2)))  # (sets, size, k)
    target_membership = pm.Data("target_membership", np.zeros((1, N_RULES + 200, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_rules = pm.Beta("lambda_rules", 2, 2)
    sigma = pm.LogNormal("sigma", mu=np.log(10.0), sigma=0.75)
    # Evidence weight of each example: the strength of the size principle.
    beta = pm.LogNormal("beta", mu=np.log(0.7), sigma=0.6)
    log_noise = np.log(NOISE / 100)
    n = n_examples[:, None]

    log_rules = pt.log(lambda_rules) - np.log(N_RULES) + beta * (
        rule_hits * np.log((1 - NOISE) / RULE_SIZES + NOISE / 100)
        + (n - rule_hits) * log_noise
    )

    log_erlang = pt.log(INTERVAL_SIZES / sigma**2) - INTERVAL_SIZES / sigma
    log_size_prior = (
        pt.log1p(-lambda_rules) + log_erlang
        - pt.logsumexp(np.log(N_INTERVALS_OF_SIZE) + log_erlang)
    )
    k = pt.arange(interval_log_count.shape[2])
    log_intervals = (
        interval_log_count
        + log_size_prior[None, :, None]
        + beta * (
            k[None, None, :] * np.log((1 - NOISE) / INTERVAL_SIZES + NOISE / 100)[None, :, None]
            + (n[:, :, None] - k[None, None, :]) * log_noise
        )
    ).reshape((interval_log_count.shape[0], -1))

    log_joint = pt.concatenate([log_rules, log_intervals], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = pt.matmul(posterior[:, None, :], target_membership)[:, 0, :]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
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
    n_k = max(len(s) for s in unique) + 1
    groups = (_ENDS - _STARTS) * n_k
    log_count = np.empty((len(unique), 100, n_k))
    membership = np.zeros((len(unique), N_RULES + 100 * n_k, width))
    for u, examples in enumerate(unique):
        group = groups + INTERVAL_MEMBERSHIP[:, np.array(examples) - 1].sum(axis=1)
        count = np.bincount(group, minlength=100 * n_k).astype("float64")
        share = np.zeros((100 * n_k, 100))
        np.add.at(share, group, INTERVAL_MEMBERSHIP)
        share /= np.maximum(count, 1)[:, None]
        log_count[u] = np.where(count > 0, np.log(np.maximum(count, 1)), -1e30).reshape(100, n_k)
        columns = np.array(asked[examples]) - 1
        membership[u, :, : len(columns)] = np.concatenate([RULE_MEMBERSHIP[:, columns], share[:, columns]])
    return {
        "n_examples": np.array([len(s) for s in unique], dtype="float64"),
        "rule_hits": np.array([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "interval_log_count": log_count,
        "target_membership": membership,
        "row_cell": np.array([index[s] * width + slot[(s, t)] for s, t in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
