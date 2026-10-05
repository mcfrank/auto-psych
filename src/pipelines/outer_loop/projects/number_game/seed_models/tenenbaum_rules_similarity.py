"""Tenenbaum's (2000) Bayesian model of the number game: rules and similarity.

"Rules and similarity in concept learning" (NIPS 12). A learner who sees the
example set X considers 5,083 hypotheses about the concept: 33 mathematical
properties (odd, even, squares, cubes, primes, multiples of 3-12, powers of
2-10, numbers ending in 1-9) and the 5,050 intervals [a, b] of 1..100. A
fraction lambda of the prior is spread uniformly over the properties, the rest
over the intervals by an Erlang density on their size,
p(h) ~ (|h| / sigma^2) exp(-|h| / sigma), so intervals of intermediate size
are favoured. Examples are sampled at random from the concept (the size
principle), and the probability of answering "yes" to a target is the
posterior probability that the concept contains it. His fit has lambda = 1/2,
sigma = 10; here both are fitted.

One addition: with pure strong sampling a target that is one of the examples
gets p(yes) = 1, and people answer "no" to such targets about a quarter of the
time. As in Bigelow & Piantadosi (2016), each example is drawn from the
concept with probability 1 - noise and uniformly from 1..100 otherwise; they
fix noise at 0.1, here it is fitted with a prior favouring small values.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


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

# An interval's likelihood depends only on its size s and the number k of
# examples it contains, ((1 - noise) / s + noise / 100)^k (noise / 100)^(n - k),
# and its prior only on s. So for each example set prepare_observed counts the
# intervals in each (s, k) group, and for each target the set was asked about
# the share of each group that contains it; the posterior over the groups is
# then exactly the one over all 5,050 intervals. Only the targets with
# responses are computed (padded to the most any set has): every target on
# the full dataset, a handful per set in a loop experiment.

with pm.Model() as model:
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    rule_hits = pm.Data("rule_hits", np.zeros((1, N_RULES)))  # (sets, rules): examples in the rule
    interval_log_count = pm.Data("interval_log_count", np.zeros((1, 100, 2)))  # (sets, size, k)
    # (sets, rules + size * k, targets asked): each asked target's membership
    # in each rule, then the share of each interval group that contains it.
    target_membership = pm.Data("target_membership", np.zeros((1, N_RULES + 200, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * targets asked + slot
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_rules = pm.Beta("lambda_rules", 2, 2)
    sigma = pm.LogNormal("sigma", mu=np.log(10.0), sigma=0.75)
    noise = pm.Beta("noise", 1, 19)

    log_noise = pt.log(noise / 100)
    n = n_examples[:, None]

    log_rules = (
        pt.log(lambda_rules) - np.log(N_RULES)
        + rule_hits * pt.log((1 - noise) / RULE_SIZES + noise / 100)
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
        + k[None, None, :] * pt.log((1 - noise) / INTERVAL_SIZES + noise / 100)[None, :, None]
        + (n[:, :, None] - k[None, None, :]) * log_noise
    ).reshape((interval_log_count.shape[0], -1))

    log_joint = pt.concatenate([log_rules, log_intervals], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = pt.matmul(posterior[:, None, :], target_membership)[:, 0, :]  # (sets, targets asked)

    # Rounding can put a sure membership a few ulps above 1.
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0))
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
    groups = (_ENDS - _STARTS) * n_k  # size index * n_k, plus k below
    log_count = np.empty((len(unique), 100, n_k))
    membership = np.zeros((len(unique), N_RULES + 100 * n_k, width))
    for u, examples in enumerate(unique):
        group = groups + INTERVAL_MEMBERSHIP[:, np.array(examples) - 1].sum(axis=1)
        count = np.bincount(group, minlength=100 * n_k).astype("float64")
        share = np.zeros((100 * n_k, 100))
        np.add.at(share, group, INTERVAL_MEMBERSHIP)
        share /= np.maximum(count, 1)[:, None]
        # Empty groups get no weight: exp(-1e30) == 0 with finite gradients.
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
