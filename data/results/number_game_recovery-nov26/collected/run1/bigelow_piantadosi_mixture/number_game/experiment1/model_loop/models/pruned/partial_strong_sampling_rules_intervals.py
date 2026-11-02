"""Tenenbaum's rules-and-similarity number-game model with partial strong sampling.

Refinement of `tenenbaum_rules_similarity`. A learner who sees the example set
X considers 33 mathematical rules (odd, even, squares, cubes, primes, multiples
of 3-12, powers of 2-10, ending in 1-9) and the 5,050 intervals of 1..100. A
fraction lambda of the prior is shared equally by the rules, the rest by the
intervals via an Erlang density on their size (scale sigma). The change: each
example is believed to have been sampled from the concept (strong sampling,
P = 1/|h|) only with probability theta, and otherwise to be an arbitrary
number of 1..100 that happens to fall in the concept (weak sampling, P =
1/100, independent of |h|). theta is fitted, so the size principle applies
only partially. As in the original, 10% of examples are believed to be pure
noise (uniform on 1..100, inside or outside the concept), so a target that is
an example is not a certain "yes". "Yes" is the posterior probability that the
concept contains the target.

Fixes the original's cross-validation crash: data shapes no longer depend on
which sets or targets a fold contains (k always runs 0..4, all 100 targets).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
NOISE = 0.1
N_K = 5  # examples in an interval: 0..4


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
# examples inside it, and its prior only on s, so intervals are grouped by
# (s, k) per set in prepare_observed; the posterior over groups is exact.

with pm.Model() as model:
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    rule_hits = pm.Data("rule_hits", np.zeros((1, N_RULES)))  # (sets, rules)
    interval_log_count = pm.Data("interval_log_count", np.zeros((1, 100, N_K)))  # (sets, size, k)
    # (sets, rules + size * k, 100 targets): membership of each target.
    target_membership = pm.Data("target_membership", np.zeros((1, N_RULES + 100 * N_K, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_rules = pm.Beta("lambda_rules", 2, 2)
    sigma = pm.LogNormal("sigma", mu=np.log(10.0), sigma=0.75)
    theta = pm.Beta("theta", 2, 2)  # probability an example was strongly sampled

    log_noise = np.log(NOISE / 100)
    n = n_examples[:, None]

    def log_in(sizes):
        # log P(x | h) for an example x inside h of the given size(s)
        return pt.log((1 - NOISE) * (theta / sizes + (1 - theta) / 100) + NOISE / 100)

    log_rules = (
        pt.log(lambda_rules) - np.log(N_RULES)
        + rule_hits * log_in(RULE_SIZES)[None, :]
        + (n - rule_hits) * log_noise
    )

    log_erlang = pt.log(INTERVAL_SIZES / sigma**2) - INTERVAL_SIZES / sigma
    log_size_prior = (
        pt.log1p(-lambda_rules) + log_erlang
        - pt.logsumexp(np.log(N_INTERVALS_OF_SIZE) + log_erlang)
    )
    k = np.arange(N_K, dtype="float64")
    log_intervals = (
        interval_log_count
        + log_size_prior[None, :, None]
        + k[None, None, :] * log_in(INTERVAL_SIZES)[None, :, None]
        + (n[:, :, None] - k[None, None, :]) * log_noise
    ).reshape((interval_log_count.shape[0], -1))

    log_joint = pt.concatenate([log_rules, log_intervals], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = pt.matmul(posterior[:, None, :], target_membership)[:, 0, :]  # (sets, 100)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    groups = (_ENDS - _STARTS) * N_K
    log_count = np.empty((len(unique), 100, N_K))
    membership = np.zeros((len(unique), N_RULES + 100 * N_K, 100))
    for u, examples in enumerate(unique):
        # k counts distinct example positions inside; repeats count each time
        group = groups + INTERVAL_MEMBERSHIP[:, np.array(examples) - 1].sum(axis=1)
        count = np.bincount(group, minlength=100 * N_K).astype("float64")
        share = np.zeros((100 * N_K, 100))
        np.add.at(share, group, INTERVAL_MEMBERSHIP.astype("float64"))
        share /= np.maximum(count, 1)[:, None]
        log_count[u] = np.where(count > 0, np.log(np.maximum(count, 1)), -1e30).reshape(100, N_K)
        membership[u] = np.concatenate([RULE_MEMBERSHIP, share])
    return {
        "n_examples": np.array([len(s) for s in unique], dtype="float64"),
        "rule_hits": np.array([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "interval_log_count": log_count,
        "target_membership": membership,
        "row_cell": np.array(
            [index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"
        ),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
