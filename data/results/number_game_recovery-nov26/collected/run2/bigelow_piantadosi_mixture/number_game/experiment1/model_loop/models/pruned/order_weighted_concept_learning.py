"""Order-weighted Bayesian concept learning (sequential attention to examples).

Refinement of `tenenbaum_digit_concepts`. People do Bayesian concept learning
over Tenenbaum's 33 mathematical rules plus 11 digit concepts ("contains the
digit d", "ends in 0"), sharing a prior fraction lambda equally, and the 5,050
intervals of 1..100 (Erlang prior on size, scale sigma), with examples
sampled from the concept (size principle, 10% uniform noise). The one change:
examples are read in sequence and attention to each shifts with its position,
so example i's log-likelihood is weighted by n * softmax(gamma * i). gamma > 0
means later examples dominate (recency), gamma < 0 earlier ones (primacy); the
weights sum to n, so the total evidence is unchanged and only its distribution
over the examples is. "Yes" is the posterior probability that the concept
contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
NOISE = 0.1
MAX_N = 4
N_PATTERNS = 2**MAX_N


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
    rules += [[n for n in NUMBERS if n % 10 == 0]]
    rules += [[n for n in NUMBERS if str(d) in str(n)] for d in range(10)]
    return rules


RULES = rule_concepts()
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, r) for r in RULES], dtype="float64")
RULE_SIZES = RULE_MEMBERSHIP.sum(axis=1)
INTERVAL_SIZES = np.arange(1, 101, dtype="float64")
N_INTERVALS_OF_SIZE = 101 - INTERVAL_SIZES
_STARTS, _ENDS = np.array([(a, b) for a in NUMBERS for b in NUMBERS if a <= b]).T
INTERVAL_MEMBERSHIP = (NUMBERS >= _STARTS[:, None]) & (NUMBERS <= _ENDS[:, None])  # (5050, 100)
# Three smooth parameters: 0.99 acceptance needlessly shrinks NUTS steps.
SAMPLER_SETTINGS = {"target_accept": 0.9}

# PATTERN_BITS[m, i] = 1 if example position i is inside the concept in hit pattern m.
PATTERN_BITS = np.array([[(m >> i) & 1 for i in range(MAX_N)] for m in range(N_PATTERNS)], dtype="float64")

# An interval's likelihood depends only on its size and on which example
# positions it contains (a bit pattern), so prepare_observed groups each set's
# intervals by (size, pattern), keeping only non-empty groups; the posterior
# over groups equals the one over all 5,050 intervals.

with pm.Model() as model:
    position_mask = pm.Data("position_mask", np.ones((1, MAX_N)))  # (sets, positions): 1 if shown
    rule_hits = pm.Data("rule_hits", np.zeros((1, N_RULES, MAX_N)))  # (sets, rules, positions)
    # Non-empty (size, pattern) interval groups of each set, padded to the most any set has.
    group_log_count = pm.Data("group_log_count", np.zeros((1, 1)))  # (sets, groups)
    group_size = pm.Data("group_size", np.zeros((1, 1), dtype="int64"))  # size index 0..99
    group_pattern = pm.Data("group_pattern", np.zeros((1, 1), dtype="int64"))  # hit pattern 0..15
    target_membership = pm.Data("target_membership", np.zeros((1, N_RULES + 1, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_rules = pm.Beta("lambda_rules", 2, 2)
    sigma = pm.LogNormal("sigma", mu=np.log(10.0), sigma=0.75)
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)  # position-attention slope
    log_noise = np.log(NOISE / 100)

    # Attention weights over the shown positions, summing to the number shown.
    n = position_mask.sum(axis=1, keepdims=True)  # (sets, 1)
    raw = pt.exp(gamma * (pt.arange(MAX_N)[None, :] - (n - 1))) * position_mask
    w = n * raw / raw.sum(axis=1, keepdims=True)  # (sets, positions)

    rule_wh = (rule_hits * w[:, None, :]).sum(axis=2)  # (sets, rules)
    log_rules = (
        pt.log(lambda_rules) - np.log(N_RULES)
        + rule_wh * np.log((1 - NOISE) / RULE_SIZES + NOISE / 100)
        + (n - rule_wh) * log_noise
    )

    log_erlang = pt.log(INTERVAL_SIZES / sigma**2) - INTERVAL_SIZES / sigma
    log_size_prior = (
        pt.log1p(-lambda_rules) + log_erlang
        - pt.logsumexp(np.log(N_INTERVALS_OF_SIZE) + log_erlang)
    )
    pattern_wh = pt.dot(w, PATTERN_BITS.T)  # (sets, patterns)
    group_wh = pt.take_along_axis(pattern_wh, group_pattern, axis=1)  # (sets, groups)
    log_size_like = pt.as_tensor(np.log((1 - NOISE) / INTERVAL_SIZES + NOISE / 100))
    log_intervals = (
        group_log_count
        + log_size_prior[group_size]
        + group_wh * log_size_like[group_size]
        + (n - group_wh) * log_noise
    )

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
    sizes = _ENDS - _STARTS  # size index 0..99
    mask = np.zeros((len(unique), MAX_N))
    hits = np.zeros((len(unique), N_RULES, MAX_N))
    per_set = []
    for examples in unique:
        ex = np.array(examples) - 1
        bits = INTERVAL_MEMBERSHIP[:, ex].astype("int64")
        group = sizes * N_PATTERNS + (bits * (2 ** np.arange(len(ex)))[None, :]).sum(axis=1)
        keys, inverse, count = np.unique(group, return_inverse=True, return_counts=True)
        share = np.zeros((len(keys), 100))
        np.add.at(share, inverse, INTERVAL_MEMBERSHIP)
        per_set.append((keys, count, share / count[:, None]))
    n_groups = max(len(k) for k, _, _ in per_set)
    log_count = np.full((len(unique), n_groups), -1e30)  # padding gets no weight
    g_size = np.zeros((len(unique), n_groups), dtype="int64")
    g_pattern = np.zeros((len(unique), n_groups), dtype="int64")
    membership = np.zeros((len(unique), N_RULES + n_groups, width))
    for u, examples in enumerate(unique):
        ex = np.array(examples) - 1
        mask[u, : len(ex)] = 1.0
        hits[u, :, : len(ex)] = RULE_MEMBERSHIP[:, ex]
        keys, count, share = per_set[u]
        log_count[u, : len(keys)] = np.log(count)
        g_size[u, : len(keys)] = keys // N_PATTERNS
        g_pattern[u, : len(keys)] = keys % N_PATTERNS
        columns = np.array(asked[examples]) - 1
        membership[u, :N_RULES, : len(columns)] = RULE_MEMBERSHIP[:, columns]
        membership[u, N_RULES : N_RULES + len(keys), : len(columns)] = share[:, columns]
    return {
        "position_mask": mask,
        "rule_hits": hits,
        "group_log_count": log_count,
        "group_size": g_size,
        "group_pattern": g_pattern,
        "target_membership": membership,
        "row_cell": np.array([index[s] * width + slot[(s, t)] for s, t in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
