"""Recency-limited memory for the examples in Bayesian concept learning.

People do conservative Bayesian concept learning over 42 arithmetic rules and
the midpoint-weighted intervals of 1..100 (size principle, 10% noise, evidence
weighted by an inferred strength), but they condition only on the examples
they still remember: the last example shown is always remembered, and an
example shown `lag` positions before it is remembered with probability
retention ** lag. A forgotten example neither rules out nor sharpens concepts.
"Yes" is the posterior probability that the concept contains the target,
averaged over which examples are remembered. Order matters: recent examples
dominate generalization.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_LAG = 3


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
N_HYP = N_RULES + N_INTERVALS
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])
LAGS = np.arange(1, MAX_LAG + 1, dtype="float64")

with pm.Model() as model:
    # One row per (set, remembered subset that includes the last example).
    sub_log_lik = pm.Data("sub_log_lik", np.zeros((1, N_HYP)))  # log P(remembered | h)
    sub_kept = pm.Data("sub_kept", np.zeros((1, MAX_LAG)))  # earlier example at lag l kept
    sub_shown = pm.Data("sub_shown", np.zeros((1, MAX_LAG)))  # an example exists at lag l
    sub_to_set = pm.Data("sub_to_set", np.ones((1, 1)))  # (sets, subsets) aggregation
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    # Probability an example one position back is still remembered.
    retention = pm.Beta("retention", 3, 1.5)

    keep_prob = pt.clip(retention ** LAGS, 1e-6, 1 - 1e-6)  # (lags,)
    log_sub_w = (sub_shown * (sub_kept * pt.log(keep_prob) + (1 - sub_kept) * pt.log1p(-keep_prob))).sum(axis=1)
    sub_w = pt.exp(log_sub_w)  # (subsets,)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT]
    log_interval_prior = log_interval - pt.logsumexp(log_interval)

    log_prior = pt.concatenate([pt.log(p_mix) + pt.log(rule_weights), pt.log1p(-p_mix) + log_interval_prior])
    log_joint = log_prior[None, :] + evidence_strength * sub_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (subsets, hyps)
    sub_pred = pt.dot(posterior, MEMBERSHIP)  # (subsets, 100)
    set_pred = pt.dot(sub_to_set * sub_w[None, :], sub_pred)  # (sets, 100)
    in_concept = set_pred[cell_set, cell_target]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    sub_ll, kept, shown, owner = [], [], [], []
    for si, s in enumerate(unique):
        n_earlier = min(len(s) - 1, MAX_LAG)
        for mask in itertools.product([0, 1], repeat=n_earlier):
            # mask[l-1] says whether the example at lag l is remembered.
            remembered = [s[-1]] + [s[-1 - l] for l in range(1, n_earlier + 1) if mask[l - 1]]
            sub_ll.append(EXAMPLE_LOG_LIK[:, np.array(remembered) - 1].sum(axis=1))
            kept.append(list(mask) + [0] * (MAX_LAG - n_earlier))
            shown.append([1] * n_earlier + [0] * (MAX_LAG - n_earlier))
            owner.append(si)
    to_set = np.zeros((len(unique), len(owner)))
    to_set[owner, np.arange(len(owner))] = 1.0
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "sub_log_lik": np.stack(sub_ll),
        "sub_kept": np.array(kept, dtype="float64"),
        "sub_shown": np.array(shown, dtype="float64"),
        "sub_to_set": to_set,
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
