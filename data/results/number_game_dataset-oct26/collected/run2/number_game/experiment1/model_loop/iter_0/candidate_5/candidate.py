"""Recency-weighted Bayesian concept learning (refines bigelow_piantadosi_interval).

People do Bayesian concept learning over 42 rules and the 5,050 intervals of
1..100 (rule prior p_mix shared by Dirichlet weights; interval prior a Beta
density over the interval's midpoint; examples drawn by the size principle,
10% uniformly from 1..100), but they weigh the examples by recency: the
example at lag L (L examples were shown after it) contributes its
log-likelihood with weight proportional to rho**L, normalised so the weights
sum to the number of examples. rho = 1 recovers the original model; rho < 1
is recency, rho > 1 primacy. "Yes" is the posterior probability that the
concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_EXAMPLES = 4


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
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")  # (H, 100)
# log P(x | h) of one example x (column x - 1) under each hypothesis.
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])

with pm.Model() as model:
    # (sets, positions, hypotheses): log P(example at that position | h), 0 for padding.
    pos_log_lik = pm.Data("pos_log_lik", np.zeros((1, MAX_EXAMPLES, N_HYP)))
    pos_lag = pm.Data("pos_lag", np.zeros((1, MAX_EXAMPLES)))  # examples shown after this one
    pos_mask = pm.Data("pos_mask", np.zeros((1, MAX_EXAMPLES)))  # 1 for a real example
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # (cells,): target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    log_rho = pm.Normal("log_rho", mu=0.0, sigma=1.0)  # recency decay per example (log scale)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_point = log_density[INTERVAL_GRID_POINT]
    log_interval_prior = log_point - pt.logsumexp(log_point)

    raw_w = pt.exp(log_rho * pos_lag) * pos_mask
    n_examples = pos_mask.sum(axis=1, keepdims=True)
    weights = raw_w * n_examples / raw_w.sum(axis=1, keepdims=True)  # (sets, positions)
    set_log_lik = pt.matmul(weights[:, None, :], pos_log_lik)[:, 0, :]  # (sets, H)

    log_prior = pt.concatenate([pt.log(p_mix) + pt.log(rule_weights), pt.log1p(-p_mix) + log_interval_prior])
    log_joint = log_prior[None, :] + set_log_lik
    posterior = pt.special.softmax(log_joint, axis=1)  # (sets, H)
    p_target = pt.dot(posterior, MEMBERSHIP)  # (sets, 100): P(target in concept | set)
    in_concept = p_target[cell_set, cell_target]  # (cells,)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    pos_ll = np.zeros((len(unique), MAX_EXAMPLES, N_HYP))
    lag = np.zeros((len(unique), MAX_EXAMPLES))
    mask = np.zeros((len(unique), MAX_EXAMPLES))
    for i, s in enumerate(unique):
        for k, x in enumerate(s):
            pos_ll[i, k] = EXAMPLE_LOG_LIK[:, x - 1]
            lag[i, k] = len(s) - 1 - k
            mask[i, k] = 1.0
    return {
        "pos_log_lik": pos_ll,
        "pos_lag": lag,
        "pos_mask": mask,
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
