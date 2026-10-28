"""Sampling-assumption inference over rules and intervals.

Refinement of `size_principle_strength_intervals`. A learner who sees example
set D considers 42 rule concepts and the 5,050 intervals of 1..100 (p_mix on the
rules with Dirichlet weights per rule, 1 - p_mix on the intervals, whose prior
is a Beta density over the interval's midpoint). The one change: rather than a
fixed per-example size-principle strength, the learner is uncertain whether the
whole set was strongly sampled (each example drawn from the concept, P = 1/|h|)
or weakly sampled (examples merely consistent with h, P = 1/100), with prior
probability `theta` on strong sampling, and infers this jointly with the
concept (Navarro et al., 2012). Both modes keep 10% uniform noise. "Yes" is the
posterior probability, marginalised over the sampling mode, that the concept
contains the target. One example barely favours strong sampling, so it is
generalized broadly; several tightly clustered examples make strong sampling
credible and generalization narrow.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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
N_HYP = N_RULES + N_INTERVALS
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
# log P(x | h) per example under strong sampling (1/|h|) and weak sampling (1/100).
STRONG_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
WEAK_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / 100 + EPSILON / 100)

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])

with pm.Model() as model:
    strong_log_lik = pm.Data("strong_log_lik", np.zeros((1, N_HYP)))  # (sets, hyps)
    weak_log_lik = pm.Data("weak_log_lik", np.zeros((1, N_HYP)))  # (sets, hyps)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    # Prior belief that the examples were strongly sampled from the concept.
    theta = pm.Beta("theta", 2, 2)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT]
    log_interval_prior = log_interval - pt.logsumexp(log_interval)
    log_prior = pt.concatenate([pt.log(p_mix) + pt.log(rule_weights), pt.log1p(-p_mix) + log_interval_prior])

    # P(D | h) marginalised over the sampling mode (strong with prior theta).
    log_lik = pt.logaddexp(pt.log(theta) + strong_log_lik, pt.log1p(-theta) + weak_log_lik)
    log_joint = log_prior[None, :] + log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, hyps)
    in_concept = pt.dot(posterior, MEMBERSHIP)[cell_set, cell_target]  # (cells,)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "strong_log_lik": np.stack([STRONG_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "weak_log_lik": np.stack([WEAK_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
