"""Bayesian concept learning with decade concepts.

Hypothesis: people do the same conservative Bayesian concept learning as
`lapsing_growing_evidence_concepts` (42 arithmetic rules and midpoint-weighted
intervals of 1-100, evidence weight growing with set size, attention lapses),
but they also hold decades (numbers sharing a tens digit: 1-9, 10-19, ...,
90-99) as named concepts with their own inferred prior share p_decade, split
equally across the ten decades. A target just outside the examples' range is
therefore endorsed more when it shares their decade than when an equally close
target crosses a decade boundary. The one change is this decade concept family.
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
    decades = [[i for i in NUMBERS if i // 10 == d and i < 100] for d in range(10)]
    return rules, intervals, decades


RULES, INTERVALS, DECADES = concept_sets()
N_RULES, N_INTERVALS, N_DECADES = len(RULES), len(INTERVALS), len(DECADES)
N_HYP = N_RULES + N_INTERVALS + N_DECADES
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS + DECADES], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])

with pm.Model() as model:
    set_log_lik = pm.Data("set_log_lik", np.zeros((1, N_HYP)))  # (sets, hyps): log P(D | h)
    set_size = pm.Data("set_size", np.ones(1, dtype="float64"))  # (sets,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    # Evidence strength: 1 is the ideal learner, below 1 conservative.
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    # Growth of per-example evidence weight with the number of examples.
    size_growth = pm.Normal("size_growth", mu=0.0, sigma=1.0)
    # Attention lapses: chance of a blind "yes" and of a blind "no".
    lapse_yes = pm.Beta("lapse_yes", 1.0, 19.0)
    lapse_no = pm.Beta("lapse_no", 1.0, 19.0)
    # Prior share of all belief held by the ten decade concepts together.
    p_decade = pm.Beta("p_decade", 1.0, 9.0)
    weight = evidence_strength * pt.exp(size_growth * pt.log(set_size))  # (sets,)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT]
    log_interval_prior = log_interval - pt.logsumexp(log_interval)

    log_other = pt.log1p(-p_decade)
    log_prior = pt.concatenate([
        log_other + pt.log(p_mix) + pt.log(rule_weights),
        log_other + pt.log1p(-p_mix) + log_interval_prior,
        pt.ones(N_DECADES) * (pt.log(p_decade) - np.log(N_DECADES)),
    ])
    log_joint = log_prior[None, :] + weight[:, None] * set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, hyps)
    in_concept = pt.dot(posterior, MEMBERSHIP)[cell_set, cell_target]  # (cells,)

    p_concept = in_concept[row_cell]
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse_yes + (1 - lapse_yes - lapse_no) * p_concept, 1e-6, 1 - 1e-6)
    )
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
        "set_log_lik": np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_size": np.array([float(len(s)) for s in unique], dtype="float64"),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
