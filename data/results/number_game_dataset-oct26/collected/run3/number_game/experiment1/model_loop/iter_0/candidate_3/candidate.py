"""Bayesian concept learning over rules and intervals, with a prior weight
inferred for every rule and a prior over where an interval lies.

A PyMC port of Eric Bigelow's numpyro model (`number_bpl_model`,
`prior_type='interval'`; reference copy in
`scripts/number_game/numpyro_reference/`, checked by
`tests/test_number_game_seed_port.py`). A learner who sees the example set D
considers 42 rule concepts (powers, multiples, final digit, contained digit,
primes, even, odd, squares, cubes) and the 5,050 intervals [m, n] of 1..100.
Each example is drawn from the concept with probability 0.9 (uniformly: the
size principle) and from 1..100 otherwise. The prior puts p_mix ~ Beta(2, 2)
on the rules, shared by Dirichlet(2, ..., 2) weights, and 1 - p_mix on the
intervals, which are not equally likely: their prior follows a Beta(a, b)
density over the interval's midpoint rescaled to [0, 1], with a, b ~
Gamma(2, 1). The probability of answering "yes" to a target is the posterior
probability that the concept contains it.

Refinement (this file): on a share lambda(t) of trials the participant does
not consult the concept and answers yes with probability 1/2. The lapse rate
grows with fatigue over the session, logit lambda(t) = lapse_base +
lapse_slope * t, with t the trial index rescaled to [0, 1]. This softens the
concept posterior's near-certain predictions and makes yeses drift upward
over the session (the yes rate is below 1/2, so more coin-flip answers mean
more yeses).

Interval [m, n] reads the density at its midpoint, (m + n) / 2 rescaled to
[0, 1]: grid point g = m + n - 2 of 199 (g / 198, clipped to [1e-5,
1 - 1e-5] where a shape below 1 makes the density infinite). The reference
reads grid point m + n + 1 of 200, an index off by three that runs past the
grid for [99, 100] and [100, 100] (JAX clamps it); this is the intended
midpoint.
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

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])
# An interval's prior depends only on its grid point, so for each example set
# the intervals at one grid point act as one hypothesis: total evidence
# sum_j P(D | interval j) and, per target, the share of it from intervals
# containing the target. prepare_observed computes both, the shares only for
# the (set, target) cells with responses; the posterior below is exactly the
# one over all 5,092 hypotheses.
GROUP_POINTS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    INTERVAL_GRID_POINT, return_inverse=True, return_counts=True
)
N_GROUPS = len(GROUP_POINTS)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | rule)
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, N_GROUPS)))  # (sets, groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,): the cell's set index
    # (cells, rules + groups): the target's membership in each rule, then each
    # group's share of evidence from intervals containing it.
    cell_in_hypothesis = pm.Data("cell_in_hypothesis", np.zeros((1, N_RULES + N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (responses,): the response's cell
    trial_time = pm.Data("trial_time", np.zeros(1))  # (responses,): trial index rescaled to [0, 1]
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    lapse_base = pm.Normal("lapse_base", mu=-2.5, sigma=1.5)
    lapse_slope = pm.Normal("lapse_slope", mu=0.0, sigma=1.5)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_grid = log_density - pt.logsumexp(log_density)
    # log p(interval) at each group's grid point, normalised over the intervals.
    log_point = log_grid[GROUP_POINTS]
    log_interval_prior = log_point - pt.logsumexp(log_point + np.log(GROUP_SIZES))

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_groups = pt.log1p(-p_mix) + log_interval_prior[None, :] + group_log_evidence
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + groups)
    in_concept = (posterior[cell_set] * cell_in_hypothesis).sum(axis=1)  # (cells,)

    # Rounding can put a sure membership a few ulps above 1.
    concept_yes = pt.clip(in_concept[row_cell], 0.0, 1.0)
    lapse = pm.math.sigmoid(lapse_base + lapse_slope * trial_time)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * concept_yes + 0.5 * lapse, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    log_evidence = np.empty((len(unique), N_GROUPS))
    in_concept = np.empty((len(unique), N_GROUPS, 100))  # (sets, groups, targets)
    for g in range(N_GROUPS):
        members = GROUP_OF_INTERVAL == g
        log_evidence[:, g] = logsumexp(interval_log_lik[:, members], axis=1)
        weights = np.exp(interval_log_lik[:, members] - log_evidence[:, g : g + 1])
        in_concept[:, g] = weights @ MEMBERSHIP[N_RULES:][members]
    cell_set = np.array([index[s] for s, _ in cells], dtype="int64")
    cell_target = np.array([t - 1 for _, t in cells])
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "group_log_evidence": log_evidence,
        "cell_set": cell_set,
        "cell_in_hypothesis": np.concatenate(
            [RULE_MEMBERSHIP[:, cell_target].T, in_concept[cell_set, :, cell_target]], axis=1
        ),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "trial_time": np.array([float(r["trial_index"]) for r in rows]) / 450.0,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
