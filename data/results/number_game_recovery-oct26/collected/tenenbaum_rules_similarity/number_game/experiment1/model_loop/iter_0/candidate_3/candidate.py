"""Tempered-evidence Bayesian concept learning over rules and intervals.

Refines `bigelow_piantadosi_interval`. The learner considers the same 42 rule
concepts and the 5,050 intervals of 1..100, with Dirichlet rule weights, a
rule share p_mix and a Beta(a, b) prior over the interval's midpoint; examples
are drawn from the concept with probability 0.9 (size principle) and from
1..100 otherwise. The single change: the learner under- (or over-) weights the
examples' evidence, so the posterior is prior x P(D | h) ** gamma with gamma
inferred (gamma = 1 is the incumbent). "Yes" is the posterior probability that
the concept contains the target.

As in the incumbent, intervals sharing a midpoint (one prior value) are pooled
into one group per example set: the group's tempered evidence
log sum_j P(D | interval j) ** gamma and, per (set, target) cell, the share of
it from intervals containing the target. Both depend on gamma only, so
prepare_observed tabulates them on a fine grid of log gamma and the model
interpolates linearly between grid points; the rules are tempered exactly.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
# log gamma grid: gamma from about 0.2 to 3.3.
LOG_GAMMA_LO, LOG_GAMMA_HI, N_KNOTS = -1.6, 1.2, 41
LOG_GAMMA_GRID = np.linspace(LOG_GAMMA_LO, LOG_GAMMA_HI, N_KNOTS)
KNOT_STEP = LOG_GAMMA_GRID[1] - LOG_GAMMA_GRID[0]


def concept_sets():
    """42 rules, then 5,050 intervals."""
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
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])
GROUP_POINTS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    INTERVAL_GRID_POINT, return_inverse=True, return_counts=True
)
N_GROUPS = len(GROUP_POINTS)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    # (knots, sets, groups): tempered log evidence of each midpoint group.
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((N_KNOTS, 1, N_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_in_rule = pm.Data("cell_in_rule", np.zeros((1, N_RULES)))  # (cells, rules)
    # (knots, cells, groups): share of a group's evidence from intervals containing the target.
    cell_in_group = pm.Data("cell_in_group", np.zeros((N_KNOTS, 1, N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    # Evidence-tempering exponent on the log scale: 0 = full Bayesian use of the examples.
    log_gamma = pm.TruncatedNormal("log_gamma", mu=0.0, sigma=0.5, lower=LOG_GAMMA_LO, upper=LOG_GAMMA_HI)
    gamma = pt.exp(log_gamma)

    pos = (log_gamma - LOG_GAMMA_LO) / KNOT_STEP
    knot = pt.clip(pt.floor(pos), 0, N_KNOTS - 2).astype("int64")
    w = pos - knot
    log_evidence = (1 - w) * group_log_evidence[knot] + w * group_log_evidence[knot + 1]
    group_share = (1 - w) * cell_in_group[knot] + w * cell_in_group[knot + 1]

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_grid = log_density - pt.logsumexp(log_density)
    log_point = log_grid[GROUP_POINTS]
    log_interval_prior = log_point - pt.logsumexp(log_point + np.log(GROUP_SIZES))

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + gamma * rule_log_lik
    log_groups = pt.log1p(-p_mix) + log_interval_prior[None, :] + log_evidence
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = (cell_post[:, :N_RULES] * cell_in_rule).sum(axis=1) + (cell_post[:, N_RULES:] * group_share).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1.0 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_set = np.array([index[s] for s, _ in cells], dtype="int64")
    cell_target = np.array([t - 1 for _, t in cells])
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]

    order = np.argsort(GROUP_OF_INTERVAL, kind="stable")
    starts = np.concatenate([[0], np.cumsum(GROUP_SIZES)[:-1]])
    sorted_groups = GROUP_OF_INTERVAL[order]
    target_member = INTERVAL_MEMBERSHIP[order][:, cell_target].T  # (cells, intervals sorted)
    log_evidence = np.empty((N_KNOTS, len(unique), N_GROUPS))
    in_group = np.empty((N_KNOTS, len(cells), N_GROUPS))
    for k, lg in enumerate(LOG_GAMMA_GRID):
        tempered = np.exp(lg) * interval_log_lik[:, order]  # (sets, intervals sorted)
        top = np.maximum.reduceat(tempered, starts, axis=1)
        weights = np.exp(tempered - top[:, sorted_groups])
        total = np.add.reduceat(weights, starts, axis=1)
        log_evidence[k] = top + np.log(total)
        share = weights / total[:, sorted_groups]
        in_group[k] = np.add.reduceat(share[cell_set] * target_member, starts, axis=1)
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "group_log_evidence": log_evidence,
        "cell_set": cell_set,
        "cell_in_rule": RULE_MEMBERSHIP[:, cell_target].T,
        "cell_in_group": in_group,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
