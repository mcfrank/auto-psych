"""Refinement of `bigelow_piantadosi_interval` with an inferred size principle.

Bayesian concept learning over 42 rules (powers, multiples, final digit,
contained digit, primes, even, odd, squares, cubes) and every interval of
1-100: per-rule Dirichlet prior weights, a rule share p_mix, and an interval
prior following a Beta density over the interval's midpoint, exactly as in
`bigelow_piantadosi_interval`. The one change: the strength of the size
principle is inferred rather than fixed at 1. Each example's likelihood
under a concept is raised to the power size_strength,

    P(D | h) = prod_x [ 0.9 * 1[x in h] / |h| + 0.1 / 100 ] ** size_strength,

so a learner may treat each example as more (or less) than one independent
draw from the concept. A strong size principle makes rich example sets pull
the posterior onto the tightest concepts covering them, and so curbs "yes"
to numbers outside the examples' range. "Yes" is the posterior probability
that the concept contains the target.

Speed: an interval's prior depends only on its midpoint grid point, so for
each set the intervals at one grid point act as one hypothesis whose total
evidence, and whose share of evidence from intervals containing the target,
depend only on size_strength. prepare_observed tabulates both (with their
exact derivatives) on a fine grid of log size_strength, and the model reads
them by cubic Hermite interpolation, which is smooth and accurate to well
below sampling noise. Rule evidence is computed exactly in the model.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)


def concept_sets():
    """42 rules, then the 5,050 intervals [m, n] of 1..100."""
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
# log P(x | h) of one example x (column x - 1) under each hypothesis, at strength 1.
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])
GROUP_POINTS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    INTERVAL_GRID_POINT, return_inverse=True, return_counts=True
)
N_GROUPS = len(GROUP_POINTS)
GROUP_ONE_HOT = np.eye(N_GROUPS)[GROUP_OF_INTERVAL]  # (intervals, groups)

# Grid over u = log(size_strength) for the interval tables.
U_MIN, U_MAX, U_STEP = -1.5, 1.5, 0.1
U_GRID = np.linspace(U_MIN, U_MAX, int(round((U_MAX - U_MIN) / U_STEP)) + 1)
N_U = len(U_GRID)


def hermite(table, slope, u):
    """Cubic Hermite interpolation of table[..., j] (slope = d table / du) at u."""
    pos = (u - U_MIN) / U_STEP
    j = pt.clip(pt.floor(pos), 0, N_U - 2).astype("int64")
    t = pos - j
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return (h00 * table[..., j] + h10 * U_STEP * slope[..., j]
            + h01 * table[..., j + 1] + h11 * U_STEP * slope[..., j + 1])


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules) at strength 1
    group_log_ev = pm.Data("group_log_ev", np.zeros((1, N_GROUPS, N_U)))  # (sets, groups, grid)
    group_log_ev_slope = pm.Data("group_log_ev_slope", np.zeros((1, N_GROUPS, N_U)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    cell_share = pm.Data("cell_share", np.zeros((1, N_GROUPS, N_U)))  # (cells, groups, grid)
    cell_share_slope = pm.Data("cell_share_slope", np.zeros((1, N_GROUPS, N_U)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    # The one change: log strength of the size principle (0 = the original model).
    log_size_strength = pm.TruncatedNormal("log_size_strength", mu=0.0, sigma=0.75, lower=U_MIN, upper=U_MAX)
    size_strength = pm.Deterministic("size_strength", pt.exp(log_size_strength))

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_grid = log_density - pt.logsumexp(log_density)
    log_point = log_grid[GROUP_POINTS]
    log_interval_prior = log_point - pt.logsumexp(log_point + np.log(GROUP_SIZES))

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + size_strength * rule_log_lik
    log_groups = (pt.log1p(-p_mix) + log_interval_prior[None, :]
                  + hermite(group_log_ev, group_log_ev_slope, log_size_strength))
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + groups)

    share = pt.clip(hermite(cell_share, cell_share_slope, log_size_strength), 0.0, 1.0)  # (cells, groups)
    cell_rule_in = pt.as_tensor_variable(RULE_MEMBERSHIP.T)[cell_target]  # (cells, rules)
    post_cell = posterior[cell_set]
    in_concept = (post_cell[:, :N_RULES] * cell_rule_in).sum(axis=1) + (post_cell[:, N_RULES:] * share).sum(axis=1)

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
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    strengths = np.exp(U_GRID)

    log_ev = np.empty((len(unique), N_GROUPS, N_U))
    log_ev_slope = np.empty_like(log_ev)
    share = np.empty((len(cells), N_GROUPS, N_U))
    share_slope = np.empty_like(share)
    cells_of_set = [[] for _ in unique]
    for c, (s, t) in enumerate(cells):
        cells_of_set[index[s]].append((c, t - 1))
    for i in range(len(unique)):
        ll = set_log_lik[i, N_RULES:]  # (intervals,)
        scaled = strengths[:, None] * ll[None, :]  # (grid, intervals)
        shift = scaled.max(axis=1, keepdims=True)
        w = np.exp(scaled - shift)  # (grid, intervals)
        ev = w @ GROUP_ONE_HOT  # (grid, groups)
        ev_ll = (w * ll) @ GROUP_ONE_HOT  # sum w * ll
        log_ev[i] = (np.log(ev) + shift).T
        mean_ll = ev_ll / ev
        log_ev_slope[i] = (strengths[:, None] * mean_ll).T  # d log ev / du
        for c, t in cells_of_set[i]:
            m = INTERVAL_MEMBERSHIP[:, t]
            num = (w * m) @ GROUP_ONE_HOT
            num_ll = (w * m * ll) @ GROUP_ONE_HOT
            sh = num / ev
            share[c] = sh.T
            # d share / du = strength * share * (E_in[ll] - E_all[ll])
            mean_in = np.where(num > 0, num_ll / np.where(num > 0, num, 1.0), 0.0)
            share_slope[c] = (strengths[:, None] * sh * (mean_in - mean_ll)).T
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "group_log_ev": log_ev,
        "group_log_ev_slope": log_ev_slope,
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "cell_share": share,
        "cell_share_slope": share_slope,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
