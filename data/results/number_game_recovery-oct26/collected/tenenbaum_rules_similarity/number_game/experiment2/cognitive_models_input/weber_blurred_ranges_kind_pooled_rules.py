"""Refinement of weber_noisy_range_membership: Bayesian concept learning over
42 arithmetic rules and every interval of 1-100 (size principle, 10% random
examples, Erlang prior over interval width), where whether a target lies inside
a range is judged on a noisy analog mental number line with scalar (Weber)
variability: the target's perceived log-magnitude is Gaussian around its true
log value with an inferred Weber fraction, so range edges are blurred, more so
for large numbers, while rule membership is symbolic and exact. "yes" =
posterior expected (fuzzy) membership. The one change: the prior over rules is
set by rule kind (powers, multiples, final digit, contained digit, special
sets) - a Dirichlet share per kind split equally among that kind's rules,
grafted from family_pooled_rule_prior - instead of a free weight per rule.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
BOUNDS = np.arange(0, 101) + 0.5  # range [m, n] spans bounds m-0.5 .. n+0.5


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
    # Rule kind of each rule: powers, multiples, final digit, contained digit, special.
    kinds = [0] * 9 + [1] * 10 + [2] * 9 + [3] * 9 + [4] * 5
    intervals = [(m, n) for m in range(1, 101) for n in range(m, 101)]
    return rules, kinds, intervals


RULES, RULE_KINDS, INTERVALS = concept_sets()
N_RULES = len(RULES)
N_KINDS = 5
KIND_MATRIX = np.eye(N_KINDS)[RULE_KINDS]
KIND_COUNTS = KIND_MATRIX.sum(axis=0)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INT_LO = np.array([m for m, _ in INTERVALS])
INT_HI = np.array([n for _, n in INTERVALS])
INT_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")
MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INT_MEMBERSHIP])
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)

INTERVAL_WIDTH = INT_HI - INT_LO + 1
GROUP_WIDTHS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    INTERVAL_WIDTH, return_inverse=True, return_counts=True
)
N_GROUPS = len(GROUP_WIDTHS)
LOG_BOUNDS = np.log(BOUNDS)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, N_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    set_bound_coef = pm.Data("set_bound_coef", np.zeros((1, N_GROUPS, 101)))
    cell_log_target = pm.Data("cell_log_target", np.zeros(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    # Prior share of each rule kind, split equally among that kind's rules.
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    weber = pm.LogNormal("weber", mu=np.log(0.1), sigma=0.7)

    log_width = np.log(GROUP_WIDTHS.astype("float64")) - GROUP_WIDTHS / width_scale
    log_interval_prior = log_width - pt.logsumexp(log_width + np.log(GROUP_SIZES))

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_groups = pt.log1p(-p_mix) + log_interval_prior[None, :] + group_log_evidence
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    post_cell = posterior[cell_set]

    # P(perceived target below bound b) for each of the 101 range bounds.
    below = 0.5 * (1 + pt.erf((LOG_BOUNDS[None, :] - cell_log_target[:, None]) / (weber * np.sqrt(2.0))))
    # Posterior-weighted bound coefficients, once per set (not per cell).
    set_coef = pt.batched_dot(posterior[:, N_RULES:], set_bound_coef)
    in_concept = (post_cell[:, :N_RULES] * cell_rule_member).sum(axis=1) + (
        set_coef[cell_set] * below
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    log_evidence = np.empty((len(unique), N_GROUPS))
    # Per set and width group: coefficient on P(perceived < bound b), i.e.
    # sum of within-group weights times (+1 at upper bound, -1 at lower bound).
    bound_coef = np.zeros((len(unique), N_GROUPS, 101))
    for g in range(N_GROUPS):
        members = np.where(GROUP_OF_INTERVAL == g)[0]
        log_evidence[:, g] = logsumexp(interval_log_lik[:, members], axis=1)
        weights = np.exp(interval_log_lik[:, members] - log_evidence[:, g : g + 1])
        np.add.at(bound_coef[:, g, :].T, INT_HI[members], weights.T)
        np.add.at(bound_coef[:, g, :].T, INT_LO[members] - 1, -weights.T)
    cell_set = np.array([index[s] for s, _ in cells], dtype="int64")
    cell_target = np.array([t for _, t in cells])
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "group_log_evidence": log_evidence,
        "cell_set": cell_set,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_target - 1].T,
        "set_bound_coef": bound_coef,
        "cell_log_target": np.log(cell_target.astype("float64")),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
