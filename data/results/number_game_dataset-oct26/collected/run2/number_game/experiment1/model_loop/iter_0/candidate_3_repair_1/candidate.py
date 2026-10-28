"""Mixed strong/weak sampling Bayesian concept learning (refines
bigelow_piantadosi_independent).

Same learner as the incumbent: 42 rule concepts and the 5,050 intervals of
1..100, prior p_mix on rules shared by Dirichlet(2) weights and 1 - p_mix
uniformly on intervals, and 10% of examples treated as noise drawn uniformly
from 1..100. The one change: a non-noise example inside concept h is believed
drawn from h (strong sampling, likelihood 1/|h|) with probability theta, and
otherwise to be an arbitrary number of 1..100 that happens to lie in h (weak
sampling, likelihood 1/100), with theta inferred (Navarro et al. 2012). theta
= 1 recovers the incumbent; smaller theta weakens the size principle and
broadens generalisation. "yes" is the posterior probability that the concept
contains the target.

For speed, each set's intervals are grouped exactly by (how many examples they
contain, their size), which fixes their likelihood; prepare_observed lists the
groups with interval counts and how many of each group's intervals contain each
target. Every log score lies in about [-30, 0], so plain sums normalise.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)


def rule_sets():
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
    return rules


RULES = rule_sets()
N_RULES = len(RULES)
N_INTERVALS = 5050
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
LOG_OUT = np.log(EPSILON / 100)
INT_M, INT_N = np.array([(a, b) for a in range(1, 101) for b in range(a, 101)]).T
INT_SIZE = INT_N - INT_M + 1


def log_in(theta, size):
    """log P(example | concept of this size) for an example inside it."""
    return pt.log((1 - EPSILON) * (theta / size + (1 - theta) / 100.0) + EPSILON / 100)


with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    rule_n_out = pm.Data("rule_n_out", np.zeros((1, N_RULES)))
    group_set = pm.Data("group_set", np.zeros(1, dtype="int64"))  # one entry per non-empty group
    group_n_in = pm.Data("group_n_in", np.zeros(1))
    group_n_out = pm.Data("group_n_out", np.zeros(1))
    group_size = pm.Data("group_size", np.ones(1))
    group_count = pm.Data("group_count", np.zeros(1))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # one entry per distinct (set, target)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    pair_cell = pm.Data("pair_cell", np.zeros(1, dtype="int64"))  # (cell, group) pairs with containing intervals
    pair_group = pm.Data("pair_group", np.zeros(1, dtype="int64"))
    pair_contain = pm.Data("pair_contain", np.zeros(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    theta = pm.Beta("theta", 2, 2)  # probability an example is strongly sampled

    rule_log_lik = rule_n_in * log_in(theta, pt.as_tensor(RULE_SIZE))[None, :] + rule_n_out * LOG_OUT
    rule_joint = p_mix * rule_weights[None, :] * pt.exp(rule_log_lik)
    one_interval = pt.exp(pt.log1p(-p_mix) - np.log(N_INTERVALS)
                          + group_n_in * log_in(theta, group_size) + group_n_out * LOG_OUT)
    n_sets = rule_n_in.shape[0]
    evidence = rule_joint.sum(axis=1) + pt.inc_subtensor(pt.zeros(n_sets)[group_set], one_interval * group_count)
    interval_yes = pt.inc_subtensor(pt.zeros(cell_set.shape[0])[pair_cell], one_interval[pair_group] * pair_contain)
    in_concept = ((rule_joint[cell_set] * cell_rule_member).sum(axis=1) + interval_yes) / evidence[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    sidx = {s: i for i, s in enumerate(unique)}
    r_in = np.zeros((len(unique), N_RULES))
    group_ids = []
    g_set, g_in, g_out, g_size, g_count = [], [], [], [], []
    for i, s in enumerate(unique):
        x = np.array(s)
        r_in[i] = RULE_MEMBERSHIP[:, x - 1].sum(axis=1)
        n_in = ((INT_M[:, None] <= x[None, :]) & (x[None, :] <= INT_N[:, None])).sum(axis=1)
        key = n_in * 101 + INT_SIZE
        keys, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        group_ids.append(inverse + len(g_set))
        for kk, c in zip(keys, counts):
            g_set.append(i); g_in.append(float(kk // 101)); g_out.append(float(len(s) - kk // 101))
            g_size.append(float(kk % 101)); g_count.append(float(c))
    lengths = np.array([len(s) for s in unique], dtype="float64")
    cells = [(sidx[s], int(r["target"])) for s, r in zip(sets, rows)]
    ucells = list(dict.fromkeys(cells))
    cidx = {c: i for i, c in enumerate(ucells)}
    p_cell, p_group, p_contain = [], [], []
    for j, (i, t) in enumerate(ucells):
        inside = (INT_M <= t) & (t <= INT_N)
        gs, cs = np.unique(group_ids[i][inside], return_counts=True)
        p_cell += [j] * len(gs); p_group += list(gs); p_contain += list(cs.astype("float64"))
    return {
        "rule_n_in": r_in,
        "rule_n_out": lengths[:, None] - r_in,
        "group_set": np.array(g_set, dtype="int64"),
        "group_n_in": np.array(g_in),
        "group_n_out": np.array(g_out),
        "group_size": np.array(g_size),
        "group_count": np.array(g_count),
        "cell_set": np.array([c[0] for c in ucells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in ucells]),
        "pair_cell": np.array(p_cell, dtype="int64"),
        "pair_group": np.array(p_group, dtype="int64"),
        "pair_contain": np.array(p_contain),
        "row_cell": np.array([cidx[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
