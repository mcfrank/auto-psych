"""Recency-weighted Bayesian concept learning (refines bigelow_piantadosi_independent).

Same learner as the incumbent: 42 rule concepts and the 5,050 intervals of
1..100, prior p_mix on rules shared by Dirichlet(2) weights and 1 - p_mix
uniformly on intervals, each example drawn from the concept with probability
0.9 (size principle) and uniformly from 1..100 otherwise; "yes" is the
posterior probability the concept contains the target. The one change: an
example's log-likelihood is weighted by its recency in memory, exp(-decay * k)
where k is how many examples were shown after it (last example: weight 1), so
earlier examples count less and the posterior favours concepts around the
most recent examples.

For speed, the intervals of each set are grouped exactly by (which examples
they contain, their size): every interval in a group has the same weighted
likelihood, so prepare_observed lists each set's non-empty groups with their
interval counts, and how many of each group's intervals contain each target.
Every log score lies in about [-40, 0], so the posterior is normalised by
plain sums, without a log-sum-exp.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_EX = 4
N_MASK = 2 ** MAX_EX
SIZES = np.arange(1, 101)


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
RULE_EX_LOG_LIK = np.log(RULE_MEMBERSHIP * (1 - EPSILON) / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
LOG_IN = np.log((1 - EPSILON) / SIZES + EPSILON / 100)  # example inside an interval of size L
LOG_OUT = np.log(EPSILON / 100)
# bit k of a mask = the example k places before the last lies inside the interval
MASK_BITS = np.array([[(m >> k) & 1 for k in range(MAX_EX)] for m in range(N_MASK)], dtype="float64")
INT_M, INT_N = np.array([(a, b) for a in range(1, 101) for b in range(a, 101)]).T
INT_SIZE = INT_N - INT_M + 1

with pm.Model() as model:
    rule_pos_log_lik = pm.Data("rule_pos_log_lik", np.zeros((1, MAX_EX, N_RULES)))  # 0 where absent
    group_set = pm.Data("group_set", np.zeros(1, dtype="int64"))  # one entry per non-empty group
    group_in = pm.Data("group_in", np.zeros((1, MAX_EX)))  # example position inside the group's intervals
    group_out = pm.Data("group_out", np.zeros((1, MAX_EX)))  # example position present but outside
    group_log_size_lik = pm.Data("group_log_size_lik", np.zeros(1))  # log P(x | interval of that size)
    group_count = pm.Data("group_count", np.zeros(1))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # one entry per distinct (set, target)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    pair_cell = pm.Data("pair_cell", np.zeros(1, dtype="int64"))  # (cell, group) pairs with containing intervals
    pair_group = pm.Data("pair_group", np.zeros(1, dtype="int64"))
    pair_contain = pm.Data("pair_contain", np.zeros(1))  # how many of the group's intervals contain the target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    decay = pm.HalfNormal("decay", sigma=1.0)

    w = pt.exp(-decay * pt.arange(MAX_EX))  # recency weight per position (last example: 1)
    rule_joint = p_mix * rule_weights[None, :] * pt.exp((rule_pos_log_lik * w[None, :, None]).sum(axis=1))
    # prior x weighted likelihood of ONE interval in each group
    one_interval = pt.exp(pt.log1p(-p_mix) - np.log(N_INTERVALS)
                          + (group_in @ w) * group_log_size_lik + (group_out @ w) * LOG_OUT)
    n_sets = rule_pos_log_lik.shape[0]
    evidence = rule_joint.sum(axis=1) + pt.inc_subtensor(pt.zeros(n_sets)[group_set], one_interval * group_count)
    interval_yes = pt.inc_subtensor(pt.zeros(cell_set.shape[0])[pair_cell], one_interval[pair_group] * pair_contain)
    in_concept = ((rule_joint[cell_set] * cell_rule_member).sum(axis=1) + interval_yes) / evidence[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    sidx = {s: i for i, s in enumerate(unique)}
    rule_pos = np.zeros((len(unique), MAX_EX, N_RULES))
    masks, group_ids = [], []
    g_set, g_in, g_out, g_loglik, g_count = [], [], [], [], []
    for i, s in enumerate(unique):
        present = np.zeros(MAX_EX)
        mask = np.zeros(N_INTERVALS, dtype="int64")
        for k, x in enumerate(reversed(s)):
            rule_pos[i, k] = RULE_EX_LOG_LIK[:, x - 1]
            present[k] = 1.0
            mask |= ((INT_M <= x) & (x <= INT_N)).astype("int64") << k
        key = mask * 100 + INT_SIZE - 1
        keys, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        group_ids.append(inverse + len(g_set))
        for kk, c in zip(keys, counts):
            bits = MASK_BITS[kk // 100]
            g_set.append(i); g_in.append(bits); g_out.append(present * (1 - bits))
            g_loglik.append(LOG_IN[kk % 100]); g_count.append(float(c))
    cells = [(sidx[s], int(r["target"])) for s, r in zip(sets, rows)]
    ucells = list(dict.fromkeys(cells))
    cidx = {c: i for i, c in enumerate(ucells)}
    p_cell, p_group, p_contain = [], [], []
    for j, (i, t) in enumerate(ucells):
        inside = (INT_M <= t) & (t <= INT_N)
        gs, cs = np.unique(group_ids[i][inside], return_counts=True)
        p_cell += [j] * len(gs); p_group += list(gs); p_contain += list(cs.astype("float64"))
    return {
        "rule_pos_log_lik": rule_pos,
        "group_set": np.array(g_set, dtype="int64"),
        "group_in": np.array(g_in),
        "group_out": np.array(g_out),
        "group_log_size_lik": np.array(g_loglik),
        "group_count": np.array(g_count),
        "cell_set": np.array([c[0] for c in ucells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in ucells]),
        "pair_cell": np.array(p_cell, dtype="int64"),
        "pair_group": np.array(p_group, dtype="int64"),
        "pair_contain": np.array(p_contain),
        "row_cell": np.array([cidx[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
