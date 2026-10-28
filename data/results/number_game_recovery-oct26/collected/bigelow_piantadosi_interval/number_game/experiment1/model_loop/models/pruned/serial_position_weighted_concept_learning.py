"""Serial-position weighted concept learning.

People read the examples one at a time and integrate them into the same
Bayesian concept learning over 42 weighted rules and all intervals of 1..100
(size principle, 10% noisy examples, occasional guessing with a yes-bias) as
concept_learning_with_guessing, but memory/attention is unequal across the
sequence: each example's log-likelihood is weighted by its serial position,
w_k proportional to exp(-recency * lag_k) (lag = how many examples were shown
after it), normalised so the weights sum to the number of examples. recency > 0
is leaky memory (recent examples dominate), recency < 0 is primacy; recency = 0
recovers the order-blind incumbent. Single change: the positional gradient.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_N = 4


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
    return rules


RULES = concept_sets()
N_RULES = len(RULES)
N_INTERVALS = 5050
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EX_LL = np.log(RULE_MEMBERSHIP * (1 - EPSILON) / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
LOG_OUT = np.log(EPSILON / 100)
INTERVALS = [(m, n) for m in range(1, 101) for n in range(m, 101)]

# Intervals: an interval's weighted log-likelihood depends only on its size and
# which examples (by position) it contains, so intervals are grouped by
# (inclusion pattern, size) per set; each group carries its count and, for each
# target, the fraction of its intervals containing it. Exact, and cheap.

with pm.Model() as model:
    lag = pm.Data("lag", np.zeros((1, MAX_N)))  # (sets, positions)
    mask = pm.Data("mask", np.ones((1, MAX_N)))
    rule_pos_ll = pm.Data("rule_pos_ll", np.zeros((1, N_RULES, MAX_N)))
    group_in = pm.Data("group_in", np.zeros((1, 1, MAX_N)))  # (sets, groups, positions)
    group_f = pm.Data("group_f", np.zeros((1, 1)))  # log P(x | interval) for x inside
    group_logcount = pm.Data("group_logcount", np.zeros((1, 1)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # set of each distinct (set, target) cell
    cell_rule_memb = pm.Data("cell_rule_memb", np.zeros((1, N_RULES)))  # target in each rule
    cell_frac = pm.Data("cell_frac", np.zeros((1, 1)))  # share of each group's intervals holding target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # cell of each response
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    recency = pm.Normal("recency", mu=0.0, sigma=1.0)

    raw = pt.exp(-recency * lag) * mask
    n_ex = mask.sum(axis=1, keepdims=True)
    w = raw * n_ex / raw.sum(axis=1, keepdims=True)  # (sets, positions)

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + (rule_pos_ll * w[:, None, :]).sum(axis=2)
    w_in = (group_in * w[:, None, :]).sum(axis=2)
    log_groups = (
        pt.log1p(-p_mix) - np.log(N_INTERVALS) + group_logcount
        + w_in * (group_f - LOG_OUT) + n_ex * LOG_OUT
    )
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # Only the (set, target) cells actually asked about are computed.
    cell_post = posterior[cell_set]
    in_concept = (cell_post[:, :N_RULES] * cell_rule_memb).sum(axis=1) + (cell_post[:, N_RULES:] * cell_frac).sum(axis=1)

    concept_p = pt.clip(in_concept[row_cell], 0.0, 1.0)
    guess_rate = pm.Beta("guess_rate", 2, 8)
    guess_bias = pm.Beta("guess_bias", 2, 2)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip((1 - guess_rate) * concept_p + guess_rate * guess_bias, 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    lag = np.zeros((S, MAX_N))
    mask = np.zeros((S, MAX_N))
    rule_pos_ll = np.zeros((S, N_RULES, MAX_N))
    groups = []
    for i, s in enumerate(unique):
        n = len(s)
        lag[i, :n] = np.arange(n - 1, -1, -1)
        mask[i, :n] = 1.0
        rule_pos_ll[i, :, :n] = RULE_EX_LL[:, np.array(s) - 1]
        g = {}
        for m, hi in INTERVALS:
            key = (tuple(m <= e <= hi for e in s), hi - m + 1)
            cnt, memb = g.setdefault(key, [0, np.zeros(100)])
            memb[m - 1:hi] += 1
            g[key][0] = cnt + 1
        groups.append(g)
    G = max(len(g) for g in groups)
    group_in = np.zeros((S, G, MAX_N))
    group_f = np.zeros((S, G))
    group_logcount = np.full((S, G), -1e4)
    group_frac = np.zeros((S, G, 100))
    for i, g in enumerate(groups):
        for j, ((pat, size), (cnt, memb)) in enumerate(g.items()):
            group_in[i, j, :len(pat)] = pat
            group_f[i, j] = np.log((1 - EPSILON) / size + EPSILON / 100)
            group_logcount[i, j] = np.log(cnt)
            group_frac[i, j] = memb / cnt
    keys = [(index[st], int(r["target"]) - 1) for st, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cs = np.array([c[0] for c in cells], dtype="int64")
    ct = np.array([c[1] for c in cells], dtype="int64")
    return {
        "lag": lag, "mask": mask, "rule_pos_ll": rule_pos_ll,
        "group_in": group_in, "group_f": group_f, "group_logcount": group_logcount,
        "cell_set": cs, "cell_rule_memb": RULE_MEMBERSHIP[:, ct].T, "cell_frac": group_frac[cs, :, ct],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
