"""Serial-position-dependent tolerance for stray examples.

People learn the program's concept by conservative Bayesian concept learning over
42 arithmetic rules (equal prior weight within the family) and the 5,050 intervals
of 1..100 (equally likely), a rules-vs-ranges share p_mix, the size principle,
evidence weight beta * n ** gamma, and guessing that falls with set size. Each
example may be a stray number drawn uniformly from 1..100, and how readily it is
dismissed as one depends on its serial position:
logit(eps_k) = a + b * (k - (n - 1) / 2), with the gradient b inferred (b > 0:
later examples are more suspect, the first ones anchor the concept; b < 0: the
most recent examples are trusted). The order-dependent exception tolerance is the
hypothesis. A concept enters only through its size and which examples it
contains, so intervals are pooled into the (examples-contained, size) groups that
occur for each distinct set; groups and counts are computed in prepare_observed.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4
SIZES = np.arange(1, 101, dtype="float64")


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
RULE_MEMBER = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (rules, 100)
RULE_SIZE_IDX = RULE_MEMBER.sum(axis=1).astype("int64") - 1
LO, HI = np.array([(m, n) for m in range(1, 101) for n in range(m, 101)]).T
N_INTERVALS = len(LO)
INTERVAL_SIZE_IDX = HI - LO  # size - 1

with pm.Model() as model:
    shown = pm.Data("shown", np.zeros((1, MAX_N)))  # (sets, positions): 1 if position used
    offset = pm.Data("offset", np.zeros((1, MAX_N)))  # (sets, positions): k - (n - 1) / 2
    set_size = pm.Data("set_size", np.ones(1))
    # Concept groups (each rule; each occurring (examples-contained, size) pool of intervals).
    group_set = pm.Data("group_set", np.zeros(1, dtype="int64"))
    group_in = pm.Data("group_in", np.zeros((1, MAX_N)))  # group's concepts contain example k
    group_lookup = pm.Data("group_lookup", np.zeros((1, MAX_N), dtype="int64"))  # flat (set, k, size)
    group_is_rule = pm.Data("group_is_rule", np.zeros(1))
    group_count = pm.Data("group_count", np.zeros(1))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    # Sparse (cell, group, count) triples: concepts of the group containing the cell's target.
    entry_cell = pm.Data("entry_cell", np.zeros(1, dtype="int64"))
    entry_group = pm.Data("entry_group", np.zeros(1, dtype="int64"))
    entry_count = pm.Data("entry_count", np.zeros(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    size_growth = pm.Normal("size_growth", mu=0.0, sigma=0.5)
    weight = evidence_strength * pt.exp(size_growth * pt.log(set_size))  # (sets,)

    # Stray-example belief by serial position.
    stray_logit = pm.Normal("stray_logit", mu=-1.5, sigma=1.0)
    stray_position_slope = pm.Normal("stray_position_slope", mu=0.0, sigma=1.0)
    eps = pm.math.sigmoid(stray_logit + stray_position_slope * offset)  # (sets, positions)
    log_out = shown * pt.log(eps / 100)  # (sets, positions)
    log_in = shown[:, :, None] * pt.log((1 - eps)[:, :, None] / SIZES + eps[:, :, None] / 100)  # (sets, k, size)

    group_ll = (group_in * log_in.flatten()[group_lookup] + (1 - group_in) * log_out[group_set]).sum(axis=1)
    log_prior = group_is_rule * (pt.log(p_mix) - np.log(N_RULES)) + (1 - group_is_rule) * (
        pt.log1p(-p_mix) - np.log(N_INTERVALS)
    )
    # Reference: the interval 1..100, which contains every example (keeps exp() in range).
    reference = pt.log1p(-p_mix) - np.log(N_INTERVALS) + weight * (shown.sum(axis=1) * np.log(0.01))
    excess = log_prior + weight[group_set] * group_ll - reference[group_set]
    u = pt.exp(pt.minimum(excess, 600.0))
    normaliser = pt.zeros(set_size.shape[0])[group_set].inc(u * group_count)
    numerator = pt.zeros(cell_set.shape[0])[entry_cell].inc(u[entry_group] * entry_count)
    in_concept = numerator / normaliser[cell_set]

    guess_single = pm.Normal("guess_single", mu=-2.0, sigma=1.5)
    guess_decline = pm.Normal("guess_decline", mu=0.0, sigma=2.0)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    guess_rate = pm.math.sigmoid(guess_single - guess_decline * pt.log(set_size))[cell_set]
    q = guess_rate * guess_yes + (1 - guess_rate) * in_concept
    p_yes = pm.Deterministic("p_yes", pt.clip(q[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


# Bound per-iteration work: the posterior has 8 well-constrained parameters.
SAMPLER_SETTINGS = {"max_treedepth": 7}


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    n_sets = len(unique)
    shown = np.zeros((n_sets, MAX_N))
    offset = np.zeros((n_sets, MAX_N))
    g_set, g_in, g_size, g_rule, g_count = [], [], [], [], []
    rule_group, interval_group = [], []  # per set: group index of each rule / interval
    for i, s in enumerate(unique):
        n = len(s)
        shown[i, :n] = 1.0
        offset[i, :n] = np.arange(n) - (n - 1) / 2.0
        start = len(g_set)
        rule_in = np.zeros((N_RULES, MAX_N))
        rule_in[:, :n] = RULE_MEMBER[:, np.array(s) - 1]
        g_set += [i] * N_RULES
        g_in += list(rule_in)
        g_size += list(RULE_SIZE_IDX)
        g_rule += [1.0] * N_RULES
        g_count += [1.0] * N_RULES
        rule_group.append(start + np.arange(N_RULES))
        mask = np.zeros(N_INTERVALS, dtype="int64")
        for k, x in enumerate(s):
            mask |= ((LO <= x) & (x <= HI)).astype("int64") << k
        key = mask * 100 + INTERVAL_SIZE_IDX
        keys, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        for kk, c in zip(keys, counts):
            m = int(kk // 100)
            g_set.append(i)
            g_in.append(np.array([(m >> k) & 1 for k in range(MAX_N)], dtype="float64"))
            g_size.append(int(kk % 100))
            g_rule.append(0.0)
            g_count.append(float(c))
        interval_group.append(start + N_RULES + inverse.ravel())
    g_set = np.array(g_set, dtype="int64")
    g_size = np.array(g_size, dtype="int64")
    lookup = (g_set[:, None] * MAX_N + np.arange(MAX_N)[None, :]) * 100 + g_size[:, None]
    entry_cell, entry_group, entry_count = [], [], []
    for c, (s, t) in enumerate(cells):
        i = index[s]
        has = (LO <= t) & (t <= HI)
        groups, counts = np.unique(interval_group[i][has], return_counts=True)
        rules = rule_group[i][RULE_MEMBER[:, t - 1] > 0]
        entry_cell.append(np.full(len(groups) + len(rules), c))
        entry_group.append(np.r_[rules, groups])
        entry_count.append(np.r_[np.ones(len(rules)), counts])
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "shown": shown,
        "offset": offset,
        "set_size": np.array([float(len(s)) for s in unique]),
        "group_set": g_set,
        "group_in": np.array(g_in, dtype="float64"),
        "group_lookup": lookup.astype("int64"),
        "group_is_rule": np.array(g_rule),
        "group_count": np.array(g_count),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "entry_cell": np.concatenate(entry_cell).astype("int64"),
        "entry_group": np.concatenate(entry_group).astype("int64"),
        "entry_count": np.concatenate(entry_count).astype("float64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
