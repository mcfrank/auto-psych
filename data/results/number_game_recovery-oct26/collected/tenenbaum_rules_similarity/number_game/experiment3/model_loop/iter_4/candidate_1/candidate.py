"""Tversky-similarity retrieval of stored number categories.

People do not infer which concept produced the examples; they retrieve
familiar number categories from memory (42 arithmetic rules with a kind-level
familiarity, and every stretch of 1-100 with a familiarity favouring moderate,
decade-bounded widths) by how similar each stored category is to the example
set under Tversky's ratio model: sim = common / (common + alpha * examples the
category lacks + beta * category members not shown). Retrieval strength is
familiarity * sim ** gamma (Luce choice over categories), and "yes" is the
retrieval-weighted share of categories containing the target. Because
similarity depends on proportions, not joint probabilities, extra examples
consistent with a small category do not tighten generalization as the size
principle would.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def concept_sets():
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    kinds = [0] * 9 + [1] * 10 + [2] * 9 + [3] * 9 + [4] * 5
    bounds = [(m, n) for m in range(1, 101) for n in range(m, 101)]
    return rules, kinds, bounds


RULES, RULE_KINDS, BOUNDS = concept_sets()
N_RULES = len(RULES)
N_KINDS = 5
KIND_MATRIX = np.eye(N_KINDS)[RULE_KINDS]
KIND_COUNTS = KIND_MATRIX.sum(axis=0)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INTERVAL_MEMBERSHIP = np.array([(NUMBERS >= m) & (NUMBERS <= n) for m, n in BOUNDS], dtype="float64")
INTERVAL_WIDTH = np.array([n - m + 1 for m, n in BOUNDS], dtype="float64")
DECADE_BOUNDED = np.array([(m - 1) % 10 == 0 and n % 10 == 0 for m, n in BOUNDS], dtype="float64")
SIZES = RULE_MEMBERSHIP.sum(axis=1)
NEG = -1e4
MAX_N = 4
# Intervals with the same width, decade alignment and number of examples they
# contain have the same retrieval strength, so they are pooled into groups.
WD_KEYS, WD_OF_INTERVAL, WD_SIZES = np.unique(
    np.stack([INTERVAL_WIDTH, DECADE_BOUNDED], axis=1), axis=0, return_inverse=True, return_counts=True
)
WD_OF_INTERVAL = WD_OF_INTERVAL.ravel()
N_WD = len(WD_KEYS)
G_WIDTH = np.repeat(WD_KEYS[:, 0], MAX_N)
G_DECADE = np.repeat(WD_KEYS[:, 1], MAX_N)
G_COMMON = np.tile(np.arange(1, MAX_N + 1, dtype="float64"), N_WD)
N_GROUPS = len(G_WIDTH)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    n_shown, rule_common, group_log_count, group_mem = [], [], [], []
    for s in unique:
        ex = np.array(sorted(set(s)))
        n_shown.append(float(len(ex)))
        rule_common.append(RULE_MEMBERSHIP[:, ex - 1].sum(axis=1))
        c = INTERVAL_MEMBERSHIP[:, ex - 1].sum(axis=1).astype(int)
        log_count = np.full(N_GROUPS, NEG)
        mem = np.zeros((N_GROUPS, 100))
        ok = c > 0
        g = WD_OF_INTERVAL[ok] * MAX_N + c[ok] - 1
        counts = np.bincount(g, minlength=N_GROUPS)
        np.add.at(mem, g, INTERVAL_MEMBERSHIP[ok])
        has = counts > 0
        log_count[has] = np.log(counts[has])
        mem[has] /= counts[has, None]
        group_log_count.append(log_count)
        group_mem.append(mem)
    cell_in = np.stack(
        [np.concatenate([RULE_MEMBERSHIP[:, t - 1], group_mem[index[s]][:, t - 1]]) for s, t in cells]
    )
    return {
        "n_shown": np.array(n_shown),
        "rule_common": np.stack(rule_common),
        "group_log_count": np.stack(group_log_count),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_in_category": cell_in,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    n_shown = pm.Data("n_shown", np.ones(1))
    rule_common = pm.Data("rule_common", np.ones((1, N_RULES)))
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_in_category = pm.Data("cell_in_category", np.zeros((1, N_RULES + N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Familiarity of stored categories: rules vs stretches, kinds of rule,
    # and preferred stretch widths (Erlang) with a decade-boundary boost.
    rule_share = pm.Beta("rule_share", alpha=2.0, beta=2.0)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    # Tversky ratio-model weights: examples a category lacks (alpha) and
    # category members not shown (beta); retrieval sharpness gamma.
    alpha = pm.LogNormal("alpha", mu=0.0, sigma=1.0)
    beta = pm.LogNormal("beta", mu=np.log(0.1), sigma=1.5)
    gamma = pm.LogNormal("gamma", mu=np.log(3.0), sigma=1.0)

    def log_similarity(common, lacked, unshown):
        safe = pt.maximum(common, 1.0)
        return pt.log(safe) - pt.log(safe + alpha * lacked + beta * unshown)

    log_rule = pt.log(rule_share) + pt.log(pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS))
    log_wd = np.log(WD_KEYS[:, 0]) - WD_KEYS[:, 0] / width_scale + decade_boost * WD_KEYS[:, 1]
    log_wd = pt.log(1 - rule_share) + log_wd - pt.logsumexp(log_wd + np.log(WD_SIZES))
    log_group_fam = pt.repeat(log_wd, MAX_N)

    rule_sim = log_similarity(rule_common, n_shown[:, None] - rule_common, SIZES[None, :] - rule_common)
    rule_strength = log_rule[None, :] + gamma * pt.where(rule_common > 0, rule_sim, NEG)
    group_sim = log_similarity(
        G_COMMON[None, :], pt.maximum(n_shown[:, None] - G_COMMON[None, :], 0.0), (G_WIDTH - G_COMMON)[None, :]
    )
    group_strength = log_group_fam[None, :] + group_log_count + gamma * group_sim
    strength = pt.concatenate([rule_strength, group_strength], axis=1)
    retrieval = pt.exp(strength - pt.logsumexp(strength, axis=1, keepdims=True))
    in_category = (retrieval[cell_set] * cell_in_category).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_category[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
