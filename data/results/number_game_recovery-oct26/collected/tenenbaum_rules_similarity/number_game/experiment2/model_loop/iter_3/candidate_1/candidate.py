"""Extreme examples attended sampling: people treat the smallest and largest
examples as samples from the program's concept (strong sampling), while the
examples lying between them are encoded only partly as samples -- each inner
example's size-principle evidence is weighted by an inferred attention
weight, the rest counting only as consistency (weak sampling). Otherwise:
Bayesian concept learning over 42 arithmetic rules (primes excluding 1,
kind-level prior) and every range of 1-100 (Erlang width prior, decade-boundary
bonus, a range topping out at the largest example gets an evidence bonus),
10% random examples, "yes" = posterior probability the concept contains the
target. So inner examples tighten generalization less than the ideal learner
predicts, while examples that extend the span tighten it fully.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
NEG = -1e4
L_OUT = np.log(EPSILON / 100)
L_WEAK_IN = np.log(1.0 / 100)

SAMPLER_SETTINGS = {"target_accept": 0.9}


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
RULE_SIZE = RULE_MEMBERSHIP.sum(1)
LO = np.array([m for m, _ in BOUNDS])
HI = np.array([n for _, n in BOUNDS])
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
INTERVAL_WIDTH = HI - LO + 1
DECADE_BOUNDED = ((LO - 1) % 10 == 0) & (HI % 10 == 0)
GROUP_KEYS, GROUP_OF_INTERVAL = np.unique(
    np.stack([INTERVAL_WIDTH, DECADE_BOUNDED.astype("int64")], axis=1), axis=0, return_inverse=True
)
GROUP_OF_INTERVAL = GROUP_OF_INTERVAL.ravel()
GROUP_SIZES = np.bincount(GROUP_OF_INTERVAL)
GROUP_WIDTHS = GROUP_KEYS[:, 0].astype("float64")
GROUP_ALIGNED = GROUP_KEYS[:, 1].astype("float64")
N_GROUPS = len(GROUP_KEYS)
# Interval cells: (at_top k, group g, extremes covered e in 0..2, inner covered i in 0..2).
CELL_SHAPE = (2, N_GROUPS, 3, 3)
N_CELLS = int(np.prod(CELL_SHAPE))
CELL_K, CELL_G, CELL_E, CELL_I = [a.ravel() for a in np.indices(CELL_SHAPE)]
CELL_LOG_STRONG_IN = np.log((1 - EPSILON) / GROUP_WIDTHS[CELL_G] + EPSILON / 100)


def split_examples(examples):
    s = sorted(examples)
    if len(s) == 1:
        return np.array(s), np.array([], dtype="int64")
    return np.array([s[0], s[-1]]), np.array(s[1:-1], dtype="int64")


def set_quantities(examples):
    ext, inner = split_examples(examples)
    # Rules: strong log-lik of extremes, strong and weak log-lik of inner examples.
    strong_in = np.log((1 - EPSILON) / RULE_SIZE + EPSILON / 100)
    def strong(xs):
        m = RULE_MEMBERSHIP[:, xs - 1] if len(xs) else np.zeros((N_RULES, 0))
        return (m * strong_in[:, None] + (1 - m) * L_OUT).sum(1)
    def weak(xs):
        m = RULE_MEMBERSHIP[:, xs - 1] if len(xs) else np.zeros((N_RULES, 0))
        return (m * L_WEAK_IN + (1 - m) * L_OUT).sum(1)
    rule_ext = strong(ext)
    rule_inner_strong = strong(inner)
    rule_inner_weak = weak(inner)
    # Intervals pooled into cells.
    e_cov = INTERVAL_MEMBERSHIP[:, ext - 1].sum(1).astype("int64")
    i_cov = INTERVAL_MEMBERSHIP[:, inner - 1].sum(1).astype("int64") if len(inner) else np.zeros(len(BOUNDS), "int64")
    at_top = (HI == max(examples)).astype("int64")
    cell = np.ravel_multi_index((at_top, GROUP_OF_INTERVAL, e_cov, i_cov), CELL_SHAPE)
    count = np.bincount(cell, minlength=N_CELLS).astype("float64")
    mem = np.zeros((N_CELLS, 100))
    np.add.at(mem, cell, INTERVAL_MEMBERSHIP)
    mem = mem / np.maximum(count, 1.0)[:, None]
    n_ext, n_in = len(ext), len(inner)
    ext_ll = CELL_E * CELL_LOG_STRONG_IN + (n_ext - CELL_E) * L_OUT
    inner_strong = CELL_I * CELL_LOG_STRONG_IN + (n_in - CELL_I) * L_OUT
    inner_weak = CELL_I * L_WEAK_IN + (n_in - CELL_I) * L_OUT
    log_count = np.where(count > 0, np.log(np.maximum(count, 1.0)), NEG)
    return (rule_ext, rule_inner_strong, rule_inner_weak,
            ext_ll + log_count, inner_strong, inner_weak, mem)


with pm.Model() as model:
    rule_ext_ll = pm.Data("rule_ext_ll", np.zeros((1, N_RULES)))
    rule_inner_strong = pm.Data("rule_inner_strong", np.zeros((1, N_RULES)))
    rule_inner_weak = pm.Data("rule_inner_weak", np.zeros((1, N_RULES)))
    cell_ext_ll = pm.Data("cell_ext_ll", np.zeros((1, N_CELLS)))
    cell_inner_strong = pm.Data("cell_inner_strong", np.zeros((1, N_CELLS)))
    cell_inner_weak = pm.Data("cell_inner_weak", np.zeros((1, N_CELLS)))
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_in_hypothesis = pm.Data("pair_in_hypothesis", np.zeros((1, N_RULES + N_CELLS)))
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    rule_share = pm.Beta("rule_share", alpha=2.0, beta=2.0)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    top_bonus = pm.Normal("top_bonus", mu=0.0, sigma=1.5)
    # How much an inner example counts as a sample (1 = ideal learner).
    inner_attention = pm.Beta("inner_attention", alpha=2.0, beta=2.0)

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))

    w = inner_attention
    log_rules = (pt.log(rule_share) + pt.log(rule_weights)[None, :] + rule_ext_ll
                 + w * rule_inner_strong + (1 - w) * rule_inner_weak)
    cell_prior = log_interval_prior[CELL_G] + top_bonus * CELL_K
    log_cells = (pt.log1p(-rule_share) + cell_prior[None, :] + cell_ext_ll
                 + w * cell_inner_strong + (1 - w) * cell_inner_weak)
    log_joint = pt.concatenate([log_rules, log_cells], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[pair_set] * pair_in_hypothesis).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_pair], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    pairs = list(dict.fromkeys(zip(sets, targets)))
    pair_index = {c: i for i, c in enumerate(pairs)}
    q = {s: set_quantities(np.array(s)) for s in unique}
    out = {k: np.stack([q[s][j] for s in unique]) for j, k in enumerate(
        ["rule_ext_ll", "rule_inner_strong", "rule_inner_weak", "cell_ext_ll", "cell_inner_strong", "cell_inner_weak"])}
    out["pair_set"] = np.array([index[s] for s, _ in pairs], dtype="int64")
    out["pair_in_hypothesis"] = np.stack(
        [np.concatenate([RULE_MEMBERSHIP[:, t - 1], q[s][6][:, t - 1]]) for s, t in pairs])
    out["row_pair"] = np.array([pair_index[c] for c in zip(sets, targets)], dtype="int64")
    out["rating"] = np.array([int(r["rating"]) for r in rows], dtype="int64")
    return out
