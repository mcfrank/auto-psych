"""Two-source partition of the examples in exception-tolerant concept learning.

People believe the hidden program may draw its outputs from more than one
source, so they may split the examples into groups: one example that does not
fit the others can be attributed to a second, separate source (prior readiness
rho to posit such a split, weighed by the tempered evidence of one source
against two), and they infer a concept for each source by the same
exception-tolerant, tempered Bayesian concept learning (45 rules with salient
final-digit rules, Erlang-width intervals on a compressed number line). A target
is endorsed if either source would produce it, so people generalise around an
off-pattern example in its own right while still extending the majority's rule.
"""

from collections import Counter

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


RULE_NAMES = list(rule_concepts().keys())
RULES = list(rule_concepts().values())
# Rules read straight off the final digit: parity and "ends in d".
RULE_SALIENT = np.array([n in ("odd", "even") or n.startswith("ends in") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# All intervals [a, b] of 1..100.
INT_A = np.array([a for a in NUMBERS for b in NUMBERS if b >= a])
INT_B = np.array([b for a in NUMBERS for b in NUMBERS if b >= a])
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
INT_SIZE_IDX = INT_B - INT_A  # width - 1
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)  # intervals of each width


MAX_SPLIT = 4  # most distinct one-example splits of a set
MAX_K = 4  # most examples in a source
KS = np.arange(1, MAX_K + 1, dtype="float64")
GRID_A = np.repeat(NUMBERS, 100).reshape(100, 100)  # [a-1, b-1] -> a
GRID_B = np.tile(NUMBERS, 100).reshape(100, 100)  # [a-1, b-1] -> b
GRID_VALID = GRID_A <= GRID_B
GRID_WIDTH = np.clip(GRID_B - GRID_A, 0, 99)  # width - 1


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    # "Sources": every example group a source may explain (whole sets, sets minus one example, single examples).
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sources, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sources,): number of examples
    # Intervals holding >= 1 example fall into blocks: those holding exactly the examples between two of
    # them. A block's likelihood is shared, so it is a rectangle (a-range x b-range) of the interval grid.
    blk_src = pm.Data("blk_src", np.zeros(1, dtype="int64"))
    blk_k = pm.Data("blk_k", np.ones(1))  # examples inside
    blk_log_x = pm.Data("blk_log_x", np.zeros(1))  # their summed log value
    blk_rect = pm.Data("blk_rect", np.ones((1, 4), dtype="int64"))  # a0, a1, b0, b1
    empty_size_count = pm.Data("empty_size_count", np.zeros((1, 100)))
    # Source cells (source, target).
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_empty_size_count = pm.Data("cell_empty_size_count", np.zeros((1, 100)))
    # (source cell, block) pairs: the part of the block's rectangle whose intervals hold the target.
    hit_cell = pm.Data("hit_cell", np.zeros(1, dtype="int64"))
    hit_blk = pm.Data("hit_blk", np.zeros(1, dtype="int64"))
    hit_rect = pm.Data("hit_rect", np.ones((1, 4), dtype="int64"))
    # Shown sets: the whole-set source and, per split, the majority-group and lone-example sources.
    shown_whole = pm.Data("shown_whole", np.zeros(1, dtype="int64"))
    split_group = pm.Data("split_group", np.zeros((1, MAX_SPLIT), dtype="int64"))
    split_single = pm.Data("split_single", np.zeros((1, MAX_SPLIT), dtype="int64"))
    split_mask = pm.Data("split_mask", np.zeros((1, MAX_SPLIT)))
    # Shown cells (shown set, target): matching source cells.
    tc_shown = pm.Data("tc_shown", np.zeros(1, dtype="int64"))
    tc_whole = pm.Data("tc_whole", np.zeros(1, dtype="int64"))
    tc_group = pm.Data("tc_group", np.zeros((1, MAX_SPLIT), dtype="int64"))
    tc_single = pm.Data("tc_single", np.zeros((1, MAX_SPLIT), dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples
    gamma = pm.HalfNormal("gamma", sigma=1.0)  # compression of the mental number line
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)  # typical interval width
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)  # log prior boost of final-digit rules
    rho = pm.Beta("rho", 1, 3)  # prior readiness to posit a second source for one example

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)  # (rules,)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    log_rule_prior = kappa * RULE_SALIENT - pm.math.logsumexp(kappa * RULE_SALIENT)
    log_rules = pt.log(lambda_0) + log_rule_prior[None, :] + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)  # per-source stabiliser
    w_rules = pt.exp(log_rules - shift[:, None])

    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    log_w = pt.log(pt.maximum(cum_w[GRID_B] - cum_w[GRID_A - 1], 1e-12))  # (100, 100) over [a, b]
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z  # (widths,)
    # Prior x size-principle grid for intervals holding k examples, and its 2-D prefix sums.
    grid = pt.switch(
        GRID_VALID[None], pt.exp(log_size_prior[GRID_WIDTH][None] - beta * KS[:, None, None] * log_w[None]), 0.0
    )  # (k, a, b)
    prefix = pt.zeros((MAX_K + 1, 101, 101))
    prefix = pt.set_subtensor(prefix[1:, 1:, 1:], pt.cumsum(pt.cumsum(grid, axis=1), axis=2))

    def rect_sum(k, r):
        k = pt.cast(k, "int64")
        a0, a1, b0, b1 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        return pt.maximum(
            prefix[k, a1, b1] - prefix[k, a0 - 1, b1] - prefix[k, a1, b0 - 1] + prefix[k, a0 - 1, b0 - 1], 0.0
        )

    blk_scale = pt.exp(
        beta * (blk_k * pt.log1p(-eps) - gamma * blk_log_x + (set_n[blk_src] - blk_k) * log_noise) - shift[blk_src]
    )
    w_blk = blk_scale * rect_sum(blk_k, blk_rect)
    size_prior = pt.exp(log_size_prior)
    empty_scale = pt.exp(beta * set_n * log_noise - shift)
    w_empty = empty_scale * (empty_size_count * size_prior[None, :]).sum(axis=1)

    z = w_rules.sum(axis=1) + _scatter_sum(n_sets, blk_src, w_blk) + w_empty  # (sources,)
    log_evidence = pt.log(z) + shift  # tempered evidence of each source's examples
    numer = (
        (w_rules[cell_set] * cell_rule_member).sum(axis=1)
        + _scatter_sum(n_cells, hit_cell, blk_scale[hit_blk] * rect_sum(blk_k[hit_blk], hit_rect))
        + empty_scale[cell_set] * (cell_empty_size_count * size_prior[None, :]).sum(axis=1)
    )
    src_in = numer / z[cell_set]  # (source cells,): P(target in the source's concept)

    # Posterior over partitions: one source, or one example split off to its own source.
    n_split = split_mask.sum(axis=1)
    has_split = pt.gt(n_split, 0)
    log_one = pt.switch(has_split, pt.log1p(-rho), 0.0) + log_evidence[shown_whole]
    log_two = (
        pt.log(rho) - pt.log(pt.maximum(n_split, 1.0))[:, None]
        + log_evidence[split_group] + log_evidence[split_single]
    )
    log_two = pt.switch(pt.gt(split_mask, 0), log_two, -1e10)
    log_all = pt.concatenate([log_one[:, None], log_two], axis=1)
    post = pt.exp(log_all - pm.math.logsumexp(log_all, axis=1, keepdims=True))  # (shown sets, 1 + splits)

    a_whole = src_in[tc_whole]
    a_union = 1 - (1 - src_in[tc_group]) * (1 - src_in[tc_single])  # either source produces it
    post_c = post[tc_shown]
    in_output = post_c[:, 0] * a_whole + (post_c[:, 1:] * a_union).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_output[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    shown = list(dict.fromkeys(sets))
    shown_index = {s: i for i, s in enumerate(shown)}
    trial_cells = [(shown_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    tcells = list(dict.fromkeys(trial_cells))
    tcell_index = {c: i for i, c in enumerate(tcells)}

    sources = {}

    def src(group):
        return sources.setdefault(tuple(sorted(group)), len(sources))

    shown_whole = np.zeros(len(shown), dtype="int64")
    split_group = np.zeros((len(shown), MAX_SPLIT), dtype="int64")
    split_single = np.zeros((len(shown), MAX_SPLIT), dtype="int64")
    split_mask = np.zeros((len(shown), MAX_SPLIT))
    for i, st in enumerate(shown):
        shown_whole[i] = src(st)
        splits = []
        if len(st) > 1:
            for j in range(len(st)):
                rest, lone = st[:j] + st[j + 1:], (st[j],)
                key = frozenset([tuple(sorted(rest)), lone])
                if key not in [k for k, _, _ in splits]:
                    splits.append((key, rest, lone))
        for j, (_, rest, lone) in enumerate(splits):
            split_group[i, j], split_single[i, j], split_mask[i, j] = src(rest), src(lone), 1.0
    unique = list(sources.keys())

    cells = {}

    def cell(s, t):
        return cells.setdefault((s, t), len(cells))

    tc_whole = np.zeros(len(tcells), dtype="int64")
    tc_group = np.zeros((len(tcells), MAX_SPLIT), dtype="int64")
    tc_single = np.zeros((len(tcells), MAX_SPLIT), dtype="int64")
    for c, (i, t) in enumerate(tcells):
        tc_whole[c] = cell(shown_whole[i], t)
        for j in range(MAX_SPLIT):
            if split_mask[i, j]:
                tc_group[c, j] = cell(split_group[i, j], t)
                tc_single[c, j] = cell(split_single[i, j], t)
    cell_list = list(cells.keys())

    blk_src, blk_k, blk_log_x, blk_rect, src_blocks, empty_size_count, set_empty = [], [], [], [], [], [], []
    for i, st in enumerate(unique):
        counts = Counter(st)
        v = sorted(counts)
        blocks = []
        for p in range(len(v)):
            for q in range(p, len(v)):
                rect = (v[p - 1] + 1 if p else 1, v[p], v[q], v[q + 1] - 1 if q + 1 < len(v) else 100)
                blocks.append((len(blk_src), rect))
                blk_src.append(i)
                blk_k.append(float(sum(counts[x] for x in v[p:q + 1])))
                blk_log_x.append(sum(counts[x] * np.log(x) for x in v[p:q + 1]))
                blk_rect.append(rect)
        src_blocks.append(blocks)
        ex = np.array(st)
        inside = (INT_A[:, None] <= ex[None, :]) & (INT_B[:, None] >= ex[None, :])
        empty = np.nonzero(inside.sum(axis=1) == 0)[0]
        empty_size_count.append(np.bincount(INT_SIZE_IDX[empty], minlength=100).astype("float64"))
        set_empty.append(empty)

    hit_cell, hit_blk, hit_rect, cell_empty_size_count = [], [], [], []
    for c, (i, t) in enumerate(cell_list):
        for b, (a0, a1, b0, b1) in src_blocks[i]:
            a1, b0 = min(a1, t), max(b0, t)
            if a0 <= a1 and b0 <= b1:
                hit_cell.append(c)
                hit_blk.append(b)
                hit_rect.append((a0, a1, b0, b1))
        empty = set_empty[i]
        holds = empty[(INT_A[empty] <= t) & (INT_B[empty] >= t)]
        cell_empty_size_count.append(np.bincount(INT_SIZE_IDX[holds], minlength=100).astype("float64"))

    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "blk_src": np.array(blk_src, dtype="int64"),
        "blk_k": np.array(blk_k, dtype="float64"),
        "blk_log_x": np.array(blk_log_x, dtype="float64"),
        "blk_rect": np.array(blk_rect, dtype="int64"),
        "empty_size_count": np.stack(empty_size_count),
        "cell_set": np.array([i for i, _ in cell_list], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cell_list]),
        "cell_empty_size_count": np.stack(cell_empty_size_count),
        "hit_cell": np.array(hit_cell, dtype="int64"),
        "hit_blk": np.array(hit_blk, dtype="int64"),
        "hit_rect": np.array(hit_rect, dtype="int64"),
        "shown_whole": shown_whole,
        "split_group": split_group,
        "split_single": split_single,
        "split_mask": split_mask,
        "tc_shown": np.array([i for i, _ in tcells], dtype="int64"),
        "tc_whole": tc_whole,
        "tc_group": tc_group,
        "tc_single": tc_single,
        "row_cell": np.array([tcell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
