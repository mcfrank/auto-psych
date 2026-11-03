"""Recognised examples in final-digit-salient concept learning.

Refinement of `final_digit_salient_rules`: the same exception-tolerant, tempered
Bayesian concept learning over the 45 rules (final-digit rules boosted by kappa)
and Erlang-width intervals on a compressed number line. The one change: when the
target is itself one of the shown examples, people recognise it from memory with
fitted probability rho and say yes outright ("it was shown, so the program makes
it"), whatever their concept inference implies; otherwise they answer from the
concept posterior. Shown examples are thus endorsed more than the exception-
tolerant posterior implies, and the exception rate eps is freed to describe
generalisation to new numbers only.
"""

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


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    # Intervals holding >= 1 example, one entry per (set, interval).
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_int = pm.Data("pair_int", np.zeros(1, dtype="int64"))
    pair_k = pm.Data("pair_k", np.zeros(1))  # examples inside the interval
    pair_log_x = pm.Data("pair_log_x", np.zeros(1))  # sum of log x over those examples
    # Intervals holding no example all share one likelihood: lumped per set.
    empty_size_count = pm.Data("empty_size_count", np.zeros((1, 100)))  # (sets, widths)
    # Cells (set, target): which listed intervals contain the target.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_empty_size_count = pm.Data("cell_empty_size_count", np.zeros((1, 100)))  # empty intervals holding target, by width
    cell_is_example = pm.Data("cell_is_example", np.zeros(1))  # target is one of the shown examples
    hit_pair = pm.Data("hit_pair", np.zeros(1, dtype="int64"))
    hit_cell = pm.Data("hit_cell", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples
    gamma = pm.HalfNormal("gamma", sigma=1.0)  # compression of the mental number line
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)  # typical interval width
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)  # log prior boost of final-digit rules
    rho = pm.Beta("rho", 2, 2)  # probability a shown example is recognised and endorsed outright

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)  # (rules,)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    log_rule_prior = kappa * RULE_SALIENT - pm.math.logsumexp(kappa * RULE_SALIENT)
    log_rules = pt.log(lambda_0) + log_rule_prior[None, :] + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)  # per-set stabiliser
    w_rules = pt.exp(log_rules - shift[:, None])

    # Interval members are drawn with weight x^-gamma: P(x | [a, b]) = x^-gamma / W(a, b).
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    log_w = pt.log(cum_w[INT_B] - cum_w[INT_A - 1])  # (intervals,)
    pair_lik = (
        pair_k * (pt.log1p(-eps) - log_w[pair_int])
        - gamma * pair_log_x
        + (set_n[pair_set] - pair_k) * log_noise
    )
    # Erlang width prior: log P(interval of width s) = log s - s / sigma - log Z.
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z  # (widths,)
    w_pairs = pt.exp(log_size_prior[pt.as_tensor(INT_SIZE_IDX)[pair_int]] + beta * pair_lik - shift[pair_set])
    size_prior = pt.exp(log_size_prior)
    empty_scale = pt.exp(beta * set_n * log_noise - shift)  # (sets,): shared likelihood of empty intervals
    w_empty = empty_scale * (empty_size_count * size_prior[None, :]).sum(axis=1)

    z = w_rules.sum(axis=1) + _scatter_sum(n_sets, pair_set, w_pairs) + w_empty  # (sets,)
    numer = (
        (w_rules[cell_set] * cell_rule_member).sum(axis=1)
        + _scatter_sum(n_cells, hit_cell, w_pairs[hit_pair])
        + empty_scale[cell_set] * (cell_empty_size_count * size_prior[None, :]).sum(axis=1)
    )
    in_concept = numer / z[cell_set]
    recog = rho * cell_is_example
    answer = recog + (1 - recog) * in_concept

    p_yes = pm.Deterministic("p_yes", pt.clip(answer[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}

    pair_set, pair_int, pair_k, pair_log_x, empty_size_count = [], [], [], [], []
    set_pairs, set_empty = [], []
    offset = 0
    for i, st in enumerate(unique):
        ex = np.array(st)
        inside = (INT_A[:, None] <= ex[None, :]) & (INT_B[:, None] >= ex[None, :])
        k = inside.sum(axis=1)
        listed = np.nonzero(k > 0)[0]
        empty = np.nonzero(k == 0)[0]
        pair_set += [i] * len(listed)
        pair_int += list(listed)
        pair_k += list(k[listed].astype(float))
        pair_log_x += list((inside[listed] * np.log(ex)[None, :]).sum(axis=1))
        empty_size_count.append(np.bincount(INT_SIZE_IDX[empty], minlength=100).astype("float64"))
        set_pairs.append((offset, listed))
        set_empty.append(empty)
        offset += len(listed)

    hit_pair, hit_cell, cell_empty_size_count = [], [], []
    for c, (i, t) in enumerate(cells):
        start, listed = set_pairs[i]
        has_t = np.nonzero((INT_A[listed] <= t) & (INT_B[listed] >= t))[0]
        hit_pair += list(start + has_t)
        hit_cell += [c] * len(has_t)
        empty = set_empty[i]
        holds = empty[(INT_A[empty] <= t) & (INT_B[empty] >= t)]
        cell_empty_size_count.append(np.bincount(INT_SIZE_IDX[holds], minlength=100).astype("float64"))

    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "pair_set": np.array(pair_set, dtype="int64"),
        "pair_int": np.array(pair_int, dtype="int64"),
        "pair_k": np.array(pair_k, dtype="float64"),
        "pair_log_x": np.array(pair_log_x, dtype="float64"),
        "empty_size_count": np.stack(empty_size_count),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_empty_size_count": np.stack(cell_empty_size_count),
        "cell_is_example": np.array([1.0 if t in unique[i] else 0.0 for i, t in cells]),
        "hit_pair": np.array(hit_pair, dtype="int64"),
        "hit_cell": np.array(hit_cell, dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
