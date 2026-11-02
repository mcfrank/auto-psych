"""Rule-plus-exception-neighbourhood concepts in exception-tolerant tempered concept learning.

People do the same exception-tolerant, tempered Bayesian concept learning over the
same 45 rules (final-digit rules boosted by kappa) and Erlang-width ranges on a
compressed number line, but when one example breaks the others' pattern they
explain it as the seed of a small numerical region the program also covers: a
rule R together with the window of numbers within h of an off-rule example x
("ends in 2, and also numbers around 27"). These concepts take a fitted share
alpha of each rule's prior; the half-width h has a geometric prior with fitted
mean tau (h = 0 is the single-exception concept). Examples are drawn uniformly
from R united with the window. So people endorse non-example neighbours of an
off-pattern example that a rule-plus-single-exception learner rejects.
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
RULE_SALIENT = np.array([n in ("odd", "even") or n.startswith("ends in") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype=bool)
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1).astype("float64")
INT_A = np.array([a for a in NUMBERS for b in NUMBERS if b >= a])
INT_B = np.array([b for a in NUMBERS for b in NUMBERS if b >= a])
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
INT_SIZE_IDX = INT_B - INT_A
MAX_H = 10  # largest half-width of an exception's neighbourhood
HALF_WIDTHS = np.arange(MAX_H + 1, dtype="float64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))
    set_n = pm.Data("set_n", np.ones(1))
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_int = pm.Data("pair_int", np.zeros(1, dtype="int64"))
    pair_k = pm.Data("pair_k", np.zeros(1))
    pair_log_x = pm.Data("pair_log_x", np.zeros(1))
    empty_size_count = pm.Data("empty_size_count", np.zeros((1, 100)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_empty_size_count = pm.Data("cell_empty_size_count", np.zeros((1, 100)))
    hit_pair = pm.Data("hit_pair", np.zeros(1, dtype="int64"))
    hit_cell = pm.Data("hit_cell", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    # Rule R plus window of half-width h around an off-rule example: one entry per (set, rule, example, h).
    nb_set = pm.Data("nb_set", np.zeros(1, dtype="int64"))
    nb_set_rule = pm.Data("nb_set_rule", np.zeros(1, dtype="int64"))  # flat set * N_RULES + rule
    nb_rule = pm.Data("nb_rule", np.zeros(1, dtype="int64"))
    nb_h = pm.Data("nb_h", np.zeros(1, dtype="int64"))
    nb_k = pm.Data("nb_k", np.zeros(1))  # examples inside the union
    nb_size = pm.Data("nb_size", np.ones(1))  # size of the union
    # Sparse (cell, entry) pairs: target lies in the window but outside the rule.
    nb_hit_cell = pm.Data("nb_hit_cell", np.zeros(1, dtype="int64"))
    nb_hit_entry = pm.Data("nb_hit_entry", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)
    eps = pm.Beta("eps", 2, 5)
    gamma = pm.HalfNormal("gamma", sigma=1.0)
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)
    alpha = pm.Beta("alpha", 1, 4)  # share of a rule's prior given to rule-plus-neighbourhood variants
    tau = pm.LogNormal("tau", mu=np.log(2.0), sigma=1.0)  # typical half-width of an exception's region

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    log_rule_prior = kappa * RULE_SALIENT - pm.math.logsumexp(kappa * RULE_SALIENT)
    log_rules = pt.log(lambda_0) + pt.log1p(-alpha) + log_rule_prior[None, :] + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)
    w_rules = pt.exp(log_rules - shift[:, None])

    # Neighbourhood prior: geometric over h with mean tau; centre uniform over 1..100.
    log_h = -HALF_WIDTHS * pt.log1p(1.0 / tau)
    log_h = log_h - pm.math.logsumexp(log_h)
    log_nb_prior = pt.log(lambda_0) + pt.log(alpha) + log_rule_prior[nb_rule] + log_h[nb_h] - np.log(100)
    nb_in = pt.log((1 - eps) / nb_size + eps / 100)
    nb_lik = nb_k * nb_in + (set_n[nb_set] - nb_k) * log_noise
    w_nb = pt.exp(log_nb_prior + beta * nb_lik - shift[nb_set])  # (entries,)
    w_nb_rule = _scatter_sum(n_sets * N_RULES, nb_set_rule, w_nb).reshape((n_sets, N_RULES))

    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    log_w = pt.log(cum_w[INT_B] - cum_w[INT_A - 1])
    pair_lik = (
        pair_k * (pt.log1p(-eps) - log_w[pair_int])
        - gamma * pair_log_x
        + (set_n[pair_set] - pair_k) * log_noise
    )
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z
    w_pairs = pt.exp(log_size_prior[pt.as_tensor(INT_SIZE_IDX)[pair_int]] + beta * pair_lik - shift[pair_set])
    size_prior = pt.exp(log_size_prior)
    empty_scale = pt.exp(beta * set_n * log_noise - shift)
    w_empty = empty_scale * (empty_size_count * size_prior[None, :]).sum(axis=1)

    z = w_rules.sum(axis=1) + w_nb_rule.sum(axis=1) + _scatter_sum(n_sets, pair_set, w_pairs) + w_empty
    numer = (
        ((w_rules + w_nb_rule)[cell_set] * cell_rule_member).sum(axis=1)
        + _scatter_sum(n_cells, nb_hit_cell, w_nb[nb_hit_entry])
        + _scatter_sum(n_cells, hit_cell, w_pairs[hit_pair])
        + empty_scale[cell_set] * (cell_empty_size_count * size_prior[None, :]).sum(axis=1)
    )
    in_concept = numer / z[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
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

    nb = {k: [] for k in ("set", "set_rule", "rule", "h", "k", "size")}
    entries = []  # per set: list of (rule, centre, h, entry index)
    for i, st in enumerate(unique):
        ex = np.array(st)
        mine = []
        for r in range(N_RULES):
            for x in dict.fromkeys(st):
                if RULE_MEMBERSHIP[r, x - 1]:
                    continue
                for h in range(MAX_H + 1):
                    union = RULE_MEMBERSHIP[r] | (np.abs(NUMBERS - x) <= h)
                    mine.append((r, x, h, len(nb["set"])))
                    nb["set"].append(i)
                    nb["set_rule"].append(i * N_RULES + r)
                    nb["rule"].append(r)
                    nb["h"].append(h)
                    nb["k"].append(float(union[ex - 1].sum()))
                    nb["size"].append(float(union.sum()))
        entries.append(mine)
    nb_hit_cell, nb_hit_entry = [], []
    for c, (i, t) in enumerate(cells):
        for r, x, h, e in entries[i]:
            if abs(t - x) <= h and not RULE_MEMBERSHIP[r, t - 1]:
                nb_hit_cell.append(c)
                nb_hit_entry.append(e)

    return {
        "nb_set": np.array(nb["set"], dtype="int64"),
        "nb_set_rule": np.array(nb["set_rule"], dtype="int64"),
        "nb_rule": np.array(nb["rule"], dtype="int64"),
        "nb_h": np.array(nb["h"], dtype="int64"),
        "nb_k": np.array(nb["k"], dtype="float64"),
        "nb_size": np.array(nb["size"], dtype="float64"),
        "nb_hit_cell": np.array(nb_hit_cell, dtype="int64"),
        "nb_hit_entry": np.array(nb_hit_entry, dtype="int64"),
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]).astype("float64"),
        "set_n": np.array([float(len(s)) for s in unique]),
        "pair_set": np.array(pair_set, dtype="int64"),
        "pair_int": np.array(pair_int, dtype="int64"),
        "pair_k": np.array(pair_k, dtype="float64"),
        "pair_log_x": np.array(pair_log_x, dtype="float64"),
        "empty_size_count": np.stack(empty_size_count),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]).astype("float64"),
        "cell_empty_size_count": np.stack(cell_empty_size_count),
        "hit_pair": np.array(hit_pair, dtype="int64"),
        "hit_cell": np.array(hit_cell, dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
