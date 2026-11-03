"""Parity-restricted compressed-line ranges with an Erlang prior on range span.

Refinement of `parity_restricted_compressed_intervals`: people do the same
concept learning over rules, plain ranges and parity-restricted ranges sampled
on a compressed number line, but they expect a range to span a moderate stretch
of the number line. The one change (grafted from
`erlang_width_compressed_intervals`): within each range family the prior over
ranges follows an Erlang density on the span s = b - a + 1,
P(range) proportional to s exp(-s / sigma) (sigma fitted), replacing the flat
prior, so tiny ranges hugging one or two examples are a priori unlikely.

Parent model: the same 45 rules, prior
lambda_0 split equally over rules, tempered size principle (beta), fitted
exception rate eps, and range members sampled along a compressed number line,
P(x | range) = x^-gamma / sum_{k in range} k^-gamma. The one change: besides
plain ranges [a, b], the magnitude concepts include parity-restricted ranges
("even numbers in [a, b]", "odd numbers in [a, b]"), which take a fitted share
rho of the prior mass given to ranges. When the examples share a parity, near
same-parity numbers are endorsed while immediate (parity-flipping) neighbours
and the neighbours of an off-pattern example are endorsed less.
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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# Range families: 0 = all numbers, 1 = even numbers, 2 = odd numbers.
FAMILY_MASK = np.array([np.ones(100), NUMBERS % 2 == 0, NUMBERS % 2 == 1], dtype="float64")
_A = np.array([a for a in NUMBERS for b in NUMBERS if b >= a])
_B = np.array([b for a in NUMBERS for b in NUMBERS if b >= a])
# Each distinct parity range is listed once, by endpoints of its own parity.
_valid = [np.arange(len(_A))] + [
    np.nonzero((FAMILY_MASK[f, _A - 1] > 0) & (FAMILY_MASK[f, _B - 1] > 0))[0] for f in (1, 2)
]
CON_A = np.concatenate([_A[v] for v in _valid])
CON_B = np.concatenate([_B[v] for v in _valid])
CON_F = np.concatenate([np.full(len(v), f) for f, v in enumerate(_valid)])
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
CON_S = CON_B - CON_A  # span - 1
SIZES = np.arange(1, 101, dtype="float64")
FAM_SIZE_COUNT = np.stack([np.bincount(CON_S[CON_F == f], minlength=100) for f in range(3)])
LOG_FAM_SIZE_COUNT = np.where(FAM_SIZE_COUNT > 0, np.log(np.maximum(FAM_SIZE_COUNT, 1)), -1e4)


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    # Range concepts holding >= 1 example, one entry per (set, concept).
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_con = pm.Data("pair_con", np.zeros(1, dtype="int64"))
    pair_k = pm.Data("pair_k", np.zeros(1))  # examples inside the concept
    pair_log_x = pm.Data("pair_log_x", np.zeros(1))  # sum of log x over those examples
    # Range concepts holding no example share one likelihood: lumped per (set, family).
    empty_size_count = pm.Data("empty_size_count", np.zeros((1, 300)))  # (sets, family x span)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_empty_size_count = pm.Data("cell_empty_size_count", np.zeros((1, 300)))  # empty ranges holding target
    hit_pair = pm.Data("hit_pair", np.zeros(1, dtype="int64"))
    hit_cell = pm.Data("hit_cell", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples
    gamma = pm.HalfNormal("gamma", sigma=1.0)  # compression of the mental number line
    rho = pm.Beta("rho", 1, 3)  # share of range prior on parity-restricted ranges
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)  # typical range span

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    log_rules = pt.log(lambda_0) - np.log(N_RULES) + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)
    w_rules = pt.exp(log_rules - shift[:, None])

    # Compressed sampling weight within each family's range.
    x_w = pt.exp(-gamma * LOG_NUMBERS)[None, :] * FAMILY_MASK  # (3, 100)
    cum_w = pt.concatenate([pt.zeros((3, 1)), pt.cumsum(x_w, axis=1)], axis=1)
    log_w = pt.log(cum_w[CON_F, CON_B] - cum_w[CON_F, CON_A - 1])  # (concepts,)
    log_fam_share = pt.log1p(-lambda_0) + pt.stack([pt.log1p(-rho), pt.log(rho / 2), pt.log(rho / 2)])
    # Erlang span prior, normalised within each family: log s - s / sigma - log Z_f.
    log_width = pt.log(SIZES) - SIZES / sigma  # (spans,)
    log_z = pm.math.logsumexp(LOG_FAM_SIZE_COUNT + log_width[None, :], axis=1, keepdims=True)  # (3, 1)
    log_size_prior = log_fam_share[:, None] + log_width[None, :] - log_z  # (3, spans)
    log_con_prior = log_size_prior[CON_F, CON_S]  # (concepts,)
    pair_lik = (
        pair_k * (pt.log1p(-eps) - log_w[pair_con])
        - gamma * pair_log_x
        + (set_n[pair_set] - pair_k) * log_noise
    )
    w_pairs = pt.exp(log_con_prior[pair_con] + beta * pair_lik - shift[pair_set])
    size_prior = pt.exp(log_size_prior).reshape((300,))
    empty_scale = pt.exp(beta * set_n * log_noise - shift)  # (sets,): shared likelihood of empty ranges
    w_empty = empty_scale * (empty_size_count * size_prior[None, :]).sum(axis=1)

    z = w_rules.sum(axis=1) + _scatter_sum(n_sets, pair_set, w_pairs) + w_empty
    numer = (
        (w_rules[cell_set] * cell_rule_member).sum(axis=1)
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

    pair_set, pair_con, pair_k, pair_log_x, empty_size_count = [], [], [], [], []
    set_pairs, set_empty = [], []
    offset = 0
    for i, st in enumerate(unique):
        ex = np.array(st)
        inside = (
            (CON_A[:, None] <= ex[None, :])
            & (CON_B[:, None] >= ex[None, :])
            & (FAMILY_MASK[CON_F][:, ex - 1] > 0)
        )
        k = inside.sum(axis=1)
        listed = np.nonzero(k > 0)[0]
        empty = np.nonzero(k == 0)[0]
        pair_set += [i] * len(listed)
        pair_con += list(listed)
        pair_k += list(k[listed].astype(float))
        pair_log_x += list((inside[listed] * np.log(ex)[None, :]).sum(axis=1))
        empty_size_count.append(np.bincount(CON_F[empty] * 100 + CON_S[empty], minlength=300).astype(float))
        set_pairs.append((offset, listed))
        set_empty.append(empty)
        offset += len(listed)

    hit_pair, hit_cell, cell_empty_size_count = [], [], []
    for c, (i, t) in enumerate(cells):
        start, listed = set_pairs[i]
        contains = (CON_A[listed] <= t) & (CON_B[listed] >= t) & (FAMILY_MASK[CON_F[listed], t - 1] > 0)
        has_t = np.nonzero(contains)[0]
        hit_pair += list(start + has_t)
        hit_cell += [c] * len(has_t)
        empty = set_empty[i]
        e_has = (CON_A[empty] <= t) & (CON_B[empty] >= t) & (FAMILY_MASK[CON_F[empty], t - 1] > 0)
        holds = empty[e_has]
        cell_empty_size_count.append(np.bincount(CON_F[holds] * 100 + CON_S[holds], minlength=300).astype(float))

    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "pair_set": np.array(pair_set, dtype="int64"),
        "pair_con": np.array(pair_con, dtype="int64"),
        "pair_k": np.array(pair_k, dtype="float64"),
        "pair_log_x": np.array(pair_log_x, dtype="float64"),
        "empty_size_count": np.stack(empty_size_count),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_empty_size_count": np.stack(cell_empty_size_count),
        "hit_pair": np.array(hit_pair, dtype="int64"),
        "hit_cell": np.array(hit_cell, dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
