"""Ratio-scale (Weber) exemplar retrieval of stored number concepts.

People hold number concepts as stored member lists (45 equally plausible rules
and every magnitude range of 1-100) and compare numbers by their ratio, not
their difference: similarity s(x, y) = r^D(x, y) with D = 20 |ln x - ln y|
(D = 1 for neighbours near 20; 8 vs 10 is as similar as 80 vs 100), the
gradient r inferred. A concept is retrieved in proportion to its prior times,
per example, a mixture of the example's mean similarity to the concept's
members and a stray (uniform, or with inferred probability mix a familiar
number: a random member of a random rule), with uncertain reliability
a ~ Beta integrated out per set; a target is endorsed by its similarity to the
nearest stored member of each retrieved concept. So near-misses around large
examples are partly endorsed, near-misses around small ones are not.

E_a prod_i (a s_i + (1 - a) q_i) = prod_i q_i * sum_S c_|S| prod_{i in S} s_i / q_i;
range evidence is tabulated per set/subset on a logit(r) grid in prepare_observed.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4
GRID = np.linspace(-8.0, -0.5, 100)  # logit(r)
LOG_SCALE = 20.0
GRID_STEP = GRID[1] - GRID[0]


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


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
LOGN = np.log(NUMBERS.astype("float64"))
RATIO_DIST = LOG_SCALE * np.abs(LOGN[:, None] - LOGN[None, :])  # (numbers, numbers)
RULE_DMIN = np.array([np.where(m[None, :] > 0, RATIO_DIST, np.inf).min(axis=1) for m in RULE_MEMBERSHIP])

STRAY_RULE_GEN = (RULE_MEMBERSHIP / RULE_SIZE[:, None]).mean(axis=0)  # (100,)
SUBSETS = np.array([[(m >> i) & 1 for i in range(MAX_N)] for m in range(2**MAX_N)], dtype=bool)  # (16, 4)
SUBSET_SIZE = SUBSETS.sum(axis=1)

_N = np.arange(1, MAX_N + 1, dtype="float64")[:, None]
_J = np.arange(MAX_N + 1, dtype="float64")[None, :]
_VALID = _J <= _N
_N_MINUS_J = np.maximum(_N - _J, 0.0)


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


def _elementary(weights, lib=pt):
    """Elementary symmetric polynomials e_0..e_4 over the last axis (4 examples)."""
    ones = lib.ones_like(weights[..., 0])
    e = [ones] + [ones * 0.0] * MAX_N
    for i in range(MAX_N):
        w = weights[..., i]
        e = [e[0]] + [e[j] + w * e[j - 1] for j in range(1, MAX_N + 1)]
    return lib.stack(e, axis=-1)


def range_tables(example_idx, example_mask, cell_set, cell_target):
    """Per logit(r) grid point: log sum over ranges of prod_{i in S} 100 s_i(h)
    per set and subset S, and the same weighted by each cell target's
    similarity to the range (cells, S)."""
    lo = np.array([m for m in NUMBERS for n in range(m, 101)])
    hi = np.array([n for m in NUMBERS for n in range(m, 101)])
    size = (hi - lo + 1).astype("float64")[:, None]
    x = NUMBERS[None, :]
    inside = (x >= lo[:, None]) & (x <= hi[:, None])
    # ratio distance from each number to the nearest member of each range
    dmin = np.where(inside, 0.0, np.minimum(RATIO_DIST[lo - 1].T, RATIO_DIST[hi - 1].T).T)
    S, C = example_idx.shape[0], len(cell_set)
    K = len(SUBSETS)
    log_set = np.zeros((len(GRID), S, K))
    log_cell = np.zeros((len(GRID), C, K))
    cells_of = [np.where(cell_set == s)[0] for s in range(S)]
    for g, logit_r in enumerate(GRID):
        r = 1.0 / (1.0 + np.exp(-logit_r))
        sim = r**RATIO_DIST  # (numbers, members)
        cum = np.concatenate([np.zeros((100, 1)), np.cumsum(sim, axis=1)], axis=1)  # (numbers, 101)
        summed = (cum[:, hi] - cum[:, lo - 1]).T  # (ranges, numbers)
        w100 = 100.0 * summed / size  # (ranges, numbers)
        endorse = r**dmin  # (ranges, numbers)
        for s in range(S):
            wts = w100[:, example_idx[s]] * example_mask[s][None, :]  # (ranges, 4)
            prods = np.prod(np.where(SUBSETS[None, :, :], wts[:, None, :], 1.0), axis=2)  # (ranges, 16)
            log_set[g, s] = np.log(np.maximum(prods.sum(axis=0), 1e-300))
            cs = cells_of[s]
            if len(cs):
                num = endorse[:, cell_target[cs]].T @ prods  # (k, 16)
                log_cell[g, cs] = np.log(np.maximum(num, 1e-300))
    return log_set, log_cell


with pm.Model() as model:
    example_idx = pm.Data("example_idx", np.zeros((1, MAX_N), dtype="int64"))  # x - 1
    example_mask = pm.Data("example_mask", np.zeros((1, MAX_N)))
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))
    range_log_e = pm.Data("range_log_e", np.zeros((len(GRID), 1, len(SUBSETS))))  # (grid, sets, S)
    range_log_cell = pm.Data("range_log_cell", np.zeros((len(GRID), 1, len(SUBSETS))))  # (grid, cells, S)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    cell_n_idx = pm.Data("cell_n_idx", np.zeros(1, dtype="int64"))  # (cells,): n - 1
    ex_onehot = pm.Data("ex_onehot", np.zeros((1, MAX_N, 100)))  # one-hot examples (masked rows zero)
    cell_flat = pm.Data("cell_flat", np.zeros(1, dtype="int64"))  # set * 100 + target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    # Ratio-scale similarity gradient s(x, y) = r^D(x, y).
    r_logit = pm.TruncatedNormal("r_logit", mu=-3.0, sigma=1.5, lower=GRID[0], upper=GRID[-1])
    r = pm.Deterministic("r", pm.math.sigmoid(r_logit))
    log_r = pt.log(r)

    c_table = pt.where(_VALID, pt.exp(_betaln(u + _J, v + _N_MINUS_J) - _betaln(u, v)), 0.0)
    c_set = c_table[set_n_idx]  # (sets, j)

    # Stray distribution: uniform mixed with a random member of a random rule.
    mix_logit = pm.Normal("mix_logit", mu=0.0, sigma=1.5)
    mix = pm.Deterministic("mix", pm.math.sigmoid(mix_logit))
    rho = 1.0 / ((1 - mix) + mix * 100.0 * STRAY_RULE_GEN)  # (numbers,): 1 / (100 q)
    rho_ex = rho[example_idx]  # (sets, 4); masked examples have zero weight anyway
    rho_sub = pt.prod(pt.where(SUBSETS[None, :, :], rho_ex[:, None, :], 1.0), axis=2)  # (sets, 16)
    w_sub = c_set[:, SUBSET_SIZE] * rho_sub  # (sets, 16)

    # Rules: mean similarity of each example to the rule's stored members.
    sim_rule = pt.dot(pt.exp(log_r * RATIO_DIST), RULE_MEMBERSHIP.T) / RULE_SIZE  # (numbers, rules)
    w_rule = 100.0 * pt.dot(ex_onehot, sim_rule)  # (sets, 4, rules), masked rows zero
    e_rule = _elementary((w_rule * rho_ex[:, :, None]).dimshuffle(0, 2, 1))  # (sets, rules, j)
    rule_lik = (e_rule * c_set[:, None, :]).sum(axis=2)
    log_rules = pt.log(lambda_0) - np.log(N_RULES) + pt.log(rule_lik)

    # Ranges: Gaussian-kernel interpolation of the tabulated logs in logit(r).
    kernel = pt.exp(-0.5 * ((r_logit - GRID) / GRID_STEP) ** 2)
    kw = kernel / kernel.sum()
    e_range = pt.exp(pt.tensordot(kw, range_log_e, axes=[[0], [0]]))  # (sets, S)
    num_range = pt.exp(pt.tensordot(kw, range_log_cell, axes=[[0], [0]]))  # (cells, S)
    range_ev = (e_range * w_sub).sum(axis=1)
    log_ranges = pt.log1p(-lambda_0) - np.log(5050.0) + pt.log(range_ev)
    range_share = (num_range * w_sub[cell_set]).sum(axis=1) / range_ev[cell_set]

    log_joint = pt.concatenate([log_rules, log_ranges[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    range_post = pt.exp(log_ranges - log_norm)

    rule_endorse = pt.exp(log_r * RULE_DMIN)  # (rules, numbers)
    set_endorse = pt.dot(rule_post, rule_endorse)  # (sets, numbers)
    rule_part = set_endorse.flatten()[cell_flat]
    p_cell = rule_part + range_post[cell_set] * range_share
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    example_idx = np.zeros((S, MAX_N), dtype="int64")
    example_mask = np.zeros((S, MAX_N))
    set_n_idx = np.zeros(S, dtype="int64")
    for i, s in enumerate(unique):
        n = len(s)
        set_n_idx[i] = n - 1
        example_idx[i, :n] = np.array(s) - 1
        example_mask[i, :n] = 1.0
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_set = np.array([i for i, _ in cells], dtype="int64")
    cell_target = np.array([t - 1 for _, t in cells], dtype="int64")
    log_e, log_cell = range_tables(example_idx, example_mask, cell_set, cell_target)
    return {
        "example_idx": example_idx,
        "example_mask": example_mask,
        "set_n_idx": set_n_idx,
        "range_log_e": log_e,
        "range_log_cell": log_cell,
        "cell_set": cell_set,
        "cell_n_idx": set_n_idx[cell_set],
        "ex_onehot": np.eye(100)[example_idx] * example_mask[..., None],
        "cell_flat": cell_set * 100 + cell_target,
        "cell_target": cell_target,
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
