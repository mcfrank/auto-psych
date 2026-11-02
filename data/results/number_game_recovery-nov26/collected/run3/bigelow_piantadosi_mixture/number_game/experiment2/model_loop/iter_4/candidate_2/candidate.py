"""Digit-position exemplar retrieval of stored number concepts.

People hold number concepts as stored member lists (45 rules -- parity, primes,
squares and cubes, powers, multiples, final digit, contained digit -- and every
magnitude range of 1-100) and compare numbers to stored members by written
form: s(x, y) = r^D(x, y), where D counts the digit positions (hundreds, tens,
ones, numbers written as three digits) in which x and y differ. A concept is
retrieved by its prior times, per example, a mixture of the example's mean
digit-wise similarity to the concept's members (size principle as r -> 0) and
a uniform stray, with uncertain reliability a ~ Beta integrated out per set;
rules plausible only by kind. A target is endorsed by r^(digit distance to the
nearest member) of each retrieved concept.

E_a prod_i (a s_i + (1 - a)/100) = 100^-n sum_j e_j(100 s) B(u+j, v+n-j)/B(u,v).
Ranges depend on parameters only through r (c_j enter linearly), so
prepare_observed tabulates their summed e_j per set and per cell on a logit(r)
grid; the graph interpolates the logs with Gaussian-kernel weights.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4
GRID = np.linspace(-8.0, -1.0, 96)
MAX_D = 3
GRID_STEP = GRID[1] - GRID[0]

_DIGITS = np.array([[n // 100, (n // 10) % 10, n % 10] for n in NUMBERS])
DIG_DIST = (_DIGITS[:, None, :] != _DIGITS[None, :, :]).sum(axis=2)  # (100, 100) in 0..3


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
RULE_DMIN = np.array(
    [np.where(m[None, :] > 0, DIG_DIST, 99).min(axis=1) for m in RULE_MEMBERSHIP], dtype="float64"
)

FAMILIES = ["parity", "primes", "squares and cubes", "powers", "multiples", "ends in", "contains digit"]


def family_of(name):
    if name in ("odd", "even"):
        return "parity"
    if name in ("squares", "cubes"):
        return "squares and cubes"
    for f in ("primes", "powers", "multiples", "ends in", "contains digit"):
        if name.startswith(f):
            return f
    raise ValueError(name)


RULE_FAMILY = np.array([FAMILIES.index(family_of(r)) for r in RULE_NAMES])

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


def _range_counts():
    """Members of each range (lo..hi) at digit distance d from each number: (ranges, numbers, d)."""
    lo = np.array([m for m in NUMBERS for n in range(m, 101)])
    hi = np.array([n for m in NUMBERS for n in range(m, 101)])
    out = np.zeros((len(lo), 100, MAX_D + 1))
    for d in range(MAX_D + 1):
        ind = (DIG_DIST == d).astype("float64")  # (x, y)
        cum = np.concatenate([np.zeros((100, 1)), np.cumsum(ind, axis=1)], axis=1)  # (x, 101)
        out[:, :, d] = (cum[:, hi] - cum[:, lo - 1]).T
    return out, (hi - lo + 1).astype("float64")


def range_tables(example_idx, example_mask, cell_set, cell_target):
    counts, size = _range_counts()
    dmin = np.argmax(counts > 0, axis=2).astype("float64")  # (ranges, numbers)
    S, C = example_idx.shape[0], len(cell_set)
    log_set = np.zeros((len(GRID), S, MAX_N + 1))
    log_cell = np.zeros((len(GRID), C, MAX_N + 1))
    for g, logit_r in enumerate(GRID):
        r = 1.0 / (1.0 + np.exp(-logit_r))
        summed = counts @ (r ** np.arange(MAX_D + 1))  # (ranges, numbers)
        w100 = 100.0 * summed / size[:, None]
        wts = w100[:, example_idx].transpose(1, 0, 2) * example_mask[:, None, :]  # (sets, ranges, 4)
        e = _elementary(wts, lib=np)
        tot = e.sum(axis=1)
        endorse = r**dmin
        num = np.einsum("chj,hc->cj", e[cell_set], endorse[:, cell_target])
        log_set[g] = np.log(np.maximum(tot, 1e-300))
        log_cell[g] = np.log(np.maximum(num, 1e-300))
    return log_set, log_cell


with pm.Model() as model:
    example_idx = pm.Data("example_idx", np.zeros((1, MAX_N), dtype="int64"))
    example_mask = pm.Data("example_mask", np.zeros((1, MAX_N)))
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))
    range_log_e = pm.Data("range_log_e", np.zeros((len(GRID), 1, MAX_N + 1)))
    range_log_cell = pm.Data("range_log_cell", np.zeros((len(GRID), 1, MAX_N + 1)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    cell_n_idx = pm.Data("cell_n_idx", np.zeros(1, dtype="int64"))
    ex_rule_dcount = pm.Data("ex_rule_dcount", np.zeros((1, MAX_N, N_RULES, MAX_D + 1)))
    cell_flat = pm.Data("cell_flat", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_level = pm.ZeroSumNormal("family_level", sigma=1.5, shape=len(FAMILIES))
    rule_weights = pm.Deterministic("rule_weights", pt.special.softmax(family_level[RULE_FAMILY]))
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    # Digit-position similarity gradient s(x, y) = r^(number of differing digits).
    r_logit = pm.TruncatedNormal("r_logit", mu=-4.0, sigma=1.0, lower=GRID[0], upper=GRID[-1])
    r = pm.Deterministic("r", pm.math.sigmoid(r_logit))
    log_r = pt.log(r)

    c_table = pt.where(_VALID, pt.exp(_betaln(u + _J, v + _N_MINUS_J) - _betaln(u, v)), 0.0)
    c_set = c_table[set_n_idx]
    c_cell = c_table[cell_n_idx]

    r_pow = pt.exp(log_r * np.arange(MAX_D + 1))
    w_rule = 100.0 * pt.dot(ex_rule_dcount, r_pow) / RULE_SIZE  # (sets, 4, rules)
    e_rule = _elementary(w_rule.dimshuffle(0, 2, 1))
    rule_lik = (e_rule * c_set[:, None, :]).sum(axis=2)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + pt.log(rule_lik)

    kernel = pt.exp(-0.5 * ((r_logit - GRID) / GRID_STEP) ** 2)
    kw = kernel / kernel.sum()
    e_range = pt.exp(pt.tensordot(kw, range_log_e, axes=[[0], [0]]))
    num_range = pt.exp(pt.tensordot(kw, range_log_cell, axes=[[0], [0]]))
    range_ev = (e_range * c_set).sum(axis=1)
    log_ranges = pt.log1p(-lambda_0) - np.log(5050.0) + pt.log(range_ev)
    range_share = (num_range * c_cell).sum(axis=1) / range_ev[cell_set]

    log_joint = pt.concatenate([log_rules, log_ranges[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    range_post = pt.exp(log_ranges - log_norm)

    rule_endorse = pt.exp(log_r * RULE_DMIN)  # (rules, numbers)
    set_endorse = pt.dot(rule_post, rule_endorse)
    rule_part = set_endorse.flatten()[cell_flat]
    p_cell = rule_part + range_post[cell_set] * range_share
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _distance_counts(example_idx, example_mask):
    """Number of each rule's members at digit distance d (0..3) from each example."""
    out = np.zeros(example_idx.shape + (N_RULES, MAX_D + 1))
    for d in range(MAX_D + 1):
        near = (DIG_DIST == d).astype("float64")
        out[..., d] = (near[example_idx] @ RULE_MEMBERSHIP.T) * example_mask[..., None]
    return out


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
        "ex_rule_dcount": _distance_counts(example_idx, example_mask),
        "cell_flat": cell_set * 100 + cell_target,
        "cell_target": cell_target,
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
