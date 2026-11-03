"""Compressed mental number line for magnitude concepts.

People do Bayesian concept learning over the 45 rules (each with its own
inferred prior weight) and every interval of 1-100, with the size principle and
an uncertain belief about how often an example is a stray draw from 1-100 --
but they think about magnitude ranges on a compressed mental number line.
A range concept is a stretch of that line, and its examples are taken as drawn
evenly along the line rather than evenly over the integers, so small numbers
occupy more room than large ones.  A range around small examples is therefore
mentally "wide" (weak size-principle evidence, broad generalisation), while a
range of the same span among large numbers is mentally narrow; examples low in
a range are more typical of it than examples high in it, so people extend a
range further upward than downward.

The mental position of x is x**c (log x at c = 0); c = 1 recovers the linear
number line of uncertain_reliability_noise_concepts.  The integer x occupies
w(x) = f(x + .5) - f(x - .5) of the line and a range [a, b] has length
L = f(b + .5) - f(a - .5), so a genuine example has p(x | [a, b]) = w(x) / L.
With the reliability Beta(u, v) integrated out,
p(X | [a, b]) = sum_j e_j / L**j * 100**-(n - j) * B(u + j, v + n - j) / B(u, v),
e_j the elementary symmetric polynomial of the in-range examples' w.  The sums
over intervals of e_j / L**j depend only on c, so prepare_observed tabulates
their logs per set (and per set-target cell, over the ranges containing the
target) on a grid of c and the graph interpolates them with Gaussian weights.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4
C_GRID = np.linspace(0.0, 1.6, 81)
C_STEP = C_GRID[1] - C_GRID[0]


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
RULE_SIZE_IDX = RULE_MEMBERSHIP.sum(axis=1).astype("int64") - 1
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Rule likelihood table over (n, k, s, j), as in the incumbent.
_N = np.arange(1, MAX_N + 1, dtype="float64")[:, None, None, None]
_K = np.arange(MAX_N + 1, dtype="float64")[None, :, None, None]
_S = np.arange(1, 101, dtype="float64")[None, None, :, None]
_J = np.arange(MAX_N + 1, dtype="float64")[None, None, None, :]
_VALID = (_J <= _K) & (_K <= _N)
_LOG_BINOM = np.where(
    _VALID,
    np.vectorize(lambda k, j: math.lgamma(k + 1) - math.lgamma(j + 1) - math.lgamma(max(k - j, 0) + 1))(_K, _J),
    -1e4,
)
_CONST = _LOG_BINOM - _J * np.log(_S) - (_N - _J) * np.log(100.0)
_N_MINUS_J = np.maximum(_N - _J, 0.0)
_J_SHAPED = _J + 0.0 * _N_MINUS_J
# Interval coefficients over (n, j): 100^-(n - j), -inf where j > n.
_NJ_N = np.arange(1, MAX_N + 1, dtype="float64")[:, None]
_NJ_J = np.arange(MAX_N + 1, dtype="float64")[None, :]
_NJ_LOG100 = np.where(_NJ_J <= _NJ_N, -(_NJ_N - _NJ_J) * np.log(100.0), -1e4)
_NJ_NMJ = np.maximum(_NJ_N - _NJ_J, 0.0)
_NJ_JS = _NJ_J + 0.0 * _NJ_NMJ


def _line(x, c):
    return np.log(x) if c == 0 else x**c


def _log_interval_sums(examples, target=None):
    """log sum over intervals (containing target) of e_j(w in range) / L^j; (grid, j)."""
    out = np.full((len(C_GRID), MAX_N + 1), -1e4)
    ex = np.array(examples)
    keep = np.ones(len(INT_LO), dtype=bool) if target is None else (INT_LO <= target) & (target <= INT_HI)
    lo, hi = INT_LO[keep], INT_HI[keep]
    inside = (ex[None, :] >= lo[:, None]) & (ex[None, :] <= hi[:, None])  # (intervals, n)
    for g, c in enumerate(C_GRID):
        w = _line(ex + 0.5, c) - _line(ex - 0.5, c)
        L = _line(hi + 0.5, c) - _line(lo - 0.5, c)
        r = np.where(inside, w[None, :] / L[:, None], 0.0)
        e = np.zeros((len(lo), MAX_N + 1))
        e[:, 0] = 1.0
        for i in range(len(ex)):  # elementary symmetric polynomials
            e[:, 1:] = e[:, 1:] + r[:, i:i + 1] * e[:, :-1]
        tot = e.sum(axis=0)
        out[g] = np.where(tot > 0, np.log(np.maximum(tot, 1e-300)), -1e4)
    return out


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules)
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))  # (sets,): n - 1
    set_log_sums = pm.Data("set_log_sums", np.zeros((1, len(C_GRID), MAX_N + 1)))
    cell_log_sums = pm.Data("cell_log_sums", np.zeros((1, len(C_GRID), MAX_N + 1)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa
    # Compression of the mental number line: 0 = logarithmic, 1 = linear.
    compression_logit = pm.Normal("compression_logit", mu=0.0, sigma=1.5)
    compression = pm.Deterministic("compression", 1.6 * pm.math.sigmoid(compression_logit))

    # Rules: examples drawn evenly over the rule's members.
    log_table = pt.logsumexp(
        _CONST + _betaln(u + _J_SHAPED, v + _N_MINUS_J) - _betaln(u, v), axis=3
    )  # (n, k, s)
    rule_ll = log_table[set_n_idx[:, None], rule_k, RULE_SIZE_IDX[None, :]]
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll

    # Intervals: interpolate the tabulated sums at the compression value.
    gw = -0.5 * ((compression - C_GRID) / C_STEP) ** 2
    gw = pt.exp(gw - pt.logsumexp(gw))  # (grid,)
    log_coef = _NJ_LOG100 + _betaln(u + _NJ_JS, v + _NJ_NMJ) - _betaln(u, v)  # (n, j)
    set_ls = pt.tensordot(gw, set_log_sums, axes=[[0], [1]])  # (sets, j)
    cell_ls = pt.tensordot(gw, cell_log_sums, axes=[[0], [1]])  # (cells, j)
    log_int_ev = pt.logsumexp(set_ls + log_coef[set_n_idx], axis=1)
    log_cell_ev = pt.logsumexp(cell_ls + log_coef[set_n_idx[cell_set]], axis=1)
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + log_int_ev

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    cell_int_share = pt.exp(pt.minimum(log_cell_ev - log_int_ev[cell_set], 0.0))
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + interval_post[cell_set] * cell_int_share
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_k = np.zeros((S, N_RULES), dtype="int64")
    set_n_idx = np.zeros(S, dtype="int64")
    set_log_sums = np.zeros((S, len(C_GRID), MAX_N + 1))
    for i, s in enumerate(unique):
        ex = np.array(s)
        set_n_idx[i] = len(ex) - 1
        rule_k[i] = RULE_MEMBERSHIP[:, ex - 1].sum(axis=1).astype("int64")
        set_log_sums[i] = _log_interval_sums(s)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_log_sums = np.stack([_log_interval_sums(unique[i], t) for i, t in cells])
    return {
        "rule_k": rule_k,
        "set_n_idx": set_n_idx,
        "set_log_sums": set_log_sums,
        "cell_log_sums": cell_log_sums,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
