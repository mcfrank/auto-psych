"""Location-biased range prior.

People infer the hidden program as a Bayesian concept learner over number rules
(a prior mass per family shared equally by its members, exception rules in
their own family, digit-sum rules "digits add up to s" as a family) and the 5,050 ranges of 1..100 (an
Erlang prior on width, the size principle, 10% of examples uniform noise). The
one claim: a range's prior plausibility is not the same wherever it sits on
the number line. People expect a magnitude-based program to live in a
particular region, so among ranges of the same width the prior rises or falls
exponentially with the range's midpoint, with direction and strength
(loc_slope) inferred from the data. Magnitude generalisation is therefore
asymmetric: after a single example or a tight cluster, targets on the favoured
side are endorsed more than equally distant targets on the other side, where
the best model generalises symmetrically.

Speed: interval likelihoods do not depend on parameters, so prepare_observed
precomputes, per set (and per observed set-target cell), sums over the ranges
of each width of exp(log-likelihood) times the Chebyshev coefficients of
exp(loc_slope * u) in loc_slope (u = scaled midpoint; degree 7 on |slope| <= 3,
relative error ~2e-5). The graph then works on (cells x 100 widths x 8) arrays
instead of (sets x 5050 ranges).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.polynomial import chebyshev

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
DEG = 7
SLOPE_RANGE = 3.0


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
    rules.update({f"digit sum {t}": [n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)})
    return rules


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("digit sum"):
        return 6
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
N_RULES = len(_RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in _RULES.values()], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)

LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS
U = (LO + HI) / 200.0 - 0.505  # range midpoint scaled to [-0.495, 0.495]
# exp(slope * u) ~= sum_k U_POW[u, k] * T_k(slope / SLOPE_RANGE), Chebyshev on
# |slope| <= 3 (max relative error ~2e-5).
_NODES = np.cos(np.pi * (np.arange(40) + 0.5) / 40)
U_POW = chebyshev.chebfit(_NODES, np.exp(SLOPE_RANGE * _NODES[:, None] * U[None, :]), DEG).T  # (ranges, k)
WIDTH_ONEHOT = np.eye(100)[WIDTH - 1]  # (ranges, widths)
# Mean over the ranges of each width of the coefficients: the location normaliser.
LOC_NORM_TABLE = (WIDTH_ONEHOT.T @ U_POW) / N_PER_WIDTH[:, None]  # (widths, k)


def interval_log_lik(examples):
    ll = np.zeros(len(LO))
    for x in examples:
        inside = (LO <= x) & (x <= HI)
        ll += np.log(ALPHA * inside / WIDTH + (1 - ALPHA) / 100)
    return ll


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules), minus set max
    set_width_pow = pm.Data("set_width_pow", np.zeros((1, 100, DEG + 1)))  # (sets, widths, k)
    cell_width_pow = pm.Data("cell_width_pow", np.zeros((1, 100, DEG + 1)))  # (cells, widths, k)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the seven rule families and of ranges.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    # Characteristic range width of the Erlang size prior.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    # Location bias: among ranges of one width, prior proportional to exp(slope * midpoint).
    loc_slope = pm.Normal("loc_slope", mu=0.0, sigma=1.0)

    # Rules: family mass shared equally among the family's members.
    rule_prior = family_weight[:7][RULE_FAMILY] / FAMILY_SIZE[RULE_FAMILY]
    rule_weight = rule_prior[None, :] * pt.exp(rule_log_lik)  # (sets, rules)
    rule_total = rule_weight.sum(axis=1)
    rule_in = (rule_weight @ pt.as_tensor(RULE_MEMBERSHIP))[cell_set, cell_target]

    # Ranges: per-range prior = Erlang(width) * exp(slope * u) / mean_width exp(slope * u).
    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_range = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))  # (widths,)
    z = loc_slope / SLOPE_RANGE
    cheb = [pt.ones(()), z]
    for _ in range(DEG - 1):
        cheb.append(2.0 * z * cheb[-1] - cheb[-2])
    slope_pow = pt.stack(cheb)
    loc_norm = pt.dot(pt.as_tensor(LOC_NORM_TABLE), slope_pow)  # (widths,)
    width_weight = family_weight[7] * pt.exp(log_per_range) / loc_norm  # (widths,)
    int_total = pt.dot(pt.dot(set_width_pow, slope_pow), width_weight)  # (sets,)
    int_in = pt.dot(pt.dot(cell_width_pow, slope_pow), width_weight)  # (cells,)

    p_cell = (rule_in + int_in) / (rule_total + int_total)[cell_set]
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    rule_ll = np.zeros((len(unique), N_RULES))
    set_width_pow = np.zeros((len(unique), 100, DEG + 1))
    range_weights = []
    for i, s in enumerate(unique):
        r_ll = RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1)
        i_ll = interval_log_lik(s)
        top = max(r_ll.max(), i_ll.max())
        rule_ll[i] = r_ll - top
        e = np.exp(i_ll - top)  # (ranges,)
        range_weights.append(e)
        set_width_pow[i] = WIDTH_ONEHOT.T @ (e[:, None] * U_POW)
    cell_width_pow = np.zeros((len(cells), 100, DEG + 1))
    for j, (s, t) in enumerate(cells):
        e = range_weights[set_index[s]] * ((LO <= t) & (t <= HI))
        cell_width_pow[j] = WIDTH_ONEHOT.T @ (e[:, None] * U_POW)
    return {
        "rule_log_lik": rule_ll,
        "set_width_pow": set_width_pow,
        "cell_width_pow": cell_width_pow,
        "cell_set": np.array([set_index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
