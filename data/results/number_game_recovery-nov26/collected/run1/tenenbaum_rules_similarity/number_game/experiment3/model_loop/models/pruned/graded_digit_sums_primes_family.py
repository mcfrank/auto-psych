"""Graded digit-sum concepts with primes as their own family.

Refinement of `graded_prior_digit_sum_concepts`, adding one component from
`centre_weighted_ranges_primes_family`: people do Bayesian concept learning over
number rules, digit-sum rules ("digits add up to s", prior over s proportional
to exp(-decay * s)) and the 5,050 ranges of 1..100 (Erlang width prior, outputs
clustered toward the range's middle), with the size principle, a prior mass per
concept family (exception rules in their own family) and 10% of examples
uniform noise. The one change is that "prime numbers" leaves the shared
special-sets family (squares, cubes, powers) and forms its own family with a
separately inferred prior mass, so reading prime-like sets as "primes" no
longer depends on how plausible squares or powers are.

Implementation: the range block depends on only two parameters (the Erlang
width scale and the centre weight), so for each example set its total evidence
and its P(target in range | a range is the concept) are tabulated once, in
prepare_observed, as 2-D Chebyshev expansions in (log scale, centre weight);
the graph only evaluates the polynomials. The width scale's log has a normal
prior truncated to the expansion's domain.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.polynomial import chebyshev
from scipy.special import logsumexp, softmax

ALPHA = 0.9
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
    if name == "primes":
        return 7
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # squares, cubes, powers


_RULES = rule_concepts()
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
IS_DSUM = RULE_FAMILY == 6
DSUM_INDEX = np.where(IS_DSUM)[0]
DSUM_VALUE = np.arange(1, 19, dtype="float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS
CHEB_DEGREE = 6
# log(u * (1 - u)) at the k-th of W positions of a width-W range, u = (k - 0.5) / W.
_K = np.arange(1, 101)[None, :]
_U = np.clip((_K - 0.5) / WIDTHS[:, None], 1e-9, 1 - 1e-9)
SHAPE_TABLE = np.where(_K <= WIDTHS[:, None], np.log(_U * (1 - _U)), 0.0)
SHAPE_MASK = np.where(_K <= WIDTHS[:, None], 0.0, -1e4)


def example_log_lik_table(kappa):
    """Log P(example at position k of a width-W range), (W, k), for centre weight kappa."""
    log_density = kappa * SHAPE_TABLE - logsumexp(kappa * SHAPE_TABLE + SHAPE_MASK, axis=1, keepdims=True)
    return np.log(ALPHA * np.exp(log_density) + (1 - ALPHA) / 100)


# An interval's log likelihood is a smooth function of kappa alone, so each
# table entry is expanded in Chebyshev polynomials of 2 * kappa - 1 on [0, 1]
# (max error ~5e-5 nats); the last row is "outside the range".
_NODES = np.cos(np.pi * (np.arange(30) + 0.5) / 30)
_TABLES = np.stack([example_log_lik_table((x + 1) / 2).reshape(-1) for x in _NODES])
CHEB_TABLE = np.concatenate(
    [
        chebyshev.chebfit(_NODES, _TABLES, CHEB_DEGREE).T,
        np.eye(1, CHEB_DEGREE + 1) * np.log((1 - ALPHA) / 100),
    ]
)  # (100 * 100 + 1, degree + 1)
OUTSIDE = 100 * 100

# Domain of the range-block expansion: log width scale in [LS_LO, LS_HI]
# (scale 1.5 .. 600), centre weight in [0, 1].
LS_LO, LS_HI = np.log(1.5), np.log(600.0)
DEG_S, DEG_K = 36, 5
N_BASIS = (DEG_S + 1) * (DEG_K + 1)


def _cheb_nodes(n):
    return np.cos(np.pi * (np.arange(n) + 0.5) / n)


def range_block(interval_cheb, log_scale, kappa):
    """Exact (log evidence of ranges, P(target in range)) per set at one parameter point."""
    x = 2.0 * kappa - 1.0
    basis = chebyshev.chebvander(np.array([x]), CHEB_DEGREE)[0]
    log_lik = interval_cheb @ basis  # (sets, intervals)
    erlang = np.log(WIDTHS) - WIDTHS / np.exp(log_scale)
    log_prior = (erlang - logsumexp(erlang + np.log(N_PER_WIDTH)))[WIDTH - 1]
    log_joint = log_lik + log_prior[None, :]
    return logsumexp(log_joint, axis=1), softmax(log_joint, axis=1) @ INTERVAL_MEMBERSHIP


def range_block_expansion(interval_cheb):
    """2-D Chebyshev coefficients of the range block: (sets, basis) and (sets, 100, basis)."""
    ns, nk = _cheb_nodes(DEG_S + 4), _cheb_nodes(DEG_K + 4)
    log_z = np.zeros((len(interval_cheb), len(ns), len(nk)))
    member = np.zeros((len(interval_cheb), 100, len(ns), len(nk)))
    for a, xs in enumerate(ns):
        for b, xk in enumerate(nk):
            ls = LS_LO + (xs + 1) / 2 * (LS_HI - LS_LO)
            log_z[:, a, b], member[:, :, a, b] = range_block(interval_cheb, ls, (xk + 1) / 2)
    vs = chebyshev.chebvander(ns, DEG_S)  # (nodes_s, DEG_S + 1)
    vk = chebyshev.chebvander(nk, DEG_K)
    # Least-squares fit on the tensor grid: coef = pinv(vs) f pinv(vk)^T.
    ps, pk = np.linalg.pinv(vs), np.linalg.pinv(vk)
    cz = np.einsum("ia,sab,jb->sij", ps, log_z, pk).reshape(len(interval_cheb), N_BASIS)
    cm = np.einsum("ia,stab,jb->stij", ps, member, pk).reshape(len(interval_cheb), 100, N_BASIS)
    return cz, cm


def _cheb_basis(x, degree):
    terms = [pt.ones_like(x), x]
    for _ in range(degree - 1):
        terms.append(2.0 * x * terms[-1] - terms[-2])
    return pt.stack(terms)


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    range_logz_coef = pm.Data("range_logz_coef", np.zeros((1, N_BASIS)))  # (sets, basis)
    # One "cell" per distinct (set, target) pair asked about.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # set index of each cell
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # does each rule contain the target
    cell_range_coef = pm.Data("cell_range_coef", np.zeros((1, N_BASIS)))  # P(target in range) expansion
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the eight rule families (incl. exception rules, digit
    # sums and primes) and of ranges.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(9))
    log_w = pt.log(family_weight)
    # Characteristic range width of the Erlang size prior (on the log scale).
    log_interval_scale = pm.TruncatedNormal(
        "log_interval_scale", mu=np.log(20.0), sigma=1.0, lower=LS_LO, upper=LS_HI
    )
    # How strongly a range's outputs are expected to cluster at its centre.
    centre_weight = pm.Beta("centre_weight", alpha=1.5, beta=6.0)
    # Gradient of prior plausibility over digit sums (0 = equal share per sum).
    dsum_decay = pm.Normal("dsum_decay", mu=0.0, sigma=1.0)

    xs = 2.0 * (log_interval_scale - LS_LO) / (LS_HI - LS_LO) - 1.0
    xk = 2.0 * centre_weight - 1.0
    basis = (_cheb_basis(xs, DEG_S)[:, None] * _cheb_basis(xk, DEG_K)[None, :]).reshape((-1,))
    range_logz = pt.dot(range_logz_coef, basis)  # (sets,)
    range_member = pt.clip(pt.dot(cell_range_coef, basis), 0.0, 1.0)  # (cells,)

    # Within-family log prior: equal shares, except digit sums follow exp(-decay * s).
    dsum_log_share = -dsum_decay * DSUM_VALUE
    dsum_log_share = dsum_log_share - pt.logsumexp(dsum_log_share)
    within = pt.set_subtensor(pt.as_tensor(-np.log(FAMILY_SIZE)[RULE_FAMILY])[DSUM_INDEX], dsum_log_share)
    log_rules = (log_w[:8][RULE_FAMILY] + within)[None, :] + rule_log_lik
    log_ranges = log_w[8] + range_logz
    log_joint = pt.concatenate([log_rules, log_ranges[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_posterior = posterior[cell_set]  # (cells, rules + 1)
    in_concept = (cell_posterior[:, :N_RULES] * cell_rule_member).sum(axis=1) + cell_posterior[:, N_RULES] * range_member

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    interval_cheb = np.zeros((len(unique), len(LO), CHEB_DEGREE + 1))
    for i, s in enumerate(unique):
        for x in s:
            hit = (LO <= x) & (x <= HI)
            interval_cheb[i] += CHEB_TABLE[np.where(hit, (WIDTH - 1) * 100 + (x - LO), OUTSIDE)]
    logz_coef, member_coef = range_block_expansion(interval_cheb)
    cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: i for i, c in enumerate(unique_cells)}
    cell_set = np.array([c[0] for c in unique_cells], dtype="int64")
    cell_target = np.array([c[1] for c in unique_cells], dtype="int64")
    return {
        "rule_log_lik": np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "range_logz_coef": logz_coef,
        "cell_set": cell_set,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_target - 1].T.copy(),
        "cell_range_coef": member_coef[cell_set, cell_target - 1],
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
