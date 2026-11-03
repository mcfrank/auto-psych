"""First-example anchored reliability.

People do Bayesian concept learning over 45 mathematical rules and every
magnitude interval of 1-100 (size principle; rules as a group get share
lambda_0, organised by a family-pooled prior; each interval gets
(1 - lambda_0) / 100^2), but they treat the first example differently from
the rest: the first number shown anchors the concept and is trusted to come
from it with its own inferred reliability a1, while only the later examples
are pooled into an uncertain (Beta(mean alpha, concentration kappa)) belief
about how reliable the program is on that set. So the same set in a different
order generalises differently. "Yes" is the posterior probability that the
concept contains the target.

p(X | h) = [a1 1(x1 in h)/s + (1 - a1)/100] * T(m, k, s), where m = n - 1
later examples of which k fall in h, and
T(m, k, s) = sum_j C(k, j) s^-j 100^-(m-j) B(u + j, v + m - j) / B(u, v).
Everything depends only on (m, first_in, k, s), so a table is built once per
gradient step and gathered; prepare_observed counts intervals by
(first_in, k, s) per set and per (set, target).
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_M = 3  # later examples (sets have at most 4)


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
INT_SIZE_IDX = INT_HI - INT_LO  # size - 1

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

# Constant parts of the (m, k, s, j) table for the later examples; m, k, j = 0..3.
_M = np.arange(MAX_M + 1, dtype="float64")[:, None, None, None]
_K = np.arange(MAX_M + 1, dtype="float64")[None, :, None, None]
_S = np.arange(1, 101, dtype="float64")[None, None, :, None]
_J = np.arange(MAX_M + 1, dtype="float64")[None, None, None, :]
_VALID = (_J <= _K) & (_K <= _M)
_LOG_BINOM = np.where(
    _VALID,
    np.vectorize(lambda k, j: math.lgamma(k + 1) - math.lgamma(j + 1) - math.lgamma(max(k - j, 0) + 1))(_K, _J),
    -1e4,
)
_CONST = _LOG_BINOM - _J * np.log(_S) - np.maximum(_M - _J, 0.0) * np.log(100.0)
_M_MINUS_J = np.maximum(_M - _J, 0.0)
_J_SHAPED = _J + 0.0 * _M_MINUS_J
_LOG_S = np.log(np.arange(1, 101, dtype="float64"))


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


with pm.Model() as model:
    rule_first = pm.Data("rule_first", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules): x1 in rule
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules): later in rule
    set_m_idx = pm.Data("set_m_idx", np.zeros(1, dtype="int64"))  # (sets,): n - 1
    interval_count = pm.Data("interval_count", np.zeros((1, 2, MAX_M + 1, 100)))  # (sets, first, k, size)
    cell_count = pm.Data("cell_count", np.zeros((1, 2, MAX_M + 1, 100)))  # (cells, first, k, size)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_level = pm.Normal("family_level", mu=0.0, sigma=1.5, shape=len(FAMILIES))
    tau = pm.HalfNormal("tau", sigma=1.0)
    rule_dev = pm.Normal("rule_dev", mu=0.0, sigma=1.0, shape=N_RULES)
    rule_weights = pm.Deterministic(
        "rule_weights", pt.special.softmax(family_level[RULE_FAMILY] + tau * rule_dev)
    )
    # Trust in the first (anchoring) example.
    first_logit = pm.Normal("first_logit", mu=2.2, sigma=1.5)
    a1 = pm.Deterministic("a1", pm.math.sigmoid(first_logit))
    # Uncertain reliability for the later examples.
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    later_log = pt.logsumexp(
        _CONST + _betaln(u + _J_SHAPED, v + _M_MINUS_J) - _betaln(u, v), axis=3
    )  # (m, k, s)
    # First example: in the concept (index 1) or not (index 0).
    first_log = pt.stack(
        [
            pt.log1p(-a1) - np.log(100.0) + 0.0 * _LOG_S,
            pt.logaddexp(pt.log(a1) - _LOG_S, pt.log1p(-a1) - np.log(100.0)),
        ]
    )  # (2, s)
    log_table = first_log[None, :, None, :] + later_log[:, None, :, :]  # (m, first, k, s)
    lik_table = pt.exp(log_table)

    rule_ll = log_table[set_m_idx[:, None], rule_first, rule_k, RULE_SIZE_IDX[None, :]]  # (sets, rules)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll

    set_lik = lik_table[set_m_idx]  # (sets, first, k, size)
    int_ev = (interval_count * set_lik).sum(axis=(1, 2, 3))
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    cell_int_share = (cell_count * set_lik[cell_set]).sum(axis=(1, 2, 3)) / int_ev[cell_set]
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + interval_post[cell_set] * cell_int_share
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_first = np.zeros((S, N_RULES), dtype="int64")
    rule_k = np.zeros((S, N_RULES), dtype="int64")
    set_m_idx = np.zeros(S, dtype="int64")
    int_first = np.zeros((S, len(INT_LO)), dtype="int64")
    int_k = np.zeros((S, len(INT_LO)), dtype="int64")
    counts = np.zeros((S, 2, MAX_M + 1, 100))
    for i, s in enumerate(unique):
        ex = np.array(s)
        first, later = ex[0], ex[1:]
        set_m_idx[i] = len(later)
        rule_first[i] = RULE_MEMBERSHIP[:, first - 1].astype("int64")
        rule_k[i] = RULE_MEMBERSHIP[:, later - 1].sum(axis=1).astype("int64")
        int_first[i] = ((first >= INT_LO) & (first <= INT_HI)).astype("int64")
        int_k[i] = ((later[None, :] >= INT_LO[:, None]) & (later[None, :] <= INT_HI[:, None])).sum(axis=1)
        np.add.at(counts[i], (int_first[i], int_k[i], INT_SIZE_IDX), 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_count = np.zeros((len(cells), 2, MAX_M + 1, 100))
    for j, (i, t) in enumerate(cells):
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(
            cell_count[j], (int_first[i][contains], int_k[i][contains], INT_SIZE_IDX[contains]), 1.0
        )
    return {
        "rule_first": rule_first,
        "rule_k": rule_k,
        "set_m_idx": set_m_idx,
        "interval_count": counts,
        "cell_count": cell_count,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
