"""Digit-position binding errors in checking final-digit rules.

People do Bayesian concept learning over 45 mathematical rules (plausibility
set by kind of rule: one inferred level per family -- parity, primes, squares
and cubes, powers, multiples, final digit, contained digit) and every
magnitude interval of 1-100 (prior (1 - lambda_0) / 100^2 each), with the size
principle and an uncertain belief over the program's reliability a (Beta with
mean alpha and concentration kappa, integrated out per set; a stray is a
uniform draw from 1-100). The one distortion is a feature-binding error when
they verify whether the target obeys a final-digit rule: they register that
it contains the digit d but sometimes lose track of where, so a number
whose tens digit (and not its ones digit) is d is taken to "end in d" with an
inferred probability eps (an illusory conjunction). The examples themselves
are read correctly and the concept is inferred from them as an ideal learner
would; the error arises when the target is checked against a remembered
final-digit rule, which is held as "a d" with its position only partly bound.
All other rules and the magnitude ranges are checked exactly.

p(X | h) = E_a prod_i (a 1[x_i in h] / s + (1 - a) / 100)
         ∝ sum_j s^-j C(k, j) 100^j B(u + j, v + n - j) / B(u, v),
with k the number of examples h contains (checked exactly), and
"Yes" = sum_h post(h) m_h(target), where m_h(t) = 1 for members, eps for a
target with d only as its tens digit under "ends in d", and 0 otherwise.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4


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

# Binding-confusable: under "ends in d", numbers with d as tens digit but not ones digit.
BIND = np.array(
    [[1.0 if (r.startswith("ends in") and x % 10 != int(r[-1]) and str(r[-1]) in str(x)) else 0.0 for x in NUMBERS]
     for r in RULE_NAMES]
)
RULE_SIZE_IDX = RULE_SIZE.astype("int64") - 1

_N = np.arange(1, MAX_N + 1, dtype="float64")[:, None]
_J = np.arange(MAX_N + 1, dtype="float64")[None, :]
_VALID = _J <= _N
_N_MINUS_J = np.maximum(_N - _J, 0.0)
_SIZES = np.arange(1, 101, dtype="float64")
_SIZE_POW = _SIZES[:, None] ** -np.arange(MAX_N + 1)[None, :]  # (size, j)
_KJ_BINOM100 = np.array(
    [[math.comb(k, j) * 100.0**j for j in range(MAX_N + 1)] for k in range(MAX_N + 1)]
)  # (k, j)
_LOG_SHIFT = np.arange(1, MAX_N + 1) * np.log(10.0)  # per-n scale so interval likelihoods stay O(1)


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules): member examples
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))  # (sets,): n - 1
    ks_count = pm.Data("ks_count", np.zeros((1, (MAX_N + 1) * 100)))  # (sets, k*100+size-1): interval counts
    cell_ks_count = pm.Data("cell_ks_count", np.zeros((1, (MAX_N + 1) * 100)))  # same, containing target
    cell_set_onehot = pm.Data("cell_set_onehot", np.ones((1, 1)))  # (cells, sets)
    set_n_onehot = pm.Data("set_n_onehot", np.ones((1, MAX_N)))  # (sets, n)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_bind = pm.Data("cell_bind", np.zeros((1, N_RULES)))  # (cells, rules): target confusable
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Family-only rule prior: every rule of a family equally plausible
    # (sum-to-zero levels, so the softmax has no unidentified shift).
    family_level = pm.ZeroSumNormal("family_level", sigma=1.5, shape=len(FAMILIES))
    rule_weights = pm.Deterministic("rule_weights", pt.special.softmax(family_level[RULE_FAMILY]))
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    # Probability that a number with d only as its tens digit passes as "ends in d".
    eps = pm.Beta("eps", 1.0, 3.0)

    # Reliability-marginal weights c_j = B(u + j, v + n - j) / B(u, v) for each n.
    c_table = pt.where(_VALID, pt.exp(_betaln(u + _J, v + _N_MINUS_J) - _betaln(u, v)), 0.0)

    # Intervals: with uniform strays an interval's likelihood depends only on
    # (n, k examples inside, size), summed with the per-set and per-cell
    # counts of intervals in each (k, size) cell.
    # L(n, k, s) = sum_j C(k, j) 100^j s^-j c_j(n), the common factor dropped.
    log_L_ks = pt.log(pt.dot(c_table[:, None, :] * _KJ_BINOM100[None, :, :], _SIZE_POW.T))  # (n, k, size)
    log_L_flat = log_L_ks.reshape((MAX_N, (MAX_N + 1) * 100))
    lik_int = pt.exp(log_L_flat - _LOG_SHIFT[:, None])  # (n, 500), scaled to O(1)
    int_ev = (pt.dot(ks_count, lik_int.T) * set_n_onehot).sum(axis=1)  # (sets,)
    log_intervals = (
        pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev) + pt.as_tensor_variable(_LOG_SHIFT)[set_n_idx]
    )

    # Rules: the same table, at the rule's size and its count of member examples
    # (examples are checked exactly).
    log_rule_lik = log_L_ks[set_n_idx[:, None], rule_k, RULE_SIZE_IDX[None, :]]  # (sets, rules)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + log_rule_lik

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    # Cells read their set's posterior through a one-hot (cells, sets) matrix.
    cell_post = pt.dot(cell_set_onehot, pt.concatenate([rule_post, interval_post[:, None], int_ev[:, None]], axis=1))
    cell_int_ev = (pt.dot(cell_ks_count, lik_int.T) * pt.dot(cell_set_onehot, set_n_onehot)).sum(axis=1)
    target_member = cell_rule_member + eps * cell_bind  # (cells, rules)
    in_concept = (cell_post[:, :N_RULES] * target_member).sum(axis=1) + cell_post[:, N_RULES] * (
        cell_int_ev / cell_post[:, N_RULES + 1]
    )
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    example_idx = np.zeros((S, MAX_N), dtype="int64")
    example_mask = np.zeros((S, MAX_N))
    set_n_idx = np.zeros(S, dtype="int64")
    ks_count = np.zeros((S, (MAX_N + 1) * 100))
    set_k = []
    for i, s in enumerate(unique):
        ex = np.sort(np.array(s))
        n = len(ex)
        set_n_idx[i] = n - 1
        example_idx[i, :n] = ex - 1
        example_mask[i, :n] = 1.0
        k = ((ex[None, :] >= INT_LO[:, None]) & (ex[None, :] <= INT_HI[:, None])).sum(axis=1)
        set_k.append(k)
        np.add.at(ks_count[i], k * 100 + INT_SIZE_IDX, 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_ks_count = np.zeros((len(cells), (MAX_N + 1) * 100))
    for j, (i, t) in enumerate(cells):
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(cell_ks_count[j], set_k[i][contains] * 100 + INT_SIZE_IDX[contains], 1.0)
    return {
        "rule_k": (RULE_MEMBERSHIP[:, example_idx].transpose(1, 0, 2) * example_mask[:, None, :])
        .sum(axis=2)
        .astype("int64"),
        "set_n_idx": set_n_idx,
        "ks_count": ks_count,
        "cell_ks_count": cell_ks_count,
        "cell_set_onehot": np.eye(S)[[i for i, _ in cells]],
        "set_n_onehot": np.eye(MAX_N)[set_n_idx],
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "cell_bind": np.stack([BIND[:, t - 1] for _, t in cells]),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
