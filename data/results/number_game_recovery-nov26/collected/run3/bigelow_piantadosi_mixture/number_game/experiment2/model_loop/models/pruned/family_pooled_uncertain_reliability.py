"""Refinement of uncertain_reliability_noise_concepts: Bayesian concept learning
over 45 rules and every interval of 1-100 (a share lambda_0 for the rules as a
group, prior (1 - lambda_0) / 100^2 for each interval), with the size principle
and stray examples drawn uniformly from 1-100, where the learner does not know
the reliability a (the chance an example comes from the concept) and integrates
a Beta(mean alpha, concentration kappa) belief over it for each example set.
The one change (from family_pooled_rule_prior) is the rule prior: each rule's
log prior weight is drawn around its family's level (parity, primes, squares and
cubes, powers, multiples, final digit, contained digit) with an inferred spread
tau (non-centred partial pooling), so rules of one kind are similarly plausible
unless the data say otherwise. "Yes" is the posterior probability that the
concept contains the target.

With n examples of which k fall in a concept of size s,
p(X | h) = sum_j C(k, j) s^-j 100^-(n-j) B(u + j, v + n - j) / B(u, v),
u = alpha kappa, v = (1 - alpha) kappa; this depends only on (n, k, s), so a
table over (n, k, s) is computed once per gradient step and gathered, and
prepare_observed counts intervals by (k, s) per set and per (set, target).
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

# Constant parts of the (n, k, s, j) likelihood table; n = 1..4, k, j = 0..4.
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


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules)
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))  # (sets,): n - 1
    interval_count = pm.Data("interval_count", np.zeros((1, MAX_N + 1, 100)))  # (sets, k, size)
    cell_count = pm.Data("cell_count", np.zeros((1, MAX_N + 1, 100)))  # (cells, k, size): containing target
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Each rule's prior log-weight is drawn around its family's level, with an
    # inferred spread (non-centred): tau = 0 gives equal weights within a family.
    family_level = pm.Normal("family_level", mu=0.0, sigma=1.5, shape=len(FAMILIES))
    tau = pm.HalfNormal("tau", sigma=1.0)
    rule_dev = pm.Normal("rule_dev", mu=0.0, sigma=1.0, shape=N_RULES)
    rule_weights = pm.Deterministic(
        "rule_weights", pt.special.softmax(family_level[RULE_FAMILY] + tau * rule_dev)
    )
    # Mean believed reliability and how certain people are of it.
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    # log p(X | h) for every (n, k, s), marginalising the reliability.
    log_table = pt.logsumexp(
        _CONST + _betaln(u + _J_SHAPED, v + _N_MINUS_J) - _betaln(u, v), axis=3
    )  # (n, k, s)
    lik_table = pt.exp(log_table)

    rule_ll = log_table[set_n_idx[:, None], rule_k, RULE_SIZE_IDX[None, :]]  # (sets, rules)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll

    set_lik = lik_table[set_n_idx]  # (sets, k, size)
    int_ev = (interval_count * set_lik).sum(axis=(1, 2))
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    cell_int_share = (cell_count * set_lik[cell_set]).sum(axis=(1, 2)) / int_ev[cell_set]
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
    int_k = np.zeros((S, len(INT_LO)), dtype="int64")
    counts = np.zeros((S, MAX_N + 1, 100))
    for i, s in enumerate(unique):
        ex = np.array(s)
        set_n_idx[i] = len(ex) - 1
        rule_k[i] = RULE_MEMBERSHIP[:, ex - 1].sum(axis=1).astype("int64")
        int_k[i] = ((ex[None, :] >= INT_LO[:, None]) & (ex[None, :] <= INT_HI[:, None])).sum(axis=1)
        np.add.at(counts[i], (int_k[i], INT_SIZE_IDX), 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_count = np.zeros((len(cells), MAX_N + 1, 100))
    for j, (i, t) in enumerate(cells):
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(cell_count[j], (int_k[i][contains], INT_SIZE_IDX[contains]), 1.0)
    return {
        "rule_k": rule_k,
        "set_n_idx": set_n_idx,
        "interval_count": counts,
        "cell_count": cell_count,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
