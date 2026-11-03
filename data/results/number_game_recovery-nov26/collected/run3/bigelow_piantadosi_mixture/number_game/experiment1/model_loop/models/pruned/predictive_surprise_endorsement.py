"""People answer "would the program produce this number?" by imagining the
target as the program's next output and judging how expected it would be,
rather than by checking concept membership. The concept is learned by
size-principle Bayesian inference over 45 rules (each with its own prior
weight, a share lambda_0 for the rules as a group) and every interval of 1-100,
with a Beta(alpha, kappa) belief over the program's reliability (stray examples
drawn uniformly from 1-100) integrated out per set, exactly as in
uncertain_reliability_noise_concepts. The readout is changed: the posterior
predictive probability of the target as the next example, P(t | X) =
P(X, t) / P(X), is compared with chance (1/100), and
p_yes = sigmoid(c + beta * log(100 P(t | X))). Members of tight concepts are
therefore endorsed more than members of broad concepts that are just as surely
consistent with the examples.

With n examples of which k fall in a concept of size s,
p(X | h) = sum_j C(k, j) s^-j 100^-(n-j) B(u + j, v + n - j) / B(u, v),
which depends only on (n, k, s). The joint P(X, t | h) is the same table at
n + 1, with k + 1 if t is in h; intervals are counted by (k, s) per set and per
(set, target) in prepare_observed.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 5  # up to 4 examples plus the target


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

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Constant parts of the (n, k, s, j) likelihood table; n = 1..5, k, j = 0..5.
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
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES), dtype="int64"))  # (cells, rules)
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
    # Readout: criterion and sensitivity to how expected the target is.
    criterion = pm.Normal("criterion", mu=0.0, sigma=2.0)
    sensitivity = pm.LogNormal("sensitivity", mu=0.0, sigma=0.75)

    log_table = pt.logsumexp(
        _CONST + _betaln(u + _J_SHAPED, v + _N_MINUS_J) - _betaln(u, v), axis=3
    )  # (n, k, s)
    lik_table = pt.exp(log_table)
    log_prior_rules = pt.log(lambda_0) + pt.log(rule_weights)
    log_prior_int = pt.log1p(-lambda_0) - 2 * np.log(100)

    # Evidence of the examples alone, per set.
    rule_ll = log_table[set_n_idx[:, None], rule_k, RULE_SIZE_IDX[None, :]]
    set_lik = lik_table[set_n_idx]  # (sets, k, size)
    int_ev = (interval_count * set_lik).sum(axis=(1, 2))
    log_px = pt.logsumexp(
        pt.concatenate([log_prior_rules[None, :] + rule_ll, (log_prior_int + pt.log(int_ev))[:, None]], axis=1),
        axis=1,
    )

    # Evidence of the examples plus the target as one more output, per cell.
    cn = set_n_idx[cell_set] + 1
    ck = rule_k[cell_set] + cell_rule_member
    rule_ll_t = log_table[cn[:, None], ck, RULE_SIZE_IDX[None, :]]
    next_lik = lik_table[cn]  # (cells, k, size)
    in_t = cell_count[:, :MAX_N, :] * next_lik[:, 1:, :]
    out_t = (interval_count[cell_set] - cell_count) * next_lik
    int_ev_t = in_t.sum(axis=(1, 2)) + out_t.sum(axis=(1, 2))
    log_pxt = pt.logsumexp(
        pt.concatenate([log_prior_rules[None, :] + rule_ll_t, (log_prior_int + pt.log(int_ev_t))[:, None]], axis=1),
        axis=1,
    )

    log_ratio = log_pxt - log_px[cell_set] + np.log(100.0)  # log(100 P(t | X))
    p_cell = pm.math.sigmoid(criterion + sensitivity * log_ratio)
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell, 1e-9, 1 - 1e-9)[row_cell])
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
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]).astype("int64"),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
