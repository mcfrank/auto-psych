"""Refinement of family_pooled_win_stay_lose_sample: people learn the concept one
example at a time, holding a single working guess (a rule or magnitude range)
picked from a size-principle Bayesian posterior whose rule prior is organised by
family (each rule's log weight drawn around its family's level with spread tau,
non-centred). They keep the guess with probability `stay` while it still
contains each new example, and otherwise re-pick from the posterior given the
examples so far. "Yes" is the probability that the final guess contains the
target.

The one change (taken from the incumbent family_pooled_uncertain_reliability)
is the posterior people re-pick from: instead of a fixed 10% stray rate, they
are unsure of the program's reliability a, hold a Beta(mean alpha,
concentration kappa) belief over it, and integrate it out per prefix:
p(X | h) = sum_j C(k, j) s^-j 100^-(n-j) B(u + j, v + n - j) / B(u, v).
That depends only on (n, k, s), so prepare_observed counts intervals by (k, s)
for every quantity the guess recursion needs:
  D_t(h) = stay * c_t(h) * D_{t-1}(h) + R_t * P_t(h),
  R_t = 1 - stay * sum_h c_t(h) D_{t-1}(h).
Sets shorter than four examples are padded with dummy steps that every concept
contains and at which nobody re-samples.
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
    rule_k = pm.Data("rule_k", np.zeros((1, MAX_N, N_RULES), dtype="int64"))  # in rule among prefix t
    pref_n_idx = pm.Data("pref_n_idx", np.zeros((1, MAX_N), dtype="int64"))  # prefix length - 1
    rule_c = pm.Data("rule_c", np.ones((1, MAX_N, N_RULES)))  # rule contains example t (1 if dummy)
    active = pm.Data("active", np.ones((1, MAX_N)))  # step t is a real example
    int_count = pm.Data("int_count", np.zeros((1, MAX_N, MAX_N + 1, 100)))  # [s, t, k, size]
    cons_count = pm.Data("cons_count", np.zeros((1, MAX_N, MAX_N, MAX_N + 1, 100)))  # [s, t', t, k, size]
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_count = pm.Data("cell_count", np.zeros((1, MAX_N, MAX_N + 1, 100)))  # [c, t', k, size]
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
    # Probability of keeping a guess that is still consistent with a new example.
    stay_logit = pm.Normal("stay_logit", mu=0.0, sigma=1.5)
    stay = pm.Deterministic("stay", pm.math.sigmoid(stay_logit))
    # Mean believed reliability and how certain people are of it.
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    log_table = pt.logsumexp(
        _CONST + _betaln(u + _J_SHAPED, v + _N_MINUS_J) - _betaln(u, v), axis=3
    )  # (n, k, s)
    lik_table = pt.exp(log_table)

    # Posterior after each prefix: rules (sets, t, rules), intervals as a group (sets, t).
    rule_ll = log_table[pref_n_idx[:, :, None], rule_k, RULE_SIZE_IDX[None, None, :]]
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, None, :] + rule_ll
    pref_lik = lik_table[pref_n_idx]  # (sets, t, k, size)
    int_ev = (int_count * pref_lik).sum(axis=(2, 3))
    log_int = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)
    log_z = pt.logaddexp(pt.logsumexp(log_rules, axis=2), log_int)
    p_rule = pt.exp(log_rules - log_z[:, :, None])
    q_int = pt.exp(log_int - log_z)
    # Share of prefix-t' interval posterior on intervals containing examples t'+1..t.
    frac_cons = (cons_count * pref_lik[:, :, None]).sum(axis=(3, 4)) / int_ev[:, :, None]

    s_step = active * stay + (1 - active)  # padded steps: everyone stays
    d_rule = p_rule[:, 0, :]
    gains = [pt.ones_like(q_int[:, 0])]  # gains[t'] = R_t' * prod_{u in t'+1..t} s_u
    for t in range(1, MAX_N):
        consistent = (d_rule * rule_c[:, t, :]).sum(axis=1)
        for tp in range(t):
            consistent = consistent + gains[tp] * q_int[:, tp] * frac_cons[:, tp, t]
        r_t = active[:, t] * pt.clip(1 - s_step[:, t] * consistent, 0.0, 1.0)
        d_rule = s_step[:, t, None] * rule_c[:, t, :] * d_rule + r_t[:, None] * p_rule[:, t, :]
        gains = [g * s_step[:, t] for g in gains] + [r_t]

    gain_mat = pt.stack(gains, axis=1)  # (sets, t')
    cell_frac = (cell_count * pref_lik[cell_set]).sum(axis=(2, 3)) / int_ev[cell_set]
    in_concept = ((d_rule[cell_set] * cell_rule_member).sum(axis=1)
                  + (gain_mat[cell_set] * q_int[cell_set] * cell_frac).sum(axis=1))
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_k = np.zeros((S, MAX_N, N_RULES), dtype="int64")
    pref_n_idx = np.zeros((S, MAX_N), dtype="int64")
    rule_c = np.ones((S, MAX_N, N_RULES))
    active = np.zeros((S, MAX_N))
    int_count = np.zeros((S, MAX_N, MAX_N + 1, 100))
    cons_count = np.zeros((S, MAX_N, MAX_N, MAX_N + 1, 100))
    pref_k, later_in = {}, {}
    for i, s in enumerate(unique):
        ex = np.array(s)
        n = len(ex)
        inside = (ex[None, :] >= INT_LO[:, None]) & (ex[None, :] <= INT_HI[:, None])  # (intervals, n)
        for t in range(MAX_N):
            m = min(t + 1, n)
            active[i, t] = 1.0 if t < n else 0.0
            pref_n_idx[i, t] = m - 1
            rule_k[i, t] = RULE_MEMBERSHIP[:, ex[:m] - 1].sum(axis=1).astype("int64")
            if t < n:
                rule_c[i, t] = RULE_MEMBERSHIP[:, ex[t] - 1]
            k = inside[:, :m].sum(axis=1)
            pref_k[i, t] = k
            np.add.at(int_count[i, t], (k, INT_SIZE_IDX), 1.0)
            for t2 in range(t + 1, MAX_N):
                cont = inside[:, t + 1:min(t2 + 1, n)].all(axis=1)
                np.add.at(cons_count[i, t, t2], (k[cont], INT_SIZE_IDX[cont]), 1.0)
            later_in[i, t] = inside[:, t + 1:n].all(axis=1)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_count = np.zeros((len(cells), MAX_N, MAX_N + 1, 100))
    for j, (i, tg) in enumerate(cells):
        has_t = (INT_LO <= tg) & (tg <= INT_HI)
        for t in range(MAX_N):
            sel = has_t & later_in[i, t]
            np.add.at(cell_count[j, t], (pref_k[i, t][sel], INT_SIZE_IDX[sel]), 1.0)
    return {
        "rule_k": rule_k, "pref_n_idx": pref_n_idx, "rule_c": rule_c, "active": active,
        "int_count": int_count, "cons_count": cons_count, "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_count": cell_count,
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
