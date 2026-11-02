"""Random-looking strays (subjective randomness).

People do Bayesian concept learning over 45 mathematical rules and every
magnitude interval of 1-100 (size principle; rules as a group get share
lambda_0, split equally within each kind of rule by one inferred level per
family; each interval gets (1 - lambda_0) / 100^2), trusting the first example
with its own reliability a1 and pooling the later ones into an uncertain
Beta(alpha, kappa) reliability. Their idea of a stray follows subjective
randomness: a stray "should look random", so stray numbers are expected to be
unremarkable rather than familiar. The stray density is
q(x) proportional to exp(-gamma * f(x)), where f is the standardised log
"familiarity" of x (probability of x when one picks one of the 45 rules at
random and then one of its members). gamma > 0 (inferred) means familiar
off-pattern examples are hard to write off as strays. "Yes" is the posterior
probability that the concept contains the target.

For the later examples (m <= 3) a concept's evidence depends on which later
examples it contains (bitmask A), its size s and the q values:
T(A, s) = sum_{J subset A} s^-|J| prod_{i not in J} q(x_i) B(u+|J|, v+m-|J|)/B(u, v).
prepare_observed counts intervals by (first in, A, s) per set and per cell.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_M = 3
N_PAT = 2**MAX_M


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
INT_SIZE_IDX = INT_HI - INT_LO

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

# Familiarity of each number: log P(x) under "pick a rule, then a member".
_fam = np.log((RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True)).mean(axis=0))
FAMILIARITY = (_fam - _fam.mean()) / _fam.std()

# Subset structure over the later examples' bitmasks.
_POP = np.array([bin(j).count("1") for j in range(N_PAT)])
_SUBSET = np.array([[(j & a) == j for j in range(N_PAT)] for a in range(N_PAT)])  # (A, J)
_IN_J = np.array([[(j >> i) & 1 for i in range(MAX_M)] for j in range(N_PAT)], dtype="float64")  # (J, i)
_LOG_S = np.log(np.arange(1, 101, dtype="float64"))


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


with pm.Model() as model:
    first_num = pm.Data("first_num", np.zeros(1, dtype="int64"))  # (sets,) x1 - 1
    later_num = pm.Data("later_num", np.zeros((1, MAX_M), dtype="int64"))  # (sets, 3) x - 1 (0 pad)
    later_valid = pm.Data("later_valid", np.zeros((1, MAX_M)))  # (sets, 3)
    set_m = pm.Data("set_m", np.zeros(1))  # (sets,) number of later examples
    j_valid = pm.Data("j_valid", np.zeros((1, N_PAT)))  # (sets, J): J uses only real examples
    rule_first = pm.Data("rule_first", np.zeros((1, N_RULES), dtype="int64"))
    rule_pat = pm.Data("rule_pat", np.zeros((1, N_RULES), dtype="int64"))
    # Sparse interval counts over the flattened (first in, A, s) table.
    int_nz_set = pm.Data("int_nz_set", np.zeros(1, dtype="int64"))
    int_nz_idx = pm.Data("int_nz_idx", np.zeros(1, dtype="int64"))
    int_nz_cnt = pm.Data("int_nz_cnt", np.zeros(1))
    cell_nz_cell = pm.Data("cell_nz_cell", np.zeros(1, dtype="int64"))
    cell_nz_idx = pm.Data("cell_nz_idx", np.zeros(1, dtype="int64"))
    cell_nz_cnt = pm.Data("cell_nz_cnt", np.zeros(1))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_level = pm.ZeroSumNormal("family_level", sigma=1.5, shape=len(FAMILIES))
    rule_weights = pm.Deterministic("rule_weights", pt.special.softmax(family_level[RULE_FAMILY]))
    first_logit = pm.Normal("first_logit", mu=2.2, sigma=1.5)
    a1 = pm.Deterministic("a1", pm.math.sigmoid(first_logit))
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa
    # Strength of the "strays look random" expectation.
    gamma = pm.HalfNormal("gamma", sigma=1.0)

    log_q = -gamma * FAMILIARITY
    log_q = log_q - pt.logsumexp(log_q)  # (100,)

    lq_later = log_q[later_num] * later_valid  # (sets, 3)
    L_J = pt.dot(lq_later, (1.0 - _IN_J).T)  # (sets, J): strays among the later examples
    bratio = _betaln(u + _POP[None, :], v + set_m[:, None] - _POP[None, :]) - _betaln(u, v)
    term = L_J + bratio + pt.log(j_valid + 1e-30)  # (sets, J)
    term = term[:, :, None] - _POP[None, :, None] * _LOG_S[None, None, :]  # (sets, J, s)
    # Sum over J subset of A in probability space (terms are >= ~e^-40, no underflow).
    later_lik = pt.tensordot(pt.exp(term), _SUBSET.astype("float64"), axes=[[1], [1]])  # (sets, s, A)
    later_log = pt.log(later_lik).dimshuffle(0, 2, 1)  # (sets, A, s)

    lq1 = log_q[first_num]  # (sets,)
    out1 = pt.log1p(-a1) + lq1
    first_log = pt.stack(
        [out1[:, None] + 0.0 * _LOG_S[None, :], pt.logaddexp(pt.log(a1) - _LOG_S[None, :], out1[:, None])],
        axis=1,
    )  # (sets, 2, s)
    log_table = first_log[:, :, None, :] + later_log[:, None, :, :]  # (sets, 2, A, s)
    log_table = pt.maximum(log_table, -700.0)
    lik_table = pt.exp(log_table)

    S_idx = pt.arange(log_table.shape[0])
    rule_ll = log_table[S_idx[:, None], rule_first, rule_pat, RULE_SIZE_IDX[None, :]]
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll

    lik_flat = lik_table.reshape((lik_table.shape[0], 2 * N_PAT * 100))
    n_sets = lik_table.shape[0]
    int_ev = pt.zeros(n_sets)[int_nz_set].inc(int_nz_cnt * lik_flat[int_nz_set, int_nz_idx])
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    cell_ev = pt.zeros(cell_set.shape[0])[cell_nz_cell].inc(
        cell_nz_cnt * lik_flat[cell_set[cell_nz_cell], cell_nz_idx]
    )
    cell_int_share = cell_ev / int_ev[cell_set]
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + interval_post[cell_set] * cell_int_share
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    first_num = np.zeros(S, dtype="int64")
    later_num = np.zeros((S, MAX_M), dtype="int64")
    later_valid = np.zeros((S, MAX_M))
    set_m = np.zeros(S)
    j_valid = np.zeros((S, N_PAT))
    rule_first = np.zeros((S, N_RULES), dtype="int64")
    rule_pat = np.zeros((S, N_RULES), dtype="int64")
    int_first = np.zeros((S, len(INT_LO)), dtype="int64")
    int_pat = np.zeros((S, len(INT_LO)), dtype="int64")
    counts = np.zeros((S, 2, N_PAT, 100))
    for i, s in enumerate(unique):
        ex = np.array(s)
        first, later = ex[0], ex[1:]
        m = len(later)
        first_num[i] = first - 1
        later_num[i, :m] = later - 1
        later_valid[i, :m] = 1.0
        set_m[i] = m
        j_valid[i] = [1.0 if j < 2**m else 0.0 for j in range(N_PAT)]
        rule_first[i] = RULE_MEMBERSHIP[:, first - 1].astype("int64")
        bits = 2 ** np.arange(m)
        rule_pat[i] = (RULE_MEMBERSHIP[:, later - 1] * bits[None, :]).sum(axis=1).astype("int64")
        int_first[i] = ((first >= INT_LO) & (first <= INT_HI)).astype("int64")
        inside = (later[None, :] >= INT_LO[:, None]) & (later[None, :] <= INT_HI[:, None])
        int_pat[i] = (inside * bits[None, :]).sum(axis=1)
        np.add.at(counts[i], (int_first[i], int_pat[i], INT_SIZE_IDX), 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_count = np.zeros((len(cells), 2, N_PAT, 100))
    for j, (i, t) in enumerate(cells):
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(cell_count[j], (int_first[i][contains], int_pat[i][contains], INT_SIZE_IDX[contains]), 1.0)
    flat_c = counts.reshape(S, -1)
    i_s, i_k = np.nonzero(flat_c)
    flat_cc = cell_count.reshape(len(cells), -1)
    c_s, c_k = np.nonzero(flat_cc)
    return {
        "int_nz_set": i_s.astype("int64"),
        "int_nz_idx": i_k.astype("int64"),
        "int_nz_cnt": flat_c[i_s, i_k],
        "cell_nz_cell": c_s.astype("int64"),
        "cell_nz_idx": c_k.astype("int64"),
        "cell_nz_cnt": flat_cc[c_s, c_k],
        "first_num": first_num,
        "later_num": later_num,
        "later_valid": later_valid,
        "set_m": set_m,
        "j_valid": j_valid,
        "rule_first": rule_first,
        "rule_pat": rule_pat,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
