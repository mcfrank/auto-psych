"""Noisy-criterion concept judgment.

People form a graded belief p that the target belongs to the hidden concept by
simple Bayesian concept learning (45 rules whose prior is set by rule family,
every interval of 1-100, size principle, one inferred chance eps that an
example is a stray draw from 1-100), but they answer "yes" only when p clears
a decision criterion c that varies from trial to trial, c ~ Beta(mu*kappa,
(1-mu)*kappa). So P(yes) = P(c < p) = I_p(mu*kappa, (1-mu)*kappa).
mu*kappa = (1-mu)*kappa = 1 (uniform criterion) is probability matching; a
concentrated criterion gives near all-or-none answers around mu.
"""

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
_N = np.arange(MAX_N + 1, dtype="float64")[:, None, None]
_K = np.arange(MAX_N + 1, dtype="float64")[None, :, None]
_S = np.arange(1, 101, dtype="float64")[None, None, :]
_NMK = np.maximum(_N - _K, 0.0)

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules)
    set_n = pm.Data("set_n", np.zeros(1, dtype="int64"))
    interval_count = pm.Data("interval_count", np.zeros((1, (MAX_N + 1) * 100)))  # (sets, k*size)
    cell_count = pm.Data("cell_count", np.zeros((1, (MAX_N + 1) * 100)))  # (cells, k*size)
    set_n_onehot = pm.Data("set_n_onehot", np.zeros((1, MAX_N + 1)))  # (sets, n)
    cell_n_onehot = pm.Data("cell_n_onehot", np.zeros((1, MAX_N + 1)))  # (cells, n)
    cell_set_onehot = pm.Data("cell_set_onehot", np.zeros((1, 1)))  # (cells, sets)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_level = pm.ZeroSumNormal("family_level", sigma=1.5, shape=len(FAMILIES))
    rule_weights = pt.special.softmax(family_level[RULE_FAMILY])
    eps = pm.Beta("eps", 2, 10)  # chance an example is a stray

    # log p(examples | concept) for n examples, k in concept, concept size s
    log_table = _K * pt.log((1 - eps) / _S + eps / 100.0) + _NMK * (pt.log(eps) - np.log(100.0))
    lik_table = pt.exp(log_table)

    rule_ll = log_table[set_n[:, None], rule_k, RULE_SIZE_IDX[None, :]]
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll
    lik_flat = lik_table.reshape((MAX_N + 1, (MAX_N + 1) * 100))  # (n, k*size)
    int_ev = (pt.dot(interval_count, lik_flat.T) * set_n_onehot).sum(axis=1)
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)
    cell_int = (pt.dot(cell_count, lik_flat.T) * cell_n_onehot).sum(axis=1)
    set_stats = pt.concatenate([rule_post, (interval_post / int_ev)[:, None]], axis=1)  # (sets, rules+1)
    cell_stats = pt.dot(cell_set_onehot, set_stats)
    belief = (cell_stats[:, :N_RULES] * cell_rule_member).sum(axis=1) + cell_stats[:, N_RULES] * cell_int
    belief = pt.clip(belief, 1e-9, 1 - 1e-9)

    # Trial-to-trial criterion c ~ Beta(mu*kappa, (1-mu)*kappa); yes iff belief > c.
    crit_mu = pm.Beta("crit_mu", 5, 5)
    log_kappa = pm.Normal("log_kappa", mu=np.log(2.0), sigma=0.7)
    kappa = pt.exp(log_kappa)
    p_cell = pt.betainc(crit_mu * kappa, (1 - crit_mu) * kappa, belief)
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_k = np.zeros((S, N_RULES), dtype="int64")
    set_n = np.zeros(S, dtype="int64")
    int_k = np.zeros((S, len(INT_LO)), dtype="int64")
    counts = np.zeros((S, MAX_N + 1, 100))
    for i, s in enumerate(unique):
        ex = np.array(s)
        set_n[i] = len(ex)
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
    cell_set = np.array([i for i, _ in cells], dtype="int64")
    eye = np.eye(MAX_N + 1)
    return {
        "rule_k": rule_k,
        "set_n": set_n,
        "interval_count": counts.reshape(S, -1),
        "cell_count": cell_count.reshape(len(cells), -1),
        "set_n_onehot": eye[set_n],
        "cell_n_onehot": eye[set_n[cell_set]],
        "cell_set_onehot": np.eye(S)[cell_set],
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
